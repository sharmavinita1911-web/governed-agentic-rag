"""Control 2 — grounding / citation gate (Self-RAG-style).

Runs *after* generation. Splits the answer into claims and keeps only those that
a retrieved chunk actually supports; unsupported claims are hard-rejected. Each
accepted claim is annotated with the id of its supporting chunk (a citation).

Two verifier backends share one interface:
  - ``llm``  : ask the local Gemma model, per claim, "is this supported? which id?"
  - ``nli``  : a cross-encoder entailment model (fallback when the LLM judge is
               too slow/flaky). Selected via ``evaluation.faithfulness_backend``.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .base import Chunk, ControlDecision, RequestContext

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_claims(answer: str) -> List[str]:
    """Naive sentence splitter — one claim per sentence."""
    return [s.strip() for s in _SENT_SPLIT.split(answer.strip()) if s.strip()]


class GroundingControl:
    name = "c2_grounding"
    stage = "grounding"

    def __init__(self, llm=None, backend: str = "llm", nli_model: Optional[str] = None):
        self.llm = llm
        self.backend = backend
        self._nli = None
        self._nli_model_name = nli_model

    # --- verifier backends ---
    def _verify_llm_multi(self, claim: str, chunks: List[Chunk]) -> Optional[Chunk]:
        """One LLM call per claim: ask which passage (if any) supports it.

        This is O(claims) calls per answer instead of O(claims x chunks) — the
        difference between minutes and hours on a CPU-served model.
        """
        evidence = "\n".join(f"[{i}] {ch.text}" for i, ch in enumerate(chunks))
        prompt = (
            "You are a strict fact-checker. Below are numbered PASSAGES and a CLAIM.\n"
            "Reply with the single number of ONE passage that directly supports the "
            "claim, or NONE if no passage supports it. Reply with only the number or NONE.\n\n"
            f"PASSAGES:\n{evidence}\n\nCLAIM: {claim}\n\nSupporting passage number (or NONE):"
        )
        out = self.llm.generate(prompt).text.strip()
        m = re.search(r"\d+", out)
        if not m:
            return None
        idx = int(m.group())
        return chunks[idx] if 0 <= idx < len(chunks) else None

    def _verify_nli(self, claim: str, evidence: str) -> bool:
        if self._nli is None:
            from sentence_transformers import CrossEncoder
            self._nli = CrossEncoder(self._nli_model_name)
        # cross-encoder NLI returns [contradiction, entailment, neutral] logits
        scores = self._nli.predict([(evidence, claim)])
        label = int(scores[0].argmax()) if hasattr(scores[0], "argmax") else 0
        return label == 1  # entailment

    def _supported_by(self, claim: str, chunks: List[Chunk]) -> Optional[Chunk]:
        if not chunks:
            return None
        if self.backend == "nli":
            for ch in chunks:
                if self._verify_nli(claim, ch.text):
                    return ch
            return None
        return self._verify_llm_multi(claim, chunks)

    # --- contract ---
    def apply(self, ctx: RequestContext, chunks: List[Chunk]) -> ControlDecision:
        answer = ctx.answer or ""
        claims = split_claims(answer)
        citations: List[Tuple[str, str]] = []   # (claim, supporting_chunk_id)
        accepted: List[str] = []
        rejected: List[str] = []
        cited_chunks: dict = {}

        for claim in claims:
            support = self._supported_by(claim, chunks)
            if support is not None:
                accepted.append(claim)
                citations.append((claim, support.chunk_id))
                cited_chunks[support.chunk_id] = support
            else:
                rejected.append(claim)

        n = len(claims) or 1
        faithfulness = len(accepted) / n
        reasons = [f"grounded {len(accepted)}/{len(claims)} claims (backend={self.backend})"]
        for claim in rejected:
            reasons.append(f"REJECTED (ungrounded): {claim[:80]}")

        grounded_answer = " ".join(
            f"{c} [{cid}]" for c, cid in citations
        ) or "I could not find supported evidence to answer this."

        return ControlDecision(
            allow=len(accepted) > 0,          # hard-reject if nothing is grounded
            reasons=reasons,
            filtered_chunks=list(cited_chunks.values()),
            metadata={
                "citations": citations,
                "grounded_answer": grounded_answer,
                "faithfulness": faithfulness,
                "accepted": len(accepted),
                "rejected": len(rejected),
            },
        )
