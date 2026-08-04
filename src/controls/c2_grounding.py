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

_SENT_SPLIT  = re.compile(r"(?<=[.!?])\s+")
_BULLET      = re.compile(r"^[-*•·]\s+(.+)$")
_NUMBERED    = re.compile(r"^\d+[.)]\s+(.+)$")
# strips markdown: headers (#), bold/italic (*_), code (`), blockquote (>),
# inline images/links, and horizontal rules
_MD_STRIP    = re.compile(r"#{1,6}\s|[*_`>~]|!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)|-{3,}")
_MIN_CLAIM   = 10   # characters; shorter fragments are noise


def split_claims(answer: str) -> List[str]:
    """Split a model answer into verifiable atomic claims.

    Handles both markdown-formatted responses (bullet/numbered lists, headers)
    and plain prose.  Strategy:

    1. Scan every line for bullet/numbered list items — if any are found, each
       item is one claim (markdown symbols stripped).  Most MLX / Gemma responses
       are list-formatted, so this branch fires first.
    2. Fall back to sentence-boundary splitting on cleaned prose for answers
       written as plain paragraphs.

    A minimum length of 10 chars filters out stray punctuation / fragments.
    """
    if not answer or not answer.strip():
        return []

    claims: List[str] = []

    # Pass 1 — list items (catches bullet and numbered Gemma outputs)
    for line in answer.splitlines():
        line = line.strip()
        m = _BULLET.match(line) or _NUMBERED.match(line)
        if m:
            text = _MD_STRIP.sub("", m.group(1)).strip()
            if len(text) >= _MIN_CLAIM:
                claims.append(text)

    if claims:
        return claims

    # Pass 2 — plain prose: strip markdown symbols then split on sentence ends
    prose = _MD_STRIP.sub("", answer)
    prose = re.sub(r"\s+", " ", prose).strip()
    return [p.strip() for p in _SENT_SPLIT.split(prose) if len(p.strip()) >= _MIN_CLAIM]


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

        if not claims:
            return ControlDecision(
                allow=False,
                reasons=["no verifiable claims extracted — answer may be empty or unparseable"],
                filtered_chunks=[],
                metadata={"citations": [], "grounded_answer": "", "faithfulness": 0.0,
                          "accepted": 0, "rejected": 0},
            )

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

        faithfulness = len(accepted) / len(claims)
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
