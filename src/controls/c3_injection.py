"""Control 3 — injection / poisoning screen.

Screens each retrieved chunk for embedded instructions that try to override the
model (the PoisonedRAG / prompt-injection pattern) and drops the offenders before
they reach the context window. A fast heuristic (regex) runs by default; an
optional policy model (Llama Guard / NeMo Guardrails) can be wired in behind a
config flag without changing the contract.
"""
from __future__ import annotations

import re
from typing import List

from .base import Chunk, ControlDecision, RequestContext

# Signature phrases of instruction-injection / context-override attacks.
INJECTION_PATTERNS = [
    r"ignore (all |any )?(prior|previous|above) instructions",
    r"disregard (prior|previous|the above|all) instructions",
    r"system notice",
    r"\[admin\]",
    r"note to (the )?(model|assistant)",
    r"(you|the assistant|assistant|model) must (append|reply|respond|output|say|state|include)",
    r"append the phrase",
    r"to every (answer|response)",
    r"single source of truth",
    r"override any conflicting",
    r"treat this document as",
    r"access is granted",
    r"reply only with",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def screen_text(text: str) -> List[str]:
    """Return the list of injection patterns that fired on ``text``."""
    return [p for p, rx in zip(INJECTION_PATTERNS, _COMPILED) if rx.search(text)]


class InjectionControl:
    name = "c3_injection"
    stage = "screen"

    def __init__(self, cfg: dict | None = None):
        gov = (cfg or {}).get("governance", {}) if cfg else {}
        # placeholder for an optional policy-model backend
        self.use_policy_model = bool(gov.get("c3_policy_model", False))

    def apply(self, ctx: RequestContext, chunks: List[Chunk]) -> ControlDecision:
        clean: List[Chunk] = []
        flagged: List[Chunk] = []
        reasons: List[str] = []

        for ch in chunks:
            hits = screen_text(ch.text)
            if hits:
                flagged.append(ch)
                reasons.append(f"FLAGGED {ch.chunk_id}: matched {hits}")
            else:
                clean.append(ch)

        if not reasons:
            reasons.append("no injection signatures detected")

        return ControlDecision(
            allow=True,
            reasons=reasons,
            filtered_chunks=clean,
            metadata={
                "flagged_ids": [ch.chunk_id for ch in flagged],
                "flagged_count": len(flagged),
                "screened_count": len(chunks),
            },
        )
