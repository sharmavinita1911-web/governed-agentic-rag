"""PII detection / redaction over retrieved chunks.

Uses Microsoft Presidio when available; falls back to a small regex detector so
the pipeline still runs on a machine without the spaCy model installed. Wired as
an optional control step (``governance.pii_redaction``) that rewrites chunk text
with entities masked before it reaches the context window.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import List

from .base import Chunk, ControlDecision, RequestContext

# regex fallback covers the entity types our Faker probes generate
_REGEX = {
    "EMAIL_ADDRESS": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "PHONE_NUMBER": re.compile(r"\+?\d[\d\s().-]{7,}\d"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


class _RegexEngine:
    def redact(self, text: str) -> tuple[str, int]:
        n = 0
        for label, rx in _REGEX.items():
            text, k = rx.subn(f"<{label}>", text)
            n += k
        return text, n


class _PresidioEngine:
    def __init__(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def redact(self, text: str) -> tuple[str, int]:
        results = self.analyzer.analyze(text=text, language="en")
        anon = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anon.text, len(results)


def _build_engine():
    try:
        return _PresidioEngine()
    except Exception:  # noqa: BLE001 - spaCy model / presidio not present
        return _RegexEngine()


class PIIControl:
    name = "pii_redaction"
    stage = "screen"

    def __init__(self):
        self.engine = _build_engine()

    def apply(self, ctx: RequestContext, chunks: List[Chunk]) -> ControlDecision:
        redacted: List[Chunk] = []
        total = 0
        for ch in chunks:
            new_text, n = self.engine.redact(ch.text)
            total += n
            redacted.append(replace(ch, text=new_text))
        return ControlDecision(
            allow=True,
            reasons=[f"redacted {total} PII entities across {len(chunks)} chunks"],
            filtered_chunks=redacted,
            metadata={"pii_redacted": total, "engine": type(self.engine).__name__},
        )
