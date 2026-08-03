"""Control 4 — tamper-evident audit logging.

A hash-chained, append-only log. Every retrieval and every control decision is
recorded; each record's ``hash`` covers the previous record's hash, so any edit,
deletion, or reordering breaks the chain and is detectable by :meth:`verify`.

FROZEN record schema (agreed in Phase 0):

    {seq, ts, actor_role, query, control, decision, reasons, chunk_ids, prev_hash, hash}

``hash = sha256(prev_hash + canonical_json(record_without_hash))``
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List

GENESIS = "0" * 64  # prev_hash of the very first record


def _canonical(record: Dict[str, Any]) -> str:
    """Deterministic JSON so the hash is stable across runs/machines."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_hash(record_without_hash: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(record_without_hash).encode()).hexdigest()


class AuditLog:
    """Append-only hash-chained log for one pipeline run/session."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    @property
    def _prev_hash(self) -> str:
        return self.records[-1]["hash"] if self.records else GENESIS

    def record(
        self,
        *,
        actor_role: str,
        query: str,
        control: str,
        decision: str,
        reasons: List[str],
        chunk_ids: List[str],
    ) -> Dict[str, Any]:
        """Append one record and return it."""
        body = {
            "seq": len(self.records),
            "ts": time.time(),
            "actor_role": actor_role,
            "query": query,
            "control": control,
            "decision": decision,
            "reasons": list(reasons),
            "chunk_ids": list(chunk_ids),
            "prev_hash": self._prev_hash,
        }
        body["hash"] = compute_hash(body)
        self.records.append(body)
        return body

    # --- integrity ---
    def verify(self) -> bool:
        """Recompute the whole chain; True iff every link is intact."""
        prev = GENESIS
        for rec in self.records:
            body = {k: v for k, v in rec.items() if k != "hash"}
            if body["prev_hash"] != prev:
                return False
            if compute_hash(body) != rec["hash"]:
                return False
            prev = rec["hash"]
        return True

    # --- persistence ---
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.records, fh, indent=2)

    def completeness(self, expected_retrievals: int) -> float:
        """Fraction of expected retrieval events that got logged (target: 1.0)."""
        logged = sum(1 for r in self.records if r["control"] == "retrieval")
        return logged / expected_retrievals if expected_retrievals else 1.0
