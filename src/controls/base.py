"""FROZEN control contract — do not change field names after Phase 0.

Every governance control plugs into the pipeline through the same tiny interface:
it receives the request context plus a list of chunks and returns a
:class:`ControlDecision` (chunks in -> decision + reasons out). Controls run in a
fixed order defined by :data:`STAGE_ORDER`:

    retrieval (C1)  ->  screen (C3)  ->  generate  ->  grounding (C2)  ->  log (C4)

C4 (audit) is special: it observes every step rather than filtering, so it is
invoked by the pipeline after each control via :meth:`Control.apply` with
``stage == "log"``. The audit-record schema it writes is frozen in
:mod:`src.controls.c4_audit`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class Chunk:
    """A retrieved passage plus its ACL / provenance metadata.

    Mirrors the Qdrant payload written during indexing (see src/ingest/index.py):
    ``dept``, ``classification``, ``sensitive``, ``is_poisoned`` and the
    ``allow_<role>`` booleans all live in ``metadata``.
    """

    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_poisoned(self) -> bool:
        return bool(self.metadata.get("is_poisoned", False))

    def allowed_for(self, role: str) -> bool:
        """True if this chunk's ACL grants the given role read access."""
        return bool(self.metadata.get(f"allow_{role}", False))


@dataclass
class RequestContext:
    """Everything a control needs about the current request."""

    actor_role: str          # caller identity: hr | finance | legal | engineering
    query: str
    config: Dict[str, Any]
    audit: Optional["object"] = None   # AuditLog instance (set by the pipeline)
    answer: Optional[str] = None       # populated after generation, for C2


@dataclass
class ControlDecision:
    """Uniform result every control returns.

    Attributes
    ----------
    allow:
        Whether the request may proceed past this control. ``False`` means the
        control hard-blocked the step (e.g. all candidates screened out).
    reasons:
        Human-readable justifications, recorded to the audit log.
    filtered_chunks:
        The chunks that survive this control. Downstream controls operate on
        these. For post-generation controls (grounding) this is the set of
        chunks that actually supported the answer.
    metadata:
        Control-specific extras (counts, scores, citations, ...).
    """

    allow: bool
    reasons: List[str] = field(default_factory=list)
    filtered_chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Fixed execution order. The runner uses this to sort active controls.
STAGE_ORDER = {"retrieval": 0, "screen": 1, "grounding": 2, "log": 3}


@runtime_checkable
class Control(Protocol):
    """The plug-in contract. Implementations live in c1_permission.py etc."""

    name: str      # short id, e.g. "c1_permission"
    stage: str     # one of STAGE_ORDER keys

    def apply(self, ctx: RequestContext, chunks: List[Chunk]) -> ControlDecision:
        ...
