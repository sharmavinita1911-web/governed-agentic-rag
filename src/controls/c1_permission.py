"""Control 1 — permission-aware retrieval.

The headline control. Given the caller's role, keep only chunks whose ACL grants
that role access; everything else is a *blocked candidate*. The blocked set is
returned in the decision so the pipeline can write it to the audit log.

Enforcement point: in production this filter is pushed down to the vector layer
(see ``VectorStore.search(acl_filter=True)``) so unauthorised chunks never leave
the store. For the ablation we filter the *same* candidate pool post-retrieval so
the on/off comparison isolates exactly this control's effect — the reasons list
records that distinction.
"""
from __future__ import annotations

from typing import List

from .base import Chunk, ControlDecision, RequestContext


class PermissionControl:
    name = "c1_permission"
    stage = "retrieval"

    def apply(self, ctx: RequestContext, chunks: List[Chunk]) -> ControlDecision:
        allowed: List[Chunk] = []
        blocked: List[Chunk] = []
        for ch in chunks:
            (allowed if ch.allowed_for(ctx.actor_role) else blocked).append(ch)

        reasons = [
            f"role={ctx.actor_role}: {len(allowed)} allowed, {len(blocked)} blocked by ACL"
        ]
        for ch in blocked:
            reasons.append(
                f"BLOCKED {ch.chunk_id} (dept={ch.metadata.get('dept')}, "
                f"class={ch.metadata.get('classification')})"
            )

        return ControlDecision(
            allow=True,  # a filter, never a hard stop
            reasons=reasons,
            filtered_chunks=allowed,
            metadata={
                "blocked_ids": [ch.chunk_id for ch in blocked],
                "blocked_count": len(blocked),
                "allowed_count": len(allowed),
            },
        )
