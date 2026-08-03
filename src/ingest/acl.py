"""ACL / RBAC overlay — the synthetic multi-department permission boundary.

Lifted from the notebook (cell 4). Each document is deterministically assigned a
department and a classification (public / internal / restricted) from a hash of
its title, which then decides the ``allow_<role>`` booleans. Determinism (via the
title hash) is what makes the ground-truth "who-can-see-what" map reproducible.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

# Frozen role set — must match configs/default.yaml and the ACL schema.
ROLES: List[str] = ["hr", "finance", "legal", "engineering"]


def _bucket(title: str, salt: str, mod: int) -> int:
    """Stable hash bucket in [0, mod)."""
    h = int(hashlib.md5((salt + title).encode()).hexdigest(), 16)
    return h % mod


def assign_acl(title: str, roles: List[str] = ROLES) -> Dict:
    """Deterministically assign ACL metadata to a document title.

    - ``public``     (score < 20): readable by every role
    - ``internal``   (20-69):      readable only by the owning department
    - ``restricted`` (>= 70):      owning department only, flagged sensitive
    """
    dept = roles[_bucket(title, "dept", len(roles))]
    r = _bucket(title, "class", 100)
    classification = "public" if r < 20 else ("internal" if r < 70 else "restricted")
    allowed = set(roles) if classification == "public" else {dept}
    meta = {
        "dept": dept,
        "classification": classification,
        "sensitive": classification == "restricted",
        "is_poisoned": False,
    }
    for role in roles:
        meta[f"allow_{role}"] = role in allowed
    return meta
