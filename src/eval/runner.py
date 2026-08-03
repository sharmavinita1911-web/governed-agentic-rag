"""Ablation runner — the governance on/off experiment.

For each governance configuration (baseline, one-control-at-a-time, full) it runs
the test suite through the pipeline, then computes the safety metrics and the
latency/token overhead. Output is a metrics dict per configuration that feeds the
trade-off curve and the report.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import metrics as M

GOV_KEYS = ["c1_permission", "c2_grounding", "c3_injection", "c4_audit", "pii_redaction"]


def _gov(**on) -> Dict[str, bool]:
    cfg = {k: False for k in GOV_KEYS}
    cfg.update(on)
    return cfg


_FULL = _gov(c1_permission=True, c2_grounding=True,
             c3_injection=True, c4_audit=True, pii_redaction=True)

# The ablation grid: baseline, each safety-relevant control alone, and full stack.
CONFIGS = [
    ("baseline", _gov()),
    ("+C1 permission", _gov(c1_permission=True)),
    ("+C3 injection", _gov(c3_injection=True)),
    ("+C2 grounding", _gov(c2_grounding=True)),
    ("+C4 audit", _gov(c4_audit=True)),
    ("full governance", _FULL),
]

# Fast sanity check: only the two endpoints of the curve.
QUICK_CONFIGS = [
    ("baseline", _gov()),
    ("full governance", _FULL),
]


def run_ablation(
    pipeline,
    suite,
    cfg: dict,
    n_boundary: int = 10,
    n_poison: Optional[int] = None,
    qa_queries: Optional[List[tuple]] = None,
    compute_ragas: bool = True,
    configs: Optional[List[tuple]] = None,
    logger=None,
) -> Dict[str, dict]:
    """Execute each config and return {config_name: metrics}.

    ``qa_queries`` is a list of (question, role) used for faithfulness + overhead.
    ``configs`` overrides the ablation grid (defaults to the full :data:`CONFIGS`).
    """
    from ..logging_utils import get_logger
    log = logger or get_logger()

    boundary = suite.boundary[:n_boundary]
    poison = suite.poison[: n_poison] if n_poison else suite.poison
    grid = configs or CONFIGS
    log.info("Ablation: %d configs x (boundary=%d, poison=%d) ragas=%s",
             len(grid), len(boundary), len(poison), compute_ragas)

    out: Dict[str, dict] = {}
    for name, gov in grid:
        log.info("=== config: %s ===", name)
        boundary_results = [pipeline.run(b.query, b.caller_role, gov) for b in boundary]
        poison_results = [pipeline.run(p.query, p.caller_role, gov) for p in poison]
        # faithfulness/overhead use the answerable QA set (public-doc questions),
        # so governed configs produce grounded answers rather than refusals. Falls
        # back to the boundary generations only if no QA set is available.
        qa_source = qa_queries or getattr(suite, "qa", None)
        qa_results = (
            [pipeline.run(q, role, gov) for q, role in qa_source]
            if qa_source else boundary_results
        )

        row = {
            "governance": gov,
            "leak_rate": M.leak_rate(boundary_results, suite.acl_map),
            "injection_success": M.injection_success(poison_results),
            "poison_exposure": M.poison_exposure(poison_results),
            "audit_completeness": M.audit_completeness(
                boundary_results + poison_results + qa_results
            ),
            "overhead": M.overhead(qa_results),
        }
        if compute_ragas:
            row["faithfulness"] = M.compute_faithfulness(qa_results, cfg)
        out[name] = row
        log.info(
            "  leak=%.2f inj=%.2f poison_exp=%.2f audit=%.2f faith=%s lat=%.0fms tok=%.0f",
            row["leak_rate"], row["injection_success"], row["poison_exposure"],
            row["audit_completeness"],
            f"{row['faithfulness']:.2f}" if "faithfulness" in row else "-",
            row["overhead"]["latency_ms"], row["overhead"]["total_tokens"],
        )
    return out
