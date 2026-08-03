"""Run the governance ablation and write metrics + plots + a sample audit log.

Usage:
    python scripts/run_eval.py [--n-boundary 10] [--no-ragas]

Requires: a built index (scripts/build_index.py) and a running Ollama daemon with
the configured model pulled (`ollama pull hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M`).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config          # noqa: E402
from src.agent.graph import GovernedPipeline  # noqa: E402
from src.eval.plots import make_safety_bars, make_tradeoff_plot  # noqa: E402
from src.eval.runner import QUICK_CONFIGS, run_ablation  # noqa: E402
from src.eval.testsuite import build_suite    # noqa: E402
from src.logging_utils import setup_logging   # noqa: E402
from src.retrieval.store import VectorStore   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boundary", type=int, default=10)
    ap.add_argument("--no-ragas", action="store_true", help="skip RAGAS/faithfulness")
    ap.add_argument("--quick", action="store_true",
                    help="fast sanity run: only baseline vs full governance, tiny N, no RAGAS")
    args = ap.parse_args()

    # --quick trims everything for a fast first pass; explicit flags still win.
    n_boundary = 2 if args.quick else args.n_boundary
    n_poison = 2 if args.quick else None
    compute_ragas = not (args.no_ragas or args.quick)
    configs = QUICK_CONFIGS if args.quick else None

    cfg = load_config()
    artifacts = cfg["paths"]["artifacts_dir"]
    logger, log_path = setup_logging(os.path.join(artifacts, "logs"), filename="run_eval.log")

    store = VectorStore(cfg)
    pipeline = GovernedPipeline(store, cfg=cfg)

    logger.info("Building test suite from the indexed corpus...")
    suite = build_suite(store, n_boundary=max(n_boundary, 5), cfg=cfg)
    logger.info("  boundary=%d poison=%d pii_probes=%d",
                len(suite.boundary), len(suite.poison), len(suite.pii_probes))

    metrics = run_ablation(
        pipeline, suite, cfg,
        n_boundary=n_boundary,
        n_poison=n_poison,
        compute_ragas=compute_ragas,
        configs=configs,
        logger=logger,
    )

    # persist metrics (drop non-serializable audit objects)
    metrics_path = os.path.join(artifacts, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Wrote %s", metrics_path)

    tradeoff = make_tradeoff_plot(metrics, os.path.join(artifacts, "tradeoff.png"))
    bars = make_safety_bars(metrics, os.path.join(artifacts, "safety_bars.png"))
    logger.info("Wrote %s and %s", tradeoff, bars)

    # save one full-governance audit log as a tamper-evidence artifact
    sample = pipeline.run(suite.boundary[0].query, suite.boundary[0].caller_role,
                          {"c1_permission": True, "c2_grounding": True,
                           "c3_injection": True, "c4_audit": True, "pii_redaction": True})
    audit_path = os.path.join(artifacts, "audit_sample.json")
    sample.audit.save(audit_path)
    logger.info("Wrote %s (chain intact: %s)", audit_path, sample.audit.verify())
    logger.info("Full log written to %s", log_path)

    store.close()


if __name__ == "__main__":
    main()
