"""Build the local Qdrant index from HotpotQA + the ACL/poison overlay.

Logs progress to the console and to a timestamped file under artifacts/logs/.

Usage:
    python scripts/build_index.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config          # noqa: E402
from src.ingest.index import build_index     # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402


def main():
    cfg = load_config()
    log_dir = os.path.join(cfg["paths"]["artifacts_dir"], "logs")
    logger, log_path = setup_logging(log_dir, filename="build_index.log")

    logger.info("Building index (downloads HotpotQA + embedding model on first run)...")
    summary = build_index(cfg, logger=logger)

    logger.info("Done:")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    logger.info("Full log written to %s", log_path)


if __name__ == "__main__":
    main()
