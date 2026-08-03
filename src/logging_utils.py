"""Shared logging setup.

Configures a named logger that writes to BOTH the console and a timestamped file
under ``artifacts/logs/`` so long-running steps (indexing, eval) leave an
inspectable trail. Import :func:`setup_logging` and call it once at the top of a
script; pass the returned logger down into the library functions.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Tuple


def setup_logging(
    log_dir: str,
    name: str = "governed_rag",
    filename: Optional[str] = None,
    level: int = logging.INFO,
) -> Tuple[logging.Logger, str]:
    """Return (logger, log_file_path). Logs to console + a timestamped file."""
    os.makedirs(log_dir, exist_ok=True)
    filename = filename or f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    path = os.path.join(log_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()          # avoid duplicate handlers on re-run
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Logging to %s", path)
    return logger, path


def get_logger(name: str = "governed_rag") -> logging.Logger:
    """Fetch the shared logger; falls back to a console logger if unconfigured."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    return logger
