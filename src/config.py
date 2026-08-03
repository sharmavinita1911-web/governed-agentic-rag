"""Central config loader.

Loads ``configs/default.yaml`` into a plain dict and resolves the handful of
paths the rest of the code needs. Everything downstream imports :func:`load_config`
so there is exactly one source of truth (the Colab notebook's scattered globals
are consolidated here).
"""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

# repo root = two levels up from this file (src/config.py -> src -> root)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "default.yaml")


@lru_cache(maxsize=4)
def load_config(path: str = DEFAULT_CONFIG) -> dict:
    """Load YAML config and resolve directory paths to absolute, creating them."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # resolve paths relative to repo root and make sure the dirs exist
    paths = cfg.setdefault("paths", {})
    for key, rel in list(paths.items()):
        absolute = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        paths[key] = absolute
        os.makedirs(absolute, exist_ok=True)

    cfg["_root"] = ROOT
    return cfg
