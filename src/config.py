"""Central config loader.

Loads ``configs/default.yaml`` into a plain dict and resolves the handful of
paths the rest of the code needs. Everything downstream imports :func:`load_config`
so there is exactly one source of truth (the Colab notebook's scattered globals
are consolidated here).

Runtime profile
---------------
Switch between machines with the ``GRAG_PROFILE`` environment variable
(``local`` default, or ``colab``). The selected profile's ``paths`` block in
``configs/default.yaml`` overrides the defaults, so you can move to a Colab T4
without editing any code:

    import os; os.environ["GRAG_PROFILE"] = "colab"   # before importing src.*

Fine-grained overrides (win over the profile):
    GRAG_BASE_DIR     -> put data/qdrant/artifacts under one base dir
    GRAG_OLLAMA_HOST  -> point at a specific Ollama daemon
    GRAG_MODEL        -> use a different model tag (e.g. a bigger one on GPU)

The compute device is auto-detected (CUDA if available), so no flag is needed to
use the T4 GPU — embedding and any local torch models pick it up automatically.
"""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

# repo root = two levels up from this file (src/config.py -> src -> root)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "default.yaml")


def _apply_profile_and_env(cfg: dict) -> str:
    """Overlay the selected profile's paths + env overrides. Returns profile name."""
    profile = os.environ.get("GRAG_PROFILE", "local")
    profiles = cfg.get("profiles", {})
    if profile in profiles and isinstance(profiles[profile], dict):
        cfg.setdefault("paths", {}).update(profiles[profile].get("paths", {}))

    # single-knob base-dir override (handy for Colab + Google Drive)
    base = os.environ.get("GRAG_BASE_DIR")
    if base:
        cfg["paths"] = {
            "data_dir": base,
            "qdrant_dir": os.path.join(base, "qdrant"),
            "artifacts_dir": os.path.join(base, "artifacts"),
        }

    # LLM overrides
    if os.environ.get("GRAG_OLLAMA_HOST"):
        cfg.setdefault("llm", {})["host"] = os.environ["GRAG_OLLAMA_HOST"]
    if os.environ.get("GRAG_MODEL"):
        cfg.setdefault("llm", {})["model"] = os.environ["GRAG_MODEL"]
    return profile


@lru_cache(maxsize=8)
def _load_cached(path: str, profile_key: str, base_key: str,
                 host_key: str, model_key: str) -> dict:
    """Cache keyed on the env values so changing GRAG_* re-resolves the config."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    profile = _apply_profile_and_env(cfg)

    # resolve paths (absolute Colab paths pass through; relative ones anchor to ROOT)
    paths = cfg.setdefault("paths", {})
    for key, rel in list(paths.items()):
        absolute = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        paths[key] = absolute
        os.makedirs(absolute, exist_ok=True)

    cfg["_root"] = ROOT
    cfg["_profile"] = profile
    return cfg


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    """Load YAML config, apply the GRAG_* runtime profile/overrides, resolve paths."""
    return _load_cached(
        path,
        os.environ.get("GRAG_PROFILE", "local"),
        os.environ.get("GRAG_BASE_DIR", ""),
        os.environ.get("GRAG_OLLAMA_HOST", ""),
        os.environ.get("GRAG_MODEL", ""),
    )
