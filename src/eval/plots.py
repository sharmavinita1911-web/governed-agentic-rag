"""Plots — the safety-cost trade-off curve (the project's novel contribution).

``safety_score`` collapses the four safety metrics into one number so each
configuration is a single point of (cost, safety); the curve traces how safety
rises as governance overhead is added.
"""
from __future__ import annotations

from typing import Dict


def safety_score(row: dict) -> float:
    """Composite safety in [0,1]: higher is safer.

    Averages: no-leak (1-leak), poison-screened (1-poison_exposure), faithfulness,
    audit. Uses poison_exposure (deterministic C3 signal) rather than raw
    injection_success, which a robust small model keeps at 0 either way.
    """
    poison = row.get("poison_exposure", row.get("injection_success", 0.0))
    parts = [
        1.0 - row.get("leak_rate", 0.0),
        1.0 - poison,
        row.get("faithfulness", 0.0),
        row.get("audit_completeness", 0.0),
    ]
    return sum(parts) / len(parts)


def make_tradeoff_plot(metrics: Dict[str, dict], out_path: str, cost_key: str = "latency_ms"):
    """Scatter each config as (overhead cost, safety) and annotate it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(metrics.keys())
    costs = [metrics[n]["overhead"].get(cost_key, 0.0) for n in names]
    safety = [safety_score(metrics[n]) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(costs, safety, s=90, color="#2E6E68", zorder=3)
    for n, x, y in zip(names, costs, safety):
        ax.annotate(n, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel(f"Governance overhead ({cost_key})")
    ax.set_ylabel("Composite safety score (higher = safer)")
    ax.set_title("Governed Agentic RAG — safety vs cost trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_safety_bars(metrics: Dict[str, dict], out_path: str):
    """Grouped bar chart of the raw safety metrics per configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(metrics.keys())
    series = {
        "leak_rate": [metrics[n].get("leak_rate", 0.0) for n in names],
        "poison_exposure": [metrics[n].get("poison_exposure", 0.0) for n in names],
        "injection_success": [metrics[n].get("injection_success", 0.0) for n in names],
        "faithfulness": [metrics[n].get("faithfulness", 0.0) for n in names],
        "audit_completeness": [metrics[n].get("audit_completeness", 0.0) for n in names],
    }
    import numpy as np

    x = np.arange(len(names))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + (i - 2) * w, vals, w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("metric value")
    ax.set_title("Per-configuration governance metrics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
