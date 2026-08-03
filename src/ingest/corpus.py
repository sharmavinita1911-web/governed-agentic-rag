"""HotpotQA loading, corpus assembly, and chunking.

Lifted from the Colab notebook (cells 2-3) with no logic changes — only the
hard-coded constants now come from the config.
"""
from __future__ import annotations

from typing import Dict, List


def load_hotpot(n: int):
    """Load the first ``n`` HotpotQA (distractor) validation questions.

    The dataset id has been namespaced differently over time on the HF Hub, so
    we try the known ids in order.
    """
    from datasets import load_dataset

    last = None
    for name in ["hotpot_qa", "hotpotqa/hotpot_qa"]:
        try:
            return load_dataset(name, "distractor", split=f"validation[:{n}]")
        except Exception as e:  # noqa: BLE001 - want to try the next id
            last = e
    raise RuntimeError(
        "Could not load HotpotQA. Check the current id on huggingface.co/datasets "
        f"and update the loop. Last error: {last}"
    )


def build_corpus(ds) -> Dict[str, str]:
    """Flatten the per-question context paragraphs into a title -> text corpus."""
    docs: Dict[str, str] = {}
    for row in ds:
        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            if title not in docs:
                docs[title] = " ".join(sentences).strip()
    # drop empties
    return {t: p for t, p in docs.items() if p}


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Sliding-window character chunking with overlap."""
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
