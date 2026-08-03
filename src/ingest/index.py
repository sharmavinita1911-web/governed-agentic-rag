"""Embed the corpus + poison set and build the on-disk Qdrant index.

Lifted from the notebook (cells 6 & 8). The one change from Colab: ``BASE`` now
points at the local ``data/`` directory (from config) instead of Google Drive, so
the whole thing runs on a laptop with no server and no Drive mount.
"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Tuple

import time

from ..config import load_config
from ..logging_utils import get_logger
from .acl import ROLES, assign_acl
from .corpus import build_corpus, chunk_text, load_hotpot
from .poison import build_poison_set


def _assemble_records(corpus: Dict[str, str], poison_docs, cfg) -> List[Tuple[str, str, Dict]]:
    """Turn the corpus + poison docs into (chunk_id, text, metadata) records."""
    c = cfg["corpus"]
    records: List[Tuple[str, str, Dict]] = []
    for title, text in corpus.items():
        base_meta = assign_acl(title)
        doc_id = "doc_" + hashlib.md5(title.encode()).hexdigest()[:10]
        for ci, chunk in enumerate(chunk_text(text, c["max_chars"], c["overlap"])):
            meta = dict(base_meta, doc_id=doc_id, title=title,
                        chunk_index=ci, source="hotpotqa")
            records.append((f"{doc_id}::{ci}", chunk, meta))

    for doc_id, text, meta in poison_docs:
        meta = dict(meta, doc_id=doc_id, title=doc_id, chunk_index=0, source="synthetic")
        records.append((f"{doc_id}::0", text, meta))
    return records


def build_index(cfg: dict | None = None, logger=None) -> dict:
    """Full ingestion + indexing pipeline. Returns a small summary dict.

    Logs each stage (corpus, poison, embedding progress, indexing) to the shared
    logger so a manual run leaves an inspectable trail in artifacts/logs/.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer
    import torch

    cfg = cfg or load_config()
    log = logger or get_logger()
    random.seed(cfg["corpus"]["seed"])

    # 1) corpus + attack set
    log.info("Loading HotpotQA subset (%d questions)...", cfg["corpus"]["subset"])
    ds = load_hotpot(cfg["corpus"]["subset"])
    corpus = build_corpus(ds)
    log.info("Corpus assembled: %d unique documents", len(corpus))
    poison_docs = (
        build_poison_set(cfg["poison"]["n_poison"], ROLES)
        if cfg["poison"]["enabled"] else []
    )
    log.info("Poison documents: %d", len(poison_docs))
    records = _assemble_records(corpus, poison_docs, cfg)
    log.info("Total chunks to embed: %d", len(records))

    # 2) embed (windowed so progress lands in the log file, not just a tqdm bar)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Embedding on %s with %s", device, cfg["embedding"]["model"])
    model = SentenceTransformer(cfg["embedding"]["model"], device=device)
    texts = [r[1] for r in records]
    embeddings = []
    window = 640
    t0 = time.perf_counter()
    for i in range(0, len(texts), window):
        part = texts[i:i + window]
        embeddings.extend(
            model.encode(part, normalize_embeddings=True, batch_size=64).tolist()
        )
        done = min(i + window, len(texts))
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed else 0
        log.info("  embedded %d/%d chunks (%.0f/s)", done, len(texts), rate)
    log.info("Embedding done in %.1fs", time.perf_counter() - t0)

    # 3) index into on-disk Qdrant (local, no server)
    collection = cfg["vector_store"]["collection"]
    log.info("Indexing into Qdrant collection '%s' at %s", collection, cfg["paths"]["qdrant_dir"])
    client = QdrantClient(path=cfg["paths"]["qdrant_dir"])
    dim = len(embeddings[0])
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Qdrant ids must be ints/UUIDs; keep the string chunk id + text in the payload.
    points = []
    for pid, (cid, text, meta) in enumerate(records):
        payload = dict(meta, chunk_id=cid, text=text)
        points.append(PointStruct(id=pid, vector=embeddings[pid], payload=payload))

    B = 500
    for i in range(0, len(points), B):
        client.upsert(collection_name=collection, points=points[i:i + B])
        log.info("  upserted %d/%d points", min(i + B, len(points)), len(points))

    count = client.count(collection_name=collection).count
    client.close()
    log.info("Indexing complete: %d chunks in '%s'", count, collection)

    summary = {
        "documents": len(corpus),
        "poison_docs": len(poison_docs),
        "chunks_indexed": count,
        "embedding_dim": dim,
        "device": device,
        "qdrant_dir": cfg["paths"]["qdrant_dir"],
    }
    return summary
