"""Vector-store access layer.

Wraps the on-disk Qdrant client and the query embedder (notebook cell 9). Exposes
a single :meth:`VectorStore.search` that can retrieve either everything (governance
OFF baseline) or ACL-filtered candidates. The actual permission *policy* lives in
Control 1 (``c1_permission.py``); this module just knows how to build the Qdrant
filter and hydrate results into :class:`~src.controls.base.Chunk` objects.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import load_config
from ..controls.base import Chunk


class VectorStore:
    def __init__(self, cfg: dict | None = None):
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        import torch

        self.cfg = cfg or load_config()
        self.collection = self.cfg["vector_store"]["collection"]
        self.top_k = self.cfg["vector_store"]["top_k"]
        self.query_prefix = self.cfg["embedding"]["query_prefix"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(self.cfg["embedding"]["model"], device=device)
        self.client = QdrantClient(path=self.cfg["paths"]["qdrant_dir"])

    # --- embedding ---
    def embed_query(self, q: str) -> list:
        return self.embedder.encode(
            self.query_prefix + q, normalize_embeddings=True
        ).tolist()

    @staticmethod
    def _acl_filter(role: str):
        """Qdrant filter that keeps only chunks whose ACL grants ``role``."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[FieldCondition(key=f"allow_{role}", match=MatchValue(value=True))]
        )

    def _to_chunk(self, point) -> Chunk:
        m = point.payload
        return Chunk(
            chunk_id=m.get("chunk_id", str(point.id)),
            doc_id=m.get("doc_id", ""),
            title=m.get("title", ""),
            text=m.get("text", ""),
            score=float(point.score),
            metadata=m,
        )

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        top_k: Optional[int] = None,
        acl_filter: bool = False,
    ) -> List[Chunk]:
        """Retrieve chunks for ``query``.

        Parameters
        ----------
        role:
            The caller's role. Only used when ``acl_filter`` is True.
        acl_filter:
            When True, apply the permission filter *at the vector layer* (this is
            Control 1's enforcement point). When False, retrieve everything — the
            ungoverned baseline that leaks.
        """
        qv = self.embed_query(query)
        k = top_k or self.top_k
        query_filter = self._acl_filter(role) if (acl_filter and role) else None
        res = self.client.query_points(
            collection_name=self.collection,
            query=qv,
            limit=k,
            query_filter=query_filter,
        )
        return [self._to_chunk(p) for p in res.points]

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
