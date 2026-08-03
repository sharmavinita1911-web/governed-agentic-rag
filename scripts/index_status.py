"""Report the state of the local Qdrant index.

Prints the total chunk count and a breakdown by classification / poison so you can
confirm the corpus + ACL overlay + attack set are all indexed.

Run this only when nothing else is using the store (the local Qdrant is
single-writer, so it will error if build_index.py or run_eval.py is still running).

Usage:
    python scripts/index_status.py
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402


def main():
    from qdrant_client import QdrantClient

    cfg = load_config()
    collection = cfg["vector_store"]["collection"]
    qdir = cfg["paths"]["qdrant_dir"]

    client = QdrantClient(path=qdir)
    if not client.collection_exists(collection):
        print(f"No collection '{collection}' at {qdir}. Run scripts/build_index.py first.")
        return

    total = client.count(collection_name=collection).count
    print(f"Collection '{collection}' @ {qdir}")
    print(f"  total chunks indexed: {total}")

    # scroll payloads for a quick breakdown
    cls = Counter()
    dept = Counter()
    poisoned = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            m = p.payload
            cls[m.get("classification", "?")] += 1
            dept[m.get("dept", "?")] += 1
            poisoned += 1 if m.get("is_poisoned") else 0
        if offset is None:
            break

    print(f"  poisoned chunks     : {poisoned}")
    print(f"  by classification   : {dict(cls)}")
    print(f"  by department       : {dict(dept)}")
    client.close()


if __name__ == "__main__":
    main()
