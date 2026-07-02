"""
ChromaDB wrapper — now indexes BUSINESSES (150K), not reviews (5.7M).

Two query patterns:
  query_businesses(text, n)      : semantic search over business documents
  query_by_category(cats, n)     : filter businesses by category string
"""

import os
import time
import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/processed/chroma")
COLLECTION = "yelp_businesses"

# On a scale-to-zero Cloud Run cold start the first touch of the baked Chroma
# files (image-layer lazy load) plus CPU pressure can exceed ChromaDB's internal
# SQLite connection-pool acquire timeout, surfacing as
# "pool timed out while waiting for an open connection". That failure is
# transient — the first attempt warms the files, so a bounded retry lets the
# next attempt connect. Costs nothing at idle; only ever runs during a cold boot.
CHROMA_INIT_RETRIES   = int(os.getenv("CHROMA_INIT_RETRIES", "4"))
CHROMA_INIT_BACKOFF_S = float(os.getenv("CHROMA_INIT_BACKOFF_S", "2.0"))


class VectorStore:
    """Singleton-safe ChromaDB wrapper for business-level semantic search."""

    def __init__(self):
        self._client = self._connect_with_retry()
        # ONNXMiniLM_L6_V2 uses onnxruntime instead of PyTorch — same model weights,
        # ~400 MB lighter. We embed queries manually and pass query_embeddings to
        # collection.query() so we never hit the name-mismatch conflict check that
        # chromadb raises when the persisted EF name (sentence_transformers) differs
        # from the new EF name (onnx_mini_lm_l6_v2).
        self._embed_fn = ONNXMiniLM_L6_V2()
        self._col = self._client.get_collection(name=COLLECTION)
        log.info(f"VectorStore ready — {self._col.count():,} businesses in '{COLLECTION}'")

    @staticmethod
    def _connect_with_retry():
        """Open the persistent client, retrying transient cold-start pool timeouts."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, CHROMA_INIT_RETRIES + 1):
            try:
                return chromadb.PersistentClient(path=CHROMA_DIR)
            except Exception as exc:                      # noqa: BLE001 — surface any init failure
                last_exc = exc
                log.warning(
                    "ChromaDB client init attempt %d/%d failed (%s)%s",
                    attempt, CHROMA_INIT_RETRIES, exc,
                    f"; retrying in {CHROMA_INIT_BACKOFF_S:.1f}s"
                    if attempt < CHROMA_INIT_RETRIES else "",
                )
                if attempt < CHROMA_INIT_RETRIES:
                    time.sleep(CHROMA_INIT_BACKOFF_S)
        raise RuntimeError(
            f"ChromaDB PersistentClient failed after {CHROMA_INIT_RETRIES} attempts"
        ) from last_exc

    def query_businesses(
        self,
        query: str,
        n: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search: returns n businesses most similar to query text.
        Each result: {business_id, name, categories, city, state, stars, review_count, distance}
        """
        embeddings = self._embed_fn([query])
        kwargs = dict(
            query_embeddings=embeddings,
            n_results=min(n, self._col.count()),
            include=["metadatas", "distances", "documents"],
        )
        if where:
            kwargs["where"] = where

        results = self._col.query(**kwargs)
        return _zip_results(results)

    def query_by_category(self, categories: list[str], n: int = 10) -> list[dict]:
        """
        Find businesses matching any of the given category strings.
        Falls back to semantic search if category filter returns nothing.
        """
        if not categories:
            return []

        query = ", ".join(categories)
        return self.query_businesses(query=query, n=n)

    def get_business(self, business_id: str) -> Optional[dict]:
        """Fetch a single business by ID."""
        try:
            result = self._col.get(
                ids=[business_id],
                include=["metadatas", "documents"],
            )
            if result["ids"]:
                return {"business_id": business_id, **result["metadatas"][0]}
        except Exception:
            pass
        return None


def _zip_results(results: dict) -> list[dict]:
    ids   = results.get("ids",       [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out = []
    for bid, meta, dist in zip(ids, metas, dists):
        out.append({
            "business_id": bid,
            "distance"   : round(float(dist), 4),
            **meta,
        })
    return out
