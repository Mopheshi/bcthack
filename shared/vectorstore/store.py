"""
shared/vectorstore/store.py
----------------------------
ChromaDB wrapper — now indexes BUSINESSES (150K), not reviews (5.7M).

Two query patterns:
  query_businesses(text, n)      : semantic search over business documents
  query_by_category(cats, n)     : filter businesses by category string
"""

import os
import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

CHROMA_DIR  = os.getenv("CHROMA_DIR",   "./data/processed/chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")
COLLECTION  = "yelp_businesses"


class VectorStore:
    """Singleton-safe ChromaDB wrapper for business-level semantic search."""

    def __init__(self):
        self._client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self._col = self._client.get_collection(
            name=COLLECTION,
            embedding_function=self._embed_fn,
        )
        log.info(f"VectorStore ready — {self._col.count():,} businesses in '{COLLECTION}'")

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
        kwargs = dict(
            query_texts=[query],
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
