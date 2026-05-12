"""
scripts/build_index.py  (v2 — business-level indexing)
-------------------------------------------------------
WHAT CHANGED AND WHY:
  Previous version tried to index 5.7M reviews → ChromaDB compaction crash.
  This version indexes ~150K businesses instead.

WHY THIS IS ACTUALLY BETTER:
  - Task A (review sim): user history comes from parquet (fast pandas filter).
                         No need to embed 5.7M reviews for this.
  - Task B (recommend):  semantic search is over BUSINESSES, not reviews.
                         "Find places similar to this query" = business-level.
  - 150K docs indexes in ~5-10 min. 5.7M took hours and crashed.

CHECKPOINT / RESUME:
  Progress saved every 100 batches to build_index_checkpoint.json.
  If it crashes, re-run with --resume.

Run:
    python -m scripts.build_index               # auto-resume if checkpoint exists
    python -m scripts.build_index --fresh       # delete and start over
"""

import os
import json
import logging
import argparse
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR   = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
RAW_DIR         = Path(os.getenv("YELP_RAW_DIR",  "./data/raw"))
CHROMA_DIR      = Path(os.getenv("CHROMA_DIR",    "./data/processed/chroma"))
EMBED_MODEL     = os.getenv("EMBED_MODEL",         "all-MiniLM-L6-v2")

COLLECTION      = "yelp_businesses"
BATCH_SIZE      = 200
CHECKPOINT_FILE = PROCESSED_DIR / "build_index_checkpoint.json"
SNIPPET_CHARS   = 300


# ── Build rich business documents ─────────────────────────────────────────────

def build_business_documents() -> pd.DataFrame:
    import json as _json

    log.info("Loading business file ...")
    biz_records = []
    with open(RAW_DIR / "yelp_academic_dataset_business.json", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _json.loads(line)
            biz_records.append({
                "business_id" : r.get("business_id", ""),
                "name"        : r.get("name", ""),
                "categories"  : r.get("categories", "") or "",
                "city"        : r.get("city", ""),
                "state"       : r.get("state", ""),
                "stars"       : float(r.get("stars", 0)),
                "review_count": int(r.get("review_count", 0)),
                "is_open"     : int(r.get("is_open", 1)),
            })

    df_biz = pd.DataFrame(biz_records)
    log.info(f"  -> {len(df_biz):,} businesses loaded")

    log.info("Loading review snippets from parquet ...")
    df_rev = pd.read_parquet(
        PROCESSED_DIR / "reviews.parquet",
        columns=["business_id", "text", "stars"]
    )

    log.info("Aggregating snippets per business (top 3 reviews each) ...")
    snippets = (
        df_rev.groupby("business_id")["text"]
        .apply(lambda texts: " | ".join(str(t)[:SNIPPET_CHARS] for t in texts.head(3)))
        .reset_index()
        .rename(columns={"text": "review_snippets"})
    )

    df = df_biz.merge(snippets, on="business_id", how="left")
    df["review_snippets"] = df["review_snippets"].fillna("")
    df["document"] = df.apply(_build_doc_text, axis=1)

    log.info(f"Business documents ready: {len(df):,}")
    return df


def _build_doc_text(row) -> str:
    parts = [
        f"Business: {row['name']}",
        f"Categories: {row['categories']}",
        f"Location: {row['city']}, {row['state']}",
        f"Rating: {row['stars']} stars ({row['review_count']} reviews)",
    ]
    if row["review_snippets"]:
        parts.append(f"Review excerpts: {row['review_snippets']}")
    return "\n".join(parts)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        start = data.get("last_completed_batch", -1) + 1
        log.info(f"Checkpoint found — resuming from batch {start}")
        return start
    return 0


def save_checkpoint(batch_idx: int):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_completed_batch": batch_idx}, f)


def clear_checkpoint():
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


# ── Main index builder ─────────────────────────────────────────────────────────

def build_index(resume: bool = True):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    df    = build_business_documents()
    total = len(df)

    client   = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    start_batch = 0

    if resume and CHECKPOINT_FILE.exists():
        start_batch = load_checkpoint()
        try:
            collection = client.get_collection(
                name=COLLECTION,
                embedding_function=embed_fn,
            )
            log.info(f"Resuming — collection currently has {collection.count():,} docs")
        except Exception:
            log.warning("Checkpoint found but collection missing — starting fresh")
            start_batch = 0
            collection  = _fresh_collection(client, embed_fn)
    else:
        try:
            client.delete_collection(COLLECTION)
            log.info(f"Deleted old '{COLLECTION}' collection")
        except Exception:
            pass
        collection = _fresh_collection(client, embed_fn)
        clear_checkpoint()

    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    log.info(f"Indexing {total:,} businesses | {num_batches} batches of {BATCH_SIZE}")

    for batch_idx in range(start_batch, num_batches):
        s     = batch_idx * BATCH_SIZE
        e     = min(s + BATCH_SIZE, total)
        batch = df.iloc[s:e]

        try:
            collection.upsert(
                documents=batch["document"].tolist(),
                ids=batch["business_id"].tolist(),
                metadatas=batch[[
                    "name", "categories", "city", "state",
                    "stars", "review_count", "is_open",
                ]].fillna("").astype(str).to_dict(orient="records"),
            )
        except Exception as exc:
            log.error(f"Upsert failed at batch {batch_idx}: {exc}")
            save_checkpoint(batch_idx - 1)
            log.error("Re-run with --resume to continue from this point.")
            raise

        if batch_idx % 100 == 0:
            save_checkpoint(batch_idx)
            log.info(f"  [{100*e/total:5.1f}%] {e:,}/{total:,}")

    log.info(f"\n✅ Done — '{COLLECTION}' has {collection.count():,} documents")
    log.info(f"   Persisted to {CHROMA_DIR}")
    clear_checkpoint()

    # Sanity check
    log.info("\nSanity query — 'good pizza downtown':")
    results = collection.query(query_texts=["good pizza downtown"], n_results=3)
    for meta in results["metadatas"][0]:
        log.info(f"  -> {meta.get('name')} | {meta.get('city')} | {meta.get('categories','')[:50]}")


def _fresh_collection(client, embed_fn):
    return client.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh",  action="store_true", help="Delete index and start over")
    args = parser.parse_args()
    build_index(resume=not args.fresh)
