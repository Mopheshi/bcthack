"""
Pre-builds the user_id -> row_indices inverted index ONCE and saves it
to disk as a compact pickle. Containers can then load this in ~3 seconds
instead of rebuilding it on every restart (which takes 90-120s).

Run this ONCE after build_index.py:
    python -m scripts.build_persona_index

Output:
    data/processed/user_to_idx.pkl   (~30 MB)

The PersonaBuilder will automatically detect and use this file if it exists.
"""

import os
import time
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
REVIEWS_PARQ  = PROCESSED_DIR / "reviews.parquet"
INDEX_PKL     = PROCESSED_DIR / "user_to_idx.pkl"


def build():
    if not REVIEWS_PARQ.exists():
        log.error(f"reviews.parquet not found at {REVIEWS_PARQ}")
        log.error("Run scripts/extract_data.py first.")
        return

    log.info(f"Reading {REVIEWS_PARQ} (user_id column only) ...")
    t0 = time.time()

    # Read only the user_id column — much faster than reading the full file
    df = pd.read_parquet(REVIEWS_PARQ, columns=["user_id"])
    log.info(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    log.info("Building inverted index with numpy argsort ...")
    t0 = time.time()
    user_ids = df["user_id"].values
    n = len(user_ids)

    perm = np.argsort(user_ids, kind="stable")
    sorted_uids = user_ids[perm]
    change_points = np.concatenate(
        ([0], np.where(sorted_uids[1:] != sorted_uids[:-1])[0] + 1, [n])
    )

    user_to_idx = {}
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i + 1]
        uid = sorted_uids[start]
        user_to_idx[uid] = perm[start:end]

    log.info(f"  Built {len(user_to_idx):,} user entries in {time.time()-t0:.1f}s")

    log.info(f"Saving to {INDEX_PKL} ...")
    t0 = time.time()
    with open(INDEX_PKL, "wb") as f:
        # protocol 5 supports out-of-band buffers and is much faster for numpy arrays
        pickle.dump(user_to_idx, f, protocol=5)
    size_mb = INDEX_PKL.stat().st_size / (1024 * 1024)
    log.info(f"  Saved {size_mb:.1f} MB in {time.time()-t0:.1f}s")
    log.info("Done. PersonaBuilder will auto-detect and use this file.")


if __name__ == "__main__":
    build()
