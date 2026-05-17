"""
Precomputes persona data at build time. Run once locally after extract_data.py.

Loading all 5.78M review rows at runtime meant 90-second boots, health-check
failures, and two containers being impossible on 16 GB. This script moves that
work offline, writing persona_store.parquet — one fully-formed persona per warm
user (those with >= MIN_REVIEWS_FOR_WARM_USER reviews). The service then loads
only this file (~150-250 MB) in 2-3 seconds.

Output — data/processed/persona_store.parquet:
    user_id           str
    review_count      int
    mean_stars        float
    std_stars         float
    rating_bias       str   (generous|neutral|critical)
    pct_5_star        float
    pct_1_star        float
    mean_word_count   float
    style_label       str   (concise|medium|verbose)
    top_categories    str   (JSON list)
    sample_reviews    str   (JSON list of short review snippets)
"""

import os
import json
import time
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
USERS_PARQ    = PROCESSED_DIR / "users.parquet"
REVIEWS_PARQ  = PROCESSED_DIR / "reviews.parquet"
STORE_PARQ    = PROCESSED_DIR / "persona_store.parquet"

MIN_WARM        = int(os.getenv("MIN_REVIEWS_FOR_WARM_USER", "5"))
# 3 samples is enough for style transfer in Task A; more inflates the store and the prompt.
SAMPLES_PER_USER = int(os.getenv("PERSONA_SAMPLES_PER_USER", "3"))
SNIPPET_LEN      = int(os.getenv("PERSONA_SNIPPET_LEN", "220"))


def _style_label(mean_wc: float) -> str:
    if mean_wc < 50:  return "concise"
    if mean_wc > 150: return "verbose"
    return "medium"


def build():
    if not USERS_PARQ.exists() or not REVIEWS_PARQ.exists():
        log.error("users.parquet or reviews.parquet missing. Run extract_data.py first.")
        return

    t0 = time.time()
    log.info("Loading users.parquet ...")
    users = pd.read_parquet(USERS_PARQ)
    log.info(f"  {len(users):,} users ({time.time()-t0:.1f}s)")

    warm = users[users["review_count"] >= MIN_WARM].copy()
    warm_ids = set(warm["user_id"].astype(str))
    log.info(f"  {len(warm):,} warm users (>= {MIN_WARM} reviews)")

    t0 = time.time()
    log.info("Loading reviews.parquet (text, stars, categories, date) ...")
    reviews = pd.read_parquet(
        REVIEWS_PARQ,
        columns=["user_id", "text", "stars", "categories", "date"],
    )
    log.info(f"  {len(reviews):,} rows ({time.time()-t0:.1f}s)")

    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews = reviews[reviews["user_id"].isin(warm_ids)]
    log.info(f"  {len(reviews):,} rows belong to warm users")

    t0 = time.time()
    log.info("Building inverted index (numpy argsort) ...")
    uids = reviews["user_id"].values
    perm = np.argsort(uids, kind="stable")
    sorted_uids = uids[perm]
    n = len(uids)
    change_points = np.concatenate(
        ([0], np.where(sorted_uids[1:] != sorted_uids[:-1])[0] + 1, [n])
    )
    user_to_pos = {}
    for i in range(len(change_points) - 1):
        s, e = change_points[i], change_points[i + 1]
        user_to_pos[sorted_uids[s]] = perm[s:e]
    log.info(f"  Indexed {len(user_to_pos):,} users ({time.time()-t0:.1f}s)")

    # Pre-extract numpy arrays for fast row access
    text_arr = reviews["text"].values
    star_arr = reviews["stars"].values
    cat_arr  = reviews["categories"].values
    date_arr = reviews["date"].values

    t0 = time.time()
    log.info("Composing personas ...")
    records = []
    warm_indexed = warm.set_index("user_id")

    for idx, (uid, positions) in enumerate(user_to_pos.items()):
        if idx % 50_000 == 0 and idx:
            log.info(f"  {idx:,}/{len(user_to_pos):,} personas ...")

        if uid not in warm_indexed.index:
            continue
        urow = warm_indexed.loc[uid]

        sub_dates = date_arr[positions]
        order = np.argsort(sub_dates)[::-1]          # newest first
        recent = positions[order][:SAMPLES_PER_USER]

        sample_reviews = [
            (str(text_arr[p])[:SNIPPET_LEN] if text_arr[p] is not None else "")
            for p in recent
        ]

        # up to 20 rows for category signal
        cat_window = positions[order][:20]
        cats = []
        for p in cat_window:
            cs = cat_arr[p]
            if isinstance(cs, str):
                cats.extend(c.strip() for c in cs.split(",") if c.strip())
        top_cats = [c for c, _ in Counter(cats).most_common(5)]

        mean_wc = float(urow.get("mean_word_count", 100) or 100)

        records.append({
            "user_id"        : uid,
            "review_count"   : int(urow.get("review_count", 0) or 0),
            "mean_stars"     : float(urow.get("mean_stars", 3.5) or 3.5),
            "std_stars"      : float(urow.get("std_stars", 1.0) or 1.0),
            "rating_bias"    : str(urow.get("rating_bias_label", "neutral") or "neutral"),
            "pct_5_star"     : float(urow.get("pct_5_star", 0.0) or 0.0),
            "pct_1_star"     : float(urow.get("pct_1_star", 0.0) or 0.0),
            "mean_word_count": mean_wc,
            "style_label"    : _style_label(mean_wc),
            "top_categories" : json.dumps(top_cats),
            "sample_reviews" : json.dumps(sample_reviews),
        })

    log.info(f"  Composed {len(records):,} personas ({time.time()-t0:.1f}s)")

    store = pd.DataFrame.from_records(records)
    store.to_parquet(STORE_PARQ, index=False, compression="zstd")
    size_mb = STORE_PARQ.stat().st_size / (1024 * 1024)
    log.info(f"Saved {STORE_PARQ}  ({size_mb:.1f} MB, {len(store):,} personas)")
    log.info("Done. The runtime PersonaBuilder will load ONLY this file.")


if __name__ == "__main__":
    build()
