"""
scripts/extract_data.py
-----------------------
Parses all 5 Yelp JSON files, joins them, and outputs two clean parquet files:

  processed/reviews.parquet   — one row per review, enriched with user + biz metadata
  processed/users.parquet     — one row per user with computed behavioural features

Run:
    python -m scripts.extract_data
    # or
    python scripts/extract_data.py
"""

import json
import os
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR       = Path(os.getenv("YELP_RAW_DIR", "./data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))

FILES = {
    "business" : "yelp_academic_dataset_business.json",
    "user"     : "yelp_academic_dataset_user.json",
    "review"   : "yelp_academic_dataset_review.json",
    "tip"      : "yelp_academic_dataset_tip.json",
}


# ── Streaming helpers ────────────────────────────────────────────────────────

def stream_ndjson(path: Path, limit: int = None):
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit and count >= limit:
                break


# ── Step 1: Load businesses ──────────────────────────────────────────────────

def load_businesses() -> dict:
    """Returns dict: business_id → {name, categories, city, state, stars, review_count}"""
    path = RAW_DIR / FILES["business"]
    log.info(f"Loading businesses from {path} ...")
    biz = {}
    for r in stream_ndjson(path):
        biz[r["business_id"]] = {
            "biz_name"       : r.get("name", ""),
            "categories"     : r.get("categories", "") or "",
            "city"           : r.get("city", ""),
            "state"          : r.get("state", ""),
            "biz_avg_stars"  : r.get("stars", 0.0),
            "biz_review_cnt" : r.get("review_count", 0),
            "is_open"        : r.get("is_open", 1),
        }
    log.info(f"  → {len(biz):,} businesses loaded")
    return biz


# ── Step 2: Load users ───────────────────────────────────────────────────────

def load_users() -> dict:
    """Returns dict: user_id → computed behavioural features"""
    path = RAW_DIR / FILES["user"]
    log.info(f"Loading users from {path} ...")
    users = {}
    for r in stream_ndjson(path):
        users[r["user_id"]] = {
            "user_review_count" : r.get("review_count", 0),
            "user_avg_stars"    : r.get("average_stars", 0.0),
            "user_useful"       : r.get("useful", 0),
            "user_funny"        : r.get("funny", 0),
            "user_cool"         : r.get("cool", 0),
            "user_fans"         : r.get("fans", 0),
            "user_elite"        : 1 if r.get("elite") else 0,
            "user_yelping_since": r.get("yelping_since", ""),
        }
    log.info(f"  → {len(users):,} users loaded")
    return users


# ── Step 3: Stream + enrich reviews ─────────────────────────────────────────

def extract_reviews(businesses: dict, users: dict, min_user_reviews: int = 5) -> pd.DataFrame:
    """
    Streams review file, enriches each row with user + business metadata.
    Filters to users with >= min_user_reviews (warm users only at this stage).
    Cold-start users are handled at inference time.
    """
    path = RAW_DIR / FILES["review"]
    log.info(f"Streaming reviews from {path} ...")

    rows = []
    skipped_cold = 0
    total = 0

    for r in stream_ndjson(path):
        total += 1
        uid = r.get("user_id", "")
        bid = r.get("business_id", "")

        udata = users.get(uid, {})

        # Skip cold users at extraction time (keep data lean)
        if udata.get("user_review_count", 0) < min_user_reviews:
            skipped_cold += 1
            continue

        bdata = businesses.get(bid, {})

        rows.append({
            # Core review fields
            "review_id"    : r.get("review_id", ""),
            "user_id"      : uid,
            "business_id"  : bid,
            "stars"        : float(r.get("stars", 0)),
            "text"         : r.get("text", ""),
            "date"         : r.get("date", ""),
            "useful"       : r.get("useful", 0),
            "funny"        : r.get("funny", 0),
            "cool"         : r.get("cool", 0),
            # User features (denormalised for fast persona building)
            **udata,
            # Business features
            **bdata,
        })

        if total % 1_000_000 == 0:
            log.info(f"  ...processed {total:,} reviews, kept {len(rows):,}")

    log.info(f"  → Total: {total:,} | Kept: {len(rows):,} | Cold-skipped: {skipped_cold:,}")
    return pd.DataFrame(rows)


# ── Step 4: Add derived features ─────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns used by the persona engine:
      - word_count          : length signal for style matching
      - rating_deviation    : how far this review deviates from user's mean
      - rating_bias_label   : 'generous' | 'critical' | 'neutral'
    """
    log.info("Computing derived features ...")

    df["word_count"] = df["text"].str.split().str.len().fillna(0).astype(int)

    # Rating deviation from user mean (key signal for RMSE calibration)
    user_means = df.groupby("user_id")["stars"].transform("mean")
    df["rating_deviation"] = df["stars"] - user_means

    # User-level bias label
    def bias_label(avg):
        if avg >= 4.0:   return "generous"
        elif avg <= 2.5: return "critical"
        else:            return "neutral"

    df["rating_bias_label"] = df["user_avg_stars"].map(bias_label)

    return df


# ── Step 5: Build per-user summary ───────────────────────────────────────────

def build_user_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per user with behavioural fingerprint columns.
    This is what PersonaBuilder reads — not raw reviews.
    """
    log.info("Building user summary table ...")

    agg = df.groupby("user_id").agg(
        review_count         = ("review_id",         "count"),
        mean_stars           = ("stars",              "mean"),
        std_stars            = ("stars",              "std"),
        mean_word_count      = ("word_count",         "mean"),
        std_word_count       = ("word_count",         "std"),
        mean_rating_dev      = ("rating_deviation",   "mean"),
        pct_5_star           = ("stars",              lambda x: (x == 5).mean()),
        pct_1_star           = ("stars",              lambda x: (x == 1).mean()),
        rating_bias_label    = ("rating_bias_label",  "first"),
        user_avg_stars       = ("user_avg_stars",     "first"),
        user_elite           = ("user_elite",         "first"),
        top_categories       = ("categories",         lambda x: _top_categories(x)),
    ).reset_index()

    log.info(f"  → {len(agg):,} user summaries built")
    return agg


def _top_categories(series: pd.Series, top_n: int = 5) -> str:
    from collections import Counter
    cats = Counter()
    for val in series.dropna():
        for cat in val.split(","):
            cat = cat.strip()
            if cat:
                cats[cat] += 1
    return ", ".join(c for c, _ in cats.most_common(top_n))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-reviews", type=int, default=5,
                        help="Min reviews for a user to be included in warm set")
    parser.add_argument("--sample", type=int, default=None,
                        help="Dev mode: only process first N reviews")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Step 1 — Businesses")
    businesses = load_businesses()

    log.info("Step 2 — Users")
    users = load_users()

    log.info("Step 3 — Reviews (streaming + enrichment)")
    df_reviews = extract_reviews(businesses, users, min_user_reviews=args.min_reviews)

    log.info("Step 4 — Derived features")
    df_reviews = add_derived_features(df_reviews)

    log.info("Step 5 — User summary table")
    df_users = build_user_summary(df_reviews)

    # ── Save ────────────────────────────────────────────────
    reviews_out = PROCESSED_DIR / "reviews.parquet"
    users_out   = PROCESSED_DIR / "users.parquet"

    df_reviews.to_parquet(reviews_out, index=False, engine="pyarrow")
    df_users.to_parquet(users_out,   index=False, engine="pyarrow")

    log.info("=" * 60)
    log.info(f"Saved reviews  → {reviews_out}  ({df_reviews.shape})")
    log.info(f"Saved users    → {users_out}    ({df_users.shape})")
    log.info("Run scripts/build_index.py next to build the ChromaDB vector store.")


if __name__ == "__main__":
    main()
