"""
Pre-computes dropdown options for the UI from the processed Yelp data.
Outputs ./data/processed/ui_metadata.json with:
  - top_users   : 50 most active warm users (user_id + review_count + signature)
  - top_cities  : 30 most-reviewed cities (with state)
  - top_states  : all states with counts
  - top_categories : 40 most common business categories
  - sample_businesses : 20 representative businesses for "name" autocomplete

Run once after extract_data.py:
    python -m scripts.build_ui_metadata
"""

import os
import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))


def build():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Reading users.parquet ...")
    df_users = pd.read_parquet(PROCESSED_DIR / "users.parquet")
    log.info(f"  {len(df_users):,} users")

    log.info("Reading reviews.parquet (selected columns) ...")
    df_rev = pd.read_parquet(
        PROCESSED_DIR / "reviews.parquet",
        columns=["business_id", "biz_name", "city", "state", "categories",
                 "biz_avg_stars", "biz_review_cnt"]
    )

    log.info("Building top users list ...")
    top_users = (
        df_users
        .sort_values("review_count", ascending=False)
        .head(50)
        [["user_id", "review_count", "mean_stars", "rating_bias_label"]]
        .to_dict(orient="records")
    )
    # Build a readable display label per user
    for u in top_users:
        bias = u.get("rating_bias_label", "neutral")
        u["label"] = (
            f"User {u['user_id'][:8]}… "
            f"({u['review_count']} reviews, "
            f"{u['mean_stars']:.1f}★ avg, {bias})"
        )

    log.info("Building top cities list ...")
    city_state = (
        df_rev.dropna(subset=["city"])
        .groupby(["city", "state"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(30)
    )
    top_cities = city_state.to_dict(orient="records")

    log.info("Building states list ...")
    state_counts = (
        df_rev.dropna(subset=["state"])
        .groupby("state").size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top_states = state_counts.to_dict(orient="records")

    log.info("Building top categories list ...")
    from collections import Counter
    cat_counter = Counter()
    for cats in df_rev["categories"].dropna().head(500_000):  # sample for speed
        for c in cats.split(","):
            c = c.strip()
            if c: cat_counter[c] += 1
    top_categories = [
        {"name": name, "count": cnt}
        for name, cnt in cat_counter.most_common(40)
    ]

    log.info("Building sample businesses list ...")
    # unique businesses, prioritise variety across categories
    unique_biz = (
        df_rev.drop_duplicates(subset=["business_id"])
        [["business_id", "biz_name", "city", "state", "categories",
          "biz_avg_stars", "biz_review_cnt"]]
        .sort_values("biz_review_cnt", ascending=False)
        .head(40)
    )
    sample_businesses = unique_biz.to_dict(orient="records")
    # Display label
    for b in sample_businesses:
        b["label"] = f"{b['biz_name']} — {b['city']}, {b['state']} ({b['biz_avg_stars']}★)"

    # Add a few Nigerian-themed entries as canonical demo options (synthetic)
    nigerian_demo = [
        {
            "business_id": "demo_mama_put", "biz_name": "Mama Put Kitchen",
            "city": "Lagos", "state": "LA", "categories": "Nigerian, African, Restaurants",
            "biz_avg_stars": 4.3, "biz_review_cnt": 88,
            "label": "Mama Put Kitchen — Lagos, LA (4.3★) [demo]"
        },
        {
            "business_id": "demo_suya_express", "biz_name": "Suya Express",
            "city": "Abuja", "state": "FC", "categories": "Nigerian, Grills, Restaurants",
            "biz_avg_stars": 4.5, "biz_review_cnt": 120,
            "label": "Suya Express — Abuja, FC (4.5★) [demo]"
        },
    ]
    sample_businesses = nigerian_demo + sample_businesses

    out = {
        "top_users"        : top_users,
        "top_cities"       : top_cities,
        "top_states"       : top_states,
        "top_categories"   : top_categories,
        "sample_businesses": sample_businesses,
    }

    out_path = PROCESSED_DIR / "ui_metadata.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"Saved UI metadata → {out_path}")
    log.info(f"  {len(top_users)} users · {len(top_cities)} cities · "
             f"{len(top_states)} states · {len(top_categories)} categories · "
             f"{len(sample_businesses)} businesses")


if __name__ == "__main__":
    build()
