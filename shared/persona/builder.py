"""
shared/persona/builder.py  (v5 — disk-cached index)
----------------------------------------------------
OPTIMISATION over v4:
  v4 rebuilt the user_id -> row_indices inverted index on every container
  startup (~90-120s for 5.78M rows). That blocked Docker healthchecks
  and caused restart loops.

  v5 checks for a pre-built index at data/processed/user_to_idx.pkl.
  If present (built once by scripts/build_persona_index.py), startup
  drops from ~90s to ~3-5s. The index is ~30MB on disk.

  If the pickle is absent, v5 falls back to building the index in memory
  exactly like v4, so the system still works without the pre-build step.
"""

import os
import time
import pickle
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
USERS_PARQ    = PROCESSED_DIR / "users.parquet"
REVIEWS_PARQ  = PROCESSED_DIR / "reviews.parquet"
INDEX_PKL     = PROCESSED_DIR / "user_to_idx.pkl"

TOP_K           = int(os.getenv("TOP_K_REVIEWS",            "10"))
MIN_WARM        = int(os.getenv("MIN_REVIEWS_FOR_WARM_USER", "5"))
MAX_SNIPPET_LEN = int(os.getenv("MAX_REVIEW_SNIPPET_LEN",    "200"))


class PersonaBuilder:
    """
    Loads everything at startup.

    Startup timing:
      - With pre-built index (data/processed/user_to_idx.pkl present): ~5-8s
      - Without pre-built index (first ever run):                      ~90-120s

    To pre-build the index, run once:
        python -m scripts.build_persona_index
    """

    def __init__(self):
        log.info("Loading PersonaBuilder ...")

        self._users:       Optional[pd.DataFrame] = None
        self._reviews:     Optional[pd.DataFrame] = None
        self._user_to_idx: Optional[dict]         = None

        self._load_users()
        self._load_reviews_and_index()

        log.info("PersonaBuilder ready")

    def _load_users(self):
        t0 = time.time()
        try:
            self._users = pd.read_parquet(USERS_PARQ).set_index("user_id")
            log.info(f"  users.parquet loaded: {len(self._users):,} users "
                     f"({time.time()-t0:.1f}s)")
        except FileNotFoundError:
            log.warning("users.parquet not found - cold-start only mode")

    def _load_reviews_and_index(self):
        t0 = time.time()
        log.info(f"  Loading reviews.parquet ...")
        df = pd.read_parquet(
            REVIEWS_PARQ,
            columns=["user_id", "text", "stars", "categories", "date"],
        )
        log.info(f"  reviews.parquet loaded: {len(df):,} rows "
                 f"({time.time()-t0:.1f}s)")
        self._reviews = df

        if INDEX_PKL.exists():
            t0 = time.time()
            log.info(f"  Loading pre-built index from {INDEX_PKL.name} ...")
            with open(INDEX_PKL, "rb") as f:
                self._user_to_idx = pickle.load(f)
            log.info(f"  Index loaded: {len(self._user_to_idx):,} users "
                     f"({time.time()-t0:.1f}s)")
        else:
            log.warning(
                "  No pre-built index found - building in memory. "
                "This adds ~20s to startup. Run "
                "'python -m scripts.build_persona_index' once to cache it."
            )
            t0 = time.time()
            self._user_to_idx = _build_inverted_index(df["user_id"].values)
            log.info(f"  Index built in memory: {len(self._user_to_idx):,} users "
                     f"({time.time()-t0:.1f}s)")

    def build(self, user_id: str) -> dict:
        if self._users is not None and user_id in self._users.index:
            row = self._users.loc[user_id]
            if int(row.get("review_count", 0)) >= MIN_WARM:
                return self._warm_persona(user_id, row)
        return self._cold_persona(user_id)

    def _warm_persona(self, user_id: str, row: pd.Series) -> dict:
        indices = self._user_to_idx.get(user_id, [])

        if len(indices) > 0:
            user_reviews = (
                self._reviews.iloc[indices]
                .sort_values("date", ascending=False)
                .head(TOP_K)
            )

            sample_texts = [
                (t[:MAX_SNIPPET_LEN] if isinstance(t, str) else "")
                for t in user_reviews["text"].tolist()
            ]

            all_cats = []
            for cat_str in user_reviews["categories"].dropna():
                all_cats.extend(c.strip() for c in cat_str.split(",") if c.strip())
            top_cats = [c for c, _ in Counter(all_cats).most_common(5)]
        else:
            sample_texts = []
            top_cats     = []

        mean_wc = float(row.get("mean_word_count", 100))

        return {
            "user_id"        : user_id,
            "is_cold"        : False,
            "review_count"   : int(row.get("review_count", 0)),
            "mean_stars"     : float(row.get("mean_stars",  3.5)),
            "std_stars"      : _safe_float(row.get("std_stars"), 1.0),
            "rating_bias"    : str(row.get("rating_bias_label", "neutral")),
            "pct_5_star"     : float(row.get("pct_5_star",  0.0)),
            "pct_1_star"     : float(row.get("pct_1_star",  0.0)),
            "mean_word_count": mean_wc,
            "style_label"    : _style_label(mean_wc),
            "sample_reviews" : sample_texts,
            "top_categories" : top_cats,
            "nigerian_mode"  : False,
        }

    def _cold_persona(self, user_id: str) -> dict:
        return {
            "user_id"        : user_id,
            "is_cold"        : True,
            "review_count"   : 0,
            "mean_stars"     : 3.63,
            "std_stars"      : 1.2,
            "rating_bias"    : "neutral",
            "pct_5_star"     : 0.462,
            "pct_1_star"     : 0.153,
            "mean_word_count": 104.8,
            "style_label"    : "medium",
            "sample_reviews" : [],
            "top_categories" : [],
            "nigerian_mode"  : False,
        }


def _build_inverted_index(user_ids: np.ndarray) -> dict:
    n = len(user_ids)
    perm = np.argsort(user_ids, kind="stable")
    sorted_uids = user_ids[perm]
    change_points = np.concatenate(
        ([0], np.where(sorted_uids[1:] != sorted_uids[:-1])[0] + 1, [n])
    )
    out = {}
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i + 1]
        uid = sorted_uids[start]
        out[uid] = perm[start:end]
    return out


def _style_label(mean_wc: float) -> str:
    if mean_wc < 50:  return "concise"
    if mean_wc > 150: return "verbose"
    return "medium"

def _safe_float(val, default: float) -> float:
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default
