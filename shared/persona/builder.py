"""
shared/persona/builder.py  (v3 — PROPER lazy loading)
------------------------------------------------------
PREVIOUS BUG: v2 read the full 5.7M-row reviews.parquet file from disk on
EVERY request via predicate pushdown. This caused 30-60s latency per call.

THIS VERSION:
  - Lazy-loads reviews.parquet ONCE into memory on first warm-user request
  - Pre-builds a user_id → row_indices map for O(1) lookup
  - All subsequent persona builds are sub-millisecond in-memory operations
  - Users.parquet always loaded eagerly at startup (only 13 columns × 1M rows)

Memory footprint after first warm request: ~4 GB on a 32 GB machine, fine.

Persona dict schema unchanged from v2.
"""

import os
import logging
from collections import Counter
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/processed")
USERS_PARQ    = os.path.join(PROCESSED_DIR, "users.parquet")
REVIEWS_PARQ  = os.path.join(PROCESSED_DIR, "reviews.parquet")
TOP_K         = int(os.getenv("TOP_K_REVIEWS", "20"))
MIN_WARM      = int(os.getenv("MIN_REVIEWS_FOR_WARM_USER", "5"))


class PersonaBuilder:
    """
    Instantiate once at app startup. Call build(user_id) per request.

    First warm-user request triggers a one-time reviews.parquet load
    (~10-15 seconds on first call). All later requests are sub-millisecond.
    """

    def __init__(self):
        log.info("Loading PersonaBuilder data into memory ...")

        # Users summary — small, load eagerly
        self._users: Optional[pd.DataFrame] = None
        self._load_users()

        # Reviews + lookup index — lazy, load on first warm call
        self._reviews: Optional[pd.DataFrame]      = None
        self._user_to_idx: Optional[dict[str, list[int]]] = None

        log.info("PersonaBuilder ready (reviews will lazy-load on first warm request)")

    # ── Eager loaders ────────────────────────────────────────────

    def _load_users(self):
        try:
            self._users = pd.read_parquet(USERS_PARQ).set_index("user_id")
            log.info(f"  users.parquet loaded: {len(self._users):,} users")
        except FileNotFoundError:
            log.warning("users.parquet not found. Cold-start only mode.")

    # ── Lazy loader (ONCE into memory) ───────────────────────────

    def _ensure_reviews_loaded(self):
        """
        First time only: loads reviews.parquet into memory and builds
        a user_id → row_indices map for O(1) per-user filtering.
        Subsequent calls are no-ops.
        """
        if self._reviews is not None:
            return

        log.info("First warm request — loading reviews.parquet into memory (~10-15s) ...")
        self._reviews = pd.read_parquet(
            REVIEWS_PARQ,
            columns=["user_id", "text", "stars", "categories", "date"],
        )
        log.info(f"  reviews.parquet loaded: {len(self._reviews):,} rows")

        log.info("  Building user_id → indices map for O(1) lookup ...")
        # Group by user_id and store row indices per user
        self._user_to_idx = (
            self._reviews
            .reset_index()
            .groupby("user_id")["index"]
            .apply(list)
            .to_dict()
        )
        log.info(f"  Index built: {len(self._user_to_idx):,} unique users mapped")

    # ── Public API ───────────────────────────────────────────────

    def build(self, user_id: str) -> dict:
        if self._users is not None and user_id in self._users.index:
            row = self._users.loc[user_id]
            if int(row.get("review_count", 0)) >= MIN_WARM:
                return self._warm_persona(user_id, row)

        return self._cold_persona(user_id)

    # ── Warm persona (fast: in-memory lookup) ────────────────────

    def _warm_persona(self, user_id: str, row: pd.Series) -> dict:
        self._ensure_reviews_loaded()

        # O(1) lookup of this user's review indices, then iloc into the df
        indices = self._user_to_idx.get(user_id, [])
        if indices:
            user_reviews = (
                self._reviews.iloc[indices]
                .sort_values("date", ascending=False)
                .head(TOP_K)
            )
            sample_texts = user_reviews["text"].tolist()

            all_cats = []
            for cat_str in user_reviews["categories"].dropna():
                all_cats.extend([c.strip() for c in cat_str.split(",") if c.strip()])
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

    # ── Cold-start persona ───────────────────────────────────────

    def _cold_persona(self, user_id: str) -> dict:
        log.debug(f"Cold-start persona for user_id={user_id}")
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


# ── Helpers ────────────────────────────────────────────────────────────────────

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
