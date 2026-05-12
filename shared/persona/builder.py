"""
shared/persona/builder.py
--------------------------
PersonaBuilder extracts a structured behavioural fingerprint for any user_id.

DATA SOURCES (v2 — no more review-level ChromaDB):
  - users.parquet      : pre-computed per-user stats (mean stars, word count, etc.)
  - reviews.parquet    : raw reviews, filtered by user_id via pandas (fast)
  - ChromaDB           : business-level index for semantic search (Task B only)

WARM user  : >= MIN_WARM reviews in dataset  -> full persona from parquet
COLD user  : < MIN_WARM or unknown           -> neutral defaults + content fallback

Persona dict schema:
{
    "user_id"         : str,
    "is_cold"         : bool,
    "review_count"    : int,
    "mean_stars"      : float,
    "std_stars"       : float,
    "rating_bias"     : str,     # "generous" | "critical" | "neutral"
    "pct_5_star"      : float,
    "pct_1_star"      : float,
    "mean_word_count" : float,
    "style_label"     : str,     # "verbose" | "concise" | "medium"
    "sample_reviews"  : list,    # list of raw review text strings (for RAG)
    "top_categories"  : list,    # top business categories this user reviews
    "nigerian_mode"   : bool,
}
"""

import os
import logging
from functools import lru_cache
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
    Loads parquet files into memory once — fast for repeated lookups.
    """

    def __init__(self):
        log.info("Loading PersonaBuilder data into memory ...")

        # User summary table — indexed by user_id for O(1) lookup
        self._users: Optional[pd.DataFrame] = None
        self._load_users()

        # Reviews — loaded lazily on first warm persona request
        self._reviews: Optional[pd.DataFrame] = None

        log.info("PersonaBuilder ready")

    # ── Data loaders ─────────────────────────────────────────

    def _load_users(self):
        try:
            self._users = pd.read_parquet(USERS_PARQ).set_index("user_id")
            log.info(f"  users.parquet loaded: {len(self._users):,} users")
        except FileNotFoundError:
            log.warning(f"users.parquet not found. Cold-start only mode.")

    def _ensure_reviews_loaded(self):
        """Lazy-load reviews parquet — only needed for sample_reviews."""
        if self._reviews is None:
            log.info("Lazy-loading reviews.parquet for sample extraction ...")
            self._reviews = pd.read_parquet(
                REVIEWS_PARQ,
                columns=["user_id", "text", "stars", "categories", "date"]
            )
            log.info(f"  reviews.parquet loaded: {len(self._reviews):,} rows")

    # ── Public API ────────────────────────────────────────────

    def build(self, user_id: str) -> dict:
        """
        Returns a Persona dict for user_id.
        Warm path if user has >= MIN_WARM reviews, else cold-start.
        """
        if self._users is not None and user_id in self._users.index:
            row = self._users.loc[user_id]
            if int(row.get("review_count", 0)) >= MIN_WARM:
                return self._warm_persona(user_id, row)

        return self._cold_persona(user_id)

    # ── Warm persona ──────────────────────────────────────────

    def _warm_persona(self, user_id: str, row: pd.Series) -> dict:
        self._ensure_reviews_loaded()

        # Filter reviews for this user directly in parquet — fast
        user_reviews = (
            self._reviews[self._reviews["user_id"] == user_id]
            .sort_values("date", ascending=False)
            .head(TOP_K)
        )

        sample_texts = user_reviews["text"].tolist()

        # Top categories from this user's review history
        all_cats = []
        for cat_str in user_reviews["categories"].dropna():
            all_cats.extend([c.strip() for c in cat_str.split(",") if c.strip()])
        from collections import Counter
        top_cats = [c for c, _ in Counter(all_cats).most_common(5)]

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

    # ── Cold-start persona ────────────────────────────────────

    def _cold_persona(self, user_id: str) -> dict:
        """
        No user history. Returns dataset-level defaults from EDA.
        Task B uses business content embeddings for cold-start recs.
        """
        log.debug(f"Cold-start persona for user_id={user_id}")
        return {
            "user_id"        : user_id,
            "is_cold"        : True,
            "review_count"   : 0,
            "mean_stars"     : 3.63,   # EDA: mean avg star rating
            "std_stars"      : 1.2,
            "rating_bias"    : "neutral",
            "pct_5_star"     : 0.462,  # EDA: 46.2% of all reviews are 5-star
            "pct_1_star"     : 0.153,
            "mean_word_count": 104.8,  # EDA: mean review word count
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
