"""
ARCHITECTURE CHANGE — this is the fix for every startup/RAM problem.

OLD (v5 and earlier):
  Loaded all 5.78M rows of reviews.parquet into RAM at startup, built an
  inverted index live, composed personas per request. Result: 90s+ boots,
  Docker health check restart loops, ~3.5 GB RAM per container, two
  containers impossible on a 16 GB laptop.

NEW (v6):
  Loads ONLY data/processed/persona_store.parquet — a compact file
  (~150-250 MB) where every warm-user persona was precomputed at build
  time by scripts/build_persona_store.py. Startup is 2-3 seconds.
  reviews.parquet is never opened at runtime.

  Cold-start users (not in the store) get dataset-mean defaults computed
  on the fly — no data file needed for them at all.

BUILD STEP (run once, locally):
    python -m scripts.build_persona_store
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
STORE_PARQ    = PROCESSED_DIR / "persona_store.parquet"

# Dataset-level fallback statistics for cold-start users.
COLD_MEAN_STARS = float(os.getenv("COLD_MEAN_STARS", "3.63"))
COLD_MEAN_WC    = float(os.getenv("COLD_MEAN_WC",    "104.8"))


class PersonaBuilder:
    """
    Runtime persona lookup. Instantiate once at app startup.

    Startup: loads persona_store.parquet (~2-3s) into a dict keyed by
    user_id. Every build() call after that is an O(1) dict lookup.
    """

    def __init__(self):
        log.info("Loading PersonaBuilder (v6 — persona store) ...")
        self._store: dict[str, dict] = {}
        self._load_store()
        log.info(f"PersonaBuilder ready — {len(self._store):,} warm personas in memory")

    def _load_store(self):
        if not STORE_PARQ.exists():
            log.warning(
                f"persona_store.parquet not found at {STORE_PARQ}. "
                "Running in COLD-START-ONLY mode. "
                "Run 'python -m scripts.build_persona_store' to enable warm users."
            )
            return

        t0 = time.time()
        df = pd.read_parquet(STORE_PARQ)
        # Convert to a plain dict for O(1), GIL-friendly lookups
        for rec in df.to_dict(orient="records"):
            uid = str(rec["user_id"])
            self._store[uid] = {
                "user_id"        : uid,
                "is_cold"        : False,
                "review_count"   : int(rec["review_count"]),
                "mean_stars"     : float(rec["mean_stars"]),
                "std_stars"      : float(rec["std_stars"]),
                "rating_bias"    : str(rec["rating_bias"]),
                "pct_5_star"     : float(rec["pct_5_star"]),
                "pct_1_star"     : float(rec["pct_1_star"]),
                "mean_word_count": float(rec["mean_word_count"]),
                "style_label"    : str(rec["style_label"]),
                "top_categories" : _safe_json_list(rec["top_categories"]),
                "sample_reviews" : _safe_json_list(rec["sample_reviews"]),
                "nigerian_mode"  : False,
            }
        log.info(f"  persona_store.parquet loaded: {len(self._store):,} personas "
                 f"({time.time()-t0:.1f}s)")

    def build(self, user_id: str) -> dict:
        """
        Return a persona dict for the given user_id.
        Warm users come from the precomputed store; everyone else gets a
        cold-start persona built from dataset means.
        """
        persona = self._store.get(str(user_id))
        if persona is not None:
            # Return a shallow copy so per-request mutation (e.g. the
            # Nigerian adapter setting nigerian_mode) never corrupts the store.
            return dict(persona)
        return self._cold_persona(user_id)

    def _cold_persona(self, user_id: str) -> dict:
        return {
            "user_id"        : str(user_id),
            "is_cold"        : True,
            "review_count"   : 0,
            "mean_stars"     : COLD_MEAN_STARS,
            "std_stars"      : 1.2,
            "rating_bias"    : "neutral",
            "pct_5_star"     : 0.462,
            "pct_1_star"     : 0.153,
            "mean_word_count": COLD_MEAN_WC,
            "style_label"    : "medium",
            "top_categories" : [],
            "sample_reviews" : [],
            "nigerian_mode"  : False,
        }

    @property
    def warm_user_count(self) -> int:
        return len(self._store)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_json_list(val) -> list:
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val:
        return []
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
