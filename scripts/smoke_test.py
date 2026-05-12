"""
scripts/smoke_test.py
----------------------
Verifies the full pipeline works before containerisation.
Runs one Task A simulation and one Task B recommendation locally.

Usage:
    python -m scripts.smoke_test
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level="INFO", format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

WARM_USER_ID   = None   # auto-detected from users.parquet
COLD_USER_ID   = "brand_new_user_xyz_000"

TEST_BUSINESS  = {
    "name"        : "Chicken Republic",
    "categories"  : "Fast Food, Chicken, Restaurants",
    "city"        : "Lagos",
    "state"       : "LA",
    "stars"       : 4.1,
    "review_count": 312,
}


def get_sample_warm_user() -> str:
    import pandas as pd
    import os
    users_path = Path(os.getenv("PROCESSED_DIR", "./data/processed")) / "users.parquet"
    df = pd.read_parquet(users_path)
    heavy = df[df["review_count"] >= 20]
    if heavy.empty:
        heavy = df
    uid = heavy.iloc[0]["user_id"]
    log.info(f"Using warm user_id: {uid} (review_count={heavy.iloc[0]['review_count']})")
    return uid


def test_task_a(user_id: str):
    log.info("\n" + "="*50)
    log.info("SMOKE TEST — TASK A (Review Simulator)")
    log.info("="*50)

    from task_a.simulator import ReviewSimulator
    sim    = ReviewSimulator()

    # Warm user
    log.info(f"\n[1/2] Warm user: {user_id}")
    result = sim.simulate(
        user_id        = user_id,
        business_id    = "smoke_test_biz_001",
        product_details= TEST_BUSINESS,
    )
    print(f"\n  Predicted stars : {result['predicted_stars']}")
    print(f"  Rating confidence: {result['rating_confidence']}")
    print(f"  Persona summary : {json.dumps(result['persona_summary'], indent=4)}")
    print(f"\n  Generated review:\n  {result['review_text']}\n")

    # Cold user
    log.info(f"[2/2] Cold user: {COLD_USER_ID}")
    cold_result = sim.simulate(
        user_id        = COLD_USER_ID,
        business_id    = "smoke_test_biz_002",
        product_details= TEST_BUSINESS,
    )
    print(f"\n  Cold-start review:\n  {cold_result['review_text']}\n")
    log.info("Task A smoke test PASSED")


def test_task_b(user_id: str):
    log.info("\n" + "="*50)
    log.info("SMOKE TEST — TASK B (Recommendation Agent)")
    log.info("="*50)

    from task_b.recommender import RecommendationAgent
    agent = RecommendationAgent()

    # Warm user, single turn
    log.info(f"\n[1/3] Warm user, single turn: {user_id}")
    result = agent.recommend(
        user_id = user_id,
        context = "Looking for somewhere good for a Friday night out with friends",
        top_k   = 5,
    )
    print(f"\n  Search intent : {result['search_intent']}")
    print(f"  Cold start    : {result['cold_start']}")
    print(f"  Recommendations:")
    for i, r in enumerate(result["recommendations"], 1):
        print(f"    {i}. {r['name']} ({r['stars']}★) — {r['reason']}")

    # Cold user
    log.info(f"\n[2/3] Cold user: {COLD_USER_ID}")
    cold = agent.recommend(
        user_id = COLD_USER_ID,
        context = "I want Nigerian food, something like suya or pepper soup",
        top_k   = 5,
    )
    print(f"\n  Search intent : {cold['search_intent']}")
    print(f"  Cold start    : {cold['cold_start']}")
    for i, r in enumerate(cold["recommendations"], 1):
        print(f"    {i}. {r['name']} — {r['reason']}")

    # Multi-turn
    log.info(f"\n[3/3] Multi-turn conversation")
    history = [
        {"role": "user",      "content": "I want somewhere for brunch"},
        {"role": "assistant", "content": "I recommend trying a cafe downtown"},
        {"role": "user",      "content": "Actually I want something more lively with live music"},
    ]
    multi = agent.recommend(
        user_id             = user_id,
        context             = "something lively with live music",
        conversation_history= history,
        top_k               = 5,
    )
    print(f"\n  Search intent (multi-turn): {multi['search_intent']}")
    for i, r in enumerate(multi["recommendations"], 1):
        print(f"    {i}. {r['name']} — {r['reason']}")

    log.info("Task B smoke test PASSED")


def main():
    warm_user = get_sample_warm_user()

    try:
        test_task_a(warm_user)
    except Exception as e:
        log.error(f"Task A smoke test FAILED: {e}", exc_info=True)
        sys.exit(1)

    try:
        test_task_b(warm_user)
    except Exception as e:
        log.error(f"Task B smoke test FAILED: {e}", exc_info=True)
        sys.exit(1)

    log.info("\n✅ All smoke tests passed. Ready for Docker build.")


if __name__ == "__main__":
    main()
