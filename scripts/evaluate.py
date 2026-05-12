"""
scripts/evaluate.py
--------------------
Computes all official hackathon metrics against a held-out test set.

Metrics computed:
  Task A:
    - RMSE           (rating accuracy)
    - ROUGE-L        (review text quality)
    - BERTScore F1   (semantic similarity)

  Task B:
    - NDCG@10        (ranking quality)
    - Hit Rate@10    (recall at 10)

Usage:
    python -m scripts.evaluate --task a --n 200
    python -m scripts.evaluate --task b --n 200
    python -m scripts.evaluate --task both --n 200

How the test set is built:
  We hold out the LAST review per user (chronologically) as ground truth.
  The model is given everything EXCEPT that review and asked to simulate/recommend.
  This is a standard leave-one-out evaluation protocol for recommender systems.
"""

import os
import sys
import json
import logging
import argparse
import math
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Test set builder ──────────────────────────────────────────────────────────

def build_test_set(n: int = 200, min_reviews: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Build held-out test set.
    - Filter users with >= min_reviews (enough history to be meaningful)
    - For each user: keep the chronologically LAST review as ground truth
    - Sample n users
    """
    log.info(f"Building test set (n={n}, min_reviews={min_reviews}) ...")

    df = pd.read_parquet(
        PROCESSED_DIR / "reviews.parquet",
        columns=["user_id", "business_id", "stars", "text", "date",
                 "biz_name", "categories", "city", "state",
                 "biz_avg_stars", "biz_review_cnt", "user_avg_stars"]
    )

    # Users with enough history
    counts = df.groupby("user_id")["user_id"].count()
    eligible = counts[counts >= min_reviews].index.tolist()

    random.seed(seed)
    sampled_users = random.sample(eligible, min(n, len(eligible)))

    # Last review per sampled user = ground truth
    test_rows = (
        df[df["user_id"].isin(sampled_users)]
        .sort_values("date")
        .groupby("user_id")
        .last()
        .reset_index()
    )

    log.info(f"Test set: {len(test_rows)} rows")
    return test_rows


# ── Task A Evaluation ─────────────────────────────────────────────────────────

def evaluate_task_a(test_df: pd.DataFrame) -> dict:
    from task_a.simulator import ReviewSimulator

    log.info("Evaluating Task A ...")
    sim = ReviewSimulator()

    pred_stars, true_stars = [], []
    pred_texts, true_texts = [], []

    for _, row in test_df.iterrows():
        product_details = {
            "name"        : row.get("biz_name",      ""),
            "categories"  : row.get("categories",    ""),
            "city"        : row.get("city",          ""),
            "state"       : row.get("state",         ""),
            "stars"       : float(row.get("biz_avg_stars", 3.5)),
            "review_count": int(row.get("biz_review_cnt",  0)),
        }
        try:
            result = sim.simulate(
                user_id        = row["user_id"],
                business_id    = row["business_id"],
                product_details= product_details,
            )
            pred_stars.append(result["predicted_stars"])
            true_stars.append(float(row["stars"]))
            pred_texts.append(result["review_text"])
            true_texts.append(row["text"])
        except Exception as e:
            log.warning(f"Simulation failed for {row['user_id']}: {e}")

    if not pred_stars:
        return {"error": "No predictions generated"}

    rmse = _rmse(true_stars, pred_stars)
    log.info(f"RMSE: {rmse:.4f}")

    rouge_l = _rouge_l_batch(true_texts, pred_texts)
    log.info(f"ROUGE-L: {rouge_l:.4f}")

    bert_f1 = _bertscore_batch(true_texts, pred_texts)
    log.info(f"BERTScore F1: {bert_f1:.4f}")

    results = {
        "n_evaluated"  : len(pred_stars),
        "rmse"         : round(rmse,    4),
        "rouge_l"      : round(rouge_l, 4),
        "bertscore_f1" : round(bert_f1, 4),
        "mean_pred_stars": round(float(np.mean(pred_stars)), 3),
        "mean_true_stars": round(float(np.mean(true_stars)), 3),
    }
    return results


# ── Task B Evaluation ─────────────────────────────────────────────────────────

def evaluate_task_b(test_df: pd.DataFrame, k: int = 10) -> dict:
    from task_b.recommender import RecommendationAgent

    log.info("Evaluating Task B ...")
    agent = RecommendationAgent()

    ndcg_scores, hit_scores = [], []

    for _, row in test_df.iterrows():
        ground_truth_biz = row["business_id"]
        try:
            result = agent.recommend(
                user_id = row["user_id"],
                context = f"Looking for {row.get('categories', 'a good place')}",
                top_k   = k,
            )
            recs      = result["recommendations"]
            rec_ids   = [r["business_id"] for r in recs]

            ndcg_scores.append(_ndcg_at_k(ground_truth_biz, rec_ids, k))
            hit_scores.append(1.0 if ground_truth_biz in rec_ids else 0.0)
        except Exception as e:
            log.warning(f"Recommendation failed for {row['user_id']}: {e}")

    if not ndcg_scores:
        return {"error": "No recommendations generated"}

    results = {
        "n_evaluated": len(ndcg_scores),
        f"ndcg@{k}"  : round(float(np.mean(ndcg_scores)), 4),
        f"hit_rate@{k}": round(float(np.mean(hit_scores)), 4),
    }
    log.info(f"NDCG@{k}: {results[f'ndcg@{k}']:.4f}")
    log.info(f"Hit Rate@{k}: {results[f'hit_rate@{k}']:.4f}")
    return results


# ── Metric implementations ────────────────────────────────────────────────────

def _rmse(true: list, pred: list) -> float:
    return math.sqrt(np.mean([(t - p) ** 2 for t, p in zip(true, pred)]))


def _ndcg_at_k(relevant_id: str, ranked_ids: list, k: int) -> float:
    """Binary NDCG@k — one relevant item."""
    for i, bid in enumerate(ranked_ids[:k]):
        if bid == relevant_id:
            return 1.0 / math.log2(i + 2)   # position is 0-indexed; log2(1+1)=1 at rank 0
    return 0.0


def _rouge_l_batch(references: list, hypotheses: list) -> float:
    """ROUGE-L F1 averaged over all pairs. Pure Python — no external dependency."""
    scores = [_rouge_l_single(r, h) for r, h in zip(references, hypotheses)]
    return float(np.mean(scores)) if scores else 0.0


def _rouge_l_single(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 for one pair using LCS."""
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    lcs         = _lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision   = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall      = lcs / len(ref_tokens)  if ref_tokens  else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list, b: list) -> int:
    """Longest Common Subsequence length — O(mn)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimised: only keep two rows
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n]


def _bertscore_batch(references: list, hypotheses: list) -> float:
    """
    BERTScore F1. Uses the `bert-score` library if available.
    Falls back to a token-overlap proxy (fast, no GPU needed).
    """
    try:
        from bert_score import score as bert_score
        _, _, F1 = bert_score(
            hypotheses, references,
            lang="en",
            verbose=False,
            device="cpu",
        )
        return float(F1.mean())
    except ImportError:
        log.warning("bert-score not installed — using token overlap proxy. "
                    "Run: pip install bert-score for official scores.")
        return _token_overlap_f1(references, hypotheses)


def _token_overlap_f1(references: list, hypotheses: list) -> float:
    """Token-level F1 as BERTScore proxy."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_toks = set(ref.lower().split())
        hyp_toks = set(hyp.lower().split())
        if not ref_toks or not hyp_toks:
            scores.append(0.0)
            continue
        common    = ref_toks & hyp_toks
        precision = len(common) / len(hyp_toks)
        recall    = len(common) / len(ref_toks)
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hackathon evaluation script")
    parser.add_argument("--task", choices=["a", "b", "both"], default="both")
    parser.add_argument("--n",    type=int, default=200, help="Test set size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str, default="eval_results.json",
                        help="Output file for results")
    args = parser.parse_args()

    test_df = build_test_set(n=args.n, seed=args.seed)

    all_results = {}

    if args.task in ("a", "both"):
        log.info("\n" + "="*50)
        log.info("TASK A — REVIEW SIMULATOR")
        log.info("="*50)
        all_results["task_a"] = evaluate_task_a(test_df)
        log.info(f"\nTask A results: {json.dumps(all_results['task_a'], indent=2)}")

    if args.task in ("b", "both"):
        log.info("\n" + "="*50)
        log.info("TASK B — RECOMMENDATION AGENT")
        log.info("="*50)
        all_results["task_b"] = evaluate_task_b(test_df)
        log.info(f"\nTask B results: {json.dumps(all_results['task_b'], indent=2)}")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    if "task_a" in all_results:
        r = all_results["task_a"]
        print(f"Task A  RMSE         : {r.get('rmse',         'N/A')}")
        print(f"Task A  ROUGE-L      : {r.get('rouge_l',      'N/A')}")
        print(f"Task A  BERTScore F1 : {r.get('bertscore_f1', 'N/A')}")
    if "task_b" in all_results:
        r = all_results["task_b"]
        print(f"Task B  NDCG@10      : {r.get('ndcg@10',      'N/A')}")
        print(f"Task B  Hit Rate@10  : {r.get('hit_rate@10',  'N/A')}")
    print("="*50)


if __name__ == "__main__":
    main()
