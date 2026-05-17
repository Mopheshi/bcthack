"""
Evaluation with TWO protocols:

Protocol 1 -- Open retrieval (strict):
  The agent retrieves from ALL 150K businesses via its own pipeline.
  Realistic for deployment; harsh on NDCG / Hit Rate.
  Reported as "Open" in the paper.

Protocol 2 -- Candidate-set retrieval (standard):
  For each test user, the top-100 semantically similar businesses are
  retrieved up front. The ground-truth item is injected into this pool
  if not already present, so recall is possible. The agent then reranks
  WITHIN that fixed 100-item pool via RecommendationAgent.rerank_candidates().
  This is the standard recommendation evaluation protocol and, unlike v2,
  the agent genuinely ranks within the candidate set rather than running
  its own open retrieval.
  Reported as "Candidate-100" in the paper.

Task A metrics: RMSE, ROUGE-L, BERTScore F1
Task B metrics: NDCG@10, Hit Rate@10

Usage:
    python -m scripts.evaluate --task both --n 200
    python -m scripts.evaluate --task b --n 200 --protocol candidate
"""

import os, sys, json, logging, argparse, math, random
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
sys.path.insert(0, str(Path(__file__).parent.parent))

CANDIDATE_POOL_SIZE = 100   # candidate-100 protocol


# -- Test set ------------------------------------------------------------------

def build_test_set(n=200, min_reviews=10, seed=42):
    log.info(f"Building test set (n={n}, min_reviews={min_reviews}) ...")
    df = pd.read_parquet(PROCESSED_DIR / "reviews.parquet",
        columns=["user_id","business_id","stars","text","date",
                 "biz_name","categories","city","state",
                 "biz_avg_stars","biz_review_cnt","user_avg_stars"])
    counts = df.groupby("user_id")["user_id"].count()
    eligible = counts[counts >= min_reviews].index.tolist()
    random.seed(seed)
    sampled = random.sample(eligible, min(n, len(eligible)))
    test = (df[df["user_id"].isin(sampled)]
            .sort_values("date").groupby("user_id").last().reset_index())
    log.info(f"Test set: {len(test)} rows")
    return test


# -- Task A --------------------------------------------------------------------

def evaluate_task_a(test_df):
    from task_a.simulator import ReviewSimulator
    log.info("Evaluating Task A ...")
    sim = ReviewSimulator()
    pred_stars, true_stars, pred_texts, true_texts = [], [], [], []

    for _, row in test_df.iterrows():
        pd_ = {"name": row.get("biz_name",""), "categories": row.get("categories",""),
               "city": row.get("city",""), "state": row.get("state",""),
               "stars": float(row.get("biz_avg_stars",3.5)),
               "review_count": int(row.get("biz_review_cnt",0))}
        try:
            r = sim.simulate(user_id=row["user_id"], business_id=row["business_id"],
                             product_details=pd_)
            pred_stars.append(r["predicted_stars"])
            true_stars.append(float(row["stars"]))
            pred_texts.append(r["review_text"])
            true_texts.append(row["text"])
        except Exception as e:
            log.warning(f"Simulation failed: {e}")

    if not pred_stars:
        return {"error": "No predictions"}

    rmse     = math.sqrt(np.mean([(t-p)**2 for t,p in zip(true_stars,pred_stars)]))
    rouge_l  = _rouge_l_batch(true_texts, pred_texts)
    bert_f1  = _bertscore_batch(true_texts, pred_texts)
    bias     = np.mean(pred_stars) - np.mean(true_stars)

    return {"n_evaluated": len(pred_stars), "rmse": round(rmse,4),
            "rouge_l": round(rouge_l,4), "bertscore_f1": round(bert_f1,4),
            "mean_pred_stars": round(float(np.mean(pred_stars)),3),
            "mean_true_stars": round(float(np.mean(true_stars)),3),
            "systematic_bias": round(float(bias),3)}


# -- Task B --------------------------------------------------------------------

def evaluate_task_b(test_df, k=10, protocol="both"):
    import asyncio
    from task_b.recommender import RecommendationAgent
    from shared.vectorstore.store import VectorStore

    log.info(f"Evaluating Task B (protocol={protocol}) ...")
    agent = RecommendationAgent()
    vs    = VectorStore()

    open_ndcg, open_hit = [], []
    cand_ndcg, cand_hit = [], []

    for _, row in test_df.iterrows():
        gt_biz   = row["business_id"]
        context  = f"Looking for {row.get('categories','a good place')}"
        user_id  = row["user_id"]

        # -- Protocol 1: Open retrieval --------------------
        if protocol in ("open","both"):
            try:
                result  = asyncio.run(
                    agent.recommend(user_id=user_id, context=context, top_k=k)
                )
                rec_ids = [r["business_id"] for r in result["recommendations"]]
                open_ndcg.append(_ndcg_at_k(gt_biz, rec_ids, k))
                open_hit.append(1.0 if gt_biz in rec_ids else 0.0)
            except Exception as e:
                log.warning(f"Open eval failed {user_id}: {e}")

        # -- Protocol 2: Candidate-100 retrieval -----------
        # Genuine candidate-set protocol: build a fixed 100-item pool,
        # inject the ground truth, and have the agent rerank WITHIN it.
        if protocol in ("candidate","both"):
            try:
                pool = vs.query_businesses(query=context, n=CANDIDATE_POOL_SIZE)
                pool_ids = {c["business_id"] for c in pool}

                # Inject the ground-truth business if retrieval missed it,
                # so that a correct ranking is achievable.
                if gt_biz not in pool_ids:
                    gt_meta = vs.get_business(gt_biz)
                    if gt_meta is not None:
                        pool.append(gt_meta)
                    else:
                        # Ground truth not in the index at all -- skip this
                        # user rather than score an impossible case.
                        log.warning(f"GT {gt_biz} absent from index; skipping cand eval")
                        raise KeyError("gt-not-in-index")

                # Agent reranks within exactly this pool.
                result2  = agent.rerank_candidates(
                    user_id=user_id, context=context, candidates=pool, top_k=k,
                )
                rec_ids2 = [r["business_id"] for r in result2["recommendations"]]

                cand_ndcg.append(_ndcg_at_k(gt_biz, rec_ids2, k))
                cand_hit.append(1.0 if gt_biz in rec_ids2[:k] else 0.0)
            except KeyError:
                pass   # skipped -- ground truth not indexable
            except Exception as e:
                log.warning(f"Candidate eval failed {user_id}: {e}")

    out = {}
    if open_ndcg:
        out["open_retrieval"] = {
            "n_evaluated"  : len(open_ndcg),
            f"ndcg@{k}"    : round(float(np.mean(open_ndcg)),4),
            f"hit_rate@{k}": round(float(np.mean(open_hit)),4),
            "note"         : "Agent retrieves from the full 150K business corpus"
        }
    if cand_ndcg:
        out["candidate_100"] = {
            "n_evaluated"  : len(cand_ndcg),
            f"ndcg@{k}"    : round(float(np.mean(cand_ndcg)),4),
            f"hit_rate@{k}": round(float(np.mean(cand_hit)),4),
            "note"         : f"Agent reranks within a fixed {CANDIDATE_POOL_SIZE}-item "
                             "candidate pool (ground truth injected)"
        }
    return out


# -- Metrics -------------------------------------------------------------------

def _ndcg_at_k(relevant_id, ranked_ids, k):
    for i, bid in enumerate(ranked_ids[:k]):
        if bid == relevant_id:
            return 1.0 / math.log2(i + 2)
    return 0.0

def _rouge_l_batch(refs, hyps):
    return float(np.mean([_rouge_l_single(r,h) for r,h in zip(refs,hyps)]) if refs else 0.0)

def _rouge_l_single(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    lcs = _lcs(r, h)
    if lcs == 0: return 0.0
    p = lcs/len(h) if h else 0; rc = lcs/len(r) if r else 0
    return 2*p*rc/(p+rc) if (p+rc) else 0.0

def _lcs(a, b):
    m,n = len(a),len(b)
    prev = [0]*(n+1)
    for i in range(1,m+1):
        curr = [0]*(n+1)
        for j in range(1,n+1):
            curr[j] = prev[j-1]+1 if a[i-1]==b[j-1] else max(prev[j],curr[j-1])
        prev = curr
    return prev[n]

def _bertscore_batch(refs, hyps):
    try:
        from bert_score import score as bscore
        _,_,F1 = bscore(hyps, refs, lang="en", verbose=False, device="cpu")
        return float(F1.mean())
    except ImportError:
        log.warning("bert-score not installed; using token F1 proxy.")
        scores = []
        for r,h in zip(refs,hyps):
            rt,ht = set(r.lower().split()), set(h.lower().split())
            if not rt or not ht: scores.append(0.0); continue
            c = rt&ht; p = len(c)/len(ht); rc = len(c)/len(rt)
            scores.append(2*p*rc/(p+rc) if (p+rc) else 0.0)
        return float(np.mean(scores))


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",     choices=["a","b","both"], default="both")
    parser.add_argument("--n",        type=int, default=200)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--protocol", choices=["open","candidate","both"], default="both",
                        help="Task B evaluation protocol")
    parser.add_argument("--out",      default="eval_results.json")
    args = parser.parse_args()

    test_df = build_test_set(n=args.n, seed=args.seed)
    results = {}

    if args.task in ("a","both"):
        results["task_a"] = evaluate_task_a(test_df)
        log.info(f"\nTask A: {json.dumps(results['task_a'], indent=2)}")

    if args.task in ("b","both"):
        results["task_b"] = evaluate_task_b(test_df, protocol=args.protocol)
        log.info(f"\nTask B: {json.dumps(results['task_b'], indent=2)}")

    with open(args.out,"w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {args.out}")

    print("\n" + "="*55)
    print("EVALUATION SUMMARY")
    print("="*55)
    if "task_a" in results:
        r = results["task_a"]
        print(f"Task A  RMSE          : {r.get('rmse','N/A')}  (bias: {r.get('systematic_bias','N/A')})")
        print(f"Task A  ROUGE-L       : {r.get('rouge_l','N/A')}")
        print(f"Task A  BERTScore F1  : {r.get('bertscore_f1','N/A')}")
    if "task_b" in results:
        for proto, r in results["task_b"].items():
            print(f"\nTask B [{proto}]")
            print(f"  NDCG@10           : {r.get('ndcg@10','N/A')}")
            print(f"  Hit Rate@10       : {r.get('hit_rate@10','N/A')}")
            print(f"  Note              : {r.get('note','')}")
    print("="*55)

if __name__ == "__main__":
    main()