"""
Yelp Open Dataset - EDA Script
DSN x BCT Hackathon 3.0
"""

import json
import os
import argparse
import statistics
from collections import Counter, defaultdict

def stream_ndjson(filepath, limit=None):
    """Stream a newline-delimited JSON file one record at a time."""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit and count >= limit:
                break

def eda_businesses(filepath):
    print("\n" + "="*60)
    print("BUSINESSES")
    print("="*60)

    categories  = Counter()
    states      = Counter()
    stars_dist  = Counter()
    open_status = Counter()
    total       = 0

    for biz in stream_ndjson(filepath):
        total += 1
        if biz.get('categories'):
            for cat in biz['categories'].split(', '):
                categories[cat.strip()] += 1
        states[biz.get('state', 'Unknown')] += 1
        stars_dist[biz.get('stars', 0)] += 1
        open_status['open' if biz.get('is_open') else 'closed'] += 1

    print(f"Total businesses       : {total:,}")
    print(f"Open / Closed          : {open_status['open']:,} / {open_status['closed']:,}")
    print(f"\nStar distribution:")
    for s in sorted(stars_dist):
        bar = '█' * (stars_dist[s] // 1000)
        print(f"  {s}★  {stars_dist[s]:>7,}  {bar}")
    print(f"\nTop 15 categories:")
    for cat, cnt in categories.most_common(15):
        print(f"  {cnt:>7,}  {cat}")
    print(f"\nTop 10 states/regions:")
    for state, cnt in states.most_common(10):
        print(f"  {cnt:>7,}  {state}")


def eda_reviews(filepath):
    print("\n" + "="*60)
    print("REVIEWS")
    print("="*60)

    star_dist    = Counter()
    word_lengths = []
    users_seen   = set()
    biz_seen     = set()
    useful_dist  = Counter()
    sample_texts = []
    total        = 0

    for review in stream_ndjson(filepath):
        total += 1
        stars = review.get('stars')
        star_dist[stars] += 1
        text = review.get('text', '')
        wlen = len(text.split())
        word_lengths.append(wlen)
        users_seen.add(review.get('user_id'))
        biz_seen.add(review.get('business_id'))
        useful = review.get('useful', 0)
        useful_dist['>=1' if useful >= 1 else '0'] += 1

        if len(sample_texts) < 5:
            sample_texts.append({
                'stars': stars,
                'length': wlen,
                'snippet': text[:150].replace('\n', ' ')
            })

        if total % 1_000_000 == 0:
            print(f"  ...streamed {total:,} reviews so far")

    print(f"Total reviews          : {total:,}")
    print(f"Unique users           : {len(users_seen):,}")
    print(f"Unique businesses      : {len(biz_seen):,}")
    print(f"Useful reviews (>=1)   : {useful_dist['>=1']:,} ({100*useful_dist['>=1']/total:.1f}%)")

    print(f"\nStar distribution:")
    for s in sorted(star_dist):
        bar = '█' * (star_dist[s] // 50_000)
        pct = 100 * star_dist[s] / total
        print(f"  {s}★  {star_dist[s]:>8,}  ({pct:4.1f}%)  {bar}")

    print(f"\nReview length (words):")
    print(f"  Mean   : {statistics.mean(word_lengths):.1f}")
    print(f"  Median : {statistics.median(word_lengths):.1f}")
    print(f"  Stdev  : {statistics.stdev(word_lengths):.1f}")
    print(f"  Min    : {min(word_lengths)}")
    print(f"  Max    : {max(word_lengths)}")

    print(f"\nSample reviews:")
    for s in sample_texts:
        print(f"  [{s['stars']}★ | {s['length']} words] {s['snippet']}...")


def eda_users(filepath):
    print("\n" + "="*60)
    print("USERS")
    print("="*60)

    review_counts = []
    avg_stars_list = []
    elite_count   = 0
    total         = 0

    for user in stream_ndjson(filepath):
        total += 1
        rc = user.get('review_count', 0)
        review_counts.append(rc)
        avg_stars_list.append(user.get('average_stars', 0.0))
        if user.get('elite'):
            elite_count += 1

    warm   = sum(1 for rc in review_counts if rc >= 5)
    medium = sum(1 for rc in review_counts if 10 <= rc < 50)
    heavy  = sum(1 for rc in review_counts if rc >= 50)
    cold   = total - warm

    print(f"Total users            : {total:,}")
    print(f"Elite users            : {elite_count:,} ({100*elite_count/total:.1f}%)")
    print(f"\nActivity segmentation:")
    print(f"  Cold  (<5 reviews)   : {cold:,}   ({100*cold/total:.1f}%)")
    print(f"  Warm  (5–49 reviews) : {warm-heavy:,}  ({100*(warm-heavy)/total:.1f}%)")
    print(f"  Heavy (>=50 reviews) : {heavy:,}   ({100*heavy/total:.1f}%)")
    print(f"\nReview count stats:")
    print(f"  Mean   : {statistics.mean(review_counts):.1f}")
    print(f"  Median : {statistics.median(review_counts):.1f}")
    print(f"  Max    : {max(review_counts)}")
    print(f"\nMean avg star rating   : {statistics.mean(avg_stars_list):.3f}")

    # Rating bias analysis (critical for Task A RMSE calibration)
    biased_high = sum(1 for s in avg_stars_list if s >= 4.0)
    biased_low  = sum(1 for s in avg_stars_list if s <= 2.5)
    print(f"\nRating bias (Task A signal):")
    print(f"  Positive biased (avg >= 4★) : {biased_high:,} ({100*biased_high/total:.1f}%)")
    print(f"  Negative biased (avg <= 2.5★): {biased_low:,} ({100*biased_low/total:.1f}%)")


def eda_tips(filepath):
    print("\n" + "="*60)
    print("TIPS")
    print("="*60)

    total        = 0
    lengths      = []
    users_seen   = set()

    for tip in stream_ndjson(filepath):
        total += 1
        lengths.append(len(tip.get('text', '').split()))
        users_seen.add(tip.get('user_id'))

    print(f"Total tips             : {total:,}")
    print(f"Unique tip authors     : {len(users_seen):,}")
    print(f"Mean tip length (words): {statistics.mean(lengths):.1f}")
    print(f"Median tip length      : {statistics.median(lengths):.1f}")


def eda_checkins(filepath):
    print("\n" + "="*60)
    print("CHECKINS")
    print("="*60)

    total   = 0
    biz_set = set()

    for checkin in stream_ndjson(filepath):
        total += 1
        biz_set.add(checkin.get('business_id'))

    print(f"Total checkin records  : {total:,}")
    print(f"Unique businesses      : {len(biz_set):,}")



def main():
    parser = argparse.ArgumentParser(description='Yelp Dataset EDA')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing Yelp JSON files')
    args = parser.parse_args()

    files = {
        'business' : 'yelp_academic_dataset_business.json',
        'review'   : 'yelp_academic_dataset_review.json',
        'user'     : 'yelp_academic_dataset_user.json',
        'tip'      : 'yelp_academic_dataset_tip.json',
        'checkin'  : 'yelp_academic_dataset_checkin.json',
    }

    print("\n📊 Yelp Open Dataset — EDA")
    print(f"Data directory: {args.data_dir}\n")

    for key, filename in files.items():
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️  File not found, skipping: {filepath}")
            continue

        size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"\n📁 {filename}  ({size_gb:.2f} GB)")

        if key == 'business': eda_businesses(filepath)
        elif key == 'review':  eda_reviews(filepath)
        elif key == 'user':    eda_users(filepath)
        elif key == 'tip':     eda_tips(filepath)
        elif key == 'checkin': eda_checkins(filepath)

    print("\n\n✅ EDA complete.")
    print("\nKey signals to carry into architecture:")
    print("  → Rating distribution skew  : needed for RMSE calibration (Task A)")
    print("  → Cold user %               : design cold-start fallback (Task B)")
    print("  → Review length distribution: set LLM max_tokens budget")
    print("  → Category distribution     : informs cross-domain transfer strategy")


if __name__ == '__main__':
    main()
