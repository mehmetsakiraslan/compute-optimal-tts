#!/usr/bin/env python3
"""Analyze lazy vs sync beam search decision traces on MATH-500.

Parses per-problem results from output_benchmark/ and tier decisions from
benchmark_lazy.err to identify exactly which decisions cause the accuracy gap.
"""

import json
import os
import glob
import re
import sys
from collections import Counter, defaultdict

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_benchmark")
RESULT_SUBDIR = "Qwen2.5-1.5B-Instruct/math-shepherd-mistral-7b-prm/40_4_1"
LAZY_DIR = os.path.join(OUTPUT_DIR, "MATH_lazy_beam_search", RESULT_SUBDIR)
SYNC_DIR = os.path.join(OUTPUT_DIR, "MATH_beam_search", RESULT_SUBDIR)
LAZY_ERR = os.path.join(BASE_DIR, "benchmark_lazy.err")
LAZY_OUT = os.path.join(BASE_DIR, "benchmark_lazy.out")


# ============================================================
# Step 1: Parse per-problem results
# ============================================================
def parse_results(result_dir):
    """Parse question_*/record_0.jsonl -> {question_id: {majority_vote, tokens}}"""
    results = {}
    pattern = os.path.join(result_dir, "question_*/record_0.jsonl")
    for jsonl_path in sorted(glob.glob(pattern)):
        qdir = os.path.basename(os.path.dirname(jsonl_path))
        qid = int(qdir.replace("question_", ""))
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                results[qid] = {
                    "majority_vote": d["result"]["majority_vote"],
                    "total_tokens": d["result"]["total_completion_tokens"],
                    "question": d["question"][:80],
                }
    return results


# ============================================================
# Step 2: Categorize disagreements
# ============================================================
def categorize(lazy_results, sync_results):
    """Categorize problems into 4 groups."""
    both_correct = []
    both_wrong = []
    lazy_wrong_sync_right = []
    lazy_right_sync_wrong = []

    all_qids = sorted(set(lazy_results.keys()) & set(sync_results.keys()))
    for qid in all_qids:
        lv = lazy_results[qid]["majority_vote"]
        sv = sync_results[qid]["majority_vote"]
        if lv and sv:
            both_correct.append(qid)
        elif not lv and not sv:
            both_wrong.append(qid)
        elif not lv and sv:
            lazy_wrong_sync_right.append(qid)
        else:
            lazy_right_sync_wrong.append(qid)

    return both_correct, both_wrong, lazy_wrong_sync_right, lazy_right_sync_wrong


# ============================================================
# Step 3: Parse tier decisions from benchmark_lazy.err
# ============================================================
def parse_tier_decisions(err_path, out_path):
    """Parse lightweight cap and lazy prune tier decisions.

    Returns:
        per_problem_tiers: {question_id: list of (depth, tiers, source)}
        global_tier_counts: Counter of tier values across all decisions
    """
    # First, parse the Cnt lines from .out to get problem boundaries (timestamps)
    # Format: [26-03-03 10:03:01]   Cnt:   1 / 500  |  Q:   0  |  ...
    cnt_pattern = re.compile(
        r"\[(\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+Cnt:\s+(\d+)\s*/\s*\d+\s*\|\s*Q:\s*(\d+)"
    )

    # Parse problem completion timestamps from .out
    problem_timestamps = []  # [(timestamp_str, cnt, qid)]
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                m = cnt_pattern.search(line)
                if m:
                    ts_str, cnt, qid = m.group(1), int(m.group(2)), int(m.group(3))
                    problem_timestamps.append((ts_str, cnt, qid))

    # Parse tier decisions from .err
    # Lightweight cap format:
    #   2026-03-03 10:02:47.489 | INFO | ... - Lightweight cap: 12 -> 1 (beam_size=1, end_nodes=0, tiers=[1])
    # Lazy prune format (only when actual pruning happens):
    #   Lazy prune: 4 -> 1 (cap=1, tiers=[2])
    lightweight_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \|.*Lightweight cap:.*tiers=\[([^\]]*)\]"
    )
    prune_tier_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \|.*Lazy prune: \d+ -> \d+ \(cap=\d+, tiers=\[([^\]]*)\]\)"
    )

    all_decisions = []  # [(timestamp_str, tiers_list, source)]
    if os.path.exists(err_path):
        with open(err_path) as f:
            for line in f:
                m = lightweight_pattern.search(line)
                if m:
                    ts = m.group(1)
                    tiers = [int(x.strip()) for x in m.group(2).split(",") if x.strip()]
                    all_decisions.append((ts, tiers, "lightweight"))
                    continue
                m = prune_tier_pattern.search(line)
                if m:
                    ts = m.group(1)
                    tiers = [int(x.strip()) for x in m.group(2).split(",") if x.strip()]
                    all_decisions.append((ts, tiers, "prune"))

    # Map decisions to problems using timestamp boundaries
    # Convert problem_timestamps to full datetime for comparison
    # .out timestamps: "26-03-03 10:03:01" -> "2026-03-03 10:03:01"
    # .err timestamps: "2026-03-03 10:02:47"
    problem_bounds = []
    for ts_str, cnt, qid in problem_timestamps:
        # Convert short year format to full
        full_ts = "20" + ts_str  # "26-03-03 10:03:01" -> "2026-03-03 10:03:01"
        problem_bounds.append((full_ts, qid))

    per_problem_tiers = defaultdict(list)
    global_tier_counts = Counter()

    if not problem_bounds:
        # Can't map to problems, just count globally
        for ts, tiers, source in all_decisions:
            for t in tiers:
                global_tier_counts[t] += 1
        return per_problem_tiers, global_tier_counts

    # For each decision, find which problem it belongs to by timestamp
    # A decision belongs to problem N if its timestamp is between
    # the completion of problem N-1 and the completion of problem N
    decision_idx = 0
    for bound_idx in range(len(problem_bounds)):
        bound_ts, qid = problem_bounds[bound_idx]
        prev_ts = problem_bounds[bound_idx - 1][0] if bound_idx > 0 else "0000-00-00 00:00:00"

        while decision_idx < len(all_decisions):
            d_ts, tiers, source = all_decisions[decision_idx]
            if d_ts <= bound_ts:
                per_problem_tiers[qid].append((tiers, source))
                for t in tiers:
                    global_tier_counts[t] += 1
                decision_idx += 1
            else:
                break

    # Remaining decisions (after last problem completion)
    while decision_idx < len(all_decisions):
        d_ts, tiers, source = all_decisions[decision_idx]
        for t in tiers:
            global_tier_counts[t] += 1
        decision_idx += 1

    return per_problem_tiers, global_tier_counts


# ============================================================
# Step 4: Output report
# ============================================================
def main():
    print("=" * 60)
    print("LAZY vs SYNC BEAM SEARCH DECISION TRACE ANALYSIS")
    print("=" * 60)
    print()

    # Parse results
    print("Parsing results...")
    lazy_results = parse_results(LAZY_DIR)
    sync_results = parse_results(SYNC_DIR)
    print(f"  Lazy results: {len(lazy_results)} problems")
    print(f"  Sync results: {len(sync_results)} problems")
    print()

    if not lazy_results or not sync_results:
        print("ERROR: Missing result directories. Expected:")
        print(f"  {LAZY_DIR}")
        print(f"  {SYNC_DIR}")
        sys.exit(1)

    # Accuracy
    lazy_correct = sum(r["majority_vote"] for r in lazy_results.values())
    sync_correct = sum(r["majority_vote"] for r in sync_results.values())
    lazy_total = len(lazy_results)
    sync_total = len(sync_results)
    print(f"--- Accuracy ---")
    print(f"  Lazy:  {lazy_correct}/{lazy_total} = {lazy_correct/lazy_total*100:.1f}%")
    print(f"  Sync:  {sync_correct}/{sync_total} = {sync_correct/sync_total*100:.1f}%")
    print(f"  Gap:   {sync_correct - lazy_correct} problems ({(sync_correct - lazy_correct)/sync_total*100:.1f}%)")
    print()

    # Categorize disagreements
    both_correct, both_wrong, lazy_wrong, lazy_right = categorize(lazy_results, sync_results)
    print(f"--- Disagreement Categories ---")
    print(f"  Both correct:            {len(both_correct)}")
    print(f"  Both wrong:              {len(both_wrong)}")
    print(f"  Lazy-wrong, sync-right:  {len(lazy_wrong)}")
    print(f"  Lazy-right, sync-wrong:  {len(lazy_right)}")
    print(f"  Net accuracy gap:        {len(lazy_wrong) - len(lazy_right)}")
    assert len(both_correct) + len(both_wrong) + len(lazy_wrong) + len(lazy_right) == min(lazy_total, sync_total), \
        "Categories don't sum to total"
    print()

    # Token comparison
    lazy_tokens = sum(r["total_tokens"] for r in lazy_results.values())
    sync_tokens = sum(r["total_tokens"] for r in sync_results.values())
    print(f"--- Token Usage ---")
    print(f"  Lazy total tokens:  {lazy_tokens:,}")
    print(f"  Sync total tokens:  {sync_tokens:,}")
    print(f"  Lazy avg/problem:   {lazy_tokens/lazy_total:,.0f}")
    print(f"  Sync avg/problem:   {sync_tokens/sync_total:,.0f}")
    print()

    # Parse tier decisions
    print("Parsing tier decisions from benchmark_lazy.err...")
    per_problem_tiers, global_tier_counts = parse_tier_decisions(LAZY_ERR, LAZY_OUT)
    total_tier_decisions = sum(global_tier_counts.values())
    print(f"--- Tier Distribution (all lightweight cap decisions) ---")
    for tier in sorted(global_tier_counts.keys()):
        count = global_tier_counts[tier]
        print(f"  Tier {tier}: {count} ({count/max(total_tier_decisions,1)*100:.1f}%)")
    print(f"  Total tier decisions: {total_tier_decisions}")
    print()

    # Per-problem tier breakdown for lazy-wrong/sync-right problems
    if per_problem_tiers:
        print(f"--- Per-Problem Tier Breakdown (lazy-wrong/sync-right) ---")
        print(f"{'QID':>5} {'#Decisions':>10} {'Tier-0':>7} {'Tier-1':>7} {'Tier-2':>7}  Question")
        print("-" * 90)
        for qid in sorted(lazy_wrong)[:50]:  # Show first 50
            decisions = per_problem_tiers.get(qid, [])
            tier_counts = Counter()
            for tiers, source in decisions:
                for t in tiers:
                    tier_counts[t] += 1
            total = sum(tier_counts.values())
            t0 = tier_counts.get(0, 0)
            t1 = tier_counts.get(1, 0)
            t2 = tier_counts.get(2, 0)
            q_text = lazy_results[qid]["question"][:40]
            print(f"{qid:>5} {total:>10} {t0:>7} {t1:>7} {t2:>7}  {q_text}")

        if len(lazy_wrong) > 50:
            print(f"  ... ({len(lazy_wrong) - 50} more problems)")
        print()

        # Aggregate tier stats for lazy-wrong vs lazy-right problems
        print(f"--- Aggregate Tier Stats by Category ---")
        for category_name, qids in [("lazy-wrong/sync-right", lazy_wrong),
                                     ("lazy-right/sync-wrong", lazy_right),
                                     ("both-correct", both_correct),
                                     ("both-wrong", both_wrong)]:
            cat_tiers = Counter()
            for qid in qids:
                for tiers, source in per_problem_tiers.get(qid, []):
                    for t in tiers:
                        cat_tiers[t] += 1
            cat_total = sum(cat_tiers.values())
            if cat_total > 0:
                t0_pct = cat_tiers.get(0, 0) / cat_total * 100
                t1_pct = cat_tiers.get(1, 0) / cat_total * 100
                t2_pct = cat_tiers.get(2, 0) / cat_total * 100
                print(f"  {category_name:30s}: {cat_total:5d} decisions "
                      f"(t0={t0_pct:.1f}%, t1={t1_pct:.1f}%, t2={t2_pct:.1f}%)")
            else:
                print(f"  {category_name:30s}: no tier data")
    else:
        print("  (No per-problem tier mapping available)")

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
