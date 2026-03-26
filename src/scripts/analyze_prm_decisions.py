#!/usr/bin/env python3
"""
Per-problem PRM decision analysis for AMC23 & AIME24 benchmarks.
Compares beam_search vs lazy_beam_search and flags problematic PRM decisions.
"""

import json
import os
import glob
import numpy as np


BASE_DIR = "/scratch/msa6093/compute-optimal-tts/output_benchmark_amc_aime"
RESULT_SUBDIR = "Qwen2.5-3B-Instruct/Skywork-o1-Open-PRM-Qwen-2.5-7B/40_4_1"

DATASETS = ["AMC23", "AIME24"]
METHODS = ["beam_search", "lazy_beam_search"]


def load_results(dataset, method):
    result_dir = os.path.join(BASE_DIR, f"{dataset}_{method}", RESULT_SUBDIR)
    results = {}
    q_dirs = sorted(glob.glob(os.path.join(result_dir, "question_*")),
                    key=lambda x: int(x.split("_")[-1]))
    for q_dir in q_dirs:
        q_idx = int(q_dir.split("_")[-1])
        record_file = os.path.join(q_dir, "record_0.jsonl")
        if not os.path.exists(record_file):
            continue
        with open(record_file) as f:
            data = json.loads(f.readline())
        output = data["output"][0] if data["output"] else {}
        rh = output.get("reward_history", [])
        text = output.get("text", "")
        results[q_idx] = {
            "question": data.get("question", ""),
            "groundtruth": data.get("groundtruth", ""),
            "predicted": output.get("extracted_answer", ""),
            "is_correct": data["result"].get("majority_vote", 0) == 1,
            "reward_history": rh,
            "token_history": output.get("token_history", []),
            "completion_tokens": output.get("completion_tokens", 0),
            "tree_completion_tokens": output.get("tree_completion_tokens", 0),
            "text": text,
            "has_boxed": "\\boxed" in text,
            "num_steps": len(rh),
            "mean_reward": np.mean(rh) if rh else 0,
            "max_reward": max(rh) if rh else 0,
            "final_reward": rh[-1] if rh else 0,
            "zero_ratio": sum(1 for r in rh if r < 0.001) / len(rh) if rh else 0,
        }
    return results


def print_separator(char="=", width=100):
    print(char * width)


def main():
    all_data = {}
    for ds in DATASETS:
        all_data[ds] = {}
        for method in METHODS:
            all_data[ds][method] = load_results(ds, method)

    # ═══════════════════════════════════════════
    # SECTION 1: Overall Summary
    # ═══════════════════════════════════════════
    print_separator()
    print("SECTION 1: OVERALL ACCURACY SUMMARY")
    print_separator()
    print(f"{'Dataset':<10} {'Method':<22} {'Correct':<10} {'Total':<8} {'Accuracy':<10} {'Avg Tokens':<12} {'Avg Steps':<10}")
    print("-" * 82)
    for ds in DATASETS:
        for method in METHODS:
            res = all_data[ds][method]
            n = len(res)
            correct = sum(1 for r in res.values() if r["is_correct"])
            avg_tok = np.mean([r["completion_tokens"] for r in res.values()])
            avg_steps = np.mean([r["num_steps"] for r in res.values()])
            print(f"{ds:<10} {method:<22} {correct:<10} {n:<8} {correct/n*100:>6.1f}%   {avg_tok:>8.0f}    {avg_steps:>6.1f}")
    print()

    # ═══════════════════════════════════════════
    # SECTION 2: Head-to-Head Comparison
    # ═══════════════════════════════════════════
    print_separator()
    print("SECTION 2: HEAD-TO-HEAD — BEAM vs LAZY PER PROBLEM")
    print_separator()

    for ds in DATASETS:
        beam = all_data[ds]["beam_search"]
        lazy = all_data[ds]["lazy_beam_search"]
        common = sorted(set(beam.keys()) & set(lazy.keys()))

        both_correct = beam_only = lazy_only = both_wrong = 0
        beam_wins = []
        lazy_wins = []

        for q in common:
            bc = beam[q]["is_correct"]
            lc = lazy[q]["is_correct"]
            if bc and lc:
                both_correct += 1
            elif bc and not lc:
                beam_only += 1
                beam_wins.append(q)
            elif not bc and lc:
                lazy_only += 1
                lazy_wins.append(q)
            else:
                both_wrong += 1

        print(f"\n{ds} ({len(common)} problems):")
        print(f"  Both correct:     {both_correct}")
        print(f"  Beam only correct: {beam_only}  ← lazy pruning lost these")
        print(f"  Lazy only correct: {lazy_only}  ← beam pruning lost these")
        print(f"  Both wrong:       {both_wrong}")

        if beam_wins:
            print(f"\n  BEAM WINS (beam=✓, lazy=✗):")
            print(f"  {'Q#':<6} {'GT':<12} {'Beam Pred':<12} {'Lazy Pred':<12} {'Beam MaxR':<10} {'Lazy MaxR':<10} {'Lazy 0%':<8} {'Beam Steps':<11} {'Lazy Steps'}")
            for q in beam_wins:
                b, l = beam[q], lazy[q]
                print(f"  Q{q:<4} {b['groundtruth']:<12} {b['predicted']:<12} {l['predicted']:<12} "
                      f"{b['max_reward']:<10.3f} {l['max_reward']:<10.3f} {l['zero_ratio']*100:>5.1f}%  "
                      f"{b['num_steps']:<11} {l['num_steps']}")

        if lazy_wins:
            print(f"\n  LAZY WINS (lazy=✓, beam=✗):")
            print(f"  {'Q#':<6} {'GT':<12} {'Beam Pred':<12} {'Lazy Pred':<12} {'Beam MaxR':<10} {'Lazy MaxR':<10}")
            for q in lazy_wins:
                b, l = beam[q], lazy[q]
                print(f"  Q{q:<4} {b['groundtruth']:<12} {b['predicted']:<12} {l['predicted']:<12} "
                      f"{b['max_reward']:<10.3f} {l['max_reward']:<10.3f}")

    # ═══════════════════════════════════════════
    # SECTION 3: PRM Misleading Cases
    # ═══════════════════════════════════════════
    print()
    print_separator()
    print("SECTION 3: PRM MISLEADING — HIGH SCORE BUT WRONG ANSWER")
    print_separator()
    print("Cases where max PRM score > 0.8 but answer is wrong:\n")

    for ds in DATASETS:
        for method in METHODS:
            res = all_data[ds][method]
            misleading = [(q, r) for q, r in res.items()
                          if not r["is_correct"] and r["max_reward"] > 0.8]
            if misleading:
                print(f"{ds} / {method}:")
                for q, r in sorted(misleading, key=lambda x: -x[1]["max_reward"]):
                    print(f"  Q{q}: GT={r['groundtruth']}, Pred={r['predicted']}, "
                          f"MaxPRM={r['max_reward']:.3f}, FinalPRM={r['final_reward']:.3f}, "
                          f"Steps={r['num_steps']}")
                    # Show reward trajectory summary
                    rh = r["reward_history"]
                    if len(rh) > 6:
                        first3 = [f"{x:.2f}" for x in rh[:3]]
                        last3 = [f"{x:.2f}" for x in rh[-3:]]
                        print(f"         Rewards: [{', '.join(first3)}, ..., {', '.join(last3)}]")
                    else:
                        print(f"         Rewards: [{', '.join(f'{x:.2f}' for x in rh)}]")
                print()

    # ═══════════════════════════════════════════
    # SECTION 4: Zero-Reward Artifact Analysis
    # ═══════════════════════════════════════════
    print_separator()
    print("SECTION 4: ZERO-REWARD ARTIFACT IN LAZY BEAM SEARCH")
    print_separator()

    for ds in DATASETS:
        lazy_res = all_data[ds]["lazy_beam_search"]
        beam_res = all_data[ds]["beam_search"]

        lazy_zeros = [r["zero_ratio"] for r in lazy_res.values()]
        beam_zeros = [r["zero_ratio"] for r in beam_res.values()]

        print(f"\n{ds}:")
        print(f"  Lazy beam — avg zero-reward ratio: {np.mean(lazy_zeros)*100:.1f}% "
              f"(min={min(lazy_zeros)*100:.0f}%, max={max(lazy_zeros)*100:.0f}%)")
        print(f"  Beam search — avg zero-reward ratio: {np.mean(beam_zeros)*100:.1f}% "
              f"(min={min(beam_zeros)*100:.0f}%, max={max(beam_zeros)*100:.0f}%)")

        # Problems with >80% zeros in lazy
        high_zero = [(q, r) for q, r in lazy_res.items() if r["zero_ratio"] > 0.8]
        if high_zero:
            print(f"  Problems with >80% zero rewards in lazy ({len(high_zero)} total):")
            for q, r in sorted(high_zero, key=lambda x: -x[1]["zero_ratio"])[:5]:
                print(f"    Q{q}: zero_ratio={r['zero_ratio']*100:.0f}%, "
                      f"correct={r['is_correct']}, steps={r['num_steps']}, "
                      f"non-zero rewards: {[f'{x:.2f}' for x in r['reward_history'] if x > 0.001]}")

    # ═══════════════════════════════════════════
    # SECTION 5: Premature Termination
    # ═══════════════════════════════════════════
    print()
    print_separator()
    print("SECTION 5: PREMATURE TERMINATION (no \\boxed{} in answer)")
    print_separator()

    for ds in DATASETS:
        for method in METHODS:
            res = all_data[ds][method]
            no_box = [(q, r) for q, r in res.items() if not r["has_boxed"]]
            if no_box:
                print(f"\n{ds} / {method}: {len(no_box)} trajectories without \\boxed{{}}:")
                for q, r in no_box:
                    last_text = r["text"][-120:].replace('\n', '\\n')
                    print(f"  Q{q}: GT={r['groundtruth']}, Pred={r['predicted']}, "
                          f"Steps={r['num_steps']}")
                    print(f"         ...{last_text}")
            else:
                print(f"\n{ds} / {method}: All trajectories contain \\boxed{{}}")

    # ═══════════════════════════════════════════
    # SECTION 6: Detailed Case Studies
    # ═══════════════════════════════════════════
    print()
    print_separator()
    print("SECTION 6: DETAILED CASE STUDIES — BEAM=✓, LAZY=✗")
    print_separator()

    for ds in DATASETS:
        beam = all_data[ds]["beam_search"]
        lazy = all_data[ds]["lazy_beam_search"]
        common = sorted(set(beam.keys()) & set(lazy.keys()))

        cases = [(q, beam[q], lazy[q]) for q in common
                 if beam[q]["is_correct"] and not lazy[q]["is_correct"]]

        if not cases:
            continue

        print(f"\n{'='*80}")
        print(f"  {ds}: {len(cases)} problems where beam search won")
        print(f"{'='*80}")

        for q, b, l in cases:
            print(f"\n  ── Q{q} (GT: {b['groundtruth']}) ──")
            print(f"  Question: {b['question'][:100]}...")
            print(f"  Beam:  pred={b['predicted']:<10} steps={b['num_steps']:<4} tokens={b['completion_tokens']:<6} "
                  f"mean_r={b['mean_reward']:.3f} max_r={b['max_reward']:.3f}")
            print(f"  Lazy:  pred={l['predicted']:<10} steps={l['num_steps']:<4} tokens={l['completion_tokens']:<6} "
                  f"mean_r={l['mean_reward']:.3f} max_r={l['max_reward']:.3f}")

            # Reward trajectory comparison
            brh = b["reward_history"]
            lrh = l["reward_history"]
            print(f"  Beam rewards ({len(brh)} steps): ", end="")
            if len(brh) > 8:
                print(f"[{', '.join(f'{x:.2f}' for x in brh[:4])}, ..., {', '.join(f'{x:.2f}' for x in brh[-4:])}]")
            else:
                print(f"[{', '.join(f'{x:.2f}' for x in brh)}]")

            print(f"  Lazy rewards ({len(lrh)} steps): ", end="")
            if len(lrh) > 8:
                print(f"[{', '.join(f'{x:.2f}' for x in lrh[:4])}, ..., {', '.join(f'{x:.2f}' for x in lrh[-4:])}]")
            else:
                print(f"[{', '.join(f'{x:.2f}' for x in lrh)}]")

            # Diagnosis
            zero_pct = l["zero_ratio"] * 100
            if zero_pct > 50:
                print(f"  ⚠ DIAGNOSIS: {zero_pct:.0f}% of lazy rewards are 0.0 — PRM guidance missing during expansion")
            if l["num_steps"] >= 38:
                print(f"  ⚠ DIAGNOSIS: Lazy hit near-max depth ({l['num_steps']} steps) — never found good path")
            if l["max_reward"] < 0.3:
                print(f"  ⚠ DIAGNOSIS: Lazy max PRM score only {l['max_reward']:.3f} — PRM never saw a promising path")

    print()
    print_separator()
    print("ANALYSIS COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()
