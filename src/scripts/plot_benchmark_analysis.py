#!/usr/bin/env python3
"""
Benchmark analysis visualization for AMC23 & AIME24:
lazy_beam_search vs beam_search with Qwen2.5-3B + Skywork PRM.
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict


BASE_DIR = "/scratch/msa6093/compute-optimal-tts/output_benchmark_amc_aime"
RESULT_SUBDIR = "Qwen2.5-3B-Instruct/Skywork-o1-Open-PRM-Qwen-2.5-7B/40_4_1"
OUTPUT_PATH = "/scratch/msa6093/compute-optimal-tts/benchmark_analysis.png"

CONFIGS = [
    ("AMC23", "beam_search"),
    ("AMC23", "lazy_beam_search"),
    ("AIME24", "beam_search"),
    ("AIME24", "lazy_beam_search"),
]

# Paper baselines (Qwen2.5-3B-Instruct, beam search)
PAPER_BASELINES = {"AMC23": 0.65, "AIME24": 0.20}


def load_results(dataset, method):
    """Load all per-question results for a dataset/method combination."""
    result_dir = os.path.join(BASE_DIR, f"{dataset}_{method}", RESULT_SUBDIR)
    results = []

    # Find all question directories
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
        gt = data.get("groundtruth", "")
        pred = output.get("extracted_answer", "")
        is_correct = data["result"].get("majority_vote", 0) == 1

        results.append({
            "q_idx": q_idx,
            "question": data.get("question", "")[:80],
            "groundtruth": gt,
            "predicted": pred,
            "is_correct": is_correct,
            "reward_history": output.get("reward_history", []),
            "token_history": output.get("token_history", []),
            "completion_tokens": output.get("completion_tokens", 0),
            "tree_completion_tokens": output.get("tree_completion_tokens", 0),
        })
    return results


def main():
    # Load all results
    all_results = {}
    for dataset, method in CONFIGS:
        key = f"{dataset}_{method}"
        all_results[key] = load_results(dataset, method)
        n = len(all_results[key])
        correct = sum(1 for r in all_results[key] if r["is_correct"])
        print(f"{key}: {correct}/{n} = {correct/n*100:.1f}%")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("AMC23 & AIME24 Benchmark Analysis\nQwen2.5-3B-Instruct + Skywork-o1-Open-PRM-Qwen-2.5-7B",
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Panel 1: Accuracy Comparison ──
    ax1 = fig.add_subplot(2, 2, 1)
    datasets = ["AMC23", "AIME24"]
    methods = ["beam_search", "lazy_beam_search"]
    colors = {"beam_search": "#2196F3", "lazy_beam_search": "#FF9800"}
    x = np.arange(len(datasets))
    width = 0.3

    for i, method in enumerate(methods):
        accs = []
        for ds in datasets:
            key = f"{ds}_{method}"
            res = all_results[key]
            accs.append(sum(1 for r in res if r["is_correct"]) / len(res) * 100)
        bars = ax1.bar(x + i * width - width/2, accs, width, label=method.replace("_", " ").title(),
                       color=colors[method], edgecolor='black', linewidth=0.5)
        for bar, acc in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Paper baselines
    for j, ds in enumerate(datasets):
        ax1.hlines(PAPER_BASELINES[ds] * 100, j - 0.5, j + 0.5,
                   colors='red', linestyles='--', linewidth=2)
        ax1.text(j + 0.35, PAPER_BASELINES[ds] * 100 + 1,
                 f'Paper: {PAPER_BASELINES[ds]*100:.0f}%', color='red', fontsize=9, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Method Comparison vs Paper Baseline", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 80)
    ax1.grid(axis='y', alpha=0.3)

    # ── Panel 2: Token Efficiency ──
    ax2 = fig.add_subplot(2, 2, 2)
    markers = {"AMC23": "o", "AIME24": "s"}

    for ds in datasets:
        for method in methods:
            key = f"{ds}_{method}"
            res = all_results[key]
            acc = sum(1 for r in res if r["is_correct"]) / len(res) * 100
            avg_tokens = np.mean([r["completion_tokens"] for r in res])
            ax2.scatter(avg_tokens, acc, s=200, marker=markers[ds], color=colors[method],
                        edgecolors='black', linewidth=1.5, zorder=5)
            label = f"{ds}\n{method.replace('_', ' ')}"
            offset_x = 100 if "lazy" in method else -100
            ax2.annotate(label, (avg_tokens, acc),
                         textcoords="offset points", xytext=(offset_x, -5),
                         fontsize=8, ha='center',
                         arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax2.set_xlabel("Avg Completion Tokens / Problem", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Token Efficiency: More Tokens ≠ Better Accuracy", fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='AMC23'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='AIME24'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["beam_search"], markersize=10, label='Beam Search'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["lazy_beam_search"], markersize=10, label='Lazy Beam'),
    ]
    ax2.legend(handles=legend_elements, fontsize=9, loc='upper right')

    # ── Panel 3: PRM Reward Trajectories (example problems) ──
    ax3 = fig.add_subplot(2, 2, 3)

    # Find problems where beam=correct, lazy=wrong for best examples
    examples = []
    for ds in datasets:
        beam_res = {r["q_idx"]: r for r in all_results[f"{ds}_beam_search"]}
        lazy_res = {r["q_idx"]: r for r in all_results[f"{ds}_lazy_beam_search"]}
        for q_idx in beam_res:
            if q_idx in lazy_res:
                b = beam_res[q_idx]
                l = lazy_res[q_idx]
                if b["is_correct"] and not l["is_correct"] and len(b["reward_history"]) > 3:
                    examples.append((ds, q_idx, b, l))

    # Pick up to 3 best examples
    examples = sorted(examples, key=lambda x: len(x[2]["reward_history"]), reverse=True)[:3]

    for i, (ds, q_idx, beam_r, lazy_r) in enumerate(examples):
        beam_rh = beam_r["reward_history"]
        lazy_rh = lazy_r["reward_history"]

        steps_b = range(1, len(beam_rh) + 1)
        steps_l = range(1, len(lazy_rh) + 1)

        ax3.plot(steps_b, beam_rh, '-o', markersize=3, color=colors["beam_search"],
                 alpha=0.7, label=f'{ds} Q{q_idx} beam (✓)' if i == 0 else f'Q{q_idx} beam (✓)')
        ax3.plot(steps_l, lazy_rh, '-x', markersize=4, color=colors["lazy_beam_search"],
                 alpha=0.7, label=f'{ds} Q{q_idx} lazy (✗)' if i == 0 else f'Q{q_idx} lazy (✗)')

    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel("Reasoning Step", fontsize=12)
    ax3.set_ylabel("PRM Score", fontsize=12)
    ax3.set_title("PRM Trajectories: Beam (correct) vs Lazy (wrong)", fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2, loc='upper left')
    ax3.grid(alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)

    # ── Panel 4: PRM Score Distribution (correct vs incorrect) ──
    ax4 = fig.add_subplot(2, 2, 4)

    box_data = []
    box_labels = []
    box_colors_list = []

    for method in methods:
        for correctness in [True, False]:
            scores = []
            for ds in datasets:
                key = f"{ds}_{method}"
                for r in all_results[key]:
                    if r["is_correct"] == correctness:
                        # Filter out 0.0 artifacts for lazy
                        rh = [s for s in r["reward_history"] if s > 0.001] if "lazy" in method else r["reward_history"]
                        if rh:
                            scores.append(np.mean(rh))
            box_data.append(scores)
            status = "Correct" if correctness else "Wrong"
            method_short = "Beam" if method == "beam_search" else "Lazy"
            box_labels.append(f"{method_short}\n{status}")
            box_colors_list.append(colors[method] if correctness else '#EF5350')

    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], box_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, data in enumerate(box_data):
        jitter = np.random.normal(0, 0.04, len(data))
        ax4.scatter(np.full(len(data), i + 1) + jitter, data, alpha=0.5, s=20, color='black', zorder=5)

    ax4.set_ylabel("Mean PRM Score (per problem)", fontsize=12)
    ax4.set_title("PRM Score Separation: Correct vs Wrong Answers", fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
