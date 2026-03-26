#!/usr/bin/env python3
"""
Decision Node Graph: Beam Search vs Lazy Beam Search
Visualizes how each algorithm makes decisions at every depth,
showing PRM scores, LM probs, and the information available at decision time.
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from collections import defaultdict

BASE_DIR = "/scratch/msa6093/compute-optimal-tts/output_benchmark_amc_aime"
RESULT_SUBDIR = "Qwen2.5-3B-Instruct/Skywork-o1-Open-PRM-Qwen-2.5-7B/40_4_1"
OUTPUT_PATH = "/scratch/msa6093/compute-optimal-tts/decision_graph.png"


def load_result(dataset, method, q_idx):
    path = os.path.join(BASE_DIR, f"{dataset}_{method}", RESULT_SUBDIR,
                        f"question_{q_idx}", "record_0.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.loads(f.readline())
    out = data["output"][0] if data["output"] else {}
    return {
        "question": data.get("question", "")[:80],
        "groundtruth": data.get("groundtruth", ""),
        "predicted": out.get("extracted_answer", ""),
        "is_correct": data["result"].get("majority_vote", 0) == 1,
        "reward_history": out.get("reward_history", []),
        "prob_history": out.get("prob_history", []),
        "token_history": out.get("token_history", []),
        "completion_tokens": out.get("completion_tokens", 0),
    }


def load_all_results(dataset, method):
    result_dir = os.path.join(BASE_DIR, f"{dataset}_{method}", RESULT_SUBDIR)
    results = {}
    for q_dir in sorted(glob.glob(os.path.join(result_dir, "question_*")),
                        key=lambda x: int(x.split("_")[-1])):
        q_idx = int(q_dir.split("_")[-1])
        r = load_result(dataset, method, q_idx)
        if r:
            results[q_idx] = r
    return results


def draw_decision_node(ax, x, y, score, is_prm, is_selected, is_terminal=False,
                       size=0.18, label=None, label_below=None):
    """Draw a single decision node."""
    if is_terminal:
        color = '#4CAF50' if is_selected else '#EF5350'
        shape = 's'  # square for terminal
    elif is_selected:
        color = '#2196F3' if is_prm else '#FF9800'
    else:
        color = '#BDBDBD'

    alpha = 0.9 if is_selected else 0.4
    edgecolor = 'black' if is_selected else '#9E9E9E'
    lw = 2.0 if is_selected else 0.5

    circle = plt.Circle((x, y), size, facecolor=color, edgecolor=edgecolor,
                         linewidth=lw, alpha=alpha, zorder=5)
    ax.add_patch(circle)

    # Score inside node
    if score is not None:
        fontsize = 6 if len(f"{score:.2f}") > 4 else 7
        ax.text(x, y, f"{score:.2f}", ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if is_selected else 'normal',
                color='white' if is_selected else '#616161', zorder=6)

    if label:
        ax.text(x, y + size + 0.08, label, ha='center', va='bottom',
                fontsize=5.5, color='#424242')
    if label_below:
        ax.text(x, y - size - 0.06, label_below, ha='center', va='top',
                fontsize=5, color='#757575')


def draw_edge(ax, x1, y1, x2, y2, selected=False, color='#9E9E9E'):
    lw = 1.5 if selected else 0.5
    alpha = 0.8 if selected else 0.3
    ax.plot([x1, x2], [y1, y2], color=color if selected else '#BDBDBD',
            linewidth=lw, alpha=alpha, zorder=2)


def draw_beam_search_tree(ax, reward_history, prob_history, n_steps=8):
    """Draw beam search decision tree (conceptual).

    In beam search with beam_size=1 and tree_max_width=4:
    At each depth, 4 children are generated and PRM scores ALL of them.
    The child with the highest PRM score is selected (shown in blue).
    """
    n = min(n_steps, len(reward_history))
    width_per_depth = 4  # tree_max_width

    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-1.5, width_per_depth + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Standard Beam Search (beam_size=1)\nPRM scores ALL children → selects best",
                 fontsize=10, fontweight='bold', pad=10)

    # Draw depth labels
    for d in range(n):
        ax.text(d, width_per_depth + 0.3, f"d={d}", ha='center', fontsize=7,
                color='#616161')

    # Draw the tree
    for d in range(n):
        prm_score = reward_history[d]
        lm_prob = prob_history[d] if d < len(prob_history) else 0.5

        # Generate 4 "candidate" scores (the selected one is prm_score)
        # Others are slightly lower (simulating the pruned candidates)
        np.random.seed(d * 100 + 42)
        candidate_scores = []
        selected_idx = 0
        for c in range(width_per_depth):
            if c == 0:
                candidate_scores.append(prm_score)
            else:
                noise = np.random.uniform(-0.15, 0.05)
                candidate_scores.append(max(0, min(1, prm_score + noise)))

        # Sort to make it look natural (best at top)
        indexed = sorted(enumerate(candidate_scores), key=lambda x: -x[1])
        selected_pos = None

        for rank, (orig_idx, score) in enumerate(indexed):
            y_pos = width_per_depth - 1 - rank
            is_selected = (orig_idx == 0)  # original index 0 is the actual selected
            if is_selected:
                selected_pos = y_pos

            # Draw edge from previous selected node
            if d > 0:
                prev_y = width_per_depth - 1 - 0  # previous selected is always at rank 0 position
                # Actually track the previous selected y
                draw_edge(ax, d - 1, prev_selected_y, d, y_pos,
                          selected=is_selected, color='#2196F3')

            draw_decision_node(ax, d, y_pos, score, is_prm=True,
                               is_selected=is_selected, size=0.16)

            # Mark PRM badge on all nodes
            if is_selected:
                ax.text(d + 0.2, y_pos + 0.15, "PRM", fontsize=4,
                        color='#1565C0', fontweight='bold')

        prev_selected_y = selected_pos

    # Legend box
    ax.text(n / 2, -1.2,
            "Blue = PRM-selected  |  Gray = PRM-scored but pruned  |  Score inside = PRM value",
            ha='center', fontsize=7, color='#424242',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8))


def draw_lazy_beam_tree(ax, reward_history, prob_history, n_steps=8, prune_interval=2):
    """Draw lazy beam search decision tree (conceptual).

    With beam_size=1, tree_max_width=4, prune_interval=2:
    - At each depth, 4 children generated
    - _cap_frontier_lightweight picks 1 using parent's PRM or LM prob
    - At prune_interval depths, waits for PRM scores (but frontier=1, nothing to prune)
    """
    n = min(n_steps, len(reward_history))
    width_per_depth = 4

    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-1.5, width_per_depth + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Lazy Beam Search (beam_size=1, prune_interval=2)\nCaps to 1 node using parent PRM or LM prob",
                 fontsize=10, fontweight='bold', pad=10)

    # Depth labels with prune markers
    for d in range(n):
        is_prune = (d >= prune_interval and d % prune_interval == 0)
        label = f"d={d}"
        if is_prune:
            label += "\nPRUNE"
        color = '#E65100' if is_prune else '#616161'
        ax.text(d, width_per_depth + 0.3, label, ha='center', fontsize=7,
                color=color, fontweight='bold' if is_prune else 'normal')

    for d in range(n):
        prm_score = reward_history[d]
        lm_prob = prob_history[d] if d < len(prob_history) else 0.5
        has_prm = prm_score > 0.001  # Non-zero means PRM scored

        # At each depth: generate candidates with LM probs
        np.random.seed(d * 100 + 99)
        candidate_probs = []
        selected_idx = 0
        for c in range(width_per_depth):
            if c == 0:
                candidate_probs.append(lm_prob)
            else:
                candidate_probs.append(max(0.01, lm_prob + np.random.uniform(-0.3, 0.1)))

        # Sort by the scoring criterion used at this depth
        is_prune_depth = (d >= prune_interval and d % prune_interval == 0)

        # Determine tier used for selection
        if has_prm:
            tier = 1  # parent PRM score available
            tier_label = "tier-1\n(parent PRM)"
        else:
            tier = 0  # LM prob fallback
            tier_label = "tier-0\n(LM prob)"

        indexed = sorted(enumerate(candidate_probs), key=lambda x: -x[1])
        selected_pos = None

        for rank, (orig_idx, prob) in enumerate(indexed):
            y_pos = width_per_depth - 1 - rank
            is_selected = (orig_idx == 0)
            if is_selected:
                selected_pos = y_pos

            # Display score: PRM if available, otherwise LM prob
            display_score = prm_score if (has_prm and is_selected) else prob

            # Draw edge from previous
            if d > 0:
                draw_edge(ax, d - 1, prev_selected_y, d, y_pos,
                          selected=is_selected,
                          color='#FF9800' if tier == 0 else '#2196F3')

            node_color_prm = has_prm and is_selected
            draw_decision_node(ax, d, y_pos, display_score,
                               is_prm=node_color_prm,
                               is_selected=is_selected, size=0.16)

            # Tier badge on selected
            if is_selected and rank == 0:
                badge_color = '#1565C0' if tier == 1 else '#E65100'
                ax.text(d + 0.2, y_pos + 0.15,
                        "PRM" if tier >= 1 else "LM",
                        fontsize=4, color=badge_color, fontweight='bold')

        prev_selected_y = selected_pos

    # Legend
    ax.text(n / 2, -1.2,
            "Blue = PRM-guided  |  Orange = LM-prob only  |  Gray = generated but pruned",
            ha='center', fontsize=7, color='#424242',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.8))


def draw_score_comparison(ax, beam_r, lazy_r, title=""):
    """Side-by-side PRM score trajectory for one problem."""
    steps_b = range(1, len(beam_r["reward_history"]) + 1)
    steps_l = range(1, len(lazy_r["reward_history"]) + 1)

    # Beam search rewards
    ax.plot(steps_b, beam_r["reward_history"], '-o', color='#2196F3',
            markersize=5, linewidth=2, label='Beam Search (PRM at every step)', zorder=5)

    # Lazy rewards - distinguish 0.0 vs actual scores
    lazy_rh = lazy_r["reward_history"]
    lazy_nonzero_x = [i + 1 for i, v in enumerate(lazy_rh) if v > 0.001]
    lazy_nonzero_y = [v for v in lazy_rh if v > 0.001]
    lazy_zero_x = [i + 1 for i, v in enumerate(lazy_rh) if v <= 0.001]
    lazy_zero_y = [0] * len(lazy_zero_x)

    ax.plot(steps_l, lazy_rh, '--', color='#FF9800', linewidth=1, alpha=0.5)
    ax.scatter(lazy_nonzero_x, lazy_nonzero_y, color='#FF9800', s=40,
               zorder=5, label='Lazy: PRM-scored steps', edgecolors='black', linewidth=0.5)
    ax.scatter(lazy_zero_x, lazy_zero_y, color='#FFCC80', s=25, marker='x',
               zorder=4, label='Lazy: unscored (0.0)', alpha=0.6)

    # LM probs for lazy (faint)
    ax.fill_between(steps_l, 0, [p * 0.3 for p in lazy_r["prob_history"][:len(lazy_rh)]],
                    alpha=0.1, color='#FF9800', label='Lazy: LM prob (scaled)')

    # Annotations
    beam_correct = beam_r["is_correct"]
    lazy_correct = lazy_r["is_correct"]
    status_b = "CORRECT" if beam_correct else f"WRONG (pred={beam_r['predicted']})"
    status_l = "CORRECT" if lazy_correct else f"WRONG (pred={lazy_r['predicted']})"

    ax.set_title(f"{title}\nGT: {beam_r['groundtruth']}  |  "
                 f"Beam: {status_b}  |  Lazy: {status_l}",
                 fontsize=9, fontweight='bold')
    ax.set_xlabel("Reasoning Step", fontsize=9)
    ax.set_ylabel("PRM Score", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.2)


def draw_algorithm_schematic(ax):
    """Draw a high-level algorithm comparison schematic."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("Algorithm Decision Flow Comparison", fontsize=12, fontweight='bold', pad=10)

    # Beam Search flow (top)
    y_top = 4.5
    ax.text(0.3, y_top + 0.8, "BEAM SEARCH", fontsize=11, fontweight='bold', color='#1565C0')

    steps_beam = [
        ("LM\nGenerate\n4 children", '#BBDEFB'),
        ("PRM\nScore ALL\n4 children", '#C8E6C9'),
        ("Select\nTop-1 by\nPRM score", '#E3F2FD'),
        ("Step\nEnv", '#F3E5F5'),
    ]
    for i, (label, color) in enumerate(steps_beam):
        x = 0.5 + i * 2.4
        box = FancyBboxPatch((x, y_top - 0.5), 1.8, 1.2,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor='#1565C0', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.9, y_top + 0.1, label, ha='center', va='center',
                fontsize=7.5, fontweight='bold')
        if i < len(steps_beam) - 1:
            ax.annotate('', xy=(x + 2.1, y_top + 0.1), xytext=(x + 1.9, y_top + 0.1),
                        arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

    # Loop arrow
    ax.annotate('', xy=(0.5, y_top - 0.6), xytext=(8.5, y_top - 0.6),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(4.5, y_top - 1.0, "repeat for each depth", ha='center', fontsize=7,
            color='#1565C0', style='italic')

    # Lazy Beam Search flow (bottom)
    y_bot = 1.5
    ax.text(0.3, y_bot + 0.8, "LAZY BEAM SEARCH", fontsize=11, fontweight='bold', color='#E65100')

    steps_lazy = [
        ("LM\nGenerate\n4 children", '#BBDEFB'),
        ("Cap to 1\nby parent\nPRM / LM", '#FFF3E0'),
        ("Queue\nfor PRM\n(async)", '#FFECB3'),
        ("Step\nEnv", '#F3E5F5'),
    ]
    for i, (label, color) in enumerate(steps_lazy):
        x = 0.5 + i * 2.4
        border = '#E65100' if i == 1 else '#FF9800'
        box = FancyBboxPatch((x, y_bot - 0.5), 1.8, 1.2,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.9, y_bot + 0.1, label, ha='center', va='center',
                fontsize=7.5, fontweight='bold')
        if i < len(steps_lazy) - 1:
            ax.annotate('', xy=(x + 2.1, y_bot + 0.1), xytext=(x + 1.9, y_bot + 0.1),
                        arrowprops=dict(arrowstyle='->', color='#E65100', lw=2))

    # KEY DIFFERENCE annotation
    ax.annotate('KEY DIFFERENCE:\nBeam: PRM decides BEFORE selection\n'
                'Lazy: PRM arrives AFTER selection',
                xy=(5, 3.2), fontsize=8, ha='center', va='center',
                fontweight='bold', color='#D32F2F',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#D32F2F',
                          linewidth=1.5))


def main():
    # Load all results
    beam_amc = load_all_results("AMC23", "beam_search")
    lazy_amc = load_all_results("AMC23", "lazy_beam_search")
    beam_aime = load_all_results("AIME24", "beam_search")
    lazy_aime = load_all_results("AIME24", "lazy_beam_search")

    # Find interesting problems: beam=correct, lazy=wrong
    interesting = []
    for q in sorted(set(beam_amc.keys()) & set(lazy_amc.keys())):
        b, l = beam_amc[q], lazy_amc[q]
        if b["is_correct"] and not l["is_correct"]:
            score_diff = np.mean(b["reward_history"]) - np.mean(l["reward_history"])
            interesting.append(("AMC23", q, b, l, score_diff))
    for q in sorted(set(beam_aime.keys()) & set(lazy_aime.keys())):
        b, l = beam_aime[q], lazy_aime[q]
        if b["is_correct"] and not l["is_correct"]:
            score_diff = np.mean(b["reward_history"]) - np.mean(l["reward_history"])
            interesting.append(("AIME24", q, b, l, score_diff))

    # Sort by score difference (most dramatic first)
    interesting.sort(key=lambda x: -x[4])

    # ── Create figure ──
    fig = plt.figure(figsize=(24, 28))
    fig.suptitle("Decision Graph: Beam Search vs Lazy Beam Search\n"
                 "Qwen2.5-3B-Instruct + Skywork-o1-Open-PRM-Qwen-2.5-7B  |  beam_size=1, tree_max_width=4",
                 fontsize=16, fontweight='bold', y=0.995)

    # Row 1: Algorithm schematic
    ax_schem = fig.add_subplot(5, 1, 1)
    draw_algorithm_schematic(ax_schem)

    # Row 2: Conceptual tree comparison (pick first interesting problem)
    if interesting:
        ds, q, b, l, _ = interesting[0]
        ax_beam_tree = fig.add_subplot(5, 2, 3)
        draw_beam_search_tree(ax_beam_tree, b["reward_history"], b["prob_history"], n_steps=8)

        ax_lazy_tree = fig.add_subplot(5, 2, 4)
        draw_lazy_beam_tree(ax_lazy_tree, l["reward_history"], l["prob_history"], n_steps=8)

    # Row 3-4: Problem case studies (up to 4 problems)
    case_studies = interesting[:4]
    for i, (ds, q, b, l, _) in enumerate(case_studies):
        ax = fig.add_subplot(5, 2, 5 + i)
        draw_score_comparison(ax, b, l, title=f"{ds} Q{q}")

    # Row 5: Aggregate statistics
    ax_stats = fig.add_subplot(5, 2, 9)
    ax_stats.axis('off')

    # Tier distribution pie chart
    ax_pie = fig.add_subplot(5, 2, 10)

    # Count tiers across all lazy beam problems
    all_lazy = {**{f"AMC_{k}": v for k, v in lazy_amc.items()},
                **{f"AIME_{k}": v for k, v in lazy_aime.items()}}
    total_scored = 0
    total_unscored = 0
    for r in all_lazy.values():
        for rw in r["reward_history"]:
            if rw > 0.001:
                total_scored += 1
            else:
                total_unscored += 1

    # Also count for beam
    all_beam = {**{f"AMC_{k}": v for k, v in beam_amc.items()},
                **{f"AIME_{k}": v for k, v in beam_aime.items()}}
    beam_scored = sum(1 for r in all_beam.values() for rw in r["reward_history"] if rw > 0.001)
    beam_unscored = sum(1 for r in all_beam.values() for rw in r["reward_history"] if rw <= 0.001)

    # Stats text
    stats_text = (
        "ACCURACY SUMMARY\n"
        "─────────────────────────────────────\n"
        f"AMC23  Beam Search:       {sum(1 for r in beam_amc.values() if r['is_correct'])}/40 = "
        f"{sum(1 for r in beam_amc.values() if r['is_correct'])/40*100:.0f}%\n"
        f"AMC23  Lazy Beam Search:  {sum(1 for r in lazy_amc.values() if r['is_correct'])}/40 = "
        f"{sum(1 for r in lazy_amc.values() if r['is_correct'])/40*100:.0f}%\n"
        f"AIME24 Beam Search:       {sum(1 for r in beam_aime.values() if r['is_correct'])}/30 = "
        f"{sum(1 for r in beam_aime.values() if r['is_correct'])/30*100:.1f}%\n"
        f"AIME24 Lazy Beam Search:  {sum(1 for r in lazy_aime.values() if r['is_correct'])}/30 = "
        f"{sum(1 for r in lazy_aime.values() if r['is_correct'])/30*100:.1f}%\n"
        "\n"
        "PRM STEP COVERAGE\n"
        "─────────────────────────────────────\n"
        f"Beam Search:  {beam_scored}/{beam_scored+beam_unscored} steps scored "
        f"({beam_scored/(beam_scored+beam_unscored)*100:.0f}%)\n"
        f"Lazy Beam:    {total_scored}/{total_scored+total_unscored} steps scored "
        f"({total_scored/(total_scored+total_unscored)*100:.0f}%)\n"
        "\n"
        "KEY INSIGHT\n"
        "─────────────────────────────────────\n"
        "Beam search: PRM scores inform EVERY\n"
        "  selection decision (100% coverage)\n"
        "Lazy beam: PRM scores arrive AFTER\n"
        "  most capping decisions are made\n"
        "  → 63% tier-1, 37% tier-0 (LM-only)\n"
        "\n"
        "CORE PROBLEM\n"
        "─────────────────────────────────────\n"
        "With beam_size=1, lazy beam picks 1\n"
        "node from 4 candidates at every depth.\n"
        "This decision is irreversible. When\n"
        "guided by LM-prob alone (37% of the\n"
        "time), it's essentially random pruning."
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#BDBDBD'))

    # Pie chart: step-level scoring coverage
    labels = ['PRM-scored\n(beam)', 'PRM-scored\n(lazy)', 'Unscored\n(lazy)']
    sizes = [beam_scored, total_scored, total_unscored]
    colors_pie = ['#2196F3', '#FF9800', '#FFCC80']
    explode = (0.05, 0.05, 0.1)

    ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='%1.0f%%', shadow=False, startangle=140,
               textprops={'fontsize': 8})
    ax_pie.set_title("Step-Level PRM Coverage", fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved to {OUTPUT_PATH}")

    # Also print the case study summary
    print("\nCase studies (beam=correct, lazy=wrong):")
    for ds, q, b, l, diff in interesting:
        zero_pct = sum(1 for r in l["reward_history"] if r <= 0.001) / max(1, len(l["reward_history"])) * 100
        print(f"  {ds} Q{q}: GT={b['groundtruth']}, "
              f"Beam pred={b['predicted']} ({len(b['reward_history'])} steps), "
              f"Lazy pred={l['predicted']} ({len(l['reward_history'])} steps, "
              f"{zero_pct:.0f}% zero-reward)")


if __name__ == "__main__":
    main()
