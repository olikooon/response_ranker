"""
Pipeline:
  1. Load dataset (synthetic or real HuggingFace dataset).
  2. Embed all contexts and candidate responses.
  3. Rank candidates by cosine similarity.
  4. Evaluate with MRR and Recall@k.
  5. Compare against random-baseline and show qualitative examples.
  6. Plot rank distribution.

Usage:
  python main.py                                         # synthetic data
  python main.py --dataset daily_dialog                 # DailyDialog
  python main.py --dataset blended_skill_talk           # BlendedSkillTalk
  python main.py --dataset conv_ai_2                    # PersonaChat/ConvAI2
  python main.py --dataset daily_dialog --max 500       # more examples
  python main.py --model all-mpnet-base-v2              # stronger model
  python main.py --distractors 19                       # 1-in-20 pool
"""

import argparse
import sys
import io
import random

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data_loader import build_dataset, get_pool_size
from real_data_loader import load_real_dataset, DATASET_REGISTRY, DATASET_INFO
from embedder import Embedder
from ranker import rank_all, baseline_random_rank
from evaluator import (
    compute_all_metrics,
    rank_distribution,
    print_metrics_table,
    print_rank_distribution,
)

random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    real_choices = list(DATASET_REGISTRY.keys())
    p = argparse.ArgumentParser(description="Response ranking case study")
    p.add_argument("--model", default="all-MiniLM-L6-v2",
                   help="sentence-transformers model name")
    p.add_argument("--distractors", type=int, default=9,
                   help="number of distractor candidates per example")
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic"] + real_choices,
                   help="dataset to use (default: synthetic)")
    p.add_argument("--split", default="test",
                   help="dataset split for real datasets (default: test)")
    p.add_argument("--max", type=int, default=300, dest="max_examples",
                   help="max examples to load from real datasets (default: 300)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip matplotlib output")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Qualitative inspection helper
# ─────────────────────────────────────────────────────────────────────────────

def show_qualitative_examples(
    examples,
    results,
    n: int = 3,
    show_top_k: int = 3,
) -> None:
    print(f"\n{'='*70}")
    print("QUALITATIVE EXAMPLES")
    print('='*70)

    indices = list(range(len(examples)))
    random.shuffle(indices)

    shown = 0
    for idx in indices:
        ex = examples[idx]
        res = results[idx]
        print(f"\n--- Example {idx + 1} (gold rank: {res.gold_rank}) ---")
        print(f"CONTEXT:\n  {ex.context[:300]}{'...' if len(ex.context) > 300 else ''}")
        print(f"\nGOLD RESPONSE (rank {res.gold_rank}, score {res.gold_score:.4f}):")
        print(f"  {ex.gold[:200]}")
        print(f"\nTOP-{show_top_k} RANKED CANDIDATES:")
        for rank, (cand, score) in enumerate(
            zip(res.ranked_candidates[:show_top_k], res.ranked_scores[:show_top_k]), 1
        ):
            marker = " <-- GOLD" if cand == ex.gold else ""
            print(f"  [{rank}] (score {score:.4f}){marker}")
            print(f"       {cand[:160]}...")
        shown += 1
        if shown >= n:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    embedding_metrics: dict,
    random_mrr: float,
    dist: dict,
    total: int,
    pool_size: int,
    model_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Response Ranking Evaluation  |  model: {model_name}  |  pool size: {pool_size}",
        fontsize=13, fontweight="bold"
    )

    # ── Left: metric comparison bar chart ────────────────────────────────────
    ax = axes[0]
    metric_names = list(embedding_metrics.keys())
    emb_values = list(embedding_metrics.values())

    # Random baselines for each metric
    random_baselines = {"MRR": random_mrr}
    for k_str in metric_names:
        if k_str.startswith("R@"):
            k = int(k_str[2:])
            random_baselines[k_str] = k / pool_size

    rand_values = [random_baselines[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, emb_values, width, label="Embedding ranker",
                   color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, rand_values, width, label="Random baseline",
                   color="#DD8452", edgecolor="white", alpha=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Metrics: Embedding vs Random")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)

    # ── Right: rank distribution histogram ───────────────────────────────────
    ax2 = axes[1]
    ranks = list(dist.keys())
    counts = list(dist.values())
    colors = ["#2ecc71" if r == 1 else "#4C72B0" if r <= 3 else "#95a5a6" for r in ranks]

    ax2.bar(ranks, [c / total * 100 for c in counts], color=colors, edgecolor="white")
    ax2.set_xlabel("Gold response rank")
    ax2.set_ylabel("% of examples")
    ax2.set_title("Distribution of Gold Response Ranks")
    ax2.set_xticks(ranks)
    ax2.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Rank 1 (exact hit)"),
        Patch(facecolor="#4C72B0", label="Rank 2–3"),
        Patch(facecolor="#95a5a6", label="Rank 4+"),
    ]
    ax2.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    out_path = "ranking_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  CHATBOT RESPONSE RANKING — CASE STUDY")
    print("=" * 60)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading dataset '{args.dataset}'  (distractors: {args.distractors})")
    if args.dataset == "synthetic":
        examples = build_dataset(num_distractors=args.distractors)
        print(f"      Built {len(examples)} synthetic examples across 10 domains")
    else:
        info = DATASET_INFO.get(args.dataset, "")
        if info:
            print(f"      {info}")
        try:
            examples = load_real_dataset(
                name=args.dataset,
                split=args.split,
                max_examples=args.max_examples,
                num_distractors=args.distractors,
            )
        except Exception as e:
            print(f"\n  ERROR loading '{args.dataset}': {e}")
            print("  Make sure 'datasets' is installed: pip install datasets")
            print("  Falling back to synthetic dataset.\n")
            examples = build_dataset(num_distractors=args.distractors)

    pool_size = get_pool_size(examples)
    print(f"      {len(examples)} examples | pool size: {pool_size} candidates each")

    # ── 2. Prepare inputs ─────────────────────────────────────────────────────
    print("\n[2/5] Preparing candidate pools (shuffled)")
    contexts = []
    candidates_list = []
    gold_indices = []

    for ex in examples:
        candidates, gold_idx = ex.candidates(shuffle=True)
        contexts.append(ex.context)
        candidates_list.append(candidates)
        gold_indices.append(gold_idx)

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    print(f"\n[3/5] Embedding with '{args.model}'")
    embedder = Embedder(model_name=args.model)

    print("      Contexts:")
    context_embs = embedder.embed_contexts(contexts)
    print("      Candidates:")
    candidate_embs_list = embedder.embed_candidates_batched(candidates_list)

    # ── 4. Rank ───────────────────────────────────────────────────────────────
    print("\n[4/5] Ranking candidates by cosine similarity")
    results = rank_all(context_embs, candidate_embs_list, candidates_list, gold_indices)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Computing metrics")

    k_values = [1, 2, 3, 5] if pool_size >= 5 else [1, 2, 3]
    metrics = compute_all_metrics(results, k_values=k_values)
    random_mrr = baseline_random_rank(pool_size)

    print_metrics_table(metrics, label=f"EMBEDDING RANKER  (pool={pool_size})")

    random_metrics = {"MRR": random_mrr}
    for k in k_values:
        random_metrics[f"R@{k}"] = round(k / pool_size, 4)
    print_metrics_table(random_metrics, label=f"RANDOM BASELINE   (pool={pool_size})")

    # Lift summary
    print("\n  LIFT OVER RANDOM:")
    for key in metrics:
        lift = (metrics[key] - random_metrics[key]) / random_metrics[key] * 100
        arrow = "^" if lift > 0 else "v"
        print(f"    {key:<6} [{arrow}] {lift:+.1f}%  "
              f"({random_metrics[key]:.4f} -> {metrics[key]:.4f})")

    # Rank distribution
    dist = rank_distribution(results)
    print(f"\nRANK DISTRIBUTION  (N={len(results)} examples):")
    print_rank_distribution(dist, len(results))

    # ── Qualitative ───────────────────────────────────────────────────────────
    show_qualitative_examples(examples, results, n=3, show_top_k=3)

    # ── Hardest cases ─────────────────────────────────────────────────────────
    worst = sorted(results, key=lambda r: r.gold_rank, reverse=True)[:3]
    worst_idx = [results.index(r) for r in worst]
    print(f"\n{'='*70}")
    print("HARDEST CASES (gold ranked lowest):")
    print('='*70)
    for idx, res in zip(worst_idx, worst):
        ex = examples[idx]
        print(f"\n  Context: {ex.context[:160]}...")
        print(f"  Gold response (rank {res.gold_rank}, score {res.gold_score:.4f}):")
        print(f"    {ex.gold[:180]}...")
        print(f"  Top-1 response (score {res.ranked_scores[0]:.4f}):")
        print(f"    {res.ranked_candidates[0][:180]}...")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            plot_results(metrics, random_mrr, dist, len(results), pool_size,
                     f"{args.model} | {args.dataset}")
        except Exception as e:
            print(f"\n(Plot skipped: {e})")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
