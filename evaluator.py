"""
Dialogue retrieval evaluation metrics.
MRR. Mean Reciprocal Rank: average of 1/rank across all examples. range [0, 1]; perfect ranker = 1.0.
R@k. Recall at k: fraction of examples where the gold response, appears in the top-k ranked candidates, R@1 = exact top-1 accuracy.Rank Distribution — histogram of gold ranks (diagnostic).
"""

from typing import List, Dict
import numpy as np
from ranker import RankingResult


def mean_reciprocal_rank(results: List[RankingResult]) -> float:
    return float(np.mean([r.reciprocal_rank for r in results]))


def recall_at_k(results: List[RankingResult], k: int) -> float:
    hits = sum(1 for r in results if r.gold_rank <= k)
    return hits / len(results)


def compute_all_metrics(
    results: List[RankingResult],
    k_values: List[int] = (1, 2, 3, 5),
) -> Dict[str, float]:
    metrics = {"MRR": mean_reciprocal_rank(results)}
    for k in k_values:
        metrics[f"R@{k}"] = recall_at_k(results, k)
    return metrics


def rank_distribution(results: List[RankingResult]) -> Dict[int, int]:
    dist: Dict[int, int] = {}
    for r in results:
        dist[r.gold_rank] = dist.get(r.gold_rank, 0) + 1
    return dict(sorted(dist.items()))


def print_metrics_table(metrics: Dict[str, float], label: str = "") -> None:
    header = f"{'Metric':<10} {'Score':>8}"
    sep = "-" * 20
    if label:
        print(f"\n{'='*20}  {label}  {'='*20}")
    print(header)
    print(sep)
    for name, value in metrics.items():
        print(f"{name:<10} {value:>8.4f}")
    print(sep)


def print_rank_distribution(dist: Dict[int, int], total: int) -> None:
    print(f"\n{'Rank':<6} {'Count':>6}  {'%':>6}  Bar")
    print("-" * 40)
    max_count = max(dist.values()) if dist else 1
    for rank, count in dist.items():
        pct = 100 * count / total
        bar_len = int(30 * count / max_count)
        bar = "#" * bar_len
        print(f"{rank:<6} {count:>6}  {pct:>5.1f}%  {bar}")
