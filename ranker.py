"""
Response ranking via cosine similarity.
Because embeddings are L2-normalised, cosine similarity reduces to
a dot product: sim(q, d) = q · d, avoiding the sqrt overhead.
For each conversation example the ranker:
  1. Computes similarity between context embedding and every candidate.
  2. Returns candidates sorted from most to least similar.
  3. Locates the gold response in the ranked list.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class RankingResult:
    gold_rank: int             # 1-based rank of the gold response
    gold_score: float          # cosine similarity score of gold
    ranked_candidates: List[str] = field(default_factory=list)
    ranked_scores: List[float] = field(default_factory=list)

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.gold_rank


def rank_candidates(
    context_emb: np.ndarray,       # shape (dim,)
    candidate_embs: np.ndarray,    # shape (num_candidates, dim)
    candidates: List[str],
    gold_idx: int,
) -> RankingResult:
    """Rank candidates by cosine similarity to context; return structured result."""
    # Dot product == cosine sim for normalised vectors
    scores = candidate_embs @ context_emb          # shape (num_candidates,)
    ranked_indices = np.argsort(-scores)            # descending

    ranked_candidates = [candidates[i] for i in ranked_indices]
    ranked_scores = [float(scores[i]) for i in ranked_indices]

    gold_rank = int(np.where(ranked_indices == gold_idx)[0][0]) + 1  # 1-based

    return RankingResult(
        gold_rank=gold_rank,
        gold_score=float(scores[gold_idx]),
        ranked_candidates=ranked_candidates,
        ranked_scores=ranked_scores,
    )


def rank_all(
    context_embs: np.ndarray,              # (N, dim)
    candidate_embs_list: List[np.ndarray], # list of N arrays, each (C, dim)
    candidates_list: List[List[str]],      # list of N candidate pools
    gold_indices: List[int],               # gold position in each pool
) -> List[RankingResult]:
    """Vectorised ranking over all examples."""
    results = []
    for i, (ctx_emb, cand_embs, candidates, gold_idx) in enumerate(
        zip(context_embs, candidate_embs_list, candidates_list, gold_indices)
    ):
        results.append(rank_candidates(ctx_emb, cand_embs, candidates, gold_idx))
    return results


def baseline_random_rank(num_candidates: int, num_trials: int = 10_000) -> float:
    """Expected MRR for a random ranker (analytical: 1/N * sum(1/k) for k=1..N)."""
    return sum(1.0 / k for k in range(1, num_candidates + 1)) / num_candidates
