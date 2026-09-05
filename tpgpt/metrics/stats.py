"""Statistical comparison of methods (paper Sec. IV-B).

The paper does not read winners off box plots. It runs a Mann-Whitney U test
(ref. [56]) between every pair of methods and awards a point for each comparison
a method wins at ``p < 0.05``; the ranking is by points, and methods that beat
the same number of others share a rank.

Kept here because the deferred Sec. IV baseline comparison needs it, and because
the ranking rule is easy to get subtly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu


@dataclass
class RankingResult:
    """Outcome of a pairwise comparison over several methods."""

    points: dict[str, int]
    ranks: dict[str, int]
    wins: list[tuple[str, str, float]]

    def ordered(self) -> list[str]:
        """Method names best-first."""
        return sorted(self.points, key=lambda name: (-self.points[name], name))


def rank_methods(
    scores: dict[str, np.ndarray], alpha: float = 0.05, lower_is_better: bool = True
) -> RankingResult:
    """Rank methods by pairwise Mann-Whitney U tests.

    Args:
        scores: Maps a method name to its sample of scores.
        alpha: Significance threshold.
        lower_is_better: ``True`` for error-like metrics.

    Returns:
        A :class:`RankingResult`. Methods that beat the same number of others
        share a rank, exactly as the paper describes.
    """
    names = list(scores)
    points = {name: 0 for name in names}
    wins: list[tuple[str, str, float]] = []

    alternative = "less" if lower_is_better else "greater"
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sample_a = np.asarray(scores[a], dtype=float)
            sample_b = np.asarray(scores[b], dtype=float)
            if sample_a.size == 0 or sample_b.size == 0:
                continue
            p_ab = mannwhitneyu(sample_a, sample_b, alternative=alternative).pvalue
            p_ba = mannwhitneyu(sample_b, sample_a, alternative=alternative).pvalue
            if p_ab < alpha:
                points[a] += 1
                wins.append((a, b, float(p_ab)))
            elif p_ba < alpha:
                points[b] += 1
                wins.append((b, a, float(p_ba)))

    # Methods with equal points share a rank.
    distinct = sorted({p for p in points.values()}, reverse=True)
    rank_of = {p: i + 1 for i, p in enumerate(distinct)}
    return RankingResult(
        points=points, ranks={n: rank_of[p] for n, p in points.items()}, wins=wins
    )
