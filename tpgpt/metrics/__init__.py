"""Evaluation metrics (paper Sec. IV-B)."""

from tpgpt.metrics.curves import (
    area_between_curves,
    compare_trajectories,
    dynamic_time_warping,
    final_approach_angle,
    final_position_error,
    frechet_distance,
)
from tpgpt.metrics.stats import RankingResult, rank_methods

__all__ = [
    "RankingResult",
    "area_between_curves",
    "compare_trajectories",
    "dynamic_time_warping",
    "final_approach_angle",
    "final_position_error",
    "frechet_distance",
    "rank_methods",
]
