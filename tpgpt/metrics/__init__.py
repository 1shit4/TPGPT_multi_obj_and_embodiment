"""Evaluation metrics (paper Sec. IV-B)."""

from tpgpt.metrics.curves import (
    area_between_curves,
    compare_trajectories,
    dynamic_time_warping,
    final_approach_angle,
    final_position_error,
    frechet_distance,
    path_deviation,
    point_to_curve_distance,
)
from tpgpt.metrics.stats import RankingResult, rank_methods
from tpgpt.metrics.transport import (
    JAW_SYMMETRY,
    UP,
    lift_deviation,
    orientation_transport_error,
    tilt_profile,
    vertical_tilt,
)

__all__ = [
    "JAW_SYMMETRY",
    "RankingResult",
    "UP",
    "area_between_curves",
    "compare_trajectories",
    "dynamic_time_warping",
    "final_approach_angle",
    "final_position_error",
    "frechet_distance",
    "lift_deviation",
    "orientation_transport_error",
    "path_deviation",
    "point_to_curve_distance",
    "rank_methods",
    "tilt_profile",
    "vertical_tilt",
]
