"""Simulation layer: MuJoCo/robosuite scenes, keypoints and control (paper Sec. V)."""

from tpgpt.sim.backend import FrameWriter, make_reshelving_env, managed_env
from tpgpt.sim.embodiments import EMBODIMENTS, Embodiment, get_embodiment
from tpgpt.sim.keypoints import (
    KeypointSet,
    cube_keypoints,
    keypoints_from_bodies,
    pair_keypoints,
)

__all__ = [
    "EMBODIMENTS",
    "Embodiment",
    "FrameWriter",
    "KeypointSet",
    "cube_keypoints",
    "get_embodiment",
    "keypoints_from_bodies",
    "make_reshelving_env",
    "managed_env",
    "pair_keypoints",
]
