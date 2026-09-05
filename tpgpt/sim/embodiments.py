"""Robot and gripper selection (groundwork for cross-embodiment transport).

robosuite registers 46 robots and 28 grippers, so the embodiment is a
configuration choice rather than something to model by hand. The paper only
uses a Franka Panda; this module exists so the project's longer-term goal --
transporting a policy between embodiments -- does not require restructuring the
simulation layer later.

Nothing here claims cross-embodiment transport works. It makes the axis
selectable and verifies the environments build.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Embodiment:
    """A robot/gripper pairing plus the control gains that suit it."""

    robot: str
    gripper: str | None = None
    #: Default translational stiffness for the Cartesian impedance controller.
    default_stiffness: float = 600.0
    #: Default rotational stiffness.
    default_rotational_stiffness: float = 40.0

    def as_make_kwargs(self) -> dict:
        kwargs: dict = {"robots": self.robot}
        if self.gripper is not None:
            kwargs["gripper_types"] = self.gripper
        return kwargs


#: Arms with a parallel-jaw gripper, which is what the reshelving task needs.
EMBODIMENTS: dict[str, Embodiment] = {
    "panda": Embodiment("Panda", "PandaGripper"),
    "panda_robotiq": Embodiment("Panda", "Robotiq85Gripper"),
    "sawyer": Embodiment("Sawyer", "RethinkGripper"),
    "ur5e": Embodiment("UR5e", "Robotiq85Gripper", default_stiffness=500.0),
    "kinova3": Embodiment("Kinova3", "Robotiq85Gripper", default_stiffness=400.0),
    "iiwa": Embodiment("IIWA", "Robotiq85Gripper"),
}

DEFAULT_EMBODIMENT = "panda"


def get_embodiment(name: str = DEFAULT_EMBODIMENT) -> Embodiment:
    """Look up an embodiment by short name."""
    if name not in EMBODIMENTS:
        raise KeyError(f"unknown embodiment {name!r}; known: {sorted(EMBODIMENTS)}")
    return EMBODIMENTS[name]
