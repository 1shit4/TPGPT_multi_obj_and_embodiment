"""robosuite ``GripperModel`` classes for hands ported from GraspGen-X.

One class covers every ported hand, because :mod:`tpgpt.grasp.port_gripper`
gives them all the same shape: position actuators on the driven joints, and one
command interpolating between the ``open`` and ``close`` poses the gripper's own
GraspGen-X description declares.

That makes a ported hand **one-DOF and position-commanded**, whatever its finger
count -- the Barrett's three fingers and the Sharpa's five are one number each.
It is the treatment `grippers.commands_position` calls *passthrough*: the action
**is** the position, so a closure fraction is ``2f - 1`` and `replay.set_closure`
does not apply. Worth stating, because `ROBOTICS_NOTES.md` 7.28 records a
closure sweep that returned identical results at every fraction on hands of this
kind, and the reason was exactly this.
"""

from __future__ import annotations

import json

import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel

from tpgpt.grasp.port_gripper import CURATED, PORTED


class PortedGripper(GripperModel):
    """A GraspGen-X hand, mounted in robosuite.

    ``format_action`` maps one number in [-1, 1] onto every actuator: -1 is the
    ``open`` pose of the description, +1 the ``close`` pose, and the actuators
    are position servos whose ``ctrlrange`` is exactly that span. So the hand
    cannot be commanded outside the two poses its own author measured, which is
    what stops a converted URDF from driving a finger through its own palm.
    """

    def __init__(self, name: str, idn=0):
        self._name = name
        config = json.loads((CURATED / name / "config.json").read_text())
        self._config = config
        path = PORTED / name / f"{name}.xml"
        super().__init__(str(path), idn=idn)

        # ``self.joints`` arrives prefixed with the robot id; keep the raw
        # names so the description can be looked up by them.
        self._raw_joints = [j.split("_", 1)[-1] for j in self.joints]
        driven = [j for j in config["close"]
                  if abs(config["close"][j] - config["open"].get(j, 0.0)) > 1e-6]
        self._driven = driven
        self._open = np.array([float(config["open"][j]) for j in driven])
        self._close = np.array([float(config["close"][j]) for j in driven])

    def format_action(self, action):
        """One command to every actuator, as a fraction of the closing motion."""
        action = np.asarray(action).reshape(-1)
        assert len(action) == self.dof, f"expected {self.dof}, got {len(action)}"
        # The gripper controller scales (-1, 1) onto each actuator's own
        # ctrlrange, and that range was written as open->close, so a single
        # normalised number is all that has to be passed through.
        return np.repeat(np.clip(action[0], -1.0, 1.0), len(self._driven))

    @property
    def init_qpos(self):
        """Start open, one entry per **joint** -- not per driven joint.

        robosuite writes this straight into ``qpos`` for the gripper's joints,
        so it has to cover the coupled distal joints and the locked spread as
        well, not only the three the actuators drive. Anything the description
        does not name starts at zero, which is where its equality constraint
        will hold it anyway.
        """
        config = self._config
        return np.array([float(config["open"].get(name.split("_", 1)[-1], 0.0))
                         for name in self._raw_joints])

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 1

    @property
    def _important_geoms(self):
        # Filled per hand once the model is loaded; the porter names every geom
        # after its body, so contact attribution works without a hand-written
        # list. An empty mapping is honest: nothing is claimed about which geom
        # is a "left" or "right" finger on a hand with three or five of them.
        return {}


def make(name: str, idn=0) -> PortedGripper:
    """Build the ported hand ``name``, converting it first if needed."""
    from tpgpt.grasp.port_gripper import convert

    if not (PORTED / name / f"{name}.xml").is_file():
        convert(name)
    return PortedGripper(name, idn=idn)
