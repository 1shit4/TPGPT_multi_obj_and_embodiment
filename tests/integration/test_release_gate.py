"""The release gate's own instrument: is the object resting on something?

A gate that opens the hand "when the object is supported" is only as good as
the support test, and a support test can be wrong in two directions -- see an
object hanging in the hand as supported (the fingers touch it from the side),
or miss one lying on the table. Both are checked here on the real scene.
"""

import numpy as np
import pytest

from tpgpt.experiments.diagnose import object_supported, release_gate
from tpgpt.experiments.pipeline import build_scene

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def scene():
    env = build_scene(("can", "milk"), gripper="panda", seed=0)
    yield env
    env.close()


def _set_object_z(env, name, dz):
    body = env.object_body_ids[name]
    joint = env.sim.model.body_jntadr[body]
    adr = env.sim.model.jnt_qposadr[joint]
    env.sim.data.qpos[adr + 2] += dz
    env.sim.forward()


class TestObjectSupported:
    def test_an_object_resting_on_the_table_is_supported(self, scene):
        for _ in range(20):
            scene.sim.step()
        assert object_supported(scene, "can")

    def test_an_object_hoisted_into_the_air_is_not(self, scene):
        _set_object_z(scene, "can", +0.30)
        scene.sim.forward()
        assert not object_supported(scene, "can")
        _set_object_z(scene, "can", -0.30)

    def test_the_gate_opens_on_support_or_on_arrival(self, scene):
        for _ in range(20):
            scene.sim.step()
        gate = release_gate(scene, "can", arrive=0.01)
        assert gate(scene, lag=0.03) == "supported"   # supported and near: opens
        assert not gate(scene, lag=0.15)               # supported but 150 mm short: holds
        _set_object_z(scene, "can", +0.30)
        assert not gate(scene, lag=0.05)               # in the air, 50 mm short: holds
        assert gate(scene, lag=0.005) == "arrived"     # in the air but arrived: opens
        _set_object_z(scene, "can", -0.30)
