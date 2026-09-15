"""Objects built to the size and mass of the articles they stand for.

These assertions are the reason the benchmark extents are hard-coded rather than
measured at import: a silent change to a robosuite asset would rescale every
object by the wrong factor and the scene would still build, the cloud would
still come back, and the campaign would still produce numbers.
"""

import numpy as np
import pytest

from tpgpt.sim.objects import (
    BENCHMARK,
    BENCHMARK_EXTENTS_MM,
    REAL,
    REAL_ARTICLES,
    _compiled_mass,
    make_object,
    real_size_mm,
    rest_quat,
    scale_for,
)

MESH_OBJECTS = ("can", "cereal", "milk")
COMPOSITE_OBJECTS = ("hammer", "mug")


def _extents_mm(obj) -> np.ndarray:
    """True extents of a built object, from MuJoCo's own bounding boxes."""
    import mujoco
    from robosuite.models.world import MujocoWorldBase

    world = MujocoWorldBase()
    world.merge_assets(obj)
    world.worldbody.append(obj.get_obj())
    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
    points = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body:
            continue
        centre, half = model.geom_aabb[geom][:3], model.geom_aabb[geom][3:]
        corners = np.array([[x, y, z] for x in (-half[0], half[0])
                            for y in (-half[1], half[1])
                            for z in (-half[2], half[2])]) + centre
        rotation = data.geom_xmat[geom].reshape(3, 3)
        points.append(corners @ rotation.T + data.geom_xpos[geom])
    stacked = np.vstack(points)
    return (stacked.max(0) - stacked.min(0)) * 1000.0


@pytest.mark.parametrize("name", sorted(BENCHMARK_EXTENTS_MM))
def test_the_recorded_benchmark_extents_still_describe_the_shipped_asset(name):
    """If robosuite's asset moves, every scale factor derived from it is wrong."""
    measured = _extents_mm(make_object(name, BENCHMARK))
    assert np.allclose(measured, BENCHMARK_EXTENTS_MM[name], atol=0.2), (
        f"{name}: shipped asset measures {measured.round(1)} against the "
        f"recorded {BENCHMARK_EXTENTS_MM[name]}"
    )


@pytest.mark.parametrize("name", MESH_OBJECTS)
def test_a_real_mesh_object_reaches_the_real_article_s_dimensions(name):
    """Within 4 mm, which is what per-axis scaling of a rounded hull achieves.

    Not exact, and the reason is geometric rather than sloppy: the axis-aligned
    box of a scaled convex hull is not the scaled box of the original, so a
    rounded object such as the can lands a millimetre or two under its target.
    A flat-faced carton lands on it.
    """
    measured = _extents_mm(make_object(name, REAL))
    assert np.allclose(measured, real_size_mm(name), atol=4.0), (
        f"{name}: {measured.round(1)} against a target of {real_size_mm(name)}"
    )


@pytest.mark.parametrize("name", MESH_OBJECTS + COMPOSITE_OBJECTS)
def test_a_real_object_weighs_what_the_real_article_weighs(name):
    """Mass, not only size.

    robosuite's declared densities are unphysical -- 100 kg/m3 for the can and
    the milk against 1000 for water -- so scaling the mesh alone leaves a
    66 x 123 mm can weighing 42 g where a full one weighs 345. Grip is friction
    against weight, so a light object is a systematically easier object.
    """
    from tpgpt.sim.objects import REAL_COMPOSITES

    target = (REAL_ARTICLES.get(name) or REAL_COMPOSITES[name]).mass_kg
    assert _compiled_mass(make_object(name, REAL)) == pytest.approx(target, rel=1e-3)


def test_the_real_objects_are_heavier_than_the_benchmark_ones_they_replace():
    """The whole point, stated as an inequality so it cannot pass vacuously."""
    for name in MESH_OBJECTS:
        benchmark = _compiled_mass(make_object(name, BENCHMARK))
        real = _compiled_mass(make_object(name, REAL))
        assert real > 3 * benchmark, f"{name}: {benchmark:.3f} -> {real:.3f} kg"


def test_the_hammer_is_reproducible_across_calls():
    """It samples its own dimensions, and used to sample them unseeded.

    ``HammerObject`` draws its handle length from 0.10-0.25 m and its radius
    from 0.015-0.020, so ``HammerObject(name="hammer")`` builds a *different
    hammer every call*. Every benchmark scene containing one was therefore
    irreproducible, which no test caught because nothing compared two builds.
    """
    first = _extents_mm(make_object("hammer", REAL))
    second = _extents_mm(make_object("hammer", REAL))
    assert np.allclose(first, second, atol=1e-6)


def test_the_hammer_rests_on_its_side_and_nothing_else_is_rotated():
    """A yaw-only placement would stand it on the end of its handle.

    At benchmark size it is 231 mm and topples during the settle, which is why
    this went unnoticed; at 333 mm and 700 g it topples off the table.
    """
    assert not np.allclose(rest_quat("hammer"), [1.0, 0.0, 0.0, 0.0])
    for name in ("can", "cereal", "milk", "mug"):
        assert np.allclose(rest_quat(name), [1.0, 0.0, 0.0, 0.0])


def test_scaling_is_per_axis_because_the_meshes_are_not_the_right_shape():
    """A uniform factor cannot fix a can that is squatter than a real one."""
    factors = scale_for("can")
    assert not np.allclose(factors, factors[0]), factors
    assert (factors > 1.0).all()


def test_an_object_with_no_recorded_article_is_refused_rather_than_guessed():
    with pytest.raises(KeyError):
        real_size_mm("pot")


def test_a_tall_object_standing_in_its_slot_is_scored_as_placed():
    """The vertical test is on the object's base, not on its origin.

    Asked of the origin, the test asks how tall the object is: a 300 mm cereal
    carton standing perfectly in its slot has its origin 150 mm above the board
    and used to fail a 120 mm test unconditionally, so **no cereal cell could
    ever be scored a success**. A 230 mm milk carton passed by 5 mm and would
    flip on a millimetre of settling. Both are checked here because the failure
    was silent in the benchmark scene -- nothing there is tall enough to trip
    it, so the clause was satisfied by every run and the verdict came from the
    horizontal test alone.
    """
    import numpy as np
    import pytest

    from tpgpt.experiments.pipeline import build_scene

    for name, origin_above_board_mm in (("cereal", 150.0), ("milk", 97.5)):
        env = build_scene((name,), gripper="panda", seed=0, world="real",
                          pick_config="P1")
        try:
            slot = env.slot_poses()["top_middle"]
            joint = env._object_by_name(name).joints[0]
            qpos = np.array(env.sim.data.get_joint_qpos(joint))
            base = env._geom_vertices(env.object_body_ids[name])[:, 2].min()
            qpos[:2] = slot[:2]
            qpos[2] += slot[2] - base
            env.sim.data.set_joint_qpos(joint, qpos)
            env.sim.forward()
            assert env.is_object_in_slot(name, "top_middle")
            # And the quantity the old test would have measured, so the reason
            # it failed is visible rather than asserted.
            origin_gap = (env.object_position(name)[2] - slot[2]) * 1000
            assert origin_gap == pytest.approx(origin_above_board_mm, abs=1.0)
        finally:
            env.close()


def test_an_object_hovering_above_the_slot_is_not_scored_as_placed():
    """The test the old one could not make: resting on the board, not over it."""
    import numpy as np

    from tpgpt.experiments.pipeline import build_scene

    env = build_scene(("can",), gripper="panda", seed=0, world="real",
                      pick_config="P1")
    try:
        slot = env.slot_poses()["top_middle"]
        joint = env._object_by_name("can").joints[0]
        qpos = np.array(env.sim.data.get_joint_qpos(joint))
        base = env._geom_vertices(env.object_body_ids["can"])[:, 2].min()
        qpos[:2] = slot[:2]
        qpos[2] += slot[2] - base + 0.10       # 100 mm in the air
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        assert not env.is_object_in_slot("can", "top_middle")
    finally:
        env.close()
