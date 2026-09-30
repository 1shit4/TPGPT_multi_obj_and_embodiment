"""The hardware bridge: frames, the planner on a measured cloud, the executor.

Three properties matter more than any other here, because each fails silently
on a robot:

* the **frame arithmetic** round-trips and matches the FR3 hand's geometry;
* the **planner** is equivariant -- moving the whole scene moves the plan by
  exactly that motion, so an extrinsic error becomes exactly that error at the
  gripper and nothing larger;
* the **executor** is the simulated execution loop, not a lookalike. It is
  checked bit-for-bit against ``rollout_impedance``, which calls the same
  functions in the same order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.hardware import frames
from tpgpt.hardware.executor import (
    ExecutorSettings,
    HardwareExecutor,
    SafetyMonitor,
    approach_trajectory,
    bounded_stiffness,
    crisp_accepts_full_stiffness,
    crisp_applied_gains,
    crisp_stiffness,
    logged_stiffness,
    stiffness_projection_loss,
)
from tpgpt.hardware.end_effector import FRANKA_HAND
from tpgpt.hardware.planner import (
    KEYPOINT_BOX,
    KEYPOINT_ORIENTATION,
    Plan,
    PlanningError,
    TargetScene,
    _hand_clearance,
    default_grasp,
    plan,
)
from tpgpt.hardware.source_bundle import SourceBundle, transform_labels, transform_placement
from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.policy.rollout import rollout_impedance
from tpgpt.sim.keypoints import GraspFrame, ObjectPlacement
from tpgpt.transport.labels import PolicyLabels
from tpgpt.utils.rotations import is_rotation

from tests.unit.test_keypoint_extraction import HALF, box_cloud, yaw_matrix

DT = 0.05
K_FREE = np.eye(3) * 350.0
D_FREE = np.eye(3) * (2 * np.sqrt(350.0) * 0.9)
REAL_BUNDLE = Path(__file__).resolve().parents[2] / "outputs/hardware/source_bundle"


# --------------------------------------------------------------------- frames
class TestFrames:
    def test_transform_round_trip(self, rng):
        for seed in range(10):
            R = Rotation.random(random_state=seed).as_matrix()
            T = frames.make_transform(R, rng.normal(size=3))
            p = rng.normal(size=(20, 3))
            back = frames.apply_to_points(frames.invert_transform(T), frames.apply_to_points(T, p))
            assert np.allclose(back, p, atol=1e-12)

    def test_quaternion_is_ros_xyzw(self):
        # 90 degrees about z, written [x, y, z, w].
        T = frames.transform_from_xyz_quat([1, 2, 3], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])
        assert np.allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0])
        xyz, q = frames.transform_to_xyz_quat(T)
        assert np.allclose(xyz, [1, 2, 3])
        assert np.allclose(np.abs(q), [0, 0, np.sqrt(0.5), np.sqrt(0.5)])

    def test_grasp_to_tcp_puts_fingers_on_y_and_approach_on_z(self):
        # A grasp closing along world x and approaching straight down.
        R_grasp = np.column_stack([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ np.diag([1, -1, -1])
        assert is_rotation(R_grasp)
        R_tcp = frames.grasp_to_fr3_tcp(R_grasp)
        assert is_rotation(R_tcp)
        assert np.allclose(R_tcp[:, 1], R_grasp[:, 0])   # fingers travel along closing
        assert np.allclose(R_tcp[:, 2], R_grasp[:, 2])   # approach unchanged
        assert np.allclose(frames.fr3_tcp_to_grasp(R_tcp), R_grasp)

    def test_stiffness_congruence_stays_symmetric(self, rng):
        A = rng.normal(size=(3, 3))
        K = A @ A.T + np.eye(3)
        T = frames.make_transform(Rotation.random(random_state=4).as_matrix())
        K2 = frames.apply_to_tensors(T, K[None])[0]
        assert np.allclose(K2, K2.T)
        assert np.allclose(np.linalg.eigvalsh(K2), np.linalg.eigvalsh(K))

    def test_slerp_ends(self):
        R_a = np.eye(3)
        R_b = Rotation.from_euler("z", 80, degrees=True).as_matrix()
        assert np.allclose(frames.slerp_rotation(R_a, R_b, 0.0), R_a)
        assert np.allclose(frames.slerp_rotation(R_a, R_b, 1.0), R_b)
        assert frames.rotation_angle(R_a, frames.slerp_rotation(R_a, R_b, 0.5)) == pytest.approx(
            np.radians(40)
        )


# --------------------------------------------------------------------- source
def grasp_rotation(yaw: float) -> np.ndarray:
    """A top-down grasp in the grasp convention: +X closing, +Z approach (down)."""
    c = yaw_matrix(yaw)[:, 0]
    a = np.array([0.0, 0.0, -1.0])
    return np.column_stack([c, np.cross(a, c), a])


def synthetic_bundle(table=-0.10, shelf=0.15, n=60) -> SourceBundle:
    """A pick on a table and a place on a shelf, in a robot base frame."""
    centre = np.array([0.45, -0.15, table + HALF[2]])
    points = box_cloud(centre, angle=0.4, faces="visible")
    grasp = GraspFrame(tcp=centre + [0, 0, HALF[2] - 0.02], approach=[0, 0, -1],
                       closing=yaw_matrix(0.4)[:, 0])
    dest = np.array([0.65, 0.15, shelf])
    place = np.array([dest[0], dest[1], shelf + HALF[2] + (HALF[2] - 0.02)])
    # approach (open) | hold at the grasp (closed) | carry (closed)
    # | hold at the place (closed) | release in place (open).
    # The jaws must close *at* the grasp pose: carry_transform reads the hand
    # pose at the first closed label, so closing anywhere else moves the object.
    approach, hold, release = 6, 4, 5
    carry = n - approach - 2 * hold - release
    positions = np.vstack([
        np.linspace(grasp.tcp + [0, 0, 0.1], grasp.tcp, approach),
        np.repeat(grasp.tcp[None], hold, axis=0),
        np.linspace(grasp.tcp, place, carry),
        np.repeat(place[None], hold + release, axis=0),
    ])
    yaws = np.concatenate([np.full(approach + hold, 0.4), np.linspace(0.4, 0.0, carry),
                           np.full(hold + release, 0.0)])
    index = np.arange(n)
    gripper = np.where((index >= approach) & (index < n - release), 1.0, -1.0)
    labels = PolicyLabels(
        positions=positions,
        velocities=np.gradient(positions, DT, axis=0),
        orientations=np.stack([grasp_rotation(y) for y in yaws]),
        stiffness=np.tile(K_FREE, (n, 1, 1)),
        damping=np.tile(D_FREE, (n, 1, 1)),
        gripper=gripper,
        time_belief=np.linspace(0.0, 1.0, n),
        time_rate=np.full(n, 1.0 / (DT * (n - 1))),
    )
    placement = ObjectPlacement(points, grasp, table, dest, shelf)
    return SourceBundle(labels, placement, {"control_freq": 20})


@pytest.fixture(scope="module")
def bundle():
    return synthetic_bundle()


class TestSourceBundle:
    def test_save_load_round_trip(self, bundle, tmp_path):
        bundle.save(tmp_path / "b")
        loaded = SourceBundle.load(tmp_path / "b")
        assert np.allclose(loaded.tool_labels.positions, bundle.tool_labels.positions)
        assert np.allclose(loaded.tool_labels.stiffness, bundle.tool_labels.stiffness)
        assert np.allclose(loaded.placement.grasp.closing, bundle.placement.grasp.closing)
        assert loaded.placement.destination_height == bundle.placement.destination_height

    def test_upright_transform_moves_heights_with_it(self, bundle):
        T = frames.make_transform(yaw_matrix(0.3), [0.1, -0.2, 0.9])
        moved = transform_placement(bundle.placement, T)
        assert moved.support_height == pytest.approx(bundle.placement.support_height + 0.9)
        assert np.allclose(moved.grasp.tcp, frames.apply_to_points(T, bundle.placement.grasp.tcp)[0])

    def test_a_tilting_transform_is_refused(self, bundle):
        T = frames.make_transform(Rotation.from_euler("x", 10, degrees=True).as_matrix())
        with pytest.raises(ValueError, match="tilts the vertical"):
            transform_placement(bundle.placement, T)

    def test_labels_keep_every_family(self, bundle):
        T = frames.make_transform(yaw_matrix(1.0), [1, 2, 3])
        moved = transform_labels(bundle.tool_labels, T)
        assert set(moved.present) == set(bundle.tool_labels.present)


# -------------------------------------------------------------------- planner
class TestPlanner:
    def test_defaults_match_the_simulation_pipeline(self):
        pipeline = pytest.importorskip("tpgpt.experiments.pipeline")
        assert KEYPOINT_BOX == pipeline.DEFAULT_KEYPOINT_BOX
        assert KEYPOINT_ORIENTATION == pipeline.DEFAULT_KEYPOINT_ORIENTATION

    def _scene(self, bundle, shift=np.zeros(3), **kw):
        p = bundle.placement
        return TargetScene(
            points=p.points + shift,
            destination=p.destination + shift,
            destination_height=p.destination_height + shift[2],
            support_height=p.support_height + shift[2],
            grasp=GraspFrame(tcp=p.grasp.tcp + shift, approach=p.grasp.approach,
                             closing=p.grasp.closing),
            **kw,
        )

    def test_same_scene_reproduces_the_demonstration(self, bundle):
        result = plan(bundle, self._scene(bundle))
        assert result.metrics["min_det"] > 0.9
        assert np.allclose(result.labels.positions, bundle.tool_labels.positions, atol=2e-3)
        expected = frames.grasp_to_fr3_tcp(bundle.tool_labels.orientations)
        angles = [frames.rotation_angle(a, b) for a, b in zip(result.labels.orientations, expected)]
        assert np.degrees(max(angles)) < 1.0

    def test_translating_the_scene_translates_the_plan_exactly(self, bundle):
        """An extrinsic offset becomes the same offset at the hand, nothing more."""
        shift = np.array([0.07, -0.05, 0.03])
        base = plan(bundle, self._scene(bundle))
        moved = plan(bundle, self._scene(bundle, shift))
        assert np.allclose(moved.labels.positions - base.labels.positions, shift, atol=2e-3)
        assert moved.metrics["aim_grasp_m"] < 3e-3

    def test_the_plan_aims_at_the_target_grasp(self, bundle):
        # A different, yawed object in a different place.
        centre = np.array([0.40, 0.05, -0.10 + 0.05])
        points = box_cloud(centre, angle=-0.6, half=np.array([0.025, 0.04, 0.05]),
                           seed=3, faces="visible")
        scene = TargetScene(points=points, destination=np.array([0.55, 0.25, 0.15]),
                            destination_height=0.15, support_height=-0.10)
        result = plan(bundle, scene)
        assert result.metrics["min_det"] > 0.3
        assert result.metrics["aim_grasp_m"] < 5e-3
        assert result.metrics["aim_release_m"] < 5e-3
        assert all(is_rotation(R, atol=1e-6) for R in result.labels.orientations)

    def test_default_grasp_sits_below_the_top(self):
        centre = np.array([0.4, 0.0, 0.1])
        points = box_cloud(centre, half=np.array([0.02, 0.03, 0.1]), faces="visible")
        grasp = default_grasp(points, support_height=0.0, depth_below_top=0.03)
        assert grasp.tcp[2] == pytest.approx(0.17, abs=5e-3)
        assert np.allclose(grasp.approach, [0, 0, -1])

    def test_a_sparse_cloud_is_refused(self, bundle):
        scene = self._scene(bundle)
        scene.points = scene.points[:10]
        with pytest.raises(PlanningError, match="object_not_seen"):
            plan(bundle, scene)

    def test_a_provably_too_wide_object_is_refused(self, bundle):
        points = box_cloud(np.array([0.4, 0.0, 0.0]), half=np.array([0.08, 0.08, 0.05]),
                           faces="visible")
        scene = TargetScene(points=points, destination=np.array([0.6, 0.2, 0.15]),
                            destination_height=0.15, support_height=-0.05)
        with pytest.raises(PlanningError, match="object_too_wide"):
            plan(bundle, scene)

    def test_plan_save_load_round_trip(self, bundle, tmp_path):
        result = plan(bundle, self._scene(bundle))
        result.save(tmp_path / "plan")
        loaded = Plan.load(tmp_path / "plan")
        assert np.allclose(loaded.labels.positions, result.labels.positions)
        assert np.allclose(loaded.labels.orientations, result.labels.orientations)
        assert loaded.metrics["min_det"] == pytest.approx(result.metrics["min_det"])
        assert np.allclose(loaded.grasp.tcp, result.grasp.tcp)

    # Learned labels (audit B1, step 6): the command leads the hand; the arm's path is checked.
    @staticmethod
    def _learned(taught, lead=np.array([0.0, 0.015, 0.0])):
        labels = taught.tool_labels
        return SourceBundle(
            labels.replace(positions=labels.positions + lead), taught.placement,
            {**taught.metadata, "learned": {"execution": {"speed_scale": 0.35}}},
            arm_path={"positions": labels.positions, "rotations": labels.orientations,
                      "margin": np.full(len(labels), 0.002)},
            taught_labels=labels)

    def test_a_learned_command_is_run_and_its_arm_path_is_measured(self, bundle):
        result = plan(self._learned(bundle), self._scene(bundle))
        assert np.allclose(result.labels.positions, bundle.tool_labels.positions + [0, 0.015, 0],
                           atol=2e-3)
        assert np.allclose(result.arm_path.positions, bundle.tool_labels.positions, atol=2e-3)
        # the aim is the arm's, not the command's 15 mm lead
        assert result.metrics["aim_grasp_m"] < 3e-3
        assert result.metrics["checked_path"] == "arm path (learned labels)"
        assert result.metadata["learned_execution"] == {"speed_scale": 0.35}
        assert plan(bundle, self._scene(bundle)).metadata["learned_execution"] is None

    def test_what_the_object_did_is_read_off_the_taught_labels(self, bundle):
        """The command's pose at the grasp and the release is not the hand's; the keypoints
        (the object carried from the grasp to the release) must not move with it."""
        learned = self._learned(bundle)
        scene = self._scene(bundle)
        taught_keys = plan(bundle, scene).source_keypoints.points
        assert np.allclose(plan(learned, scene).source_keypoints.points, taught_keys)
        learned.taught_labels = None                       # read off the command instead
        assert not np.allclose(plan(learned, scene).source_keypoints.points, taught_keys,
                               atol=1e-3)

    def test_a_learned_plan_and_bundle_round_trip(self, bundle, tmp_path):
        learned = self._learned(bundle)
        loaded_bundle = SourceBundle.load(learned.save(tmp_path / "b"))
        assert np.allclose(loaded_bundle.taught_labels.positions, bundle.tool_labels.positions)
        assert np.allclose(loaded_bundle.arm_path["margin"], 0.002)
        result = plan(loaded_bundle, self._scene(bundle))
        result.save(tmp_path / "plan")
        loaded = Plan.load(tmp_path / "plan")
        assert np.allclose(loaded.arm_path.positions, result.arm_path.positions)
        assert np.allclose(loaded.arm_path.orientations, result.arm_path.orientations)
        assert np.allclose(loaded.arm_margin, 0.002)
        assert Plan.load(plan(bundle, self._scene(bundle)).save(tmp_path / "plain")).arm_path is None

    def test_the_path_check_checks_the_arm_path_with_its_margin(self, bundle, monkeypatch):
        from tpgpt.hardware import path_check
        seen = {}
        monkeypatch.setattr(path_check, "path_feasibility",
                            lambda positions, orientations, *a, **kw: seen.update(
                                positions=positions, orientations=orientations, **kw))
        result = plan(self._learned(bundle), self._scene(bundle))
        path_check.check_plan(result, None, FRANKA_HAND, clearance=0.003)
        assert np.allclose(seen["positions"], result.arm_path.positions)
        assert np.allclose(seen["orientations"], result.arm_path.orientations)
        assert np.allclose(seen["clearance"], 0.005)
        assert seen["table_clearance"] is False
        plain = plan(bundle, self._scene(bundle))
        seen.clear()
        path_check.check_plan(plain, None, FRANKA_HAND)
        assert np.allclose(seen["positions"], plain.labels.positions)
        assert seen.get("clearance") is None and seen.get("held_shift") is None
        assert "table_clearance" not in seen

    def test_the_jaws_centre_the_object_the_arm_closes_off_centre_on(self, bundle, monkeypatch):
        """The arm grasps 4 mm off across the jaws and 3 mm short along the approach: the
        coupled jaws push the object 4 mm across to their centre, not along the approach."""
        from tpgpt.hardware import path_check
        seen = {}
        monkeypatch.setattr(path_check, "path_feasibility",
                            lambda *a, **kw: seen.update(kw))
        learned = self._learned(bundle)
        c = np.asarray(bundle.placement.grasp.closing, dtype=float)
        a = np.asarray(bundle.placement.grasp.approach, dtype=float)
        learned.arm_path["positions"] = learned.arm_path["positions"] + 0.004 * c - 0.003 * a
        result = plan(learned, self._scene(bundle))
        path_check.check_plan(result, None, FRANKA_HAND)
        closing = np.asarray(result.grasp.closing) / np.linalg.norm(result.grasp.closing)
        shift = np.asarray(seen["held_shift"])
        assert abs(abs(shift @ closing) - 0.004) < 5e-4
        assert np.linalg.norm(shift - (shift @ closing) * closing) < 1e-9

    @pytest.mark.skipif(not REAL_BUNDLE.exists(), reason="export_source has not been run")
    def test_the_exported_mujoco_bundle_reproduces_itself(self):
        real = SourceBundle.load(REAL_BUNDLE)
        result = plan(real, self._scene(real))
        assert result.metrics["min_det"] > 0.9
        assert np.allclose(result.labels.positions, real.tool_labels.positions, atol=3e-3)


# ------------------------------------------------------------------- executor
def line_labels(n=120, speed=0.168, closed_from=60):
    x = np.zeros((n, 3))
    x[:, 0] = np.linspace(0.0, speed * DT * (n - 1), n)
    return PolicyLabels(
        positions=x, velocities=np.gradient(x, DT, axis=0),
        stiffness=np.tile(K_FREE, (n, 1, 1)), damping=np.tile(D_FREE, (n, 1, 1)),
        gripper=np.where(np.arange(n) >= closed_from, 1.0, -1.0),
        time_belief=np.linspace(0, 1, n), time_rate=np.full(n, 1 / (DT * (n - 1))),
    )


@pytest.fixture(scope="module")
def line_policy():
    return GPPolicy().fit(line_labels())


def run_executor_on_surrogate(policy, settings=None, n_steps=400, load=None):
    """Drive the executor with the same first-order arm ``rollout_impedance`` uses."""
    ex = HardwareExecutor(policy, DT, settings)
    x = policy.labels.positions[0].copy()
    attractors, phases = [], []
    for _ in range(n_steps):
        cmd = ex.step(x, np.eye(3))
        attractors.append(cmd.attractor)
        phases.append(cmd.phase)
        A = np.linalg.solve(cmd.damping, cmd.stiffness)
        x = x + A @ (cmd.tcp_position - x) * DT
        if load is not None:
            x = x + load * DT
        if ex.finished:
            break
    return ex, np.array(attractors), np.array(phases)


class TestExecutor:
    def test_it_is_the_simulated_loop_bit_for_bit(self, line_policy):
        reference = rollout_impedance(line_policy, dt=DT, n_steps=400, substeps=1)
        ex, attractors, phases = run_executor_on_surrogate(line_policy)
        assert len(attractors) == len(reference.attractors)
        assert np.array_equal(attractors, reference.attractors)
        assert np.array_equal(phases, reference.time_belief)
        assert ex.finished and not ex.stalled

    def test_it_matches_under_load_too(self, line_policy):
        load = np.array([0.0, 0.0, -0.08])
        reference = rollout_impedance(line_policy, dt=DT, n_steps=400, substeps=1, load=load)
        _, attractors, _ = run_executor_on_surrogate(line_policy, load=load)
        assert np.array_equal(attractors, reference.attractors)

    def test_half_speed_takes_about_twice_as_long(self, line_policy):
        fast, a_fast, _ = run_executor_on_surrogate(line_policy)
        slow, a_slow, _ = run_executor_on_surrogate(
            line_policy, ExecutorSettings(speed_scale=0.5), n_steps=900
        )
        assert slow.finished and not slow.stalled
        assert 1.6 < len(a_slow) / len(a_fast) < 2.4
        assert np.linalg.norm(a_slow[-1] - a_fast[-1]) < 0.01

    def test_gripper_follows_the_labels(self, line_policy):
        ex = HardwareExecutor(line_policy, DT)
        x = line_policy.labels.positions[0].copy()
        grippers, phases = [], []
        while not ex.finished:
            cmd = ex.step(x, np.eye(3))
            grippers.append(cmd.gripper)
            phases.append(cmd.phase)
            x = cmd.tcp_position
        grippers, phases = np.array(grippers), np.array(phases)
        assert np.all(grippers[phases < 0.4] < 0)
        assert np.all(grippers[phases > 0.6] > 0)

    def test_tool_offset_is_applied_both_ways(self, line_policy):
        off = (0.0, 0.0, 0.01)
        ex = HardwareExecutor(line_policy, DT, ExecutorSettings(tool_offset=off))
        R = yaw_matrix(0.7)
        tcp = np.array([0.1, 0.2, 0.3])
        assert np.allclose(ex.tcp_from_tool(ex.tool_position(tcp, R), R), tcp)
        assert np.allclose(ex.tool_position(tcp, np.eye(3)) - tcp, off)

    def test_a_motionless_arm_finishes_exactly_as_in_simulation(self, line_policy):
        """The shared loop does not notice an arm that never moves -- by design
        of the simulated rollout, which this must reproduce. SafetyMonitor is
        what stops it on hardware."""
        reference = rollout_impedance(line_policy, dt=DT, n_steps=1000,
                                      blocked_steps=(0, 10_000))
        ex = HardwareExecutor(line_policy, DT)
        x = line_policy.labels.positions[0].copy()
        steps = 0
        while not ex.finished and steps < 1000:
            ex.step(x, np.eye(3))
            steps += 1
        assert steps == reference.metadata["steps"]
        assert reference.metadata["terminated_on_phase"] and not ex.stalled

    def test_the_safety_monitor_stops_a_motionless_arm(self, line_policy):
        ex = HardwareExecutor(line_policy, DT)
        monitor = SafetyMonitor()
        x = line_policy.labels.positions[0].copy()
        reason = None
        while not ex.finished:
            cmd = ex.step(x, np.eye(3))
            reason = monitor.check(x, cmd.attractor)
            if reason:
                break
        assert reason is not None and ex.phase < 0.5

    def test_the_safety_monitor_is_quiet_on_a_following_arm(self, line_policy):
        ex = HardwareExecutor(line_policy, DT)
        monitor = SafetyMonitor()
        x = line_policy.labels.positions[0].copy()
        while not ex.finished:
            cmd = ex.step(x, np.eye(3))
            assert monitor.check(x, cmd.attractor) is None
            A = np.linalg.solve(cmd.damping, cmd.stiffness)
            x = x + A @ (cmd.tcp_position - x) * DT

    def test_speed_scale_is_validated(self, line_policy):
        with pytest.raises(ValueError):
            HardwareExecutor(line_policy, DT, ExecutorSettings(speed_scale=1.5))


class TestApproachAndStiffness:
    def test_approach_ends_on_the_goal_and_respects_speed(self):
        R0, R1 = np.eye(3), yaw_matrix(1.0)
        p, R = approach_trajectory([0, 0, 0], R0, [0.3, 0, 0], R1, DT, max_linear_speed=0.05)
        assert np.allclose(p[0], 0) and np.allclose(p[-1], [0.3, 0, 0])
        assert np.allclose(R[-1], R1, atol=1e-9)
        speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
        assert speed.max() <= 0.05 * 1.01

    def test_crisp_stiffness_is_in_the_tcp_frame(self):
        K = np.diag([100.0, 400.0, 800.0])
        # Hand rotated 90 degrees about z: tcp x is base y.
        R = yaw_matrix(np.pi / 2)
        values = crisp_stiffness(K, R, rotational_stiffness=30.0)
        assert np.allclose(values[:3], [400.0, 100.0, 800.0])
        assert np.allclose(values[3:], 30.0)
        assert stiffness_projection_loss(K, R) == pytest.approx(0.0, abs=1e-12)

    def test_crisp_stiffness_is_clamped(self):
        values = crisp_stiffness(np.diag([5.0, 5000.0, 300.0]), np.eye(3), 500.0)
        assert np.allclose(values, [50.0, 900.0, 300.0, 60.0, 60.0, 60.0])

    def test_dropped_off_diagonal_is_reported(self):
        R45 = yaw_matrix(np.pi / 4)
        K = R45 @ np.diag([100.0, 900.0, 500.0]) @ R45.T
        assert stiffness_projection_loss(K, np.eye(3)) > 0.3

    # --- the 12-value form (patched CRISP, crisp-full-stiffness.patch) ---------------
    @staticmethod
    def _directional(u, k_across=600.0, k_along=350.0):
        """Soft along u, stiff across it: the directional rule of the execution-gap plan."""
        u = np.asarray(u, dtype=float) / np.linalg.norm(u)
        return k_across * np.eye(3) - (k_across - k_along) * np.outer(u, u)

    def test_full_message_is_the_whole_tcp_frame_block(self):
        K = self._directional([0.0, np.sin(0.42), np.cos(0.42)])
        R = Rotation.from_euler('xyz', [0.3, -0.7, 1.1]).as_matrix()
        values = crisp_stiffness(K, R, rotational_stiffness=30.0, full=True)
        assert values.shape == (12,)
        block = values[:9].reshape(3, 3)
        assert np.allclose(block, R.T @ K @ R, atol=1e-9)
        assert np.allclose(block, block.T, atol=1e-12)
        assert np.allclose(values[9:], 30.0)
        # the diagonal form of the same K loses exactly the part the full form keeps
        assert stiffness_projection_loss(K, R) > 0.05

    def test_full_message_bounds_principal_values_not_entries(self):
        R = Rotation.from_euler('xyz', [0.2, 0.5, -0.4]).as_matrix()
        K = R @ np.diag([10.0, 400.0, 1500.0]) @ R.T
        block = crisp_stiffness(K, np.eye(3), 30.0, full=True)[:9].reshape(3, 3)
        assert np.allclose(np.linalg.eigvalsh(block), [50.0, 400.0, 900.0], atol=1e-9)
        assert np.allclose(bounded_stiffness(K, 50.0, 900.0), block, atol=1e-9)
        inside = R @ np.diag([350.0, 600.0, 600.0]) @ R.T
        assert np.array_equal(bounded_stiffness(inside, 50.0, 900.0), 0.5 * (inside + inside.T))

    def test_applied_gains_invert_the_message_with_crisp_damping(self):
        K = self._directional([1.0, -2.0, 0.5])
        R = Rotation.from_euler('xyz', [-0.6, 0.1, 0.9]).as_matrix()
        K_run, D_run = crisp_applied_gains(crisp_stiffness(K, R, 30.0, full=True), R)
        assert np.allclose(K_run, K, atol=1e-9)
        assert np.allclose(D_run @ D_run, 4.0 * K, rtol=1e-9)   # D = 2 K^(1/2)
        # along the soft direction the time constant is 2/sqrt(350); across it 2/sqrt(600)
        u = np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5])
        assert np.allclose(np.linalg.solve(K_run, D_run) @ u, 2.0 / np.sqrt(350.0) * u)

    def test_diagonal_message_applies_only_its_diagonal(self):
        K = self._directional([0.0, 1.0, 1.0])
        K_run, D_run = crisp_applied_gains(crisp_stiffness(K, np.eye(3), 30.0), np.eye(3))
        assert np.allclose(K_run, np.diag(np.diag(K)))
        assert np.allclose(D_run, np.diag(2.0 * np.sqrt(np.diag(K))))

    def test_isotropic_stiffness_is_the_same_in_either_form(self):
        """A fixed stiffness (what every run so far used) gives the same arm in both forms."""
        R = yaw_matrix(0.8)
        K0, D0 = 400.0 * np.eye(3), 40.0 * np.eye(3)
        for full in (False, True):
            K_run, D_run = crisp_applied_gains(crisp_stiffness(K0, R, 30.0, full=full), R)
            assert np.allclose(K_run, K0) and np.allclose(D_run, D0)

    def test_full_stiffness_support_is_read_from_crisps_own_description(self):
        # verbatim, as `ros2 param describe` printed them on the two images (2026-09-29)
        patched = ('Topic name for receiving variable stiffness updates (std_msgs/Float64MultiArray). '
                   '6 elements: [k_pos_x, k_pos_y, k_pos_z, k_rot_x, k_rot_y, k_rot_z], a diagonal '
                   'stiffness. 12 elements: a symmetric 3x3 translational stiffness, row-major, ...')
        stock = ('Topic name for receiving variable stiffness updates (std_msgs/Float64MultiArray '
                 'with 6 elements: [k_pos_x, k_pos_y, k_pos_z, k_rot_x, k_rot_y, k_rot_z])')
        assert crisp_accepts_full_stiffness(patched)
        assert not crisp_accepts_full_stiffness(stock)
        assert not crisp_accepts_full_stiffness('')
        assert not crisp_accepts_full_stiffness(None)

    def test_logged_stiffness_reads_both_forms(self):
        K = np.diag([100.0, 400.0, 800.0])
        for full in (False, True):
            k_diag, k_rot = logged_stiffness(crisp_stiffness(K, np.eye(3), 30.0, full=full))
            assert np.allclose(k_diag, [100.0, 400.0, 800.0]) and k_rot == 30.0


# ------------------------------------------------ the infeasible-pose fallback
class _StubArm:
    """Reachability on demand: ``orientation`` decides whether a pose with a
    commanded rotation is reachable, ``beyond_x`` makes positions past it reachable
    (with or without orientation)."""

    def __init__(self, orientation=False, beyond_x=np.inf):
        self.orientation, self.beyond_x = orientation, beyond_x

    def solve_ik(self, target, rotation, seed=None, max_iterations=120, position_tolerance=0.005):
        from types import SimpleNamespace

        ok = target[0] > self.beyond_x or rotation is None or self.orientation
        return SimpleNamespace(reachable=bool(ok))


def _stuck_run(policy, arm, joints=np.zeros(7), n=60):
    """An arm that never moves, so the gate shuts and the fallback is consulted."""
    ex = HardwareExecutor(policy, DT, kinematics=arm)
    x = policy.labels.positions[0].copy()
    cmds = [ex.step(x, np.eye(3), joints) for _ in range(n)]
    return ex, cmds


class TestInfeasibleFallback:
    def test_only_the_hand_angle_out_of_reach_relaxes_the_orientation(self):
        # A policy with an orientation channel: without one there is no hand
        # angle to give up, in rollout_policy as here.
        labels = line_labels()
        labels = labels.replace(orientations=np.tile(np.eye(3), (len(labels.positions), 1, 1)))
        policy = GPPolicy().fit(labels)
        ex, cmds = _stuck_run(policy, _StubArm(orientation=False))
        relaxed = [c for c in cmds if c.relaxed]
        assert relaxed and ex.relaxed_steps > 0
        assert all(np.array_equal(c.tcp_rotation, np.eye(3)) for c in relaxed)

    def test_a_point_out_of_reach_steers_to_the_next_reachable_label(self, line_policy):
        # Nothing is reachable at the start; labels past x = 0.05 are.
        arm = _StubArm(orientation=False, beyond_x=0.05)
        arm.solve_ik = lambda target, rotation, **k: __import__("types").SimpleNamespace(
            reachable=bool(target[0] > 0.05))
        ex, cmds = _stuck_run(line_policy, arm)
        chosen = [c.trace["skip_index"] for c in cmds if c.trace["skip_index"] is not None]
        assert chosen
        assert all(line_policy.labels.positions[i][0] > 0.05 for i in chosen)
        # The attractor is being steered toward that label, never past it.
        assert cmds[-1].attractor[0] <= line_policy.labels.positions[chosen[-1]][0] + 1e-9

    def test_a_reachable_pose_changes_nothing(self, line_policy):
        ex, cmds = _stuck_run(line_policy, _StubArm(orientation=True))
        assert not any(c.relaxed for c in cmds) and ex.skips == 0 and ex.skipped_steps == 0

    def test_without_joint_angles_the_fallback_is_off_and_says_so(self, line_policy):
        ex, cmds = _stuck_run(line_policy, _StubArm(orientation=False), joints=None)
        assert not any(c.relaxed for c in cmds)
        assert ex.feasibility_unavailable > 0

    def test_the_real_fr3_is_the_default_model(self, line_policy):
        from tpgpt.hardware.fr3 import FR3

        assert isinstance(HardwareExecutor(line_policy, DT).kinematics, FR3)

    def test_the_settings_are_rollout_policys(self):
        import inspect

        from tpgpt.sim.rollout import rollout_policy

        defaults = inspect.signature(rollout_policy).parameters
        s = ExecutorSettings()
        assert s.infeasible_fallback == defaults["infeasible_fallback"].default
        assert s.feasibility_stride == defaults["feasibility_stride"].default
        assert s.feasibility_tolerance == defaults["feasibility_tolerance"].default


# ------------------------------------------------- the hand against a surface
class TestHandClearance:
    """The destination is a shelf board, not a plane that reaches to the horizon.

    Measured 2026-09-24: with the 23:01 banana demonstration the taught retreat
    comes back down *in front of* the shelf -- 215 to 314 mm from the slot, 35 to
    53 mm below the board it had just placed on -- and judging that against an
    infinite plane at the board's height refused all 25 planned candidates for
    each upper slot, on each of four captures, while the exact path check passed
    them.
    """

    DEST = np.array([0.65, 0.15, 0.30])

    def labels(self, retreat):
        """Descend, grasp, place at the destination, then retreat as given."""
        positions = np.array([[0.65, 0.15, 0.45], [0.65, 0.15, 0.35],
                              [*self.DEST[:2], 0.33], *retreat])
        n = len(positions)
        down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).T
        return PolicyLabels(
            positions=positions, velocities=np.zeros_like(positions),
            orientations=np.tile(down, (n, 1, 1)), stiffness=np.tile(K_FREE, (n, 1, 1)),
            damping=np.tile(D_FREE, (n, 1, 1)), gripper=np.full(n, -1.0),
            time_belief=np.linspace(0, 1, n), time_rate=np.full(n, 1.0))

    def clearance(self, retreat, **kw):
        labels = self.labels(retreat)
        return _hand_clearance(labels, 1, 2, 0.04, -0.01, float(self.DEST[2]), **kw)[1]

    def test_a_retreat_that_dips_far_from_the_slot_is_not_the_destination(self):
        away = [[0.65, 0.15, 0.40], [0.64, 0.40, 0.36], [0.62, 0.45, 0.25]]  # 300 mm away, low
        assert self.clearance(away) < -0.04, "unbounded, the plane refuses it"
        bounded = self.clearance(away, destination_xy=self.DEST[:2], footprint_radius=0.15)
        assert bounded > 0.0, "bounded to the board, there is nothing under the hand to hit"

    def test_a_hand_driven_into_the_board_is_still_refused(self):
        into = [[*self.DEST[:2], 0.26], [0.66, 0.16, 0.24]]        # straight down, over the slot
        assert self.clearance(into, destination_xy=self.DEST[:2], footprint_radius=0.15) < 0.0

    def test_the_release_is_measured_whatever_the_radius(self):
        away = [[0.20, -0.40, 0.10]]
        assert self.clearance(away, destination_xy=self.DEST[:2],
                              footprint_radius=0.001) == pytest.approx(
            self.clearance([[*self.DEST[:2], 0.33]], destination_xy=self.DEST[:2],
                           footprint_radius=0.5), abs=1e-9)
