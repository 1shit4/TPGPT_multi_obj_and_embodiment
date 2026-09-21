"""The FR3 model: kinematics against the real robot, IK against the simulation's rules."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.hardware.fr3 import ARM_LINKS, FR3, HOME_Q, TCP_LINK, default_model

DATA = Path(__file__).resolve().parents[1] / "data" / "fr3_still_samples.npz"


@pytest.fixture(scope="module")
def fr3() -> FR3:
    return default_model()


def test_forward_kinematics_matches_the_real_robot(fr3):
    """40 recorded poses of the real arm at rest: joint angles in, /current_pose out.

    At rest the two topics describe the same instant, so the only error left is
    the model's. Measured 0.01 mm median, 0.18 mm worst over five trials.
    """
    d = np.load(DATA)
    for q, ee in zip(d["joint_positions"], d["ee_pose_xyz_xyzw"]):
        T = fr3.tcp(q)
        assert np.linalg.norm(T[:3, 3] - ee[:3]) < 5e-4
        R = Rotation.from_quat(ee[3:]).as_matrix()
        angle = np.linalg.norm(Rotation.from_matrix(R @ T[:3, :3].T).as_rotvec())
        assert np.degrees(angle) < 0.1


def test_home_pose_is_the_familiar_one(fr3):
    assert np.allclose(fr3.tcp(HOME_Q)[:3, 3], [0.3069, 0.0, 0.4869], atol=1e-3)


def test_fr3_joint_limits_come_from_its_own_description(fr3):
    # Joints 4 and 6 are where the FR3 differs from the Panda.
    assert fr3.lower[3] == pytest.approx(-3.0421) and fr3.upper[3] == pytest.approx(-0.1518)
    assert fr3.lower[5] == pytest.approx(0.5445) and fr3.upper[5] == pytest.approx(4.5169)


def test_jacobian_matches_finite_differences(fr3):
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(fr3.lower, fr3.upper)
        J = fr3.jacobian(q)
        T0 = fr3.tcp(q)
        eps = 1e-6
        for k in range(7):
            dq = np.zeros(7)
            dq[k] = eps
            T1 = fr3.tcp(q + dq)
            assert np.allclose((T1[:3, 3] - T0[:3, 3]) / eps, J[:3, k], atol=1e-4)
            w = Rotation.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec() / eps
            assert np.allclose(w, J[3:, k], atol=1e-4)


def test_ik_recovers_poses_the_arm_can_reach(fr3):
    rng = np.random.default_rng(1)
    solved = 0
    for _ in range(20):
        q = np.clip(HOME_Q + rng.normal(scale=0.4, size=7), fr3.lower, fr3.upper)
        T = fr3.tcp(q)
        result = fr3.solve_ik(T[:3, 3], T[:3, :3], seed=HOME_Q, max_iterations=400)
        if result.reachable:
            solved += 1
            got = fr3.tcp(result.qpos)
            assert np.linalg.norm(got[:3, 3] - T[:3, 3]) < 5e-3
            assert np.all(result.qpos >= fr3.lower - 1e-9)
            assert np.all(result.qpos <= fr3.upper + 1e-9)
    # A random pose 0.4 rad per joint away from home is sometimes past a limit
    # from that seed; most are not.
    assert solved >= 16


def test_ik_refuses_a_pose_out_of_reach(fr3):
    result = fr3.solve_ik(np.array([1.5, 0.0, 0.5]), None)
    assert not result.reachable


def test_link_hulls_contain_the_links_and_move_with_them(fr3):
    poses = fr3.link_poses(HOME_Q)
    assert set(ARM_LINKS) <= set(poses) and TCP_LINK in poses
    for hull, T in fr3.posed_hulls(poses=poses):
        centre = hull.world_vertices(T).mean(axis=0)
        assert hull.contains(centre[None], T)[0]
        far = centre + np.array([0.0, 0.0, 1.0])
        assert not hull.contains(far[None], T)[0]
