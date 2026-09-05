"""Transport of a full label set through a scene-like keypoint change."""

import numpy as np
from scipy.spatial.transform import Rotation

from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.sim.keypoints import cube_keypoints, pair_keypoints
from tpgpt.transport.labels import PolicyLabels, transport_labels
from tpgpt.transport.maps import TransportMap
from tpgpt.utils.rotations import is_rotation


def _scene_keypoints(product_pos, product_yaw, goal_pos):
    product = cube_keypoints(
        product_pos,
        Rotation.from_euler("z", product_yaw).as_matrix(),
        0.025,
        name="product",
    )
    goal = cube_keypoints(goal_pos, None, 0.025, name="goal")
    return KeypointUnion(product, goal)


class KeypointUnion:
    """Concatenate two keypoint sets while preserving label pairing."""

    def __init__(self, a, b):
        from tpgpt.sim.keypoints import KeypointSet

        self.set = KeypointSet(
            points=np.vstack([a.points, b.points]), labels=[*a.labels, *b.labels]
        )


def test_two_objects_moving_differently_need_the_nonlinear_stage():
    """Two objects displaced differently cannot be matched by one rigid motion.

    This is the case the paper is built for, and the one the prototype never
    tested: its target scene was a pure translation, which the affine stage
    solves exactly.
    """
    source = _scene_keypoints([-0.11, -0.17, 0.85], 0.0, [0.20, -0.13, 1.10]).set
    target = _scene_keypoints([-0.15, 0.09, 0.85], 0.6, [0.20, 0.00, 1.06]).set
    S, T = pair_keypoints(source, target)

    affine_only = TransportMap(residual=None).fit(S, T)
    full = TransportMap().fit(S, T)

    affine_error = np.linalg.norm(T - affine_only.transport_positions(S), axis=1).max()
    full_error = full.keypoint_residual().max()
    assert affine_error > 0.02  # a rigid motion cannot do it
    assert full_error < 1e-4  # the residual stage can


def test_full_label_set_survives_transport():
    source = _scene_keypoints([-0.11, -0.17, 0.85], 0.0, [0.20, -0.13, 1.10]).set
    target = _scene_keypoints([-0.15, 0.09, 0.85], 0.6, [0.20, 0.00, 1.06]).set
    S, T = pair_keypoints(source, target)
    tm = TransportMap().fit(S, T)

    m = 60
    phase = np.linspace(0, 1, m)
    labels = PolicyLabels(
        positions=np.c_[
            np.linspace(-0.11, 0.20, m), np.linspace(-0.17, -0.13, m),
            np.linspace(0.85, 1.10, m),
        ],
        velocities=np.tile([0.1, 0.0, 0.05], (m, 1)),
        orientations=np.tile(np.eye(3), (m, 1, 1)),
        stiffness=np.tile(np.diag([350.0, 350.0, 350.0]), (m, 1, 1)),
        damping=np.tile(np.diag([33.0, 33.0, 33.0]), (m, 1, 1)),
        gripper=np.where(phase < 0.5, -1.0, 1.0),
        time_belief=phase,
        time_rate=np.full(m, 1.0),
    )
    out = transport_labels(tm, labels)

    assert all(is_rotation(r, atol=1e-8) for r in out.orientations)
    assert np.linalg.eigvalsh(out.stiffness).min() > 0
    assert np.allclose(np.sort(np.linalg.eigvalsh(out.stiffness), axis=1), 350.0, atol=1e-6)
    assert out.metadata["jacobian_fraction_positive"] == 1.0
    # A policy fits and reproduces them.
    policy = GPPolicy().fit(out)
    predicted = policy.predict(out.positions, out.time_belief)
    assert np.abs(predicted.velocity - out.velocities).max() < 0.05
