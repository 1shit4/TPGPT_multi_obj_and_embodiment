"""Policy labels and their transportation -- paper Sec. III-A, III-F, III-G."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.transport.labels import PolicyLabels, transport_labels
from tpgpt.transport.maps import TransportMap
from tpgpt.utils.rotations import is_rotation


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    m = 15
    return PolicyLabels(
        positions=rng.uniform(-0.2, 0.2, (m, 3)),
        velocities=rng.normal(0, 0.1, (m, 3)),
        orientations=Rotation.random(m, random_state=1).as_matrix(),
        stiffness=np.stack([np.diag([600.0, 400.0, 200.0])] * m),
        damping=np.stack([np.diag([40.0, 30.0, 20.0])] * m),
        gripper=np.sign(rng.normal(size=m)),
        time_belief=np.linspace(0, 1, m),
        metadata={"control_freq": 20},
    )


def test_length_and_present_families(labels):
    assert len(labels) == 15
    assert "positions" in labels.present and "stiffness" in labels.present
    assert "position_std" not in labels.present


def test_validation_rejects_inconsistent_label_counts():
    with pytest.raises(ValueError, match="labels but positions has"):
        PolicyLabels(positions=np.zeros((5, 3)), velocities=np.zeros((4, 3)))


def test_indexing_slices_every_family(labels):
    sub = labels[2:5]
    assert len(sub) == 3
    assert sub.orientations.shape == (3, 3, 3)
    assert sub.gripper.shape == (3,)
    assert len(labels[0]) == 1


def test_roundtrip_through_disk(labels, tmp_path):
    path = labels.save(tmp_path / "labels.npz")
    loaded = PolicyLabels.load(path)
    assert np.allclose(loaded.positions, labels.positions)
    assert np.allclose(loaded.orientations, labels.orientations)
    assert loaded.metadata["control_freq"] == 20


def test_optional_families_stay_absent(tmp_path):
    minimal = PolicyLabels(positions=np.zeros((4, 3)))
    loaded = PolicyLabels.load(minimal.save(tmp_path / "m.npz"))
    assert loaded.velocities is None and loaded.stiffness is None


def test_transport_carries_every_family(labels, rigid_keypoints):
    S, T, A, t = rigid_keypoints
    tm = TransportMap().fit(S, T)
    out = transport_labels(tm, labels)

    assert len(out) == len(labels)
    assert np.allclose(out.positions, labels.positions @ A.T + t, atol=1e-8)
    assert np.allclose(out.velocities, labels.velocities @ A.T, atol=1e-8)
    assert np.allclose(
        out.orientations, np.einsum("ij,njk->nik", A, labels.orientations), atol=1e-7
    )
    assert all(is_rotation(r, atol=1e-7) for r in out.orientations)
    assert np.allclose(
        np.sort(np.linalg.eigvalsh(out.stiffness), axis=1), [200.0, 400.0, 600.0], atol=1e-7
    )
    assert np.allclose(
        np.sort(np.linalg.eigvalsh(out.damping), axis=1), [20.0, 30.0, 40.0], atol=1e-7
    )


def test_scalar_channels_pass_through_unchanged(labels, rigid_keypoints):
    """The gripper command and phase are not spatial quantities."""
    S, T, _, _ = rigid_keypoints
    out = transport_labels(TransportMap().fit(S, T), labels)
    assert np.array_equal(out.gripper, labels.gripper)
    assert np.array_equal(out.time_belief, labels.time_belief)


def test_transport_attaches_uncertainty_and_diagnostics(labels, curved_keypoints):
    S, T = curved_keypoints
    out = transport_labels(TransportMap().fit(S, T), labels)
    assert out.position_std.shape == (len(labels),)
    assert out.velocity_std.shape == (len(labels), 3)
    assert out.metadata["transported"] is True
    assert out.metadata["jacobian_fraction_positive"] == 1.0
    assert out.metadata["keypoint_residual_max"] < 1e-4


def test_transport_handles_position_only_labels(rigid_keypoints):
    S, T, A, t = rigid_keypoints
    labels = PolicyLabels(positions=np.zeros((5, 3)))
    out = transport_labels(TransportMap().fit(S, T), labels)
    assert out.velocities is None and out.orientations is None
    assert out.positions.shape == (5, 3)


def test_source_labels_are_not_mutated(labels, rigid_keypoints):
    S, T, _, _ = rigid_keypoints
    before = labels.positions.copy()
    transport_labels(TransportMap().fit(S, T), labels)
    assert np.array_equal(labels.positions, before)

class TestReplace:
    """A frame change recomputes one family and must carry the rest untouched."""

    def make(self):
        return PolicyLabels(
            positions=np.zeros((4, 3)),
            velocities=np.ones((4, 3)),
            orientations=np.tile(np.eye(3), (4, 1, 1)),
            stiffness=np.tile(np.eye(3) * 300.0, (4, 1, 1)),
            damping=np.tile(np.eye(3) * 30.0, (4, 1, 1)),
            gripper=np.array([-1.0, -1.0, 1.0, 1.0]),
            time_belief=np.linspace(0, 1, 4),
            metadata={"scene": "source"},
        )

    def test_the_named_family_changes(self):
        replaced = self.make().replace(positions=np.ones((4, 3)))
        assert np.allclose(replaced.positions, 1.0)

    def test_every_other_family_survives(self):
        """The failure this guards against is silent: a dropped gripper channel
        gives a policy that never closes its fingers."""
        original = self.make()
        replaced = original.replace(positions=np.ones((4, 3)))
        for name in original.present:
            if name == "positions":
                continue
            assert np.allclose(getattr(replaced, name), getattr(original, name)), name

    def test_metadata_is_copied_not_shared(self):
        original = self.make()
        replaced = original.replace(positions=np.ones((4, 3)))
        replaced.metadata["scene"] = "target"
        assert original.metadata["scene"] == "source"

    def test_an_unknown_family_is_refused(self):
        with pytest.raises(ValueError, match="unknown label families"):
            self.make().replace(postions=np.ones((4, 3)))
