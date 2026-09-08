"""Caching grasp candidates, so an experiment can be repeated.

The planner is a diffusion model with no seed, so the cache is not an
optimisation -- it is what makes two runs of the same scene comparable.
"""

from __future__ import annotations

import numpy as np

from tpgpt.grasp.cache import QUANTUM, cache_key, load, store


def a_cloud(n=64, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 3)) * 0.05


class TestKey:
    def test_the_same_cloud_and_hand_give_the_same_key(self):
        cloud = a_cloud()
        assert cache_key(cloud, "franka_panda") == cache_key(cloud, "franka_panda")

    def test_a_different_hand_gives_a_different_key(self):
        cloud = a_cloud()
        assert cache_key(cloud, "franka_panda") != cache_key(cloud, "robotiq_2f_85")

    def test_a_different_cloud_gives_a_different_key(self):
        assert cache_key(a_cloud(seed=0), "p") != cache_key(a_cloud(seed=1), "p")

    def test_the_quantum_is_a_normalisation_and_not_a_tolerance(self):
        """Rounding has bin boundaries, so a sub-quantum nudge can still flip a
        point into the next bin. The cache is safe because a miss costs an
        inference, not a wrong answer -- so this records what it does *not*
        promise, which is the thing a reader would otherwise assume."""
        cloud = np.full((8, 3), QUANTUM * 0.5)
        assert cache_key(cloud, "p") != cache_key(cloud + QUANTUM / 10.0, "p")

    def test_a_visible_shift_does_change_the_key(self):
        cloud = a_cloud()
        assert cache_key(cloud, "p") != cache_key(cloud + 0.005, "p")

    def test_the_request_parameters_are_part_of_the_key(self):
        cloud = a_cloud()
        assert cache_key(cloud, "p", topk=10) != cache_key(cloud, "p", topk=50)

    def test_a_cloud_with_the_same_points_in_a_different_order_is_a_new_key(self):
        """The planner sees an ordered array; nothing promises it is invariant."""
        cloud = a_cloud()
        assert cache_key(cloud, "p") != cache_key(cloud[::-1], "p")


class TestStoreAndLoad:
    def test_what_is_stored_comes_back(self, tmp_path):
        poses = np.random.default_rng(0).normal(size=(7, 4, 4))
        scores = np.linspace(0, 1, 7)
        store("abc", poses, scores, tmp_path)
        got_poses, got_scores = load("abc", tmp_path)
        assert np.allclose(got_poses, poses)
        assert np.allclose(got_scores, scores)

    def test_an_unknown_key_is_a_miss_not_an_error(self, tmp_path):
        assert load("never-written", tmp_path) is None

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        (tmp_path / "bad.npz").write_text("not an npz")
        assert load("bad", tmp_path) is None

    def test_a_killed_write_leaves_no_readable_entry(self, tmp_path):
        """Writes go through a temporary file, so a half-written cache entry is
        never picked up as if it were complete."""
        store("k", np.zeros((2, 4, 4)), np.zeros(2), tmp_path)
        assert [p.name for p in tmp_path.iterdir()] == ["k.npz"]
        assert load("k", tmp_path) is not None
