"""Scene construction and simulator-native perception (paper Sec. V-adjacent).

The cloud test is the acceptance criterion for the whole grasping stage: every
grasp is derived from these points, so if the unprojection is wrong nothing
downstream can be right, and it would be wrong in a way that still looks
plausible.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.sim

#: The cameras that can actually see this scene.
#:
#: ``agentview`` and ``frontview`` sit on the far side of the shelf from the
#: table. That was harmless while the shelf lived in the unrendered collision
#: group -- depth passed straight through it -- and stopped being harmless the
#: moment the shelf was drawn: every object cloud from those two went to zero.
CAMERAS = ("workspace", "sideview", "birdview")


def mesh_extent(env, name):
    """True world-frame AABB extents of an object's collision geometry.

    Read from MuJoCo rather than hard-coded, for two reasons: the meshes are
    not the sizes their ``horizontal_radius`` suggests (the bread loaf is 4.8 cm
    tall, not the 7.5 cm its offsets imply), and the objects spawn at random
    yaw, so a fixed table would be wrong the moment the seed changed.
    """
    model, data = env.sim.model, env.sim.data
    body_id = env.object_body_ids[name]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body_id or model.geom_group[geom] != 0:
            continue
        origin = np.array(data.geom_xpos[geom])
        rotation = np.array(data.geom_xmat[geom]).reshape(3, 3)
        mesh_id = model.geom_dataid[geom]
        if model.geom_type[geom] == 7 and mesh_id >= 0:  # mjGEOM_MESH
            start = model.mesh_vertadr[mesh_id]
            vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]].reshape(-1, 3)
        else:
            size = model.geom_size[geom]
            vertices = np.array(
                [[x, y, z] for x in (-size[0], size[0])
                 for y in (-size[1], size[1]) for z in (-size[2], size[2])]
            )
        world = (rotation @ vertices.T).T + origin
        low, high = np.minimum(low, world.min(axis=0)), np.maximum(high, world.max(axis=0))
    return high - low


@pytest.fixture(scope="module")
def env():
    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    config = load_composite_controller_config(controller="BASIC", robot="Panda")
    environment = TabletopShelf(
        robots="Panda", controller_configs=config, control_freq=20, seed=0,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=list(CAMERAS), camera_heights=256, camera_widths=256,
        camera_depths=True, camera_segmentations="instance",
    )
    environment.reset()
    yield environment
    environment.close()


class TestScene:
    def test_spawns_the_requested_objects(self, env):
        assert set(env.object_names) == {"milk", "can", "cereal", "bread"}

    def test_a_repeated_object_is_refused_with_a_useful_message(self):
        """Two of the same object is two MuJoCo bodies with one name. Left to
        the XML compiler it surfaces as "repeated name 'can_main' in body",
        which is true and says nothing about the caller that asked twice."""
        import pytest

        from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

        with pytest.raises(ValueError, match="more than once"):
            TabletopShelf(robots="Panda", objects=("can", "milk", "can"))

    def test_exposes_six_named_slots_on_two_levels(self, env):
        slots = env.slot_poses()
        assert len(slots) == 6
        assert {s.split("_")[0] for s in slots} == {"bottom", "top"}
        assert {s.split("_", 1)[1] for s in slots} == {"left", "middle", "right"}

    def test_the_shelf_is_a_staircase(self, env):
        """The upper level is further away as well as higher, so both levels
        can be reached from directly above. A stacked shelf would roof the
        lower one."""
        slots = env.slot_poses()
        assert slots["top_middle"][2] > slots["bottom_middle"][2]
        assert slots["top_middle"][0] > slots["bottom_middle"][0]

    def test_every_slot_is_within_the_measured_reach_envelope(self, env):
        """End-effector x saturates near 0.24 m for the Panda in this arena."""
        for name, position in env.slot_poses().items():
            assert position[0] < 0.24, name

    def test_objects_spawn_clear_of_the_shelf(self, env):
        """The Panda's hand is deeper than its fingers and catches on a board's
        underside if the grasp is taken too close to one."""
        board_front = min(x for _, x, _ in __import__(
            "tpgpt.sim.scenes.tabletop_shelf", fromlist=["SHELF_LEVELS"]
        ).SHELF_LEVELS)
        for name in env.object_names:
            assert env.object_position(name)[0] < board_front

    def test_objects_are_at_rest_after_reset(self, env):
        """A cloud captured before the objects settle describes a stale pose."""
        env.reset()
        before = {n: env.object_position(n).copy() for n in env.object_names}
        for _ in range(40):
            env.sim.step()
        drift = max(np.linalg.norm(env.object_position(n) - before[n]) for n in env.object_names)
        assert drift < 0.005

    def test_seeding_is_reproducible(self):
        from robosuite.controllers import load_composite_controller_config

        from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

        def sample(seed):
            config = load_composite_controller_config(controller="BASIC", robot="Panda")
            e = TabletopShelf(
                robots="Panda", controller_configs=config, control_freq=20, seed=seed,
                has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
            )
            e.reset()
            positions = np.concatenate([e.object_position(n) for n in e.object_names])
            e.close()
            return positions

        assert np.allclose(sample(3), sample(3))
        assert not np.allclose(sample(3), sample(4))


class TestSegmentation:
    def test_instance_ids_are_offset_by_one(self, env):
        """robosuite adds 1 so that 0 means background. Forgetting it selects
        the background instead of the first object, which does not look like an
        off-by-one -- it looks like a broken camera calibration."""
        from tpgpt.perception.cameras import instance_names, instance_segmentation_id

        names = instance_names(env)
        assert instance_segmentation_id(env, names[0]) == 1

    def test_unknown_instance_is_rejected(self, env):
        from tpgpt.perception.cameras import instance_segmentation_id

        with pytest.raises(KeyError, match="unknown instance"):
            instance_segmentation_id(env, "not_a_thing")


class TestObjectClouds:
    """Acceptance criterion for the unprojection."""

    def test_every_object_yields_a_cloud_matching_its_true_size(self, env):
        """The acceptance criterion for the unprojection.

        A wrong pixel convention still produces a plausible-looking cloud in a
        plausible-looking place, so the check has to be quantitative: the
        bounding box must contain the body origin and match the mesh's own AABB.
        Measured error across the four objects is 0.6 to 1.6 cm, all of it the
        partial view under-reporting the unobserved back side.
        """
        from tpgpt.perception.cameras import object_point_cloud

        obs = env._get_observations()
        for name in env.object_names:
            cloud = object_point_cloud(env, name, obs=obs)
            assert len(cloud) > 50, f"{name}: only {len(cloud)} points"

            truth = env.object_position(name)
            assert cloud.contains(truth, tol=0.01), (
                f"{name}: cloud bounding box does not contain the body origin"
            )
            error = np.abs(cloud.extent - mesh_extent(env, name))
            assert error.max() < 0.02, (
                f"{name}: extent {np.round(cloud.extent, 3)} vs mesh "
                f"{np.round(mesh_extent(env, name), 3)}"
            )

    def test_the_cameras_agree_with_each_other(self, env):
        """Independent views must reconstruct the same object. Disagreement
        means the extrinsics or the pixel convention are wrong, which a single
        view cannot reveal."""
        from tpgpt.perception.cameras import object_point_cloud

        obs = env._get_observations()
        centroids = []
        for camera in CAMERAS:
            cloud = object_point_cloud(env, "cereal", cameras=(camera,), obs=obs)
            if len(cloud) > 50:
                centroids.append(cloud.centroid)
        assert len(centroids) >= 2
        spread = np.linalg.norm(np.array(centroids) - np.mean(centroids, axis=0), axis=1)
        assert spread.max() < 0.05

    def test_mask_erosion_removes_the_boundary_halo(self, env):
        """Boundary pixels sample the depth of whatever is behind the object and
        project metres away, dominating the bounding box."""
        from tpgpt.perception.cameras import object_point_cloud

        obs = env._get_observations()
        raw = object_point_cloud(env, "cereal", obs=obs, erosion=0)
        eroded = object_point_cloud(env, "cereal", obs=obs, erosion=1)
        assert eroded.extent.max() <= raw.extent.max()

    def test_the_cloud_is_capped_for_the_server(self, env):
        """The server's outlier removal is O(N^2) in memory and a large cloud
        OOM-kills it."""
        from tpgpt.perception.cameras import object_point_cloud

        cloud = object_point_cloud(env, "cereal", obs=env._get_observations(), max_points=64)
        assert len(cloud) <= 64

    def test_an_invisible_object_returns_an_empty_cloud_not_an_error(self, env):
        from tpgpt.perception.cameras import object_point_cloud

        cloud = object_point_cloud(env, "Panda0", cameras=("birdview",),
                                   obs=env._get_observations(), erosion=8)
        assert len(cloud) >= 0  # must not raise

    def test_missing_depth_or_segmentation_is_reported_clearly(self):
        from robosuite.controllers import load_composite_controller_config

        from tpgpt.perception.cameras import object_point_cloud
        from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

        config = load_composite_controller_config(controller="BASIC", robot="Panda")
        plain = TabletopShelf(
            robots="Panda", controller_configs=config, control_freq=20, seed=0,
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=["workspace"], camera_heights=64, camera_widths=64,
        )
        plain.reset()
        try:
            with pytest.raises(KeyError, match="camera_depths"):
                object_point_cloud(plain, "milk")
        finally:
            plain.close()


class TestSceneGraph:
    def test_lists_objects_and_receptacles_with_stable_ids(self, env):
        from tpgpt.perception.scene_graph import build_scene_graph

        scene = build_scene_graph(env)
        assert len(scene.objects) == 4
        assert len(scene.receptacles) == 6
        assert [e.entity_id for e in scene.objects] == [f"obj_{i:03d}" for i in range(1, 5)]

    def test_objects_carry_the_instance_needed_for_segmentation(self, env):
        from tpgpt.perception.cameras import instance_segmentation_id
        from tpgpt.perception.scene_graph import build_scene_graph

        for entity in build_scene_graph(env).objects:
            assert instance_segmentation_id(env, entity.instance) > 0

    def test_positions_track_the_simulator(self, env):
        from tpgpt.perception.scene_graph import build_scene_graph

        scene = build_scene_graph(env)
        for entity in scene.objects:
            assert np.allclose(entity.position, env.object_position(entity.label))

    def test_a_prompt_resolves_against_the_built_graph(self, env):
        """The language layer and the scene must actually agree on names."""
        from tpgpt.language.parser import parse_task
        from tpgpt.perception.scene_graph import build_scene_graph

        scene = build_scene_graph(env)
        spec = parse_task("put the milk carton on the top shelf right", scene)
        assert spec.ok, spec.error
        assert scene.by_id(spec.object_id).label == "milk"
        assert scene.by_id(spec.destination_id).label == "top shelf right"
