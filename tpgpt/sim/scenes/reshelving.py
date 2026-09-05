"""Robot reshelving scene (paper Sec. V-A).

"Robot reshelving refers to picking an object in one location, moving it, and
placing it in a desired position on a shelf." The paper's assumptions are
reproduced here:

* one movement primitive is learned from a **single** demonstration and
  transported into other object/goal configurations;
* **corner points** of the object and of the shelf slot are tracked, not poses,
  because "to describe the 3D pose of an object, we need at least 2 non-parallel
  vectors; hence, we need at least 3 non-overlapping and non-collinear points".

The randomisation follows Table II where the arm's workspace allows. The object
varies over 0.225 m in x, 0.366 m in y and 94.6 deg of yaw, exactly as
tabulated. The goal slot varies over 0.26 m in y and 0.24 m in z, against the
paper's 0.036 m and 0.675 m: with the Panda mounted at the edge of a 0.8 m
table, the reachable envelope saturates at about x = 0.24 m, and the full 0.675 m
of vertical travel would put the upper slots outside it. See ROBOTICS_NOTES.md.

The shelf is a board on legs with a low back lip, approached from above. A
multi-level shelf would require horizontal insertion under an overhanging board,
which is outside the reachable envelope here; randomising the board height
instead recovers the vertical variation that matters for the transportation map.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

#: Half extents of the product box that gets reshelved. The width is set by the
#: gripper: yaw is randomised, so the box diagonal (0.071 m) has to clear the
#: Panda gripper's 0.08 m maximum opening even when the grasp yaw is slightly
#: off.
PRODUCT_HALF_SIZE = (0.025, 0.025, 0.045)

#: Shelf geometry, in table-surface coordinates. SHELF_X is set by the arm's
#: reachable envelope: measured on the Panda in this arena, end-effector x
#: saturates near 0.24 m at every usable height.
SHELF_X = 0.20
SHELF_THICKNESS = 0.012
SHELF_WIDTH_Y = 0.42
SHELF_DEPTH_X = 0.12
SLOT_Y_POSITIONS = (-0.13, 0.0, 0.13)
#: The product spawn range is shifted away from the shelf rather than shortened:
#: the Panda's hand is deeper than its fingers, so descending to a grasp within
#: about 0.12 m of the board's front edge catches the hand on the underside of
#: the board. The full 0.225 m of Table II x-variation is preserved by extending
#: the near end instead.
#: Board height above the table, randomised per episode. The ceiling keeps the
#: hover pose above the top slot inside the arm's reachable envelope.
SHELF_HEIGHT_RANGE = (0.08, 0.30)


class Reshelving(ManipulationEnv):
    """Pick a product from the table and place it into a shelf slot.

    Args:
        robots: Robot name or list of names.
        shelf_x: Distance of the shelf front from the table centre.
        placement_seed: Deprecated alias; use ``seed``.
        success_tolerance: Planar distance below which a placement counts.
        Remaining arguments follow the robosuite ``ManipulationEnv`` API.
    """

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        hard_reset=False,
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        success_tolerance=0.05,
        **kwargs,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.success_tolerance = float(success_tolerance)

        self.slot_poses: np.ndarray | None = None
        self.goal_index: int = 0
        self.goal_position: np.ndarray = np.zeros(3)
        self.shelf_height: float = float(np.mean(SHELF_HEIGHT_RANGE))

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            **kwargs,
        )

    # ------------------------------------------------------------- geometry
    @property
    def table_top(self) -> float:
        """World z of the table surface."""
        return float(self.table_offset[2] + self.table_full_size[2] / 2)

    def _compute_slot_poses(self) -> np.ndarray:
        """World positions of every slot at the current board height."""
        z = (
            self.table_top
            + self.shelf_height
            + SHELF_THICKNESS / 2
            + PRODUCT_HALF_SIZE[2]
        )
        return np.array([[SHELF_X, y, z] for y in SLOT_Y_POSITIONS])

    def _add_shelf(self, arena: TableArena) -> None:
        """Append static shelf geometry to the arena's worldbody.

        The shelf is scene furniture, not a manipulable object, so it goes into
        the arena as welded geometry rather than as a free-jointed object.
        """
        # The body origin sits at the board; its z is set per episode via
        # sim.model.body_pos, which moves every child geom with it.
        shelf = ET.SubElement(
            arena.worldbody,
            "body",
            name="shelf",
            pos=f"{SHELF_X} 0 {self.table_top + self.shelf_height}",
        )
        half_x, half_y = SHELF_DEPTH_X / 2, SHELF_WIDTH_Y / 2
        leg_half_h = SHELF_HEIGHT_RANGE[1] / 2
        panels = [
            # name, local pos, half size
            ("shelf_board", "0 0 0", f"{half_x} {half_y} {SHELF_THICKNESS / 2}"),
            ("shelf_lip", f"{half_x - 0.008} 0 0.031",
             f"0.008 {half_y} 0.025"),
            ("shelf_leg_l", f"0 {half_y - 0.012} {-leg_half_h}",
             f"{half_x} 0.012 {leg_half_h}"),
            ("shelf_leg_r", f"0 {-(half_y - 0.012)} {-leg_half_h}",
             f"{half_x} 0.012 {leg_half_h}"),
        ]
        for name, pos, size in panels:
            ET.SubElement(
                shelf, "geom", name=name, type="box", pos=pos, size=size,
                rgba="0.55 0.38 0.22 1", group="0",
                friction=f"{self.table_friction[0]} {self.table_friction[1]} {self.table_friction[2]}",
                solimp="0.998 0.998 0.001", solref="0.001 1",
            )

        # Visual-only marker for the active goal slot. Its position is updated
        # at reset via sim.model.body_pos, so it never affects the physics.
        marker = ET.SubElement(arena.worldbody, "body", name="goal_marker", pos="0 0 -1")
        ET.SubElement(
            marker, "geom", name="goal_marker_g", type="box",
            size=f"{PRODUCT_HALF_SIZE[0]} {PRODUCT_HALF_SIZE[1]} {PRODUCT_HALF_SIZE[2]}",
            rgba="0.1 0.9 0.3 0.28", group="1", contype="0", conaffinity="0",
        )

    # ---------------------------------------------------------------- model
    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        self._add_shelf(arena)

        material = CustomMaterial(
            texture="WoodRed",
            tex_name="product_tex",
            mat_name="product_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        self.product = BoxObject(
            name="product",
            size_min=PRODUCT_HALF_SIZE,
            size_max=PRODUCT_HALF_SIZE,
            rgba=[0.8, 0.15, 0.15, 1],
            material=material,
        )

        # Table II ranges: 0.225 m in x, 0.366 m in y, 94.6 deg of yaw.
        self.placement_initializer = UniformRandomSampler(
            name="ProductSampler",
            mujoco_objects=[self.product],
            x_range=[-0.205, 0.02],
            y_range=[-0.183, 0.183],
            rotation=[-np.deg2rad(94.6) / 2, np.deg2rad(94.6) / 2],
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.002,
            rng=self.rng,
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.product],
        )

    def _setup_references(self):
        super()._setup_references()
        self.product_body_id = self.sim.model.body_name2id(self.product.root_body)
        self.goal_marker_body_id = self.sim.model.body_name2id("goal_marker")
        self.shelf_body_id = self.sim.model.body_name2id("shelf")
        self.slot_poses = self._compute_slot_poses()

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            for obj_pos, obj_quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )
            self.shelf_height = float(self.rng.uniform(*SHELF_HEIGHT_RANGE))
            self.goal_index = int(self.rng.integers(len(SLOT_Y_POSITIONS)))

        self.sim.model.body_pos[self.shelf_body_id] = [
            SHELF_X, 0.0, self.table_top + self.shelf_height
        ]
        self.slot_poses = self._compute_slot_poses()
        self.goal_position = self.slot_poses[self.goal_index].copy()
        self.sim.model.body_pos[self.goal_marker_body_id] = self.goal_position
        self.sim.forward()

    # ------------------------------------------------------------ observables
    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables
        modality = "object"

        @sensor(modality=modality)
        def product_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.product_body_id])

        @sensor(modality=modality)
        def product_quat(obs_cache):
            return convert_quat(
                np.array(self.sim.data.body_xquat[self.product_body_id]), to="xyzw"
            )

        @sensor(modality=modality)
        def goal_pos(obs_cache):
            return self.goal_position.copy()

        @sensor(modality=modality)
        def product_to_goal(obs_cache):
            return self.goal_position - np.array(self.sim.data.body_xpos[self.product_body_id])

        for s in (product_pos, product_quat, goal_pos, product_to_goal):
            observables[s.__name__] = Observable(
                name=s.__name__, sensor=s, sampling_rate=self.control_freq
            )
        return observables

    # ----------------------------------------------------------------- task
    @property
    def product_position(self) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.product_body_id])

    def _check_success(self) -> bool:
        """Product resting in the goal slot and released."""
        offset = self.product_position - self.goal_position
        placed = (
            np.linalg.norm(offset[:2]) < self.success_tolerance
            and abs(offset[2]) < PRODUCT_HALF_SIZE[2]
        )
        grasping = self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.product.contact_geoms
        )
        return bool(placed and not grasping)

    def reward(self, action=None):
        """Sparse success reward, optionally shaped by distance to the slot."""
        reward = 1.0 if self._check_success() else 0.0
        if self.reward_shaping and reward == 0.0:
            distance = np.linalg.norm(self.product_position - self.goal_position)
            reward = 0.25 * (1 - np.tanh(4.0 * distance))
        return reward * self.reward_scale
