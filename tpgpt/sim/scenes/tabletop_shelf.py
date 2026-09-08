"""Tabletop scene with several distinct objects and named shelf slots.

Built for the cross-object, cross-embodiment stage of the project.
:class:`~tpgpt.sim.scenes.reshelving.Reshelving` is deliberately left untouched
so the validated transportation result stays reproducible; this scene differs
from it in the two ways that stage needs:

* **Several mesh objects**, not one box. A milk carton, a can, a cereal box, a
  loaf, a bottle and a lemon have genuinely different shapes, so a grasp cannot
  be a hard-coded top-down pinch and has to come from the object's geometry.
* **Named destinations.** Every shelf slot is a first-class scene entity with a
  label, so "put the milk on the top shelf" has something to resolve against.

The shelf is a **staircase**: the lower level sits nearer the robot and the
upper level further away and higher. Both levels are therefore reachable from
directly above, which a conventional stacked bookshelf would not be -- the upper
board would roof the lower one.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    BottleObject,
    BreadObject,
    CanObject,
    CerealObject,
    LemonObject,
    MilkObject,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

#: Selectable objects, keyed by the short name the language layer resolves to.
OBJECT_CLASSES = {
    "milk": MilkObject,
    "can": CanObject,
    "cereal": CerealObject,
    "bread": BreadObject,
    "bottle": BottleObject,
    "lemon": LemonObject,
}

DEFAULT_OBJECTS = ("milk", "can", "cereal", "bread")

#: Shelf levels: (label, x of the board centre, height above the table).
#: x is bounded by the arm: end-effector x saturates near 0.24 m in this arena,
#: measured on the Panda, so the far level sits at 0.22 m and no further.
SHELF_LEVELS = (
    ("bottom", 0.13, 0.09),
    ("top", 0.22, 0.25),
)

#: Lateral slot labels and their y positions on every level.
SHELF_SLOTS = (("left", 0.13), ("middle", 0.0), ("right", -0.13))

SHELF_BOARD_DEPTH = 0.10
SHELF_BOARD_WIDTH = 0.42
SHELF_THICKNESS = 0.012

#: Interior height of a closed shelf level, keyed by variant and level label.
#:
#: The ``enclosed`` numbers are chosen so the lower cubby's roof clears the
#: upper board: bottom sits at 0.09 with a 0.14 interior, so its roof is at
#: 0.23, and the upper board is at 0.25. A taller lower cubby would intersect
#: the shelf above it.
#:
#: A 15.4 cm cereal box does not fit the enclosed lower cubby with a hand
#: around it. That is left as it is: it is a real constraint of a real shelf,
#: and finding out what the system does about it is the point of the variant.
CUBBY_HEIGHT = {"bottom": 0.18, "top": 0.18}
ENCLOSED_HEIGHT = {"bottom": 0.14, "top": 0.22}

#: Shelf variants.
#:
#: ``open``      the original four floating boards, kept so earlier results stay
#:               reproducible.
#: ``cubby``     back panel and two side walls, open at the top and the front.
#:               A top-down grasp still works, but the walls are real obstacles
#:               and a side approach into a slot is now genuinely blocked.
#: ``enclosed``  adds a roof, so the only way in is horizontally from the front.
#:               Top-down grasps become impossible.
SHELF_VARIANTS = ("open", "cubby", "enclosed")

#: An extra camera that can actually see the workspace.
#:
#: robosuite's stock ``agentview`` and ``frontview`` sit on the far side of the
#: shelf from the table, so once the shelf became solid they saw a wall: object
#: clouds went from 1433 points to zero. The shelf was always there -- it was in
#: the collision group, which is not rendered, so depth passed straight through
#: it. This camera looks down the length of the table from above and to one
#: side, past the shelf rather than through it.
WORKSPACE_CAMERA = "workspace"
WORKSPACE_CAMERA_POSE = ((-0.15, -1.05, 1.65), (-0.05, 0.0, 0.95))


def _look_at(position, target, up=(0.0, 0.0, 1.0)) -> str:
    """MuJoCo camera quaternion (w x y z) for a camera at ``position``.

    A MuJoCo camera looks along its own ``-Z`` with ``+Y`` up, so the frame's
    ``+Z`` points back towards where it came from.
    """
    position = np.asarray(position, dtype=float)
    backward = position - np.asarray(target, dtype=float)
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(up, dtype=float), backward)
    right /= np.linalg.norm(right)
    rotation = np.column_stack([right, np.cross(backward, right), backward])

    trace = np.trace(rotation)
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    if w < 1e-8:  # pragma: no cover - degenerate, not reachable for our poses
        w = 1e-8
    quat = np.array([
        w,
        (rotation[2, 1] - rotation[1, 2]) / (4 * w),
        (rotation[0, 2] - rotation[2, 0]) / (4 * w),
        (rotation[1, 0] - rotation[0, 1]) / (4 * w),
    ])
    return " ".join(f"{v:.6f}" for v in quat)


class TabletopShelf(ManipulationEnv):
    """Several objects on a table in front of a two-level staircase shelf.

    Args:
        objects: Short names from :data:`OBJECT_CLASSES` to spawn.
        shelf_variant: One of :data:`SHELF_VARIANTS`. ``"cubby"`` by default --
            a shelf with a back and sides is the honest case, and the open
            boards only ever existed because nothing was colliding with them
            yet.
        Remaining arguments follow the robosuite ``ManipulationEnv`` API.
    """

    def __init__(
        self,
        robots="Panda",
        objects=DEFAULT_OBJECTS,
        shelf_variant="cubby",
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
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        **kwargs,
    ):
        unknown = [name for name in objects if name not in OBJECT_CLASSES]
        if unknown:
            raise ValueError(
                f"unknown objects {unknown}; available: {sorted(OBJECT_CLASSES)}"
            )
        # Each name becomes a MuJoCo body name, so a repeat is a duplicate body
        # and the failure surfaces as "repeated name 'can_main' in body" from
        # deep inside the XML compiler -- true, but no help at all in finding
        # the caller that asked for two cans.
        repeated = sorted({n for n in objects if list(objects).count(n) > 1})
        if repeated:
            raise ValueError(
                f"objects {repeated} requested more than once; a scene holds at "
                "most one of each"
            )
        if shelf_variant not in SHELF_VARIANTS:
            raise ValueError(
                f"unknown shelf variant {shelf_variant!r}; expected one of {SHELF_VARIANTS}"
            )
        self.shelf_variant = shelf_variant
        self.object_names = tuple(objects)
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.objects: list = []

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
            camera_segmentations=camera_segmentations,
            **kwargs,
        )

    # ------------------------------------------------------------- geometry
    @property
    def table_top(self) -> float:
        return float(self.table_offset[2] + self.table_full_size[2] / 2)

    def slot_poses(self) -> dict[str, np.ndarray]:
        """World position of every named slot, keyed ``"<level>_<slot>"``.

        The z is the resting height of a small object placed on that board, so
        the pose is directly usable as a placement target.
        """
        poses = {}
        for level, x, height in SHELF_LEVELS:
            z = self.table_top + height + SHELF_THICKNESS / 2
            for slot, y in SHELF_SLOTS:
                poses[f"{level}_{slot}"] = np.array([x, y, z])
        return poses

    def _panel(self, body, name, pos, size, friction):
        ET.SubElement(
            body, "geom", name=name, type="box", pos=pos, size=size,
            # Group 1, not 0. Group 0 is the collision group, which robosuite
            # does not draw unless asked, so the shelf was physically present
            # and invisible in every render ever made of this scene. Group
            # membership is only about drawing; contype and conaffinity decide
            # collision, and both are left at their defaults here.
            rgba="0.55 0.38 0.22 1", group="1", friction=friction,
            solimp="0.998 0.998 0.001", solref="0.001 1",
        )

    def _add_shelf(self, arena: TableArena) -> None:
        """Append the staircase shelf to the arena as static geometry."""
        friction = " ".join(str(f) for f in self.table_friction)
        for level, x, height in SHELF_LEVELS:
            body = ET.SubElement(
                arena.worldbody,
                "body",
                name=f"shelf_{level}",
                pos=f"{x} 0 {self.table_top + height}",
            )
            half_x, half_y = SHELF_BOARD_DEPTH / 2, SHELF_BOARD_WIDTH / 2
            leg_half = height / 2
            panels = [
                ("board", "0 0 0", f"{half_x} {half_y} {SHELF_THICKNESS / 2}"),
                ("lip", f"{half_x - 0.008} 0 0.028", f"0.008 {half_y} 0.022"),
                ("leg_l", f"0 {half_y - 0.012} {-leg_half}", f"{half_x} 0.012 {leg_half}"),
                ("leg_r", f"0 {-(half_y - 0.012)} {-leg_half}", f"{half_x} 0.012 {leg_half}"),
            ]
            if self.shelf_variant != "open":
                interior = (
                    CUBBY_HEIGHT if self.shelf_variant == "cubby" else ENCLOSED_HEIGHT
                )[level]
                half_t = SHELF_THICKNESS / 2
                mid = SHELF_THICKNESS / 2 + interior / 2
                panels = [p for p in panels if p[0] != "lip"]
                panels += [
                    # Back panel, on the far side from the robot.
                    ("back", f"{half_x - half_t} 0 {mid}",
                     f"{half_t} {half_y} {interior / 2}"),
                    ("wall_l", f"0 {half_y - half_t} {mid}",
                     f"{half_x} {half_t} {interior / 2}"),
                    ("wall_r", f"0 {-(half_y - half_t)} {mid}",
                     f"{half_x} {half_t} {interior / 2}"),
                ]
                if self.shelf_variant == "enclosed":
                    panels.append(
                        ("roof", f"0 0 {SHELF_THICKNESS / 2 + interior}",
                         f"{half_x} {half_y} {half_t}")
                    )

            for part, pos, size in panels:
                self._panel(body, f"shelf_{level}_{part}", pos, size, friction)

        # Visual-only markers, one per slot, so a chosen destination is visible
        # in renders. contype/conaffinity 0 keeps them out of the physics.
        for name, position in self.slot_poses().items():
            marker = ET.SubElement(
                arena.worldbody, "body", name=f"slot_{name}", pos=" ".join(map(str, position))
            )
            ET.SubElement(
                marker, "geom", name=f"slot_{name}_g", type="box",
                size="0.045 0.045 0.002", rgba="0.1 0.6 0.9 0.25",
                group="1", contype="0", conaffinity="0",
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
        position, target = WORKSPACE_CAMERA_POSE
        ET.SubElement(
            arena.worldbody, "camera", name=WORKSPACE_CAMERA, mode="fixed",
            pos=" ".join(str(v) for v in position), quat=_look_at(position, target),
            fovy="55",
        )

        self.objects = [
            OBJECT_CLASSES[name](name=name) for name in self.object_names
        ]
        # Objects stay clear of the shelf: the Panda's hand is deeper than its
        # fingers, so a grasp taken within ~0.12 m of a board's front edge
        # catches the hand on the underside of the board.
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.objects,
            x_range=[-0.22, -0.04],
            y_range=[-0.18, 0.18],
            rotation=None,
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
            mujoco_objects=self.objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self.object_body_ids = {
            obj.name: self.sim.model.body_name2id(obj.root_body) for obj in self.objects
        }

    #: Simulation steps run after placement so objects come to rest.
    #:
    #: Mesh objects are dropped from a small height and settle by up to 3 cm.
    #: A point cloud captured before they settle describes a pose the object is
    #: no longer in, which would silently corrupt every grasp derived from it.
    SETTLE_STEPS = 60

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            for pos, quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(pos), np.array(quat)])
                )
            self.sim.forward()
            for _ in range(self.SETTLE_STEPS):
                self.sim.step()
        self.sim.forward()

    # ------------------------------------------------------------ observables
    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables
        modality = "object"

        def make_sensors(name):
            @sensor(modality=modality)
            def obj_pos(obs_cache, _name=name):
                return self.object_position(_name)

            @sensor(modality=modality)
            def obj_quat(obs_cache, _name=name):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.object_body_ids[_name]]),
                    to="xyzw",
                )

            return (f"{name}_pos", obj_pos), (f"{name}_quat", obj_quat)

        for name in self.object_names:
            for obs_name, fn in make_sensors(name):
                observables[obs_name] = Observable(
                    name=obs_name, sensor=fn, sampling_rate=self.control_freq
                )
        return observables

    # ------------------------------------------------------------------ task
    def object_position(self, name: str) -> np.ndarray:
        if name not in self.object_body_ids:
            raise KeyError(f"unknown object {name!r}; scene has {self.object_names}")
        return np.array(self.sim.data.body_xpos[self.object_body_ids[name]])

    def object_instance(self, name: str) -> str:
        """Segmentation instance name for a spawned object."""
        return name

    def is_object_in_slot(self, name: str, slot: str, tolerance: float = 0.06) -> bool:
        """Whether an object is resting in a named slot."""
        poses = self.slot_poses()
        if slot not in poses:
            raise KeyError(f"unknown slot {slot!r}; scene has {sorted(poses)}")
        offset = self.object_position(name) - poses[slot]
        return bool(np.linalg.norm(offset[:2]) < tolerance and abs(offset[2]) < 0.12)

    def reward(self, action=None):
        """Placeholder: this scene is for perception and grasping, not a task.

        Task success is defined by the caller, since what counts depends on the
        prompt that was parsed.
        """
        return 0.0
