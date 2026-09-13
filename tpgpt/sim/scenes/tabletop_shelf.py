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

import warnings

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    BottleObject,
    BreadObject,
    CanObject,
    CerealObject,
    HammerObject,
    HollowCylinderObject,
    LemonObject,
    MilkObject,
    PotWithHandlesObject,
    RatchetingWrenchObject,
    RoundNutObject,
    SquareNutObject,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

#: Selectable objects, keyed by the short name the language layer resolves to.
#:
#: The first six are the grocery meshes every campaign so far has used. The rest
#: are **shapes a two-finger jaw and a multi-finger hand should disagree about**,
#: which the grocery set cannot show: a box, a carton, a can and a loaf are all
#: things a parallel jaw is good at, so a fleet measured only on them cannot
#: demonstrate that more fingers buy anything.
#:
#: ``hammer`` and ``wrench`` have a handle to grip and mass somewhere else, so
#: the grasp has a moment about it. ``pot`` has handles, which is the case a
#: hand with opposable fingers should win outright. ``mug`` is a thin wall and a
#: rim rather than a solid body. ``nut_square`` and ``nut_round`` are small and
#: have a hole through them.
#:
#: Adding a key here changes nothing on its own: ``DEFAULT_OBJECTS`` is
#: unchanged, and every campaign names its own object list. But note that the
#: placement sampler lays objects out **in the order it is given**, so any run
#: with a different set is a different scene and is not cell-by-cell comparable
#: with one that has a different set.
OBJECT_CLASSES = {
    "milk": MilkObject,
    "can": CanObject,
    "cereal": CerealObject,
    "bread": BreadObject,
    "bottle": BottleObject,
    "lemon": LemonObject,
    "hammer": HammerObject,
    "wrench": RatchetingWrenchObject,
    "pot": PotWithHandlesObject,
    "mug": HollowCylinderObject,
    "nut_square": SquareNutObject,
    "nut_round": RoundNutObject,
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

#: How deep each board is, front to back, keyed by level.
#:
#: **Per level, and the top one is deep on purpose.** A shelf board only has to
#: be as deep as the object it holds; the *cubby* around it has to be deep
#: enough for the hand that puts the object there, and a hand is much bigger
#: than a carton. Measured on robosuite's own models, the grippers in this
#: registry are 83 to 204 mm across their jaw axis and 63 to 75 mm across the
#: perpendicular:
#:
#: ============  ==================  ====================
#: hand          across the jaws     across the other way
#: ============  ==================  ====================
#: yumi                     83 mm                   69 mm
#: robotiq85               148 mm                   75 mm
#: xarm                    172 mm                   75 mm
#: robotiq140              202 mm                   75 mm
#: panda                   204 mm                   63 mm
#: ============  ==================  ====================
#:
#: At 100 mm the top cubby left **88 mm** of usable depth between the board's
#: front edge and the back panel, so every hand fitted only when turned exactly
#: sideways, and nothing at all fitted turned front-to-back. The hand is driven
#: into the back panel on 16 of 20 cells of `FINDINGS.md` 8o, the arm jams, and
#: the object is released 20 to 208 mm short of where it was commanded. That is
#: the scene refusing the task, not the method failing it.
#:
#: 280 mm leaves **268 mm** of usable depth: the widest hand spans 202 mm and
#: clears both the front edge and the back panel by about 30 mm in **either**
#: orientation. The bottom level is left at 100 mm -- it is not used as a
#: destination and deepening it would push its boards out over the table.
#:
#: **This is a clearance fix, not a realism fix.** The objects here are still
#: robosuite's benchmark meshes, which are 1.4 to 2.5 times smaller than the
#: articles they represent, and a shelf 280 mm deep is still shallower than a
#: real one. Scaling the whole scene consistently is a separate experiment --
#: see the open item in `ROBOTICS_NOTES.md` -- because real-sized objects change
#: which grasps exist at all, and that has to be measured rather than mixed in
#: here.
SHELF_BOARD_DEPTH = {"bottom": 0.10, "top": 0.28}
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
#: **The bottom figure is 0.148, not 0.18, and that was a modelling defect.**
#: A cubby's interior cannot extend past the shelf above it. The bottom board
#: sits 90 mm above the table and the top board's underside 244 mm above it, so
#: the space between them is 148 mm. At 180 the lower cubby's back panel and
#: side walls poked **20 mm up through the top board**, which put a solid
#: obstacle inside the slot the arm was trying to place into -- and it is what
#: ``free_corridor`` kept reporting as ``shelf_bottom_back`` when asked whether
#: a hand could arrive at ``top_middle`` from the front.
CUBBY_HEIGHT = {"bottom": 0.148, "top": 0.18}
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
            half_x, half_y = SHELF_BOARD_DEPTH[level] / 2, SHELF_BOARD_WIDTH / 2
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

    #: Minimum simulation steps run after placement so objects come to rest.
    #:
    #: Mesh objects are dropped from a small height and settle by up to 3 cm.
    #: A point cloud captured before they settle describes a pose the object is
    #: no longer in, which would silently corrupt every grasp derived from it.
    SETTLE_STEPS = 60

    #: Hard cap on settling, in simulation steps. 2000 steps is 4 s at a 2 ms
    #: timestep -- four times the slowest case measured, so it is a guard against
    #: a never-resting scene rather than a working limit.
    MAX_SETTLE_STEPS = 2000

    #: An object counts as still when it moves less than this, in metres, over
    #: :data:`SETTLE_WINDOW` steps.
    #:
    #: **Displacement, not velocity.** Velocity is the obvious criterion and it
    #: does not work: an object resting on the table carries 8 to 23 mm/s and
    #: 0.18 to 0.65 rad/s of solver jitter indefinitely, oscillating in sign, so
    #: it never falls below any threshold tight enough to be meaningful -- while
    #: the object's actual position does not change by 0.1 mm over a second.
    #: Asking whether it *went anywhere* sidesteps the jitter entirely.
    SETTLE_TOLERANCE = 5e-4

    #: Steps over which that displacement is measured.
    SETTLE_WINDOW = 50

    #: Consecutive still windows required before the scene counts as settled.
    #:
    #: **Momentarily still is not stable.** With a single window, the UMI
    #: scene's cereal box passed at 150 steps and then toppled 63.8 mm on its
    #: own -- a tall box balanced on an edge pauses before it tips, and one
    #: window cannot tell that pause from rest. Requiring the stillness to
    #: persist for 4 windows (200 steps) means a box that is about to go has to
    #: hold station for a fifth of a second first, which a tipping box does not.
    SETTLE_CONFIRMATIONS = 4

    #: Where every arm's **fingertips** start, in world coordinates.
    #:
    #: **The same point for every hand, not the same joint configuration.** A
    #: shared joint configuration puts a 97 mm-deep Panda hand and a 270 mm-deep
    #: Robotiq 2F-140 in completely different places, which is what made the
    #: scene depend on the gripper. Solving IK per hand for a shared *fingertip*
    #: pose is what actually makes the start position identical, and the
    #: fingertip is the right point because the transported labels are a
    #: fingertip path (7.20).
    #:
    #: Chosen above and behind the sampling region so that no hand overlaps an
    #: object at reset. The old rest pose sat *inside* it: measured
    #: interpenetration at placement was 4.85 mm (yumi), 12.85 (robotiq85),
    #: 20.64 (xarm) and 26.77 (robotiq140), and MuJoCo ejects an object it finds
    #: inside a body, which is what scattered the scene differently for every
    #: hand. Section 7.32.
    #:
    #: **Eight of the nine registered hands reach this, and every other point
    #: tried.** Swept over 48 candidates -- x in -0.05 to -0.20, y in -0.05 to
    #: 0.05, z in 1.05 to 1.20 -- the panda, yumi, xarm, robotiq85, robotiq140,
    #: rethink, robotiq3f and inspire each reached **48 of 48**. The **UMI
    #: reached 0 of 48**: its 117.2 mm contact offset, the registry's only one
    #: with a large lateral component, puts the wrist target outside the arm's
    #: envelope everywhere in that volume. That is the same limit that made 8 of
    #: its 8 Tier 2 paths unreachable, so it is an embodiment limit rather than a
    #: bad choice of point, and the UMI is excluded from cross-gripper campaigns
    #: on this measurement rather than by omission.
    HOME_TCP = np.array([-0.10, 0.0, 1.15])

    #: Orientation the hand starts in: approach straight down, jaws along y.
    HOME_ROTATION = np.array([[1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0],
                              [0.0, 0.0, -1.0]])

    def _gripper_short_name(self):
        """Registry short name of the mounted hand, or ``None``."""
        try:
            from tpgpt.grasp.grippers import resolve_pair
            gripper = self.robots[0].gripper
            gripper = gripper["right"] if isinstance(gripper, dict) else gripper
            return resolve_pair(type(gripper).__name__)
        except Exception:  # pragma: no cover - an unregistered hand
            return None

    def _move_arm_home(self):
        """Put the fingertips at :data:`HOME_TCP`, whatever hand is mounted."""
        from tpgpt.grasp.grasps import contact_offset
        from tpgpt.sim.kinematics import solve_ik

        pair = self._gripper_short_name()
        if pair is None:
            return
        try:
            offset = contact_offset(pair)
        except Exception:  # pragma: no cover - an unmeasured hand
            return
        # ``solve_ik`` targets the grip_site, so step back from the fingertips
        # by this hand's own contact offset -- the same conversion the replay
        # does, and the reason every hand lands with its *fingertips* together.
        wrist = self.HOME_TCP - self.HOME_ROTATION @ offset
        result = solve_ik(self, wrist, self.HOME_ROTATION, arm="right")
        controller = self.robots[0].composite_controller.part_controllers["right"]
        index = np.asarray(controller.qpos_index)
        self.sim.data.qpos[index] = np.asarray(result.qpos, dtype=float)
        self.sim.data.qvel[np.asarray(controller.qvel_index)] = 0.0
        self.sim.forward()
        self.home_reachable = bool(result.reachable)

    def _hold_arm(self):
        """Cancel gravity on the arm so it does not sag while objects settle.

        The settle loop calls ``sim.step()`` directly -- raw physics, no
        controller -- so the arm is unactuated and falls. It falls *differently*
        for each hand, because a Robotiq 2F-140 is much heavier than a Panda
        hand, which is why the fingertips ended up 200 mm apart across grippers
        even after being placed at a common point. Applying the bias force holds
        the configuration without any controller.

        **Only the arm's degrees of freedom.** A first version wrote
        ``qfrc_applied[:] = qfrc_bias`` across the whole model, which cancels
        gravity on the *objects* as well: they hung at their placement heights
        and never settled at all, while the scene reported itself perfectly
        reproducible and perfectly upright precisely because nothing had moved.
        A settle that freezes what it is meant to settle looks exactly like a
        settle that works.
        """
        data = self.sim.data
        data.qfrc_applied[:] = 0.0
        try:
            controller = self.robots[0].composite_controller.part_controllers["right"]
            dofs = np.asarray(controller.qvel_index)
        except Exception:  # pragma: no cover - an embodiment without that arm
            return
        data.qfrc_applied[dofs] = data.qfrc_bias[dofs]

    #: Gap left between an object's lowest point and the table when placing it.
    #:
    #: One millimetre, enough that the solver sees a clean approaching contact
    #: rather than an initial interpenetration, and small enough that the object
    #: does not fall far enough to bounce.
    PLACEMENT_CLEARANCE = 1e-3

    #: Placement attempts before giving up on a robot-free arrangement.
    PLACEMENT_ATTEMPTS = 20

    def _geom_vertices(self, body_id):
        """World-frame corners of every geom's bounding box on a body.

        From ``model.geom_aabb``, MuJoCo's own local-frame bounding box, rather
        than from the mesh vertex array. A first version read ``mesh_vert``
        directly and was wrong: MuJoCo keeps each mesh's own frame in
        ``mesh_pos`` / ``mesh_quat``, and the cereal's and the bread's are a
        90-degree rotation, so transforming the raw vertices by ``geom_xmat``
        alone put their extents on the wrong axes. Objects were then seated
        through the table and fell to the floor -- the cereal ended at 15 mm.

        The box is conservative for a tilted object, but these are placed with a
        yaw-only rotation, for which its z extent is exact.
        """
        model, data = self.sim.model, self.sim.data
        out = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body_id:
                continue
            centre, half = np.array(model.geom_aabb[g][:3]), np.array(model.geom_aabb[g][3:])
            corners = np.array([[x, y, z] for x in (-half[0], half[0])
                                for y in (-half[1], half[1])
                                for z in (-half[2], half[2])]) + centre
            rot = np.array(data.geom_xmat[g]).reshape(3, 3)
            out.append(corners @ rot.T + np.array(data.geom_xpos[g]))
        return np.vstack(out) if out else np.zeros((0, 3))

    def _seat_objects(self):
        """Lower every object until it just rests on the table.

        **The sampler drops them, and it drops them much further than it says.**
        ``z_offset`` is 2 mm, but robosuite places an object at
        ``table_z + z_offset + |bottom_offset|`` and each of these objects
        declares a ``bottom_offset`` about 25 mm larger than its true half
        height. Measured: the cereal is placed at 902.0 mm and comes to rest at
        874.6, the milk at 887.0 resting at 860.9, the can 862.0 -> 840.1, the
        bread 847.0 -> 822.2 -- a **22 to 27 mm** drop in every case, landing at
        roughly 0.7 m/s.

        That is more than enough to bounce a 150 mm cereal box onto its side,
        which is exactly what happened: with the old fixed 60-step settle the
        scene handed the box over *mid-topple*, so the cloud, the grasp planned
        on it and the keypoint box all described a pose the object was leaving.
        Section 7.32.

        Seating them from their own geometry removes the impact entirely, so
        they stay in the upright pose the sampler intended.
        """
        table_z = float(self.table_offset[2])
        for obj in self.objects:
            body = self.object_body_ids[obj.name]
            vertices = self._geom_vertices(body)
            if not len(vertices):
                continue
            drop = float(vertices[:, 2].min()) - (table_z + self.PLACEMENT_CLEARANCE)
            if abs(drop) < 1e-9:
                continue
            address = self.sim.model.jnt_qposadr[
                self.sim.model.joint_name2id(obj.joints[0])
            ]
            self.sim.data.qpos[address + 2] -= drop
        self.sim.forward()

    def _robot_penetration(self):
        """Worst interpenetration between an object and the robot, in metres.

        Zero when they are merely touching or apart. A placement that starts
        *inside* the arm is ejected violently on the first step: measured at
        **32 mm** of penetration between a Robotiq 2F-85's inner knuckle and the
        can, which throws the can across the table before anything is commanded.
        It is gripper-dependent, because a deeper hand reaches further into the
        sampling region at the same joint configuration.
        """
        model, data = self.sim.model, self.sim.data
        object_bodies = set(self.object_body_ids.values())
        worst = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            b1 = model.geom_bodyid[contact.geom1]
            b2 = model.geom_bodyid[contact.geom2]
            hits_object = (b1 in object_bodies) != (b2 in object_bodies)
            if not hits_object:
                continue
            other = contact.geom2 if b1 in object_bodies else contact.geom1
            name = model.geom_id2name(other) or ""
            if name.startswith(("robot", "gripper")):
                worst = max(worst, -float(contact.dist))
        return worst

    def _object_positions(self):
        """World position of every object, as one array."""
        return np.array([
            np.array(self.sim.data.body_xpos[self.object_body_ids[obj.name]])
            for obj in self.objects
        ]) if self.objects else np.zeros((0, 3))

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # **Arm out of the way before the objects exist.** Otherwise they are
            # created inside it and thrown clear on the first step.
            self._move_arm_home()
            for pos, quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(pos), np.array(quat)])
                )
            self.sim.forward()
            # Recorded, not acted on. Seating the objects on the table was tried
            # and made things worse -- see 7.32 -- so the scene still drops them
            # ~25 mm and this says how bad the starting state is. It reads 0.00
            # for every registered hand now that the arm starts clear.
            self.placement_penetration = self._robot_penetration()
            # **Settle until the objects stop, not for a fixed count.**
            #
            # A fixed 60 steps -- 0.12 s -- was the original rule and it is not
            # enough. Measured with the arm held completely still, the 150 mm
            # cereal box needs 60 steps in one scene and **480** in another, and
            # in the four scenes where 60 was short it did not merely drift: it
            # **toppled over**, falling 35 to 78 mm entirely on its own.
            #
            # The reason it is scene-dependent is that a different gripper is a
            # different MuJoCo model, so the constraint solver's arithmetic
            # differs and a marginally balanced tall box tips one way or the
            # other. The same seed put the cereal's origin anywhere from 804 to
            # 889 mm across six grippers.
            #
            # Everything downstream reads a pose the object does not hold: the
            # cloud, the grasp planned on it, the keypoint box fitted to it, and
            # the physics. Section 7.32.
            self._hold_arm()
            self.settle_steps_taken = 0
            reference = self._object_positions()
            moved, still = float("inf"), 0
            for step in range(self.MAX_SETTLE_STEPS):
                self.sim.step()
                self.settle_steps_taken = step + 1
                if (step + 1) % self.SETTLE_WINDOW:
                    continue
                current = self._object_positions()
                moved = (float(np.abs(current - reference).max())
                         if len(current) else 0.0)
                reference = current
                still = still + 1 if moved < self.SETTLE_TOLERANCE else 0
                if (step + 1 >= self.SETTLE_STEPS
                        and still >= self.SETTLE_CONFIRMATIONS):
                    break
            else:
                # Never a silent pass: a scene that will not settle is a scene
                # whose cloud cannot be trusted.
                warnings.warn(
                    f"objects still moving after {self.MAX_SETTLE_STEPS} settle "
                    f"steps ({moved * 1000:.2f} mm in the last "
                    f"{self.SETTLE_WINDOW}); any cloud captured now describes a "
                    "pose they will leave",
                    RuntimeWarning, stacklevel=2,
                )
            # Release the hold and re-assert the home pose, so every hand ends
            # the reset with its fingertips at exactly the same point whatever
            # happened during settling.
            self.sim.data.qfrc_applied[:] = 0.0
            self._move_arm_home()
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
