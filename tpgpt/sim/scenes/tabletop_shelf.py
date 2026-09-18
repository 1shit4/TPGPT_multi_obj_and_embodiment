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

from tpgpt.sim.objects import BENCHMARK, REAL, WORLDS, YCB, make_object, rest_quat

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

# ---------------------------------------------------------------------------
# The real-sized world.
#
# Everything above this line describes robosuite's benchmark scene and is left
# exactly as it was, so every result measured in it stays reproducible. What
# follows is the same scene built to the dimensions of the real articles and the
# real furniture, selected with ``world="real"``.
#
# **Why a second set of numbers rather than a scale factor on the first.** The
# benchmark shelf is not a small real shelf: its 12 mm boards are thinner than
# any real board, its 100 mm bottom level is shallower than any real shelf, and
# its 280 mm top level was widened as a *clearance fix* for the hand rather than
# chosen as a depth. Scaling it would preserve those choices. These are chosen
# from what a real shelf is, and then checked against what the arm can reach.
# ---------------------------------------------------------------------------

#: Real shelf board thickness: 18 mm, the standard furniture-board thickness.
REAL_SHELF_THICKNESS = 0.018

#: Real board depth, front to back. The bottom is a standard 300 mm pantry
#: shelf; the top is **380 mm**, and the extra 80 mm is not a style choice.
#:
#: **The destination must admit the longest object at every orientation it can
#: be commanded into.** At 300 mm the top board leaves 282 mm of clear depth
#: between its front edge and the back panel, and the YCB hammer's footprint
#: along that axis reaches **336.6 mm** at 81 degrees of yaw. Measured on the
#: campaign before this, 21 of 42 hammer cells were commanded into a pose that
#: intersects the panels -- through the back by up to 36.7 mm -- which is the
#: scene refusing the task rather than the method failing it. 380 mm leaves
#: 362 mm clear, which clears the worst yaw by 25 mm.
#:
#: Deepening the **top** board only, because it is the only level used as a
#: destination and it sits 160 mm above the table, so growing it forwards costs
#: nothing on the tabletop. Deepening the bottom board would bring its front
#: edge to within 119 mm of the pick region, against the ~120 mm at which a
#: Panda's hand catches the underside of a board (2.9). The benchmark scene's 100 mm bottom and 280 mm top were a
#: clearance fix for the hand (see ``SHELF_BOARD_DEPTH``); 300 mm is what a
#: shelf is. Depth costs nothing in reach, because it extends *backwards* from
#: the slot: the slot centre is the placement target and the board grows away
#: from the robot behind it.
REAL_SHELF_BOARD_DEPTH = {"bottom": 0.30, "top": 0.38}

#: Real board width: 640 mm, a standard bookcase bay.
REAL_SHELF_BOARD_WIDTH = 0.64

#: Real shelf levels: ``(label, x of the board centre, height above the table)``.
#:
#: **Stacked, not a staircase, and only the top level is a destination.** The
#: benchmark scene offsets the upper level further out so that both are
#: reachable from directly above. At real size that is not achievable and the
#: staircase buys nothing, because a real object is too tall for the lower
#: level anyway. The measurement:
#:
#: A real cereal carton is 300 mm tall, so a hand grasping one standing on a
#: board sits about 150 mm above that board. Swept with ``solve_ik`` over the
#: whole workspace for a top-down hand, the deepest hand in the fleet (the
#: Robotiq 2F-140, contact offset 38.2 mm) reaches:
#:
#: ===========  ===============================================
#: board x      highest fingertip z, at y = 0 / 0.15 / 0.20
#: ===========  ===============================================
#: 0.18                    1.25 / 1.25 / 1.20
#: 0.20                    1.20 / 1.20 / 1.15
#: 0.22                    1.15 / 1.10 / --
#: 0.24                    1.10 / --   / --
#: ===========  ===============================================
#:
#: The board height was then chosen by sweeping it against all seven hands
#: rather than argued: for each candidate height the check asks whether every
#: hand can reach both the placement pose (the object's mid-height above the
#: board) and a clearance pose 80 mm above it, at the middle slot and at the
#: outer one. Measured, with the cereal carton as the tallest object:
#:
#: ==========  =======  =========  ==================================
#: board       place z  clear z    hands reaching both, of seven
#: ==========  =======  =========  ==================================
#: 0.14          1.124     1.204   7
#: **0.16**    **1.144** **1.224** **7**
#: 0.18          1.164     1.244   5  (robotiq3f, robotiq3f_dex out)
#: 0.20          1.184     1.264   4
#: ==========  =======  =========  ==================================
#:
#: So 0.16, the highest that every hand clears, with 20 mm of margin below the
#: first failure. The benchmark scene's ``x = 0.22`` would be out of reach for
#: every hand in the fleet at any of these heights, which is why the board moves
#: in as it grows deeper.
#:
#: A second level is kept because a real shelf unit has one, and because it is a
#: real obstacle below the destination. It is not used as a destination: with
#: the top board 160 mm up and 18 mm thick, the lower compartment is 102 mm
#: clear, which holds nothing in this object set standing up. That is a property of a
#: real shelf, not a modelling shortcut.
REAL_SHELF_LEVELS = (
    ("bottom", 0.18, 0.04),
    ("top", 0.18, 0.16),
)

#: Real lateral slot positions. 160 mm apart, against the benchmark's 130.
#:
#: Wide enough that the three destinations are genuinely different places for a
#: 190 mm cereal carton, and narrow enough that the outer slots stay inside the
#: envelope measured above (``y = 0.16`` reaches 1.25, ``y = 0.20`` only 1.20).
#: Only one object is ever in the scene, so slots never have to hold two
#: objects side by side.
REAL_SHELF_SLOTS = (("left", 0.16), ("middle", 0.0), ("right", -0.16))

#: Interior clear height of each real cubby.
#:
#: The top is 320 mm, a standard pantry shelf spacing, which clears a 300 mm
#: cereal carton. The bottom is what is left between the two boards, 102 mm, and
#: is derived rather than chosen: a cubby's interior cannot extend past the
#: shelf above it, which is the modelling defect recorded against
#: ``CUBBY_HEIGHT``.
REAL_CUBBY_HEIGHT = {"bottom": 0.102, "top": 0.32}

#: ``enclosed`` adds a roof, so each interior loses the roof's own thickness.
REAL_ENCLOSED_HEIGHT = {"bottom": 0.084, "top": 0.30}

#: Where a real-world object is put on the table, as ``(x, y, yaw degrees)``
#: relative to the table centre.
#:
#: **Three fixed poses, written down once and reused for every cell**, rather
#: than three draws of a random sampler. This is the rule that decides whether a
#: cross-gripper experiment measures anything, and it is stated in
#: ``PAPER_PLAN.md``: the factors the claim is about -- gripper and object --
#: are fully crossed, and the nuisance factors -- where the object starts and
#: where it has to go -- are sampled *once* and the identical sample used
#: everywhere. Drawing a pick pose per cell confounds the gripper with the
#: difficulty of the pose it happened to draw, and no analysis afterwards
#: separates them.
#:
#: Three rather than two, because two cannot distinguish "works anywhere" from
#: "works at these two points". They span the reachable table region and differ
#: in yaw by roughly 55 degrees each, so a fixed-yaw grasp cannot serve all
#: three: ``P0`` is near and to the right at 0 degrees, ``P1`` central at 55,
#: ``P2`` far and to the left at 110.
#:
#: The x values keep every object clear of the shelf's front edge, which at real
#: size sits at ``x = 0.03``: the nearest pose is 170 mm in front of it, against
#: the ~120 mm at which a Panda's hand catches the underside of a board
#: (``ROBOTICS_NOTES.md`` 2.9).
REAL_PICK_CONFIGS = {
    "P0": (-0.14, -0.15, 0.0),
    "P1": (-0.21, 0.02, 55.0),
    "P2": (-0.28, 0.16, 110.0),
}

#: The destinations the campaign varies over. Both on the top level, because the
#: lower one is 142 mm clear and does not admit a real carton.
REAL_DESTINATIONS = ("top_middle", "top_left")

#: Where a YCB object is put on the table, as ``(x, y, yaw degrees)`` relative
#: to the table centre.
#:
#: **Chosen by search, not by hand, and the constraint that decides them is the
#: robot's own base.** Objects at the previous ``REAL_PICK_CONFIGS`` poses were
#: knocked backwards into ``fixed_mount0_pedestal_col`` -- traced on
#: ``robotiq3f/cereal/P2``, the box slides 4.6 mm per control step for 17 steps
#: and stops at 99.3 mm when it reaches the pedestal, which is 400 mm behind the
#: table centre. An object with the pedestal behind it cannot be nudged clear;
#: it jams.
#:
#: Two things had to be got right to search for better ones:
#:
#: * **Only geometry that reaches above the table can obstruct an object on
#:   it.** ``fixed_mount0_pedestal_feet_col`` reaches forward to x = -0.298 and
#:   sits at z = 0.009 to 0.319 -- on the floor. Taking it as the bound left a
#:   150 mm strip in which the 332.7 mm hammer does not fit at any yaw, and only
#:   72 poses with at most 10 mm of clearance. The geom that actually matters is
#:   ``fixed_mount0_pedestal_col`` at x = -0.400, which gives a 270 mm strip and
#:   1125 feasible poses.
#: * **The hand at its home pose is not an obstacle.** Including it put the
#:   bound at x = +0.005, which is the parked gripper, and excluded everything.
#:
#: Each pose is required to clear the pedestal by 40 mm, the shelf's front edge
#: by 120 mm (2.9: a Panda's hand catches the underside of a board within about
#: that), and the table edge -- for **every object in the set**, not on average.
#: **The footprint is measured in the scene, not in an isolated model.** A first
#: search rotated each object's collision vertices about its own origin in a
#: standalone model and produced poses that looked clear by 39 mm; built in the
#: scene, the hammer's rear edge sat at x = -0.483, which is 83 mm *behind* the
#: pedestal. The object origins are not the centres of their footprints -- the
#: hammer spans -121.8 to +150.2 mm about its own -- so the two frames disagree.
#: Measuring ``_geom_vertices`` relative to ``body_xpos`` in a built scene is
#: what makes the numbers mean anything.
#:
#: The three below are the best of their yaw band at 39.8, 45.8 and 37.1 mm of
#: worst-object clearance, chosen to differ in y as well as in yaw. Feasible
#: yaws run 45 to 130 degrees: outside that the hammer's long axis points across
#: the strip and no position clears both ends.
YCB_PICK_CONFIGS = {
    "P0": (-0.230, -0.140, 50.0),
    "P1": (-0.200, 0.020, 90.0),
    "P2": (-0.220, 0.160, 125.0),
}

#: Clearance every pick pose is required to leave, in metres.
PICK_BASE_CLEARANCE = 0.040
PICK_SHELF_CLEARANCE = 0.120

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
        world=BENCHMARK,
        pick_config=None,
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
        if world == YCB:
            from tpgpt.sim.ycb import YCB_ARTICLES

            known = set(YCB_ARTICLES)
        else:
            known = set(OBJECT_CLASSES)
        unknown = [name for name in objects if name not in known]
        if unknown:
            raise ValueError(
                f"unknown objects {unknown} for world {world!r}; "
                f"available: {sorted(known)}"
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
        if world not in WORLDS:
            raise ValueError(
                f"unknown world {world!r}; expected one of {WORLDS}"
            )
        if pick_config is not None:
            if world not in (REAL, YCB):
                raise ValueError(
                    "pick_config is only defined for the real-sized world; the "
                    "benchmark scene samples its placements"
                )
            if pick_config not in self.pick_configs_for(world):
                raise ValueError(
                    f"unknown pick configuration {pick_config!r}; expected one "
                    f"of {sorted(self.pick_configs_for(world))}"
                )
            if len(objects) != 1:
                raise ValueError(
                    "a pick configuration places one object; got "
                    f"{len(objects)}. One object per scene is the campaign "
                    "default -- a neighbouring object is a failure mode that "
                    "has nothing to do with the keypoints."
                )
        self.shelf_variant = shelf_variant
        self.world = world
        self.pick_config = pick_config
        # Every piece of geometry below is read from these, not from the module
        # constants, so the two worlds share one implementation.
        real = world in (REAL, YCB)
        self.shelf_levels = REAL_SHELF_LEVELS if real else SHELF_LEVELS
        self.shelf_slots = REAL_SHELF_SLOTS if real else SHELF_SLOTS
        self.shelf_board_depth = REAL_SHELF_BOARD_DEPTH if real else SHELF_BOARD_DEPTH
        self.shelf_board_width = REAL_SHELF_BOARD_WIDTH if real else SHELF_BOARD_WIDTH
        self.shelf_thickness = REAL_SHELF_THICKNESS if real else SHELF_THICKNESS
        self.cubby_height = REAL_CUBBY_HEIGHT if real else CUBBY_HEIGHT
        self.enclosed_height = REAL_ENCLOSED_HEIGHT if real else ENCLOSED_HEIGHT
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
        for level, x, height in self.shelf_levels:
            z = self.table_top + height + self.shelf_thickness / 2
            for slot, y in self.shelf_slots:
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
        for level, x, height in self.shelf_levels:
            body = ET.SubElement(
                arena.worldbody,
                "body",
                name=f"shelf_{level}",
                pos=f"{x} 0 {self.table_top + height}",
            )
            half_x, half_y = (self.shelf_board_depth[level] / 2,
                              self.shelf_board_width / 2)
            leg_half = height / 2
            panels = [
                ("board", "0 0 0", f"{half_x} {half_y} {self.shelf_thickness / 2}"),
                ("lip", f"{half_x - 0.008} 0 0.028", f"0.008 {half_y} 0.022"),
                ("leg_l", f"0 {half_y - 0.012} {-leg_half}", f"{half_x} 0.012 {leg_half}"),
                ("leg_r", f"0 {-(half_y - 0.012)} {-leg_half}", f"{half_x} 0.012 {leg_half}"),
            ]
            if self.shelf_variant != "open":
                interior = (
                    self.cubby_height if self.shelf_variant == "cubby"
                    else self.enclosed_height
                )[level]
                half_t = self.shelf_thickness / 2
                mid = self.shelf_thickness / 2 + interior / 2
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
                        ("roof", f"0 0 {self.shelf_thickness / 2 + interior}",
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

        # ``make_object`` builds either robosuite's shipped asset unchanged or
        # the same asset scaled to the real article and given its real mass.
        # It also seeds the objects that sample their own dimensions -- the
        # hammer draws its handle length from a range, so before this every
        # scene containing one built a *different hammer on every call*.
        self.objects = [
            make_object(name, self.world, rng=np.random.default_rng(0))
            for name in self.object_names
        ]
        # Objects stay clear of the shelf: the Panda's hand is deeper than its
        # fingers, so a grasp taken within ~0.12 m of a board's front edge
        # catches the hand on the underside of the board. At real size the
        # shelf's front edge moves in to x = 0.03 and the objects are up to
        # 330 mm long, so the region moves back with it.
        x_range = [-0.32, -0.12] if self.world in (REAL, YCB) else [-0.22, -0.04]
        y_range = [-0.20, 0.20] if self.world in (REAL, YCB) else [-0.18, 0.18]
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.objects,
            x_range=x_range,
            y_range=y_range,
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

    #: Where the fingertips start in the real-sized world.
    #:
    #: **The same point 100 mm higher, and the height is the whole change.**
    #: :data:`HOME_TCP` was chosen above the benchmark objects, the tallest of
    #: which is a 150 mm cereal carton topping out at z = 0.975. The real carton
    #: is 300 mm and tops out at **1.0997**, leaving the old home pose 50 mm of
    #: clearance -- enough for the fingertips, not enough for a hand whose
    #: unactuated fingers sag under gravity while the objects settle.
    #:
    #: Measured, that cost exactly one hand one cell: at the old home pose the
    #: **Robotiq 3F** moved the cereal carton **4.79 mm** at pick P1 within 250
    #: steps of reset, while the other six hands moved it 0.061 mm -- and it
    #: registered **no interpenetration at placement at all**, so the check that
    #: caught the original defect could not have caught this one. The 3F's own
    #: dexterous variant was unaffected, which is what points at the hand's pose
    #: rather than its model: the two share a MuJoCo model and differ by a
    #: 25.6 mm contact offset, so IK puts their wrists at different heights.
    #:
    #: At 1.25 -- 150 mm above the tallest object -- all three hands tried move
    #: it 0.061 mm at all three picks, and so do 1.30 and 1.35, so the choice is
    #: not perched on a threshold. The x is left alone so nothing else about the
    #: start pose moves.
    REAL_HOME_TCP = np.array([-0.10, 0.0, 1.25])

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
        """Put the fingertips at the world's home pose, whatever hand is mounted."""
        from tpgpt.grasp.grasps import contact_offset
        from tpgpt.sim.kinematics import solve_ik

        home = self.REAL_HOME_TCP if self.world in (REAL, YCB) else self.HOME_TCP
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
        wrist = home - self.HOME_ROTATION @ offset
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
        """World-frame points bounding every geom on a body.

        **A mesh geom is measured by its vertices, not by its bounding box**,
        and the difference decides whether an object is seated or dropped. A
        mesh's ``geom_aabb`` is a local axis-aligned box, so for a convex
        decomposition the union of the parts' boxes is far larger than the
        object: the YCB hammer's 5 parts read 251.6 x 235.7 x **132.3** mm
        against a true 332.7 x 182.2 x **32.9**. Seating from that put the
        hammer about 50 mm above the table, and it then fell, bounced and
        rolled -- measured tilting 89.7 degrees at one pick pose and coming to
        rest 91 mm behind the robot base's front face. Dropped on a bare plane
        the same object settles within 0.26 degrees of its scan orientation, so
        the instability was the seating, not the object.

        Primitive geoms keep the box, which for a box, a cylinder or a sphere
        is exact.

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
        import mujoco

        model, data = self.sim.model, self.sim.data
        out = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body_id:
                continue
            rot = np.array(data.geom_xmat[g]).reshape(3, 3)
            origin = np.array(data.geom_xpos[g])
            mesh = int(model.geom_dataid[g])
            if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH) and mesh >= 0:
                start = int(model.mesh_vertadr[mesh])
                count = int(model.mesh_vertnum[mesh])
                local = np.array(model.mesh_vert[start:start + count])
            else:
                centre = np.array(model.geom_aabb[g][:3])
                half = np.array(model.geom_aabb[g][3:])
                local = np.array([[x, y, z] for x in (-half[0], half[0])
                                  for y in (-half[1], half[1])
                                  for z in (-half[2], half[2])]) + centre
            out.append(local @ rot.T + origin)
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
            # **Everything the robot is made of, not a name prefix.** This used
            # to test ``name.startswith(("robot", "gripper"))``, which silently
            # excluded the robot's own base: its geoms are named
            # ``fixed_mount0_...``. The pedestal is what a disturbed object
            # actually jams against -- measured on ``robotiq3f/cereal``, the box
            # slides 99.3 mm and stops on ``fixed_mount0_pedestal_col`` -- so a
            # check that cannot see it is checking the wrong thing.
            if name.startswith(("robot", "gripper", "fixed_mount")):
                worst = max(worst, -float(contact.dist))
        return worst

    #: Where recorded rest poses live, relative to the repository root.
    #:
    #: A file rather than a computation at reset, and read-only from the scene,
    #: because the whole point is that every gripper sees the **same** bytes.
    #: A scene that quietly settled and cached its own answer would record
    #: whichever hand happened to build the scene first, and the block would
    #: silently become that hand's scene.
    SETTLED_STATES = "configs/settled_states.json"

    @staticmethod
    def pick_configs_for(world: str) -> dict:
        """The written-down pick poses for ``world``."""
        return YCB_PICK_CONFIGS if world == YCB else REAL_PICK_CONFIGS

    def _object_by_name(self, name):
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(name)

    def _state_key(self) -> str:
        """Identifies a settled scene: world, shelf, objects and pick pose."""
        return "|".join([
            self.world, self.shelf_variant, ",".join(self.object_names),
            self.pick_config or "sampled",
        ])

    def _recorded_settled_state(self):
        """Rest poses recorded for this exact scene, or ``None``.

        ``None`` is the benchmark scene's answer and always will be: the file
        only ever holds real-world scenes with an explicit pick configuration,
        because a sampled placement depends on the seed and is not a condition
        anything is blocked on.
        """
        if self.pick_config is None:
            return None
        from pathlib import Path
        import json

        path = Path(__file__).resolve().parents[3] / self.SETTLED_STATES
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())[self._state_key()]
        except KeyError:
            return None

    def _initial_poses(self):
        """Pose every object starts at, before settling.

        Two sources, and which one is used is the difference between a sampled
        scene and a condition. With no ``pick_config`` the placement sampler
        draws from its range, which is what every benchmark campaign has done.
        With one, the object is placed at a **written-down pose reused by every
        cell** -- see :data:`REAL_PICK_CONFIGS` for why that is the rule and not
        a convenience.

        The height comes from the object's own ``bottom_offset`` exactly as the
        sampler computes it, so a placed object falls the same small distance a
        sampled one does and nothing else about the scene changes.
        """
        if self.pick_config is None:
            return {obj.name: (pos, quat)
                    for pos, quat, obj in self.placement_initializer.sample().values()}
        from robosuite.utils.transform_utils import quat_multiply

        x, y, yaw = self.pick_configs_for(self.world)[self.pick_config]
        obj = self.objects[0]
        # Generous: the exact height is set by ``_seat_objects`` from the
        # object's own geometry in whatever orientation it ends up in, which
        # ``bottom_offset`` cannot give once a rest rotation is composed in.
        z = float(self.table_offset[2]) + 0.5
        half = np.deg2rad(yaw) / 2.0
        # robosuite's ``quat_multiply`` works in ``(x, y, z, w)``; the scene
        # stores ``(w, x, y, z)``. Rest first, then yaw about the world z.
        def to_xyzw(q):
            return np.array([q[1], q[2], q[3], q[0]])

        def to_wxyz(q):
            return np.array([q[3], q[0], q[1], q[2]])

        yaw_q = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
        quat = to_wxyz(quat_multiply(yaw_q, to_xyzw(rest_quat(obj.name, self.world))))
        return {obj.name: (
            (float(self.table_offset[0]) + x, float(self.table_offset[1]) + y, z),
            quat,
        )}

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
            for name, (pos, quat) in self._initial_poses().items():
                self.sim.data.set_joint_qpos(
                    self._object_by_name(name).joints[0],
                    np.concatenate([np.array(pos), np.array(quat)]),
                )
            self.sim.forward()
            # Recorded, not acted on. Seating the objects on the table was tried
            # and made things worse -- see 7.32 -- so the scene still drops them
            # ~25 mm and this says how bad the starting state is. It reads 0.00
            # for every registered hand now that the arm starts clear.
            if self.world in (REAL, YCB):
                # **Seat the object, do not drop it.** The sampler places an
                # object at ``table_z + z_offset + |bottom_offset|`` and every
                # one of these assets declares a ``bottom_offset`` about 25 mm
                # larger than its true half height, so ``z_offset = 2 mm``
                # actually drops it 25 and it lands at roughly 0.7 m/s (7.32).
                # A 150 mm benchmark carton survives that by toppling; a 300 mm
                # real one does not, and a 333 mm hammer laid on its side
                # bounces off the table entirely -- measured resting at
                # z = 0.488 against a table surface at 0.800.
                #
                # Seating also makes the rest rotations usable at all:
                # ``bottom_offset`` is quoted in the asset's own upright frame
                # and says nothing about how far below the origin a hammer lying
                # on its side reaches. ``_seat_objects`` reads that from the
                # geometry as placed.
                self._seat_objects()
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
            self.settled_state_restored = False
            recorded = self._recorded_settled_state()
            if recorded is not None:
                # **The block is exact, not nominal.** Settling *in this scene*
                # would settle against this gripper's model, and the same seed
                # does not settle identically across hands -- a can comes to
                # rest 14.6 mm apart between a Panda and a Robotiq 2F-140
                # (ROBOTICS_NOTES 7.28), because a different hand is a different
                # MuJoCo model and the constraint solver's arithmetic differs.
                # That 15 mm travels through every cross-gripper comparison as a
                # confound. Restoring one recorded rest pose makes "the same
                # condition" mean the same scene to machine precision, which is
                # what a randomised block design requires.
                for name, qpos in recorded.items():
                    self.sim.data.set_joint_qpos(
                        self._object_by_name(name).joints[0], np.asarray(qpos))
                    self.sim.data.set_joint_qvel(
                        self._object_by_name(name).joints[0], np.zeros(6))
                self.sim.forward()
                self.settled_state_restored = True
                self.sim.data.qfrc_applied[:] = 0.0
                self._move_arm_home()
                self.sim.forward()
                return
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

    #: How far above or below the board an object may rest and still count as
    #: being on it, in metres.
    #:
    #: Measured against the object's **lowest point**, so it is the gap under
    #: the object rather than anything about its size. 30 mm admits an object
    #: that has come to rest slightly proud -- on the lip of another, or on a
    #: crumb of solver jitter -- and refuses one balanced on top of something
    #: else.
    SLOT_HEIGHT_TOLERANCE = 0.03

    def is_object_in_slot(self, name: str, slot: str, tolerance: float = 0.06) -> bool:
        """Whether an object is resting in a named slot.

        **The vertical test is on the object's base, not on its origin, and the
        difference is not a refinement.** This used to ask whether the object's
        *body origin* was within 120 mm of the board. A body origin sits half
        the object's height above whatever it rests on, so that test asked how
        tall the object was: a 300 mm cereal carton standing perfectly in its
        slot has its origin 150 mm up and **could never be scored a success**,
        while a 230 mm milk carton passed by 5 mm and would flip on a
        millimetre of settling. Measured in the real-sized scene, a cell whose
        object came to rest **5 mm** from the centre of its slot was scored
        ``placed_in_the_wrong_place``.

        It survived in the benchmark scene because nothing there is tall enough
        to trip it -- the largest object's origin sits 75 mm up, comfortably
        inside 120 -- so the vertical clause was satisfied by every run and the
        verdict came from the horizontal test alone. The same defect is
        recorded against ``stage_outcome``'s ``placed_on_shelf`` flag in
        ``docs/dynamics_execution.md`` section 12, where it disagreed with the
        scored outcome on 8 of 28 cells; what was missed is that it was in the
        **scored outcome** as well, not only in the diagnostic.

        Taking the base makes the test say what it is for: the object is over
        the slot, and it is resting on the board rather than hovering above it
        or perched on something else. It is also size-independent, which is the
        property a cross-object comparison needs.
        """
        poses = self.slot_poses()
        if slot not in poses:
            raise KeyError(f"unknown slot {slot!r}; scene has {sorted(poses)}")
        offset = self.object_position(name) - poses[slot]
        vertices = self._geom_vertices(self.object_body_ids[name])
        if not len(vertices):  # pragma: no cover - an object with no geoms
            return False
        gap = float(vertices[:, 2].min()) - float(poses[slot][2])
        return bool(np.linalg.norm(offset[:2]) < tolerance
                    and abs(gap) < self.SLOT_HEIGHT_TOLERANCE)

    def reward(self, action=None):
        """Placeholder: this scene is for perception and grasping, not a task.

        Task success is defined by the caller, since what counts depends on the
        prompt that was parsed.
        """
        return 0.0
