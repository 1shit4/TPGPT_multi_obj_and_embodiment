"""Bring a GraspGen-X gripper into robosuite, so MuJoCo can grasp with it.

Why this direction
------------------
Every other route in `docs/gripper_configs.md` goes robosuite -> GraspGen-X:
take a hand the simulator already has and write it a description. That route is
exhausted. The robosuite hands not already paired are almost all anthropomorphic
(`ability`, `schunk`, `fourier`, `inspire`), GraspGen-X was trained only on
two- and three-finger families, and a five-finger hand has no single closing
axis to describe -- section 13 measures that, and section 19 confirms it in
physics with **0 of 5** on a curated description. So the supply of hands on that
side is empty.

This module goes the other way. GraspGen-X ships **26** hands, each with a URDF
and a `config.json` its own authors curated -- and section 16 measures curated
descriptions at 39% against authored at 37%, so the half that already works is
the half these hands arrive with. Fourteen of the fifteen this project does not
already have compile in MuJoCo unchanged, four of them three-finger. What is
missing is only the robosuite side: actuators, a `grip_site`, and the body
naming robosuite mounts against.

What the config supplies, and why almost nothing has to be invented
------------------------------------------------------------------
The same file that conditions the planner also answers the modelling questions:

* ``open`` and ``close`` give each driven joint's two end states, which become
  the actuator range **and** the rest pose. No guessing what "closed" means.
* ``fingertip`` gives the depth of the tool centre along the approach axis,
  which is where robosuite's ``eef`` body and its ``grip_site`` go.
* ``type`` gives the kinematic family, and ``base_rotation`` -- when a hand
  carries one -- the rotation into the canonical frame.

Three joint roles, read from the config rather than assumed
-----------------------------------------------------------
* **driven**: ``open[j] != close[j]``. Gets a position actuator.
* **locked**: named in the config with ``open[j] == close[j]``. Held there by an
  equality constraint. The Barrett hand's ``bh_j11_joint`` is a finger *spread*
  and sits at 0 in both states; actuating it would splay the hand every time the
  jaws were told to close.
* **coupled**: present in the URDF and absent from the config. These are the
  distal links a real hand drives through a mechanical linkage, and left free
  they flop under contact. Each is tied to the driven joint above it in the
  body tree, with the ratio taken from the two joints' own limits -- the
  Barrett's distal range is 0.84 against its proximal 2.44, which is the 1/3
  coupling its hardware actually has.

The resulting hand is **position-commanded and one-DOF**: one number in, every
actuator interpolated between the open and closed poses the config declares. In
this project's terms that makes it a *passthrough* hand -- see
``grippers.commands_position`` -- so a fraction is ``2f - 1`` and `set_closure`
does not apply.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import xml.etree.ElementTree as ET

import numpy as np

#: Where GraspGen-X's curated hands live.
CURATED = pathlib.Path(
    "/home/ishita/task_embod_aware_grasp/6dof_GraspMAS/assets/"
    "gripper_descriptions/gripper_descriptions/assets/x_grippers"
)

#: Where ported robosuite models are written.
PORTED = pathlib.Path(__file__).resolve().parents[2] / "assets" / "ported_grippers"

#: Actuator gain. robosuite's own hands use 1000 for a position servo on a
#: finger joint; matching it keeps contact behaviour comparable rather than
#: making a ported hand stiffer or softer than the registry's.
KP = 1000.0

#: Actuator force limit, N or Nm. robosuite's Panda uses 20.
FORCE = 20.0


def _tree_parent(root, joint_to_body, body_parent, joint):
    """The nearest driven joint above ``joint`` in the body tree."""
    body = joint_to_body.get(joint)
    while body is not None:
        body = body_parent.get(body)
        for other, owner in joint_to_body.items():
            if owner == body:
                return other
    return None


def convert(name: str, out_root: pathlib.Path = PORTED) -> dict:
    """Write a robosuite gripper XML for the GraspGen-X hand ``name``.

    Returns a summary dict: the joint roles, the actuator list, the tool depth
    and the path written. Nothing here is specific to one hand -- everything
    that varies is read from ``config.json`` or from the URDF's own joint
    limits.
    """
    import mujoco

    src = CURATED / name
    cfg = json.loads((src / "config.json").read_text())
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    # 1. MuJoCo converts the URDF. Fixed links are merged into their parent, so
    #    a palm with no joint of its own arrives as loose geoms on worldbody.
    model = mujoco.MjModel.from_xml_path(str(src / "gripper.urdf"))
    raw = out / f"_{name}_mujoco.xml"
    mujoco.mj_saveLastXML(str(raw), model)
    tree = ET.parse(raw)
    root = tree.getroot()

    open_js, close_js = cfg["open"], cfg["close"]
    urdf_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(model.njnt)]
    limits = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
              tuple(float(v) for v in model.jnt_range[i]) for i in range(model.njnt)}

    driven = [j for j in urdf_joints
              if j in close_js and abs(close_js[j] - open_js.get(j, 0.0)) > 1e-6]
    locked = {j: float(open_js[j]) for j in urdf_joints
              if j in close_js and abs(close_js[j] - open_js.get(j, 0.0)) <= 1e-6}
    coupled = [j for j in urdf_joints if j not in close_js]

    # 2. Body-tree bookkeeping, so a coupled joint can find its driver.
    joint_to_body, body_parent = {}, {}

    def index(element, parent=None):
        for body in element.findall("body"):
            body_parent[body.get("name")] = parent
            for joint in body.findall("joint"):
                joint_to_body[joint.get("name")] = body.get("name")
            index(body, body.get("name"))

    worldbody = root.find("worldbody")
    index(worldbody)

    # 3. Wrap everything in the body robosuite mounts against, and hang the
    #    tool frame off it at the depth the description declares.
    gripper = ET.Element("body", {"name": "right_gripper", "pos": "0 0 0"})
    ET.SubElement(gripper, "site", {
        "name": "ft_frame", "pos": "0 0 0", "size": "0.01 0.01 0.01",
        "rgba": "1 0 0 1", "type": "sphere", "group": "1"})
    ET.SubElement(gripper, "inertial", {
        "pos": "0 0 0.05", "mass": "0.3", "diaginertia": "0.09 0.07 0.05"})

    depth = float(cfg["fingertip"][-1])
    eef = ET.SubElement(gripper, "body", {"name": "eef", "pos": f"0 0 {depth}",
                                          "quat": "1 0 0 0"})
    ET.SubElement(eef, "site", {"name": "grip_site", "pos": "0 0 0",
                                "size": "0.01 0.01 0.01", "rgba": "1 0 0 0.5",
                                "type": "sphere", "group": "1"})
    for axis, quat, rgba in (("x", "0.707105 0 0.707108 0", "1 0 0 0"),
                             ("y", "0.707105 0.707108 0 0", "0 1 0 0"),
                             ("z", "1 0 0 0", "0 0 1 0")):
        pos = {"x": "0.1 0 0", "y": "0 0.1 0", "z": "0 0 0.1"}[axis]
        ET.SubElement(eef, "site", {"name": f"ee_{axis}", "pos": pos,
                                    "size": "0.005 .1", "quat": quat,
                                    "rgba": rgba, "type": "cylinder", "group": "1"})
    ET.SubElement(eef, "site", {"name": "grip_site_cylinder", "pos": "0 0 0",
                                "size": "0.005 10", "rgba": "0 1 0 0.3",
                                "type": "cylinder", "group": "1"})

    # 4. Move the hand's own geometry inside, naming every geom. robosuite
    #    identifies contacts by name, and MuJoCo's URDF conversion leaves them
    #    anonymous -- an unnamed geom cannot be reported as touching anything.
    n_geom = 0
    for geom in list(worldbody.findall("geom")):
        worldbody.remove(geom)
        geom.set("name", f"palm_collision_{n_geom}")
        geom.set("group", "0")
        geom.set("condim", "4")
        geom.set("friction", "1 0.005 0.0001")
        gripper.append(geom)
        n_geom += 1
    for body in list(worldbody.findall("body")):
        worldbody.remove(body)
        gripper.append(body)

    def name_geoms(element, stem):
        nonlocal n_geom
        for body in element.findall("body"):
            for i, geom in enumerate(body.findall("geom")):
                if geom.get("name") is None:
                    geom.set("name", f"{body.get('name')}_collision_{i}")
                geom.set("group", "0")
                geom.set("condim", "4")
                # Fingers need grip. robosuite's own pads use friction 2.
                geom.set("friction", "2 0.05 0.0001")
                geom.set("solref", "0.01 0.5")
                n_geom += 1
            name_geoms(body, stem)

    name_geoms(gripper, name)
    worldbody.append(gripper)

    # 5. Damping on every joint. A URDF converted straight across has none, and
    #    an undamped finger oscillates on contact instead of holding.
    for joint_name in urdf_joints:
        body_name = joint_to_body.get(joint_name)
        for body in gripper.iter("body"):
            if body.get("name") != body_name:
                continue
            for joint in body.findall("joint"):
                if joint.get("name") == joint_name:
                    joint.set("damping", "1.0")
                    joint.set("armature", "0.1")
                    joint.set("frictionloss", "0.1")

    # 6. Position actuators on the driven joints only.
    actuator = ET.SubElement(root, "actuator")
    for joint_name in driven:
        lo, hi = sorted((float(open_js[joint_name]), float(close_js[joint_name])))
        ET.SubElement(actuator, "position", {
            "name": f"gripper_{joint_name}", "joint": joint_name,
            "ctrllimited": "true", "ctrlrange": f"{lo} {hi}", "kp": str(KP),
            "forcelimited": "true", "forcerange": f"{-FORCE} {FORCE}"})

    # 7. Equality constraints: lock what the config holds still, couple what it
    #    does not mention to the driven joint above it.
    equality = ET.SubElement(root, "equality")
    for joint_name, value in locked.items():
        ET.SubElement(equality, "joint", {
            "joint1": joint_name, "polycoef": f"{value} 0 0 0 0"})
    couplings = {}
    for joint_name in coupled:
        driver = _tree_parent(root, joint_to_body, body_parent, joint_name)
        while driver is not None and driver not in driven:
            driver = _tree_parent(root, joint_to_body, body_parent, driver)
        if driver is None:
            ET.SubElement(equality, "joint", {
                "joint1": joint_name, "polycoef": "0 0 0 0 0"})
            couplings[joint_name] = None
            continue
        span_c = limits[joint_name][1] - limits[joint_name][0]
        span_d = limits[driver][1] - limits[driver][0]
        ratio = (span_c / span_d) if span_d else 1.0
        ET.SubElement(equality, "joint", {
            "joint1": joint_name, "joint2": driver,
            "polycoef": f"0 {ratio:.6f} 0 0 0"})
        couplings[joint_name] = (driver, round(ratio, 4))

    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "force", {"name": "force_ee", "site": "ft_frame"})
    ET.SubElement(sensor, "torque", {"name": "torque_ee", "site": "ft_frame"})

    root.set("model", name)
    # The meshes have to sit next to the XML. robosuite's ``MujocoXML`` rewrites
    # every mesh path to an absolute one **against the XML's own folder**
    # (`models/base.py:54`) and ignores the compiler's ``meshdir``, so pointing
    # that at the source checkout loads fine in bare MuJoCo and then fails the
    # moment robosuite merges the hand into a robot.
    meshes = src / "meshes"
    if meshes.is_dir():
        shutil.copytree(meshes, out / "meshes", dirs_exist_ok=True)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.set("strippath", "false")

    path = out / f"{name}.xml"
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=False)
    raw.unlink(missing_ok=True)

    return {"name": name, "path": str(path), "type": cfg["type"],
            "tool_depth_mm": depth * 1000, "driven": driven,
            "locked": list(locked), "coupled": couplings,
            "open": [float(open_js[j]) for j in driven],
            "close": [float(close_js[j]) for j in driven],
            "aperture_mm": float(cfg["sweep_volume"]["extents"][0]) * 1000}
