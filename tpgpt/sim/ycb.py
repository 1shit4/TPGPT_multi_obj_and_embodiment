"""Scene objects built from the YCB Object and Model Set's own scans.

Sizing the scene by hand, as :mod:`tpgpt.sim.objects` does, gets the objects to
plausible real dimensions but leaves three things unsatisfying. The numbers are
mine rather than a citable standard; the *shapes* are still robosuite's
benchmark meshes, so a "cereal box" is a scaled 30 mm-thin carton; and nothing
outside this project can reproduce the object set. The YCB set fixes all three:
it is the standard manipulation benchmark, every object is a laser-and-RGBD scan
of a real article, and each one is published with a measured mass.

**The scans are the authority on geometry, not the published dimension table.**
Measured from ``nontextured.stl``, the scans run a few millimetres larger than
the caliper column (a cracker box measures 71.8 x 164.0 x 213.4 against a
published 60 x 158 x 210) because a scan includes the bulge of a real cardboard
box. For one object the difference is not millimetres: the hammer's published
row reads ``34 x 32 x 135``, which is its **head**, while the scan is
**332.7 mm** long -- a 135 mm hammer weighing 665 g would be implausibly dense.
Any sizing decision here therefore comes from the mesh.

Collision geometry, and why it cannot be the raw mesh
-----------------------------------------------------

MuJoCo collides a mesh as its **convex hull**. For the boxes and cans that is
faithful. For the two objects in this set that earn their place it is ruinous:

* the **mug**'s hull fills both the cup's cavity and the gap inside its handle,
  turning the one object whose graspable features are a 5 mm wall and a handle
  into an 81 mm solid lump that the narrow hands cannot hold at all;
* the **hammer**'s hull bridges head to handle into a solid wedge, so the 33 mm
  handle -- the entire reason a hammer is in the set -- stops existing.

So each scan is decomposed into convex parts with CoACD and mounted as one geom
per part. The visual geom stays the original scan, so what is rendered and what
is point-sampled are the real article.
"""

from __future__ import annotations

import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Where the downloaded YCB scans live, relative to the repository root.
YCB_ROOT = Path("assets/ycb")

#: Where the generated MJCF and the convex parts are written.
BUILD_ROOT = Path("assets/ycb_mjcf")


@dataclass(frozen=True)
class YcbArticle:
    """One YCB object, and what this project calls it.

    Args:
        ycb: The set's own directory name, which is also its citation.
        mass_kg: The published measured mass. Taken from YCB's object list
            rather than from the mesh, because a scan has no density.
        note: Why this object is in the set, and anything the scan does not
            capture.
    """

    ycb: str
    mass_kg: float
    note: str = ""


#: Every YCB object downloaded for this project, keyed by the short name the
#: scene and the language layer use.
#:
#: The **narrowest** extent of each scan decides which hands can hold it, and
#: the narrowest jaw in the fleet is the Rethink's at 66 mm. Measured on the
#: scans, that rules out the obvious grocery choices outright: a master chef can
#: is 102.4 mm across, a cracker box 71.8, a tomato soup can 67.7, a mustard
#: bottle 66.6 and a bleach cleanser 67.7. Keeping any of them would mean one
#: hand could not attempt one column of the grid, and a cross-gripper claim
#: cannot be made on a grid with holes in it.
YCB_ARTICLES = {
    "sugar": YcbArticle("004_sugar_box", 0.514,
                        "A 176 mm tall box, 49.5 mm across its narrowest face "
                        "and the heaviest object every hand can span. Takes the "
                        "role the milk carton had: tall, heavy, flat-faced."),
    "meat": YcbArticle("010_potted_meat_can", 0.370,
                       "102 x 83.5 x 60.1 mm. A squat rectangular can -- the "
                       "narrowest face is 60.1 mm, which clears the 66 mm jaw "
                       "by 5.9 mm."),
    "mug": YcbArticle("025_mug", 0.118,
                      "The only object here that is not a solid body. Its "
                      "graspable features are a ~5 mm wall and a handle, and "
                      "its 81.3 mm body is wider than three of the seven jaws "
                      "-- so it is held by a feature rather than by its "
                      "silhouette, which is exactly what convex decomposition "
                      "has to preserve."),
    "hammer": YcbArticle("048_hammer", 0.665,
                         "332.7 mm long with its mass in the head and its grip "
                         "on a 32.9 mm handle, so a grasp carries a moment. "
                         "The published dimension row for this object is its "
                         "head, not the tool."),
    "banana": YcbArticle("011_banana", 0.066,
                         "Curved, with no flat face and no axis of symmetry -- "
                         "the shape the rest of the set does not cover."),
    # Downloaded and measured, kept out of the default set. Each is recorded
    # with the reason so the choice can be checked rather than taken on trust.
    "cracker": YcbArticle("003_cracker_box", 0.411,
                          "71.8 mm narrowest: too wide for the Rethink's 66 mm."),
    "soup": YcbArticle("005_tomato_soup_can", 0.349,
                       "67.7 mm diameter: 1.7 mm too wide for the Rethink."),
    "mustard": YcbArticle("006_mustard_bottle", 0.603,
                          "66.6 mm narrowest: 0.6 mm too wide for the Rethink."),
    "bleach": YcbArticle("021_bleach_cleanser", 1.131,
                         "67.7 mm narrowest: too wide for the Rethink."),
    "masterchef": YcbArticle("002_master_chef_can", 0.414,
                             "102.4 mm diameter: too wide for four of seven."),
    "tuna": YcbArticle("007_tuna_fish_can", 0.171, "A flat disc, 33.5 mm tall."),
    "pudding": YcbArticle("008_pudding_box", 0.187, "38.9 mm narrowest."),
    "gelatin": YcbArticle("009_gelatin_box", 0.097, "30.1 mm narrowest."),
    "brick": YcbArticle("061_foam_brick", 0.028, "51.2 mm narrowest, very light."),
}

#: The five the campaign runs on. Every one is graspable by every hand in the
#: fleet, which ``tpgpt.experiments.graspability`` verifies rather than assumes.
DEFAULT_YCB_OBJECTS = ("sugar", "meat", "mug", "hammer", "banana")


def scan_path(name: str, filename: str = "nontextured.stl") -> Path:
    return YCB_ROOT / YCB_ARTICLES[name].ycb / "google_16k" / filename


def read_stl(path: Path) -> np.ndarray:
    """Vertices of a binary STL, as ``(3 * ntri, 3)``.

    Written out rather than pulled from a mesh library because it is fifteen
    lines and the project's dependency policy is deliberately narrow.
    """
    with open(path, "rb") as handle:
        header = handle.read(84)
        count = struct.unpack("<I", header[80:84])[0]
        raw = np.frombuffer(handle.read(count * 50), dtype=np.uint8).reshape(count, 50)
    return raw[:, 12:48].copy().view(np.float32).reshape(count, 3, 3).reshape(-1, 3)


def read_stl_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)`` of a binary STL, with duplicate vertices merged."""
    flat = read_stl(path)
    vertices, inverse = np.unique(np.round(flat, 7), axis=0, return_inverse=True)
    return vertices.astype(np.float64), inverse.reshape(-1, 3)


def extents_mm(name: str) -> np.ndarray:
    """The scan's bounding extents in millimetres, largest first."""
    points = read_stl(scan_path(name))
    return np.sort((points.max(axis=0) - points.min(axis=0)) * 1000.0)[::-1]


def _centred(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift a scan so its bounding box is centred on the body origin.

    The YCB scans are expressed about the scanner's own origin, which for some
    objects sits well off the object. robosuite's placement, this project's
    seating and every keypoint construction assume the body origin is somewhere
    sensible inside the object, so the shift is applied once here and recorded
    so the point cloud can be shifted identically.
    """
    centre = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    return vertices - centre, centre


def decompose(name: str, threshold: float = 0.05, force: bool = False) -> list[Path]:
    """Convex parts of ``name``'s scan, written as STL files.

    Cached on disk keyed by the scan's hash and the threshold, because CoACD
    takes seconds to minutes per object and the result is deterministic.

    ``threshold`` is CoACD's concavity tolerance: lower means more parts and a
    closer fit. 0.05 keeps the mug's cavity and handle and the hammer's claw
    while leaving the boxes as one or two parts.
    """
    import coacd

    source = scan_path(name)
    vertices, faces = read_stl_faces(source)
    vertices, centre = _centred(vertices)
    digest = hashlib.sha1(
        source.read_bytes() + f"{threshold}".encode()).hexdigest()[:12]
    out = BUILD_ROOT / name
    stamp = out / f"parts-{digest}.json"
    if stamp.exists() and not force:
        return [out / p for p in json.loads(stamp.read_text())["parts"]]

    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("part_*.stl"):
        stale.unlink()
    for stale in out.glob("parts-*.json"):
        stale.unlink()
    parts = coacd.run_coacd(coacd.Mesh(vertices, faces), threshold=threshold)
    names = []
    for index, (part_v, part_f) in enumerate(parts):
        path = out / f"part_{index:02d}.stl"
        _write_stl(path, np.asarray(part_v), np.asarray(part_f))
        names.append(path.name)
    stamp.write_text(json.dumps(
        {"parts": names, "threshold": threshold,
         "centre_offset": centre.tolist()}, indent=1) + "\n")
    return [out / n for n in names]


def _write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.where(norms > 0, norms, 1.0))
    with open(path, "wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(faces)))
        for normal, triangle in zip(normals, triangles):
            handle.write(struct.pack("<3f", *normal.astype(np.float32)))
            for vertex in triangle:
                handle.write(struct.pack("<3f", *vertex.astype(np.float32)))
            handle.write(b"\0\0")


#: Contact parameters: **MuJoCo's defaults, not robosuite's grocery settings.**
#:
#: robosuite's own grocery objects declare ``solref="0.001 1"`` with
#: ``solimp="0.998 0.998 0.001"`` -- a 1 ms contact time constant, very stiff.
#: That is fine for a single convex mesh and it is not fine for a convex
#: *decomposition* resting on its own slightly non-planar underside: the five
#: or six simultaneous contacts fight each other and the object enters a limit
#: cycle instead of coming to rest. Measured on the sugar box, which sits on a
#: two-part bottom: it rocks through 0.05 to 0.80 degrees of tilt indefinitely,
#: still moving 0.9 mm per 50 steps after 2000, so the scene never hands over a
#: settled pose at all.
#:
#: Softening to MuJoCo's defaults settles all three of the awkward objects:
#:
#: ===============================  ==========  ==========  =========
#: contact                          sugar       hammer      mug
#: ===============================  ==========  ==========  =========
#: robosuite groceries (stiff)      never       450         250
#: **MuJoCo defaults**              **1100**    **400**     **300**
#: intermediate                     never       700         250
#: ===============================  ==========  ==========  =========
#:
#: with residual motion over the last 400 steps falling from 0.73 mm to 0.32 on
#: the sugar box and from 0.37 to 0.13 on the hammer.
SOLIMP = "0.9 0.95 0.001"
SOLREF = "0.02 1"
FRICTION = "0.95 0.3 0.1"


def write_mjcf(name: str, threshold: float = 0.05, force: bool = False) -> Path:
    """Write a robosuite-loadable MJCF for ``name`` and return its path.

    The body carries **one collision geom per convex part** and a single visual
    geom holding the original scan, so the hand feels the decomposition and the
    cameras and the point sampler see the real article.

    Mass is set on the body rather than through a density, because YCB
    publishes a measured mass and a scan has no density to derive one from.
    Distributing it across the parts by volume would also be wrong for the mug
    and the hammer, whose mass is emphatically not uniform -- so the body's
    total is set and MuJoCo distributes the inertia over the parts it was given.
    """
    parts = decompose(name, threshold=threshold, force=force)
    stamp = json.loads(next((BUILD_ROOT / name).glob("parts-*.json")).read_text())
    centre = np.asarray(stamp["centre_offset"], dtype=float)
    article = YCB_ARTICLES[name]

    root = ET.Element("mujoco", model=name)
    asset = ET.SubElement(root, "asset")
    for index, part in enumerate(parts):
        ET.SubElement(asset, "mesh", name=f"{name}_part{index}",
                      file=str(Path("..") / "ycb_mjcf" / name / part.name))
    ET.SubElement(asset, "mesh", name=f"{name}_visual",
                  file=str(Path("..") / "ycb_mjcf" / name / "visual.stl"))

    # The visual scan, shifted by the same offset the parts were.
    visual_out = BUILD_ROOT / name / "visual.stl"
    if not visual_out.exists() or force:
        vertices, faces = read_stl_faces(scan_path(name))
        _write_stl(visual_out, vertices - centre, faces)

    world = ET.SubElement(root, "worldbody")
    outer = ET.SubElement(world, "body")
    body = ET.SubElement(outer, "body", name="object")
    for index in range(len(parts)):
        # **Unnamed, like robosuite's own object XMLs.** robosuite prefixes a
        # named geom with the object's name when it registers it but leaves the
        # XML alone, so a geom called ``sugar_visual_g`` is looked up as
        # ``sugar_sugar_visual_g`` and the scene fails to build. Its shipped
        # assets name no geoms at all and are auto-named ``<object>_g0``.
        ET.SubElement(
            body, "geom", type="mesh",
            mesh=f"{name}_part{index}", group="0", condim="4",
            solimp=SOLIMP, solref=SOLREF, friction=FRICTION,
            rgba="0.7 0.6 0.5 1",
        )
    ET.SubElement(
        body, "geom", type="mesh",
        mesh=f"{name}_visual", group="1", contype="0", conaffinity="0",
        mass="0", rgba="0.78 0.72 0.6 1",
    )

    points = read_stl(scan_path(name)) - centre
    low, high = points.min(axis=0), points.max(axis=0)
    radius = float(np.hypot(*np.abs([low[:2], high[:2]]).max(axis=0)))
    for site, pos in (("bottom_site", (0, 0, low[2])),
                      ("top_site", (0, 0, high[2])),
                      ("horizontal_radius_site", (radius, radius, 0))):
        ET.SubElement(outer, "site", name=site, rgba="0 0 0 0", size="0.005",
                      pos=" ".join(f"{v:.6f}" for v in pos))

    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    path = BUILD_ROOT / f"{name}.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def make_ycb_object(name: str, threshold: float = 0.05):
    """A robosuite ``MujocoXMLObject`` for YCB object ``name``.

    Its mass is set to YCB's published measurement after the model compiles,
    the same two-pass solve :mod:`tpgpt.sim.objects` uses, because MuJoCo
    computes mass from density and mesh volume at compile time and neither is
    known beforehand.
    """
    from robosuite.models.objects import MujocoXMLObject

    from tpgpt.sim.objects import _compiled_mass, _set_mass

    path = BUILD_ROOT / f"{name}.xml"
    if not path.exists():
        path = write_mjcf(name, threshold=threshold)
    obj = MujocoXMLObject(
        str(path), name=name, joints=[dict(type="free", damping="0.0005")],
        obj_type="all", duplicate_collision_geoms=False,
    )
    _set_mass(obj, YCB_ARTICLES[name].mass_kg)
    return obj


def surface_cloud(name: str, count: int = 4096, seed: int = 0) -> np.ndarray:
    """A **full** point cloud of ``name``, sampled uniformly over the scan.

    Uniform by triangle area, so flat faces are not under-sampled relative to
    the dense curvature of a scan's rounded corners.

    This is the object as it *is*, not as the cameras see it. Every cloud used
    by the method so far has been the fused depth image from three cameras: one
    sided, dominated by the top surface, and 523 to 5106 points depending on the
    object. Swapping it for this removes perception from the loop entirely,
    which is a deliberate change of what the experiment measures -- it isolates
    the map, the grasp generator and the executor from the question of whether
    the object was seen properly. It is **not** a filter change and it is not a
    method improvement: a robot facing a real shelf does not have this.
    """
    vertices, faces = read_stl_faces(scan_path(name))
    vertices, _ = _centred(vertices)
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(faces), size=count, p=areas / areas.sum())
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    over = (u + v) > 1.0
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    return a[pick] + u * (b[pick] - a[pick]) + v * (c[pick] - a[pick])


def world_surface_cloud(env, name: str, count: int = 4096, seed: int = 0) -> np.ndarray:
    """:func:`surface_cloud` for ``name``, placed at its current pose in ``env``.

    The cloud is sampled in the object's body frame once and then carried by the
    body's current rotation and position, so it describes the object where it
    actually is rather than where the scan was taken.
    """
    body = env.object_body_ids[name]
    rotation = np.array(env.sim.data.body_xmat[body]).reshape(3, 3)
    position = np.array(env.sim.data.body_xpos[body])
    return surface_cloud(name, count=count, seed=seed) @ rotation.T + position


#: Where the settled resting orientations are cached.
REST_FILE = Path("configs/ycb_rest.json")


def stable_rest_quat(name: str) -> np.ndarray:
    """The orientation ``name`` actually comes to rest in, as ``(w, x, y, z)``.

    **A scan's own frame is not a resting pose.** The YCB scans are expressed in
    the scanner's frame, and for a compact object that happens to be close
    enough to upright that nothing notices. For the hammer it is not: placed at
    its scan orientation it rolls onto its head, and the scene's verification
    measured it tilting **24.6 degrees at pick P0 and 89.7 at P1** and coming to
    rest **91 mm behind the robot base's front face** -- inside the region the
    pick poses were chosen to keep it out of.

    So the orientation is measured rather than assumed: the object is dropped on
    its own onto a plane, settled until it stops, and the orientation it ends in
    is cached. An object placed in it starts stable, which is what makes a
    written-down pick pose mean anything.
    """
    cache = json.loads(REST_FILE.read_text()) if REST_FILE.exists() else {}
    if name in cache:
        return np.asarray(cache[name], dtype=float)
    quat = _settle_alone(name)
    cache[name] = [float(v) for v in quat]
    REST_FILE.parent.mkdir(parents=True, exist_ok=True)
    REST_FILE.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return quat


def _settle_alone(name: str, steps: int = 40000, window: int = 500,
                  still: float = 5e-5) -> np.ndarray:
    """Drop ``name`` on a plane and return the orientation it stops in."""
    import mujoco
    from robosuite.models.world import MujocoWorldBase

    obj = make_ycb_object(name)
    world = MujocoWorldBase()
    world.merge_assets(obj)
    body = obj.get_obj()
    # ``make_ycb_object`` already gives the body a free joint; adding another
    # is "more than 6 dofs in body".
    body.set("pos", "0 0 0.25")
    world.worldbody.append(body)
    ET.SubElement(world.worldbody, "geom", name="ground", type="plane",
                  size="2 2 0.1", pos="0 0 0")
    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    joint = model.body_jntadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)]
    address = model.jnt_qposadr[joint]
    held = 0
    while held < steps:
        before = data.qpos[address:address + 3].copy()
        for _ in range(window):
            mujoco.mj_step(model, data)
        held += window
        if held >= 2000 and np.abs(data.qpos[address:address + 3] - before).max() < still:
            break
    return np.array(data.qpos[address + 3:address + 7], dtype=float)


#: Where the scans come from. A stable, citable URL, which is why the raw
#: downloads are not committed: 196 MB of third-party scan data that anyone can
#: re-fetch. What *is* committed is ``assets/ycb_mjcf`` -- the convex
#: decomposition the scene actually collides against -- because that is
#: generated by a specific version of CoACD at a specific threshold and a reader
#: reproducing a campaign needs the same geometry, not merely the same recipe.
YCB_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{ycb}_google_16k.tgz"


def fetch(names=None) -> list[str]:
    """Download the ``google_16k`` scans for ``names`` if they are not present."""
    import subprocess
    import tarfile
    import urllib.request

    got = []
    for name in names or DEFAULT_YCB_OBJECTS:
        article = YCB_ARTICLES[name]
        if (YCB_ROOT / article.ycb).exists():
            continue
        YCB_ROOT.mkdir(parents=True, exist_ok=True)
        archive = YCB_ROOT / f"{article.ycb}.tgz"
        urllib.request.urlretrieve(YCB_URL.format(ycb=article.ycb), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(YCB_ROOT)
        archive.unlink()
        got.append(article.ycb)
    return got


def main() -> int:
    """``python -m tpgpt.sim.ycb`` -- fetch the scans and build the MJCF."""
    print("fetching scans:", fetch() or "already present")
    for name in DEFAULT_YCB_OBJECTS:
        path = write_mjcf(name)
        print(f"  {name:8s} {extents_mm(name).round(1)} mm  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
