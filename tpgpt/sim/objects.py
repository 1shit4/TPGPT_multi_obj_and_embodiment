"""Scene objects at the size of the real articles they stand for.

robosuite's benchmark meshes are **not** the size of the things they depict.
Measured from MuJoCo's own ``geom_aabb`` (see :data:`BENCHMARK_EXTENTS_MM`), the
cereal box is 100 x 30 x 150 mm against a real family carton's 190 x 80 x 300,
and the milk carton 40 x 40 x 144 against a real 1 L gable-top's 70 x 70 x 230.
The gripper, meanwhile, is full size -- a Panda's jaws open 80 mm whatever the
scene contains. So every campaign run before this module existed asked full-size
hardware to pick up 1.4 to 2.4 times miniature groceries, which is the open item
recorded in ``ROBOTICS_NOTES.md`` section 8.

That is not a cosmetic complaint. It decides which grasps exist at all:

* A real cereal carton is 80 mm across its narrowest face, which is exactly a
  Panda's full jaw opening, so a Panda cannot take one across that face. The
  miniature is 30 mm across and can be grasped almost anywhere.
* A real 1 L milk carton is 70 mm across against the same 80 mm jaw -- 5 mm of
  clearance a side. The miniature has 20 mm a side.

Every filter in the grasp funnel has therefore been exercised on objects that
are easy to grasp from any direction, so "the funnel kept enough candidates"
has meant very little. ``PAPER_PLAN.md`` states the same argument from the other
end: shrinking objects until the narrowest jaw closes on them *sizes the world
to the hardware*, and hides a fact that is physics rather than a limitation of
the method -- a hand's aperture decides which objects it can hold at all.

**Mass is part of "real size", and it was further out than the dimensions.**
robosuite's XML objects declare a *density*, and the declared densities are not
physical: 100 kg/m3 for the can and the milk, 150 for the cereal, 50 for the
bottle, against 1000 for water. Scaling the mesh alone would have left a
66 x 123 mm can weighing 42 g where a full one weighs 345. Since grip is
friction against weight, a light object is a systematically easier object, so
each article here is given its **real mass** and the density is solved for it
(:func:`_solve_density`).

Nothing here changes the benchmark scene. :data:`BENCHMARK` is the default
everywhere, so every result measured before this module stays reproducible, and
the real-sized scene is selected explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The two worlds a scene can be built in.
BENCHMARK = "benchmark"
REAL = "real"
WORLDS = (BENCHMARK, REAL)


@dataclass(frozen=True)
class Article:
    """A real article, and the robosuite asset used to stand for it.

    Args:
        article: What the object is in the world, in enough detail that the
            dimensions below can be checked against a shop shelf.
        size_mm: Its real size, ``(x, y, z)``, in the same axis order the
            benchmark asset uses -- widest face across ``x``, depth across
            ``y``, height along ``z``. Used as the *target* extent; the achieved
            extent is measured and asserted by the unit suite, because scaling a
            mesh per axis does not reproduce the requested box exactly for a
            rounded shape (a scaled convex hull has a slightly different
            axis-aligned box).
        mass_kg: Its real mass, as sold and full. The density of every geom is
            solved to hit this once the mesh has been scaled.
        note: Why these numbers, and anything the asset cannot represent.
    """

    article: str
    size_mm: tuple[float, float, float]
    mass_kg: float
    note: str = ""


#: True extents of the unscaled robosuite meshes, in millimetres, measured from
#: ``model.geom_aabb`` transformed into the body frame.
#:
#: Hard-coded rather than measured at import because they are fixed properties
#: of robosuite's shipped assets and measuring costs an XML compile per object.
#: ``tests/unit/test_objects.py`` re-measures them and fails if an asset moves,
#: which is the point of writing them down: a silent asset change would rescale
#: every object by the wrong factor and the scene would still build.
BENCHMARK_EXTENTS_MM = {
    "can": (52.1, 52.3, 80.2),
    "cereal": (100.0, 30.1, 150.0),
    "milk": (39.7, 39.7, 144.0),
    "bottle": (55.1, 58.5, 160.0),
    "lemon": (69.9, 40.0, 43.4),
    "bread": (48.0, 40.0, 48.0),
}

#: The real articles. Dimensions are the standard retail sizes; each is stated
#: as the article rather than as a scale factor so a reader can check it.
REAL_ARTICLES = {
    "can": Article(
        article="330 ml beverage can",
        size_mm=(66.0, 66.0, 123.0),
        mass_kg=0.345,
        note="66 mm is the standard 211-diameter can body; 330 g of liquid plus "
             "about 15 g of aluminium. The benchmark mesh is 52 x 52 x 80, so "
             "its aspect ratio is squatter than a real can's (1.53 against "
             "1.86); scaling per axis fixes both.",
    ),
    "cereal": Article(
        article="family-size cereal carton",
        size_mm=(190.0, 80.0, 300.0),
        mass_kg=0.500,
        note="The 80 mm depth is the number that matters: it is exactly a "
             "Panda's full jaw opening, so a Panda cannot close across it. That "
             "is the point of sizing the object honestly.",
    ),
    "milk": Article(
        article="1 L gable-top milk carton",
        size_mm=(70.0, 70.0, 230.0),
        mass_kg=1.030,
        note="1 kg of milk plus board. The heaviest object in the set, which is "
             "what makes it the test of grip rather than of aim.",
    ),
    "bottle": Article(
        article="500 ml PET water bottle",
        size_mm=(65.0, 65.0, 230.0),
        mass_kg=0.520,
        note="Held in reserve: a narrow body every jaw in the fleet can close "
             "on, for use if the five chosen objects do not leave four that "
             "every hand can grasp.",
    ),
    "lemon": Article(
        article="lemon",
        size_mm=(70.0, 55.0, 55.0),
        mass_kg=0.100,
        note="Also in reserve. Small enough for every jaw, and the only object "
             "here with no flat face at all.",
    ),
    "bread": Article(
        article="800 g sandwich loaf",
        size_mm=(200.0, 110.0, 110.0),
        mass_kg=0.800,
        note="At real size its narrowest axis is 110 mm, wider than five of the "
             "seven hands can open, so it is expected to be excluded by the "
             "graspability screen rather than to fail in physics.",
    ),
}

#: Real articles for the two objects robosuite builds from primitives rather
#: than from a mesh. They take dimensions directly, so there is no scale factor.
REAL_COMPOSITES = {
    "hammer": Article(
        article="16 oz claw hammer",
        size_mm=(130.0, 38.0, 330.0),
        mass_kg=0.700,
        note="Axes as the asset lays them out: the handle runs along **z** at "
             "330 mm overall, the head spans about 130 mm across x, and the "
             "handle is a 32 mm box which with the head makes the object 38 mm "
             "thick in y. Its mass is in the head and the grip is on the "
             "handle, so a grasp has a moment about it -- which is why it is "
             "in the set. The head span is set by ``head_halfsize``, which "
             "robosuite draws from ``[r, 1.2r]``; it is reproducible here only "
             "because the generator is seeded.",
    ),
    "mug": Article(
        article="ceramic mug",
        size_mm=(90.0, 90.0, 95.0),
        mass_kg=0.350,
        note="90 mm outside diameter, 6 mm wall, 95 mm tall. robosuite's "
             "HollowCylinderObject is a bare tube: **no handle and no base**. "
             "So the graspable features are the 90 mm outside diameter, which "
             "only the wider hands can span, and the 6 mm wall, which any hand "
             "can pinch with one finger inside the tube. A width test run on "
             "the bounding box would call it 90 mm and be wrong about every "
             "hand, which is why the graspability screen asks the generator "
             "for a candidate rather than comparing boxes.",
    ),
}


#: How an object rests on a table, as a rotation applied before the pick
#: configuration's yaw. Axis-angle, ``(x, y, z, degrees)``; absent means upright
#: in the asset's own frame, which is right for a carton, a can and a mug.
#:
#: **The hammer needs one and the benchmark scene never noticed.** Its asset
#: lays the handle along the body's ``z``, so a yaw-only placement stands it on
#: the end of its handle. In the benchmark scene it is 231 mm tall and light
#: enough that it simply toppled during the settle and was measured lying down;
#: at real size, 333 mm and 700 g, it topples off the table -- measured resting
#: at z = 0.488 against a table surface at 0.800. A quarter turn about ``x``
#: lays it on its side, which is how a hammer rests.
REST_ROTATIONS = {
    "hammer": (1.0, 0.0, 0.0, 90.0),
}


def rest_quat(name: str) -> np.ndarray:
    """Resting orientation for ``name`` as a ``(w, x, y, z)`` quaternion."""
    spec = REST_ROTATIONS.get(name)
    if spec is None:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(spec[:3], dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(spec[3]) / 2.0
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def real_size_mm(name: str) -> tuple[float, float, float]:
    """The real article's size for ``name``, in millimetres."""
    if name in REAL_ARTICLES:
        return REAL_ARTICLES[name].size_mm
    if name in REAL_COMPOSITES:
        return REAL_COMPOSITES[name].size_mm
    raise KeyError(f"no real article recorded for {name!r}")


def scale_for(name: str) -> np.ndarray:
    """Per-axis mesh scale taking the benchmark asset to the real article.

    Per axis, not uniform, because the benchmark meshes do not share the real
    articles' aspect ratios: the can is squatter, the cereal box thinner.
    robosuite's ``MujocoXMLObject`` accepts a 3-vector, and measured on all six
    mesh objects the axes do not cross-talk beyond about 2 mm -- a scale of
    ``[2, 1, 1]`` doubles the x extent and leaves y and z alone.
    """
    base = np.asarray(BENCHMARK_EXTENTS_MM[name], dtype=float)
    return np.asarray(real_size_mm(name), dtype=float) / base


def _compiled_mass(obj) -> float:
    """Mass MuJoCo gives ``obj`` once compiled, in kilograms.

    Compiles the object on its own in an empty world. That costs one XML
    compile, which is milliseconds, and it is the only way to learn the mass:
    MuJoCo computes it from each geom's density and the mesh's volume at compile
    time, and neither the mesh volume nor the total is available before that.
    """
    import mujoco
    from robosuite.models.world import MujocoWorldBase

    world = MujocoWorldBase()
    world.merge_assets(obj)
    world.worldbody.append(obj.get_obj())
    model = world.get_model(mode="mujoco")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
    subtree = [b for b in range(model.nbody) if model.body_rootid[b] == body]
    return float(sum(model.body_mass[b] for b in subtree))


def _set_mass(obj, target_kg: float) -> float:
    """Scale every geom's density on ``obj`` so it weighs ``target_kg``.

    Density rather than an explicit ``mass`` attribute, so the *inertia* scales
    with it too. Writing ``mass`` on a geom makes MuJoCo rescale the inertia to
    match, which is the same thing; writing it on only some geoms would not be.

    Returns the achieved mass, which is exact to compile precision because mass
    is linear in density -- one measurement gives the factor, and no iteration
    is needed.
    """
    before = _compiled_mass(obj)
    if before <= 0:
        raise ValueError(f"{obj.name} compiled to zero mass; cannot set density")
    factor = target_kg / before
    for geom in obj._obj.iter("geom"):
        # A geom with an explicit mass ignores density, so clear it first.
        geom.attrib.pop("mass", None)
        density = float(geom.get("density", 1000.0))
        geom.set("density", f"{density * factor:.6g}")
    return _compiled_mass(obj)


def _solve_density(cls, name: str, target_kg: float,
                   density_key: str = "density", **kwargs):
    """Build a ``CompositeObject`` twice to hit ``target_kg``.

    A composite takes its density as a *constructor* argument and builds its
    geometry from it, so unlike a mesh object it cannot be edited after the
    fact. Mass is still linear in density, so two builds are enough: one at a
    reference density to learn the volume, one at the solved density.
    """
    reference = 1000.0
    probe = cls(name=name, **{density_key: reference}, **_replay(kwargs))
    mass = _compiled_mass(probe)
    return cls(name=name, **{density_key: reference * target_kg / mass},
               **_replay(kwargs))


def _seeded(rng):
    """A recorded seed, so the two builds of a density solve see one stream."""
    return int(np.asarray(rng.integers(0, 2 ** 31 - 1)))


def _replay(kwargs: dict) -> dict:
    """``kwargs`` with any recorded seed turned back into a fresh generator.

    The two builds inside :func:`_solve_density` must draw *identically*. Passing
    one generator does the opposite: the probe consumes it and the second build
    draws a different object, which for the hammer changed the head size and
    missed the requested mass by 4%.
    """
    out = dict(kwargs)
    if isinstance(out.get("rng"), (int, np.integer)):
        out["rng"] = np.random.default_rng(int(out["rng"]))
    return out


def make_object(name: str, world: str = BENCHMARK, rng=None):
    """Build scene object ``name`` in ``world``.

    Args:
        name: A key of :data:`~tpgpt.sim.scenes.tabletop_shelf.OBJECT_CLASSES`.
        world: :data:`BENCHMARK` for robosuite's shipped sizes, unchanged, or
            :data:`REAL` for the real article.
        rng: Seeded generator for the objects robosuite samples dimensions for.
            **The hammer is one of them**, and it matters: its handle length is
            drawn uniformly from 0.10-0.25 m and its handle radius from
            0.015-0.020, so ``HammerObject(name="hammer")`` with no generator
            builds a *different hammer every call*. Every benchmark scene
            containing one has therefore been irreproducible, which is why this
            argument exists and why the real hammer is built from fixed scalars
            instead of ranges.

    Returns:
        A robosuite ``MujocoObject``, unregistered and unplaced.
    """
    from robosuite.models.objects import HammerObject, HollowCylinderObject
    from tpgpt.sim.scenes.tabletop_shelf import OBJECT_CLASSES

    if world not in WORLDS:
        raise ValueError(f"unknown world {world!r}; expected one of {WORLDS}")
    if rng is None:
        rng = np.random.default_rng(0)

    cls = OBJECT_CLASSES[name]

    if world == BENCHMARK:
        # Unchanged, except that anything sampling its own dimensions is given a
        # seeded generator so the benchmark scene stops being a lottery too.
        if cls is HammerObject:
            return cls(name=name, rng=rng)
        return cls(name=name)

    if cls is HammerObject:
        spec = REAL_COMPOSITES["hammer"]
        _, thickness, length = spec.size_mm
        # ``handle_length`` is the handle alone; the head is added beyond it, so
        # the overall length is ``handle_length + 2 * head_halfsize`` and
        # ``head_halfsize`` is drawn from ``[r, 1.2r]``. The draw is left in
        # place and seeded rather than removed, so the object stays a robosuite
        # HammerObject; ``_seeded`` hands both builds of the density solve the
        # *same* stream, without which the second build draws a different head
        # and the solved mass misses by about 4%.
        radius = 0.016
        return _solve_density(
            HammerObject, "hammer", spec.mass_kg,
            handle_shape="box",
            handle_radius=radius,
            handle_length=length / 1000.0 - 2 * 1.1 * radius,
            handle_friction=4.0,
            head_density_ratio=4.0,
            density_key="handle_density",
            rng=_seeded(rng),
        )

    if cls is HollowCylinderObject:
        spec = REAL_COMPOSITES["mug"]
        diameter, _, height = spec.size_mm
        wall = 6.0
        return _solve_density(
            HollowCylinderObject, "mug", spec.mass_kg,
            outer_radius=diameter / 2000.0,
            inner_radius=(diameter - 2 * wall) / 2000.0,
            # ``height`` is the half height: a cylinder built with 0.05 measures
            # 100 mm tall.
            height=height / 2000.0,
            ngeoms=16,
        )

    if name not in BENCHMARK_EXTENTS_MM:
        raise KeyError(
            f"{name!r} has no real-world size recorded; add it to "
            "REAL_ARTICLES with the article it stands for"
        )
    obj = cls.__mro__[1](
        _asset_path(name), name=name,
        joints=[dict(type="free", damping="0.0005")],
        obj_type="all", duplicate_collision_geoms=True,
        scale=list(scale_for(name)),
    )
    _set_mass(obj, REAL_ARTICLES[name].mass_kg)
    return obj


def _asset_path(name: str) -> str:
    """robosuite's XML for ``name``.

    robosuite's convenience classes (``CanObject`` and friends) hard-code their
    own asset path and do **not** expose ``scale``, so a scaled object has to be
    built from ``MujocoXMLObject`` directly with the same path.
    """
    from robosuite.utils.mjcf_utils import xml_path_completion

    return xml_path_completion(f"objects/{name}.xml")
