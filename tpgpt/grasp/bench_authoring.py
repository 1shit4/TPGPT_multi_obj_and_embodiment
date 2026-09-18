"""Score a sweep-volume authoring rule against GraspGen-X's 26 curated hands.

Why this exists
---------------
`tpgpt.grasp.authoring` has to guess the box a person drew in a GUI. There is
no way to argue that guess is right; there is a way to *measure* it, because
GraspGen-X ships 26 gripper descriptions its authors curated by hand, each with
the URDF they curated it from. That is 26 labelled examples, so authoring
becomes a scored problem rather than a judgement call -- and it is pure
geometry, so a variant costs seconds instead of a physics campaign
(`CLAUDE.md`: prefer a geometric measurement to an end-to-end one).

The baseline to beat is GraspGen-X's own `estimate_inner_sweep_volume`, the
function its wizard uses to seed the box before a human adjusts it.

Two interpreters
----------------
Reading the URDFs needs `yourdfpy` and `trimesh`, which live in the GraspGen-X
environment, not in `mujoco_env`. Loading 26 hands with meshes costs about ten
minutes, so `extract` caches the finger point sets once and `score` reads the
cache -- and `score` needs nothing but numpy, so it runs anywhere::

    # once, in the GraspGen-X environment
    /home/ishita/miniconda3/envs/graspgenx/bin/python \\
        -m tpgpt.grasp.bench_authoring extract --out /tmp/fingers.npz

    # thereafter, in mujoco_env, in about a second
    /home/ishita/mujoco_env/bin/python \\
        -m tpgpt.grasp.bench_authoring score --cache /tmp/fingers.npz

What the score means, and what it does not
------------------------------------------
A curated description is treated as ground truth because those hands are the
ones the paper reports working, not because a person's slider is exact. So a
small error here is evidence the rule agrees with people who knew the hands,
and it is **not** evidence a grasp will hold: the aperture and the depth enter
the model's conditioning, and whether the arm then lifts the object is a
physics question this bench cannot answer. Use `tpgpt.grasp.validate_config`
for that.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

CURATED = pathlib.Path(
    "/home/ishita/task_embod_aware_grasp/6dof_GraspMAS/assets/"
    "gripper_descriptions/gripper_descriptions/assets/x_grippers"
)

#: Vertices kept per finger link. The estimator slices along the approach axis
#: and reads a min and a max per slice, so it is insensitive to density well
#: below this; the cap keeps the cache small.
MAX_POINTS = 4000

TWO_FINGER = ("parallel_2f", "revolute_2f")


def extract(out: pathlib.Path, root: pathlib.Path = CURATED):
    """Cache each curated hand's finger geometry at its declared open pose.

    Finger links are found with GraspGen-X's own `detect_finger_geoms` -- the
    links that move when any actuated joint is perturbed -- so the comparison
    is against their notion of a finger, not ours.
    """
    import importlib.util
    import types

    for mod in ("viser", "viser.transforms"):
        if importlib.util.find_spec(mod.split(".")[0]) is None:
            sys.modules.setdefault(mod, types.ModuleType(mod))
    wizard = (pathlib.Path(__file__).resolve().parents[3]
              / "task_embod_aware_grasp/6dof_GraspMAS/GraspGenX/scripts"
              / "gripper_config_wizard.py")
    spec = importlib.util.spec_from_file_location("wiz", wizard)
    wiz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wiz)

    import trimesh
    import yourdfpy

    store, meta = {}, {}
    for d in sorted(root.iterdir()):
        cfg_f, urdf_f = d / "config.json", d / "gripper.urdf"
        if not (cfg_f.is_file() and urdf_f.is_file()):
            continue
        cfg = json.load(open(cfg_f))
        try:
            robot = yourdfpy.URDF.load(str(urdf_f), load_meshes=True,
                                       build_collision_scene_graph=False,
                                       load_collision_meshes=False)
            open_js = cfg["open"]
            if isinstance(open_js, list):
                open_js = dict(zip(wiz.get_joint_names(robot), open_js))
            names = wiz.detect_finger_geoms(robot, open_js)
            robot.update_cfg(open_js)
            scene, rng = robot.scene, np.random.default_rng(0)
            for i, g in enumerate(names):
                v = trimesh.transformations.transform_points(
                    scene.geometry[g].vertices, scene.graph.get(g)[0])
                if len(v) > MAX_POINTS:
                    v = v[rng.choice(len(v), MAX_POINTS, replace=False)]
                store[f"{d.name}|finger|{i}"] = v.astype(np.float32)
            meta[d.name] = {"n_finger": len(names), "shipped": cfg}
            print(f"  {d.name:18s} {len(names):3d} finger links", flush=True)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {d.name:18s} SKIP {type(exc).__name__}: {exc}"[:90], flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **store)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"cached {len(meta)} grippers -> {out}")
    return out


def _fingers(store, name):
    out, i = [], 0
    while f"{name}|finger|{i}" in store:
        out.append(store[f"{name}|finger|{i}"])
        i += 1
    return out


def score(cache: pathlib.Path, verbose: bool = True, **kwargs) -> dict:
    """Run :func:`tpgpt.grasp.authoring.estimate_sweep_box` over the cache.

    Extra keyword arguments go straight to the estimator, which is how a
    variant is swept without editing it.
    """
    from tpgpt.grasp.authoring import estimate_sweep_box

    store = np.load(cache)
    meta = json.loads(pathlib.Path(cache).with_suffix(".meta.json").read_text())

    rows, refused = [], []
    for name, m in meta.items():
        shipped = m["shipped"]
        # A cache may hold the whole curated config or just the fields this
        # bench reads; accept either so an old cache stays scoreable.
        sv = shipped.get("sweep_volume", shipped)
        se = np.array(sv["extents"], float)
        so = np.array(sv["offset"], float)
        try:
            box = estimate_sweep_box(_fingers(store, name), closing_axis=0, **kwargs)
        except Exception as exc:                      # noqa: BLE001
            refused.append((name, shipped["type"], f"{type(exc).__name__}: {exc}"))
            continue
        e, o = box.extents, box.offset
        rows.append({
            "gripper": name, "type": shipped["type"],
            "shipped_extents_mm": (se * 1000).round(1).tolist(),
            "extents_mm": (e * 1000).round(1).tolist(),
            "d_aperture_mm": float((e[0] - se[0]) * 1000),
            "d_depth_mm": float((e[2] - se[2]) * 1000),
            "d_offset_mm": float((o[2] - so[2]) * 1000),
            "pocket_slices": box.diagnostics["pocket_slices"],
        })

    if verbose:
        print(f"{'gripper':18s} {'type':12s} {'shipped ext (mm)':22s} "
              f"{'ours (mm)':22s} {'d_ap':>7s} {'d_z':>7s} {'d_off':>7s} {'slc':>4s}")
        for r in rows:
            s1 = " ".join(f"{v:6.1f}" for v in r["shipped_extents_mm"])
            s2 = " ".join(f"{v:6.1f}" for v in r["extents_mm"])
            print(f"{r['gripper']:18s} {r['type']:12s} {s1:22s} {s2:22s} "
                  f"{r['d_aperture_mm']:7.1f} {r['d_depth_mm']:7.1f} "
                  f"{r['d_offset_mm']:7.1f} {r['pocket_slices']:4d}")
        for name, typ, why in refused:
            print(f"{name:18s} {typ:12s} REFUSED: {why}")

    summary = {}
    for label, pick in (
        ("2-finger", lambda r: r["type"] in TWO_FINGER),
        ("3f / multi", lambda r: r["type"] not in TWO_FINGER),
        ("all", lambda r: True),
    ):
        sel = [r for r in rows if pick(r)]
        if not sel:
            continue
        ap = np.abs([r["d_aperture_mm"] for r in sel])
        summary[label] = {
            "n": len(sel),
            "median_aperture_err_mm": float(np.median(ap)),
            "within_10mm": int((ap <= 10).sum()),
            "median_depth_err_mm": float(np.median(np.abs([r["d_depth_mm"] for r in sel]))),
            "median_offset_err_mm": float(np.median(np.abs([r["d_offset_mm"] for r in sel]))),
        }
    if verbose:
        print(f"\n{'group':14s} {'n':>3s} {'med |d_ap|':>11s} {'<=10mm':>8s} "
              f"{'med |d_z|':>10s} {'med |d_off|':>12s}")
        for label, v in summary.items():
            print(f"{label:14s} {v['n']:3d} {v['median_aperture_err_mm']:11.1f} "
                  f"{v['within_10mm']:4d}/{v['n']:<3d} "
                  f"{v['median_depth_err_mm']:10.1f} {v['median_offset_err_mm']:12.1f}")
        print("\nBaseline, GraspGen-X's own estimate_inner_sweep_volume on the same "
              "26 hands:\n  2-finger 9.5 mm (10/20), multi 58.8 mm (1/6), all 19.9 mm "
              "(11/26); depth 2.67x too large; offset 26.1 mm.")
    return {"rows": rows, "refused": refused, "summary": summary}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="cache finger geometry (GraspGen-X env)")
    e.add_argument("--out", type=pathlib.Path, required=True)
    s = sub.add_parser("score", help="score the estimator against the cache")
    s.add_argument("--cache", type=pathlib.Path, required=True)
    s.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "extract":
        extract(args.out)
        return 0
    out = score(args.cache)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
