"""Record the source demonstration in MuJoCo and export it for the real robot.

Run once, in the simulation environment::

    MUJOCO_GL=egl python -m tpgpt.hardware.export_source --out outputs/hardware/source_bundle

This calls exactly what ``tpgpt.experiments.pipeline.run`` calls when it has no
cached source -- :func:`record_source_placement`, then ``_to_tool_frame`` with the
Panda's contact offset, then ``to_grasp_convention`` -- and adds one step: the
change from robosuite's world frame to the source robot's base frame (see
:mod:`tpgpt.hardware.source_bundle`). Nothing in the simulation code is
modified.

The run refuses to export a demonstration that failed its own task, since a
policy transported from a failed teacher cannot be expected to succeed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tpgpt.hardware.frames import invert_transform, make_transform
from tpgpt.hardware.source_bundle import SourceBundle, transform_labels, transform_placement


def source_robot_base(seed: int = 0, embodiment: str = "panda") -> np.ndarray:
    """``T_world_base`` of the source robot, read from a fresh scene.

    ``record_source_placement`` closes its environment before returning, so the
    base pose is read from a second one built with the same arguments. The base
    is fixed by the arena, not sampled, so the seed cannot change it -- asserted
    by the caller comparing two seeds in the unit tests' spirit, cheaply here.
    """
    from tpgpt.sim.backend import make_reshelving_env

    env = make_reshelving_env(embodiment=embodiment, seed=seed)
    try:
        env.reset()
        robot = env.robots[0]
        body = robot.robot_model.root_body
        position = np.array(env.sim.data.get_body_xpos(body))
        rotation = np.array(env.sim.data.get_body_xmat(body)).reshape(3, 3)
    finally:
        env.close()
    return make_transform(rotation, position)


def export(out: Path, seed: int = 0, n_points: int = 1200) -> SourceBundle:
    from tpgpt.experiments.pipeline import SOURCE_GRIPPER, _to_tool_frame
    from tpgpt.experiments.reshelving_pipeline import record_source_placement
    from tpgpt.grasp.grasps import contact_offset, to_grasp_convention
    from tpgpt.reporting.provenance import provenance

    labels, placement, ok = record_source_placement(seed, n_points=n_points)
    if not ok:
        raise RuntimeError(
            f"the source demonstration at seed {seed} did not complete its own "
            "task; refusing to export it"
        )

    # --- identical to pipeline.run -------------------------------------------
    offset = contact_offset(SOURCE_GRIPPER)
    tool_labels = _to_tool_frame(labels, offset)
    tool_labels = tool_labels.replace(
        orientations=to_grasp_convention(tool_labels.orientations, SOURCE_GRIPPER)
    )

    # --- new: world -> source robot base -------------------------------------
    T_world_base = source_robot_base(seed)
    T_base_world = invert_transform(T_world_base)
    bundle = SourceBundle(
        tool_labels=transform_labels(tool_labels, T_base_world),
        placement=transform_placement(placement, T_base_world),
        metadata={
            "description": "TPGPT source demonstration, fingertip labels in the grasp "
                           "convention, expressed in the source robot base frame",
            "source_seed": seed,
            "source_gripper": SOURCE_GRIPPER,
            "n_cloud_points": n_points,
            "control_freq": int(labels.metadata.get("control_freq", 20)),
            "n_labels": len(labels),
            "contact_offset_grip_site": np.asarray(offset).tolist(),
            "T_world_source_base": T_world_base.tolist(),
            "frame": "source_robot_base",
            "orientation_convention": "grasp (+X closing, +Z approach)",
            "provenance": provenance(),
        },
    )
    bundle.save(out)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/hardware/source_bundle"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-points", type=int, default=1200)
    args = parser.parse_args(argv)
    bundle = export(args.out, args.seed, args.n_points)
    p = bundle.placement
    print(f"wrote {args.out}: {bundle.tool_labels.describe()}")
    print(f"  source grasp tcp (base frame): {np.round(p.grasp.tcp, 4)}")
    print(f"  support height {p.support_height:.4f} m, destination {np.round(p.destination, 4)} "
          f"at height {p.destination_height:.4f} m")


if __name__ == "__main__":
    main()
