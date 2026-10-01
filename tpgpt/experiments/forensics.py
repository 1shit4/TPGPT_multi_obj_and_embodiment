"""Serial YCB investigation with frozen source, pinned grasps and checkpoints.

python -m tpgpt.experiments.forensics cells.json outputs/grasp_audit/baseline \
    --source outputs/grasp_audit/source --variant legacy --render

Cells are dictionaries (gripper, obj, pick_config, slot, optional seed/world,
forced_grasp_index, expected_cache_hash, expected_effective_pose, rollout_kwargs).
Old five-element lists remain readable; their last field is an expectation,
not authorization to force a rejected candidate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import os
import sys
import traceback

import numpy as np

from tpgpt.experiments.audit_artifacts import (
    atomic_json, load_source, outcome_summary, save_source, sha256, write_event_frames,
)

VARIANTS = {
    "legacy": dict(selection_mode="legacy", target_cloud_correction=False,
                   path_stride=4, critical_waypoints=False),
    "target": dict(selection_mode="legacy", target_cloud_correction=True,
                   path_stride=4, critical_waypoints=False),
    "waypoints": dict(selection_mode="legacy", target_cloud_correction=True,
                      path_stride=1, critical_waypoints=True),
    "task": dict(selection_mode="legacy", target_cloud_correction=True,
                 path_stride=4, critical_waypoints=True, require_task_waypoints=True),
    "fallback": dict(selection_mode="strict", target_cloud_correction=True,
                     path_stride=1, critical_waypoints=True,
                     validate_path_entry=False, strict_target_contacts=False, simulator_place_filter=True,
                     filter_effective_poses=False),
    "strict": dict(selection_mode="strict", target_cloud_correction=True,
                   path_stride=1, critical_waypoints=True, filter_effective_poses=False),
    "effective": dict(selection_mode="strict", target_cloud_correction=True,
                      path_stride=1, critical_waypoints=True, filter_effective_poses=True),
    "edge": dict(selection_mode="strict", target_cloud_correction=True,
                 path_stride=1, critical_waypoints=True, filter_effective_poses=True,
                 target_edge_margin_pixels=1),
}


def normalize_cell(cell):
    if isinstance(cell, list):
        g, o, pick, slot, expected = cell
        return dict(gripper=g, obj=o, pick_config=pick, slot=slot,
                    expected_index=expected, world="ycb")
    cell = dict(cell)
    if "forced_grasp_index" in cell and not all(
            k in cell for k in ("expected_cache_hash", "expected_effective_pose")):
        raise ValueError("forced diagnostics require both cache hash and effective pose")
    return cell


def run_cells(cells, out, source_dir, variant="legacy", executor="policy", render=False,
              cache_only=True):
    from tpgpt.experiments.pipeline import build_scene, run
    from tpgpt.experiments.run_experiments import prompt_for
    from tpgpt.experiments.diagnose import object_probe
    from tpgpt.experiments.reshelving_pipeline import record_source_placement
    out, source_dir = Path(out), Path(source_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (source_dir / "source.json").exists():
        labels, placement, ok = record_source_placement(0, capture=True)
        save_source(source_dir, labels, placement, ok)
    source = load_source(source_dir)
    cells = [normalize_cell(c) for c in cells]
    config = dict(cells=cells, variant=variant, executor=executor, render=render,
                  cache_only=cache_only,
                  source_sha256=sha256(source_dir / "source.json"), settings=VARIANTS[variant],
                  code_hashes={str(p): sha256(p) for p in sorted(Path("tpgpt").rglob("*.py"))})
    manifest = out / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("checkpoint configuration changed; use a new output directory")
    atomic_json(manifest, config)
    provenance = out / "code.patch"
    if not provenance.exists():
        provenance.write_text(subprocess.check_output(["git", "diff"], text=True))
        for path in sorted(Path("tpgpt").rglob("*.py")):
            destination = out / "code" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        atomic_json(out / "provenance.json", {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "status": subprocess.check_output(["git", "status", "--short"], text=True),
            "source": str(source_dir.resolve()), "python": sys.version,
            "environment": {k: os.environ.get(k) for k in (
                "MUJOCO_GL", "NUMBA_CACHE_DIR", "OPENBLAS_NUM_THREADS",
                "TPGPT_GRASP_CACHE", "GRASPGENX_GRIPPER_CFG_DIR")},
        })
    rows = []
    for i, cell in enumerate(cells):
        directory = out / f"cell_{i:03d}"
        directory.mkdir(exist_ok=True)
        checkpoint = directory / "result.json"
        if checkpoint.exists():
            rows.append(json.loads(checkpoint.read_text()))
            continue
        env, writers = None, []
        original_action = None
        row = dict(cell=cell, executed=False, refused=False)
        try:
            controller_config = None
            if executor in ("replay", "joint_policy"):
                from robosuite.controllers import load_composite_controller_config
                from tpgpt.sim.replay import make_position_controller_config
                controller_config = make_position_controller_config(
                    load_composite_controller_config(controller="BASIC", robot="Panda"))
                if "position_gain" in cell:
                    controller_config["body_parts"]["right"]["kp"] = float(cell["position_gain"])
            g, o = cell["gripper"], cell["obj"]
            env = build_scene((o,), gripper=g, seed=cell.get("seed", 0),
                              world=cell.get("world", "ycb"),
                              shelf_variant=cell.get("shelf_variant", "cubby"),
                              pick_config=(None if "pick_pose" in cell else cell.get("pick_config", "P0")),
                              pick_pose=cell.get("pick_pose"),
                              object_rest_quat=cell.get("object_rest_quat"),
                              shelf_translation=cell.get("shelf_translation", (0., 0., 0.)),
                              controller_config=controller_config)
            obj = next(obj for obj in env.objects if obj.name == o)
            def scene_state():
                return {"object_pose": np.asarray(env.sim.data.get_joint_qpos(obj.joints[0])).copy(),
                        "object_velocity": np.asarray(env.sim.data.get_joint_qvel(obj.joints[0])).copy(),
                        "slot_position": np.asarray(env.slot_poses()[cell["slot"]]).copy()}
            row["initial_scene"] = scene_state()
            probe = object_probe(env, o, g)
            frame_records, step_records = [], []
            if render:
                import imageio.v2 as imageio
                writers = [imageio.get_writer(directory / f"{cam}.mp4", fps=env.control_freq)
                           for cam in ("workspace", "sideview")]
            from tpgpt.sim.controllers.cartesian_impedance import CartesianImpedanceController
            original_action = CartesianImpedanceController.action
            commanded = {}
            def traced_action(controller, position_desired, *args, **kwargs):
                commanded["position"] = np.asarray(position_desired).copy()
                rotation = kwargs.get("rotation_desired")
                commanded["rotation"] = None if rotation is None else np.asarray(rotation).copy()
                return original_action(controller, position_desired, *args, **kwargs)
            CartesianImpedanceController.action = traced_action
            original_step = env.step
            def traced_step(action):
                if not step_records:
                    disable_execution_cameras(env)
                observation = original_step(action)
                data, model = env.sim.data, env.sim.model
                contacts = []
                for j in range(data.ncon):
                    contact = data.contact[j]
                    contacts.append({"a": model.geom_id2name(contact.geom1),
                                     "b": model.geom_id2name(contact.geom2),
                                     "distance": float(contact.dist)})
                grip = env.robots[0].gripper["right"]
                site = model.site_name2id(grip.naming_prefix + "grip_site")
                step_records.append({"frame": len(step_records),
                    "simulation_time": float(data.time), "action": np.asarray(action).copy(),
                    "commanded_pose": dict(commanded),
                    "actual_hand": np.asarray(data.site_xpos[site]).copy(),
                    "actual_rotation": np.asarray(data.site_xmat[site]).reshape(3, 3).copy(),
                    "contacts": contacts, **probe(env)})
                for cam, writer in zip(("workspace", "sideview"), writers):
                    if cam == "sideview":
                        # Render-only camera change, restored before the next
                        # observation. It cannot change candidate generation.
                        import mujoco
                        from tpgpt.sim.scenes.tabletop_shelf import _look_at
                        camera = model.camera_name2id(cam)
                        old_p, old_q = model.cam_pos[camera].copy(), model.cam_quat[camera].copy()
                        target = .5 * (env.object_position(o) + np.asarray(data.site_xpos[site]))
                        position = target + np.array([-.65, 0., .25])
                        try:
                            model.cam_pos[camera] = position
                            model.cam_quat[camera] = np.fromstring(_look_at(position, target), sep=" ")
                            mujoco.mj_camlight(model._model, data._data)
                            step_records[-1]["side_camera_position"] = position
                            step_records[-1]["side_camera_target"] = target
                            writer.append_data(env.sim.render(width=512, height=512, camera_name=cam)[::-1])
                        finally:
                            model.cam_pos[camera], model.cam_quat[camera] = old_p, old_q
                            mujoco.mj_camlight(model._model, data._data)
                    else:
                        writer.append_data(env.sim.render(width=512, height=512, camera_name=cam)[::-1])
                return observation
            env.step = traced_step
            def synchronized_probe(e):
                values = probe(e)
                frame_records.append({"frame": len(frame_records), "simulation_time": float(e.sim.data.time),
                                      **values})
                return values
            rollout_options = dict(cell.get("rollout_kwargs", {}))
            if executor in ("replay", "joint_policy"):
                rollout_options.pop("close_gate", None)
                rollout_options.pop("release_gate", None)
            options = {k: cell[k] for k in (
                "forced_grasp_index", "expected_cache_hash", "expected_effective_pose", "expected_labels_hash",
                "flip_target", "ik_seed_strategy") if k in cell}
            result = run(prompt_for(o, cell["slot"].replace("_", " ")),
                         gripper=g, seed=cell.get("seed", 0), objects=(o,), source=source,
                         shelf_variant=cell.get("shelf_variant", "cubby"),
                         max_steps=cell.get("max_steps", 600), env=env, executor=executor, cache_only=cache_only,
                         simulator_diagnostics=True,
                         rollout_kwargs={**rollout_options, "probe": synchronized_probe},
                         **VARIANTS[variant], **options)
            if cell.get("post_settle_seconds", 0) and result.rollout is not None:
                if executor not in ("replay", "joint_policy"):
                    raise ValueError("post-settling requires a position-controller executor")
                from tpgpt.experiments.settling import observe_settling
                settling = observe_settling(env, o, cell["slot"], cell["post_settle_seconds"])
                atomic_json(directory / "settling.json", settling)
                result.metrics["pre_settle_outcome"] = result.outcome
                result.metrics["post_settle"] = {k: v for k, v in settling.items() if k != "samples"}
                if result.success and not settling["passed"]:
                    result.outcome = "post_release_unstable"
                    result.detail = "did not remain released, supported and stable in the slot"
            row["final_scene"] = scene_state()
            np.savez_compressed(directory / "final_sim_state.npz", qpos=env.sim.data.qpos,
                                qvel=env.sim.data.qvel, ctrl=env.sim.data.ctrl,
                                time=float(env.sim.data.time))
            row.update(outcome=result.outcome, detail=result.detail,
                       refused=result.refused, executed=result.rollout is not None,
                       metrics=result.metrics, forced_diagnostic="forced_grasp_index" in cell)
            expected = cell.get("expected_index")
            row["reproduction_index_match"] = (None if expected is None else
                expected == result.metrics.get("grasp_chosen_index"))
            if result.transported_labels is not None:
                result.transported_labels.save(directory / "transported_labels.npz")
            ro = result.rollout
            if ro is not None:
                np.savez_compressed(directory / "trace.npz", hand=ro.positions,
                    commanded=ro.attractors, grip=ro.gripper, phase=ro.time_belief)
                atomic_json(directory / "frames.json", frame_records)
                atomic_json(directory / "simulation_steps.json", step_records)
                atomic_json(directory / "execution.json", ro.metadata)
                if result.diagnosis is not None:
                    atomic_json(directory / "measurements.json", result.diagnosis.measurements)
        except Exception as exc:
            row.update(outcome="setup_error" if env is None or isinstance(exc, FileNotFoundError) else "diagnostic_error",
                       detail=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        finally:
            if original_action is not None:
                CartesianImpedanceController.action = original_action
            for writer in writers:
                writer.close()
            if env is not None:
                env.close()
        atomic_json(checkpoint, row)
        if row.get("executed"):
            try:
                write_event_frames(directory, step_records, cell["obj"])
            except Exception as exc:
                row["event_artifact_error"] = f"{type(exc).__name__}: {exc}"
                atomic_json(checkpoint, row)
        rows.append(row)
        atomic_json(out / "rows.json", rows)
        atomic_json(out / "summary.json", outcome_summary(rows, len(cells)))
        print(f"[{i+1}/{len(cells)}] {cell['gripper']}/{cell['obj']}: {row['outcome']}", flush=True)
    atomic_json(out / "rows.json", rows)
    atomic_json(out / "summary.json", outcome_summary(rows, len(cells)))
    return rows


def disable_execution_cameras(env):
    """Selection has finished; avoid unused RGB/depth observations in execution.

    This changes no physical/control parameter. Explicit video renders remain
    enabled, and the selection's observed cloud has already been captured.
    """
    for name, observable in env._observables.items():
        if observable.modality == "image":
            env.modify_observable(name, "enabled", False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cells", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--variant", choices=VARIANTS, default="legacy")
    p.add_argument("--executor", choices=("policy", "replay", "joint_policy", "none"), default="policy")
    p.add_argument("--render", action="store_true")
    p.add_argument("--generate-candidates", action="store_true",
                   help="Allow inference on cache misses; store every new draw for reproduction")
    a = p.parse_args()
    run_cells(json.loads(a.cells.read_text()), a.out, a.source, a.variant, a.executor,
              a.render, cache_only=not a.generate_candidates)


if __name__ == "__main__":
    main()
