"""Portable, hash-checked source artifacts and atomic diagnostic checkpoints."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_value))
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_source(directory, labels, placement, success):
    if not success:
        raise ValueError("cannot freeze an unsuccessful source demonstration")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels.save(directory / "labels.npz")
    atomic_json(directory / "placement.json", {
        "points": placement.points, "grasp": {
            f.name: getattr(placement.grasp, f.name) for f in fields(placement.grasp)},
        "support_height": placement.support_height,
        "destination": placement.destination,
        "destination_height": placement.destination_height,
        "metadata": placement.metadata,
    })
    atomic_json(directory / "source.json", {
        "success": True, "provenance": "new local recording; seed 0",
        "files": {p: sha256(directory / p) for p in ("labels.npz", "placement.json")},
    })
    return load_source(directory)


def load_source(directory):
    from tpgpt.sim.keypoints import GraspFrame, ObjectPlacement
    from tpgpt.transport.labels import PolicyLabels
    directory = Path(directory)
    record = json.loads((directory / "source.json").read_text())
    if record.get("success") is not True:
        raise ValueError("source success was not verified")
    for name, digest in record["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"source hash mismatch: {name}")
    p = json.loads((directory / "placement.json").read_text())
    p["grasp"] = GraspFrame(**p["grasp"])
    for key in ("points", "destination"):
        p[key] = np.asarray(p[key], dtype=float)
    # Capture is optional metadata; restore arrays used by plotting code.
    if "capture" in p["metadata"]:
        p["metadata"]["capture"] = {
            k: np.asarray(v) if isinstance(v, list) else v
            for k, v in p["metadata"]["capture"].items()}
    return PolicyLabels.load(directory / "labels.npz"), ObjectPlacement(**p)


def outcome_summary(rows, scheduled):
    successes = sum(r.get("outcome") == "success" for r in rows)
    refused = sum(bool(r.get("refused")) for r in rows)
    setup = sum(r.get("outcome") == "setup_error" for r in rows)
    executed = sum(bool(r.get("executed")) for r in rows)
    return {"scheduled": scheduled, "completed": len(rows), "successes": successes,
            "refusals": refused, "setup_errors": setup,
            "diagnostic_errors": sum(r.get("outcome") == "diagnostic_error" for r in rows),
            "not_executed": sum(r.get("outcome") == "not_executed" for r in rows),
            "execution_failures": sum(bool(r.get("executed")) and r.get("outcome") != "success" for r in rows),
            "overall_success": successes / scheduled if scheduled else None,
            "conditional_execution_success": successes / executed if executed else None}


def validate_pin(cache_hash, effective_pose, expected_hash=None, expected_pose=None):
    """Reject a cache or post-clamp pose mismatch before diagnostic execution."""
    if expected_hash is not None and cache_hash != expected_hash:
        raise ValueError("candidate cache hash changed")
    if expected_pose is not None:
        pose, expected = np.asarray(effective_pose), np.asarray(expected_pose)
        if pose.shape != (4, 4) or expected.shape != (4, 4) or not np.allclose(
                pose, expected, rtol=0, atol=1e-9):
            raise ValueError("effective grasp pose changed")


def write_event_frames(directory, steps, object_name):
    """Extract synchronized control-step events; absent events stay null."""
    directory = Path(directory)
    if not steps:
        return
    first_contact = next((i for i, s in enumerate(steps) if any(
        object_name in (c.get("a") or "") and "gripper0" in (c.get("b") or "") or
        object_name in (c.get("b") or "") and "gripper0" in (c.get("a") or "")
        for c in s.get("contacts", []))), None)
    close = next((i for i, s in enumerate(steps) if s["action"][-1] > 0), None)
    release = (next((i for i in range(close + 1, len(steps))
                     if steps[i]["action"][-1] < 0), None) if close is not None else None)
    initial_z = steps[0]["object_z"]
    lift = next((i for i, s in enumerate(steps) if s["object_z"] > initial_z + .02), None)
    events = dict(first_contact=first_contact, commanded_closure=close,
                  lift_20mm=lift, release=release, settling=len(steps)-1)
    atomic_json(directory / "events.json", {
        name: None if i is None else {"frame": i, "simulation_time": steps[i]["simulation_time"]}
        for name, i in events.items()})
    import imageio.v2 as imageio
    for camera in ("workspace", "sideview"):
        path = directory / f"{camera}.mp4"
        if not path.exists():
            continue
        with imageio.get_reader(path) as reader:
            for name, i in events.items():
                if i is not None:
                    imageio.imwrite(directory / f"{camera}_{name}.png", reader.get_data(i))


def paired_outcomes(before, after):
    """Paired transitions only when cell identity, source and caches agree externally."""
    def key(r):
        c = r["cell"]
        # Explicit placements and shelf translations are experimental factors,
        # not annotations. Omitting them silently paired different scenes.
        fields = ("gripper", "obj", "pick_config", "pick_pose", "slot", "seed", "world",
                  "shelf_variant", "shelf_translation", "object_rest_quat")
        return json.dumps({k: c.get(k) for k in fields}, sort_keys=True)
    right = {key(r): r for r in after}
    if len(right) != len(after) or len({key(r) for r in before}) != len(before):
        raise ValueError("duplicate cell identity in paired comparison")
    transitions, pairs = {}, []
    for left in before:
        r = right.get(key(left))
        if r is None:
            continue
        a = left.get("metrics", {}).get("candidate_cache", {}).get("candidate_sha256")
        b = r.get("metrics", {}).get("candidate_cache", {}).get("candidate_sha256")
        match = a is not None and a == b
        transition = f"{left['outcome']} -> {r['outcome']}"
        if match:
            transitions[transition] = transitions.get(transition, 0) + 1
        pairs.append({"cell": left["cell"], "same_candidate_cache": match,
                      "transition": transition})
    return {"transitions": transitions, "pairs": pairs}


def labels_digest(labels):
    digest = hashlib.sha256()
    for name in sorted(labels.present):
        array = np.asarray(getattr(labels, name), dtype="<f8")
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
