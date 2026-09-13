"""Campaigns: run the pipeline many times and write up what happened.

Four campaigns, each varying one thing:

``keypoints``  all seven combinations of the keypoint families, to settle which
               set actually works rather than which one is tidiest.
``grippers``   every hand whose frame is verified in physics, on one object.
``objects``    every object, on one hand, from a 5 cm lemon to a 15 cm cereal box.
``shelves``    the open boards against a real cubby against a closed shelf.

Reports are written for the runs worth reading -- one per distinct setting, and
every failure -- while repeats contribute a row to the index and their raw
numbers. Writing a full report for every run of a few hundred would bury the
interesting ones.

The source demonstration is recorded **once** per campaign and reused, since
re-recording it per run costs a full episode and would be the same episode
every time.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from tpgpt.experiments.diagnose import (
    CARRY_FRACTION,
    CLOSING_BUDGET_FRACTION,
    LIFT_HEIGHT,
    PLACE_TOLERANCE,
    REACH_TOLERANCE,
    tally,
    tally_text,
)
from tpgpt.experiments.pipeline import (
    DEFAULT_KEYPOINTS,
    SCENE_OBJECTS,
    build_scene,
    run,
)
from tpgpt.experiments.run_keypoint_replay import REPLAY_GRIPPERS
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.grasp.grippers import VERIFIED_PAIRS
from tpgpt.reporting.html import write_index, write_manifest, write_report
from tpgpt.reporting.overlay import build_figures

#: The seven non-empty combinations of the keypoint families.
#:
#: Two are expected to be degenerate rather than merely bad: the support point
#: alone is two points across the whole task and the contacts alone are four,
#: and neither determines an affine map. That is a result, not a crash, so the
#: runner records the reason and moves on.
KEYPOINT_SETS = [
    combination
    for size in (1, 2, 3)
    for combination in itertools.combinations(("box", "contacts", "support"), size)
]

OBJECTS = ("cereal", "milk", "can", "bread", "lemon")
SLOTS = ("top middle", "bottom left", "top right", "bottom right")
ALL_SLOTS = ("top left", "top middle", "top right",
             "bottom left", "bottom middle", "bottom right")

#: Enough steps for the longest successful run measured (361) with room for
#: an arm that is merely slow rather than stuck.
MAX_STEPS = 800

PROMPTS = {
    "cereal": "put the cereal box on the {slot}",
    "milk": "put the milk carton on the {slot}",
    "can": "put the can on the {slot}",
    "bread": "put the bread on the {slot}",
    "lemon": "put the lemon on the {slot}",
}


def prompt_for(obj: str, slot: str) -> str:
    return PROMPTS[obj].format(slot=f"{slot} shelf" if "shelf" not in slot else slot)


#: Keys that describe the run rather than the scene contents.
LABEL_KEYS = ("keypoints", "gripper", "shelf_variant", "obj", "slot", "seed")



def _settings_record(name, settings, objects) -> dict:
    """What this campaign varied and what it held fixed.

    Split that way on purpose. A campaign is only interpretable if a reader can
    see its *axis* -- the thing deliberately changed -- against its
    *background*, everything held constant. Section 7.21 was rewritten in full
    and then withdrawn because its background (which keypoint set was selected)
    had moved without anyone recording it, so the study measured the background
    and attributed it to the axis.

    ``varied`` lists, per key, the distinct values that key took across the
    settings list, so it is derived from what actually ran rather than from what
    was intended. ``fixed`` lists the keys that took exactly one value, which is
    precisely the set someone would otherwise have to reconstruct from the code.
    """
    settings = [dict(s) for s in settings]
    keys = sorted({k for s in settings for k in s})
    seen = {}
    for key in keys:
        values = []
        for s in settings:
            v = s.get(key, "<absent>")
            v = list(v) if isinstance(v, tuple) else v
            if v not in values:
                values.append(v)
        seen[key] = values
    return {
        "campaign": name,
        "n_settings": len(settings),
        "varied": {k: v for k, v in seen.items() if len(v) > 1},
        "fixed": {k: v[0] for k, v in seen.items() if len(v) == 1},
        "scene_objects": list(objects),
        "default_keypoints": list(DEFAULT_KEYPOINTS),
        "max_steps": MAX_STEPS,
    }


def _label(setting: dict) -> str:
    parts = []
    for key in LABEL_KEYS:
        if key not in setting:
            continue
        value = setting[key]
        parts.append("+".join(value) if isinstance(value, tuple) else str(value))
    return " ".join(parts)


def _row(result, label: str) -> dict:
    """One line of the index, including *where* the run broke.

    ``blame`` is the stage that failed rather than the point the run stopped at;
    see :mod:`tpgpt.experiments.diagnose` for why those are different questions.
    """
    diagnosis = result.diagnosis
    row = {
        "label": label,
        "outcome": result.outcome,
        "detail": result.detail,
        "blame": diagnosis.blame if diagnosis is not None else result.outcome,
        "stages": (
            {s.name: s.ok for s in diagnosis.stages} if diagnosis is not None else {}
        ),
        "stage_detail": (
            diagnosis.summary() if diagnosis is not None else result.detail
        ),
        "placement_error_xy": result.metrics.get("placement_error_xy"),
        "n_keypoints": result.metrics.get("n_keypoints"),
        "min_det": result.metrics.get("min_det"),
        "cloud_points": result.metrics.get("cloud_points"),
        "grasps_generated": result.metrics.get("grasps_generated"),
        # Which constraints had to be dropped to leave any candidate at all.
        # A jaw-width fallback means the chosen grasp is on a face wider than
        # the hand opens, which is a grip that cannot hold -- and that is
        # invisible in the outcome.
        "fallbacks": (
            [st.name for st in result.funnel.stages if st.fallback]
            if result.funnel is not None else []
        ),
        "survivors": (
            len(result.funnel.survivors) if result.funnel is not None else None
        ),
        # Retired with the whole-path check; absent from new runs, kept so an
        # older rows.json still reads. See pipeline.CRITICAL_SAMPLES.
        "executable_fraction": result.metrics.get("executable_fraction"),
        "path_fell_back": result.metrics.get("path_fell_back"),
        "path_rank_examined": result.metrics.get("path_rank_examined"),
        "steps": result.metrics.get("steps"),
        "seconds": round(result.seconds, 1),
    }
    if diagnosis is not None:
        row.update({f"m_{k}": v for k, v in diagnosis.measurements.items()})
    return row


def campaign(
    name: str,
    settings: list[dict],
    out_dir: Path,
    objects=("milk", "can", "cereal", "bread"),
    report_every: bool = False,
    verbose: bool = True,
) -> dict:
    """Run one campaign and write its reports, manifest and rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, placement, ok = record_source_placement(0, capture=True)
    if not ok:
        raise RuntimeError("the source demonstration failed; nothing to transport")

    rows, reported, diagnoses = [], set(), []
    for index, setting in enumerate(settings):
        setting = dict(setting)
        label = _label(setting)
        scene_objects = tuple(setting.pop("objects", objects))
        if setting["obj"] not in scene_objects:
            # A prompt naming an object the scene does not contain is a broken
            # experiment, not a language failure: the parser is right to refuse.
            scene_objects = scene_objects + (setting["obj"],)
        env = build_scene(
            scene_objects,
            gripper=setting.get("gripper", "panda"),
            shelf_variant=setting.get("shelf_variant", "cubby"),
            seed=setting.get("seed", 0),
        )
        try:
            result = run(
                prompt_for(setting["obj"], setting["slot"]),
                gripper=setting.get("gripper", "panda"),
                shelf_variant=setting.get("shelf_variant", "cubby"),
                seed=setting.get("seed", 0),
                source=(labels, placement),
                keypoint_parts=setting.get("keypoints", DEFAULT_KEYPOINTS),
                max_steps=MAX_STEPS,
                env=env,
            )
            rows.append(_row(result, label))
            if result.diagnosis is not None:
                diagnoses.append(result.diagnosis)
            if verbose:
                print(f"  {label:<34} {result.summary()}", flush=True)

            key = (setting.get("keypoints"), setting.get("gripper"),
                   setting.get("shelf_variant"), setting["obj"])
            worth_reporting = report_every or not result.success or key not in reported
            if worth_reporting:
                reported.add(key)
                run_dir = out_dir / f"run_{index:03d}"
                figures = build_figures(result, env, run_dir, obs=env._get_observations())
                write_report(result, run_dir, figures,
                             title=f"{label} - {result.prompt}")
                rows[-1]["report"] = f"run_{index:03d}/report.html"
        finally:
            env.close()

    successes = sum(1 for r in rows if r["outcome"] == "success")
    counts = tally(diagnoses)
    write_manifest(
        out_dir,
        title=f"{name} campaign",
        description=CAMPAIGN_TEXT.get(name, ""),
        # What the campaign *was*, separately from how it turned out. These
        # used to be one field holding only run counts and stage tallies --
        # results, with nothing saying what produced them, which is half the
        # reason the earlier campaigns could not be interpreted six weeks later
        # (ROBOTICS_NOTES section 7.26). "Varied" is the campaign's own axis;
        # "fixed" is everything a reader would otherwise have to guess at.
        settings=_settings_record(name, settings, objects),
        results={"runs": len(rows), "succeeded": successes, "stages": counts},
        thresholds={
            "reach_tolerance_m": REACH_TOLERANCE,
            "closing_budget_fraction": CLOSING_BUDGET_FRACTION,
            "lift_height_m": LIFT_HEIGHT,
            "carry_fraction": CARRY_FRACTION,
            "place_tolerance_m": PLACE_TOLERANCE,
        },
        runs=rows,
    )
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=float))
    if verbose:
        print(f"\n{name}: {successes}/{len(rows)} succeeded -> {out_dir}")
        if diagnoses:
            print(tally_text(counts, len(diagnoses)))
        print(_blame_table(rows))
    return {"rows": rows, "successes": successes, "stages": counts}


def _blame_table(rows: list[dict]) -> str:
    """Which stage stopped each run, counted up. The headline of a campaign."""
    from collections import Counter

    counter = Counter(r["blame"] for r in rows)
    lines = ["", "stopped at:"]
    for blame, n in counter.most_common():
        lines.append(f"  {blame:<12}{n:>3}")
    return "\n".join(lines)


CAMPAIGN_TEXT = {
    "execution": (
        "Experiment R's grid -- five hands, four objects -- executed by the "
        "policy instead of replayed pose by pose. R placed 16/20 with the "
        "executor removed; any shortfall here is the executor's."
    ),
    "keypoints": "Every combination of the three keypoint families, to settle "
                 "which set describes an unseen object well enough to transport onto it.",
    "grippers": "One task, every hand whose frame is verified in physics. Jaw "
                "openings span 50 to 125 mm and fingertip depths 103 to 195 mm.",
    "objects": "One hand, every object, from a 5 cm lemon to a 15 cm cereal box.",
    "shelves": "The same task against floating boards, a real cubby with a back "
               "and sides, and a fully closed shelf that can only be entered from the front.",
    "slots": "One hand and one object, into every one of the six destinations, "
             "to separate a placement problem from a grasping one.",
}


def keypoint_settings(objects=("can", "cereal"), slot="top middle", seeds=(0, 1, 2)):
    """Every keypoint combination, everything else held fixed.

    The keypoint set is the only thing that varies within a seed, and the grasp
    cache makes the candidate pool identical across the seven, so a difference
    between two rows is a difference between the two keypoint sets rather than
    between two draws of a diffusion model (section 7.15).
    """
    return [
        {"keypoints": parts, "obj": obj, "slot": slot, "seed": seed}
        for parts in KEYPOINT_SETS
        for obj in objects
        for seed in seeds
    ]


def gripper_settings(obj="can", slot="top middle", seeds=(0, 1)):
    """One object, one slot, every hand. Isolates the embodiment."""
    return [
        {"gripper": g, "obj": obj, "slot": slot, "seed": seed}
        for g in VERIFIED_PAIRS
        for seed in seeds
    ]


def object_settings(slot="top middle", gripper="panda", seeds=(0, 1, 2)):
    """One hand, one slot, every object. Isolates the object geometry.

    Each object is tested with the same two distractors alongside it, so the
    amount of clutter -- and so the amount of occlusion in the point cloud -- is
    the same for the 5 cm lemon as for the 15 cm cereal box.
    """
    return [
        {"obj": obj, "slot": slot, "gripper": gripper, "seed": seed,
         "objects": (obj, *[o for o in ("can", "milk") if o != obj][:2])}
        for obj in OBJECTS
        for seed in seeds
    ]


def slot_settings(obj="can", gripper="panda", seeds=(0, 1)):
    """One hand, one object, every destination. Isolates the placement."""
    return [
        {"obj": obj, "gripper": gripper, "slot": slot, "seed": seed}
        for slot in ALL_SLOTS
        for seed in seeds
    ]


def shelf_settings(obj="can", seeds=(0, 1)):
    """One hand, one object, three shelf variants. Isolates the obstacle.

    The destinations are held to the side the demonstration actually places on
    (``y = -0.13``) rather than being spread over both. The reason is a design
    principle, not a measured result: a campaign that varies the shelf must hold
    the destination fixed, or it measures the two together and can attribute
    neither. The first version of this campaign paired variants with mirrored
    slots and could not tell the walls from the mirroring.

    An earlier version of this docstring cited a measured failure rate for
    mirrored slots. That measurement is withdrawn -- see ``ROBOTICS_NOTES``
    section 7.26 -- but the principle it was used to justify stands on its own.
    """
    return [
        {"shelf_variant": v, "obj": obj, "slot": s, "seed": seed}
        for v in ("open", "cubby", "enclosed")
        for s in ("top middle", "bottom middle")
        for seed in seeds
    ]


def execution_settings(slot="top middle", seeds=(0,)):
    """Experiment R's grid, executed by the **policy** instead of replayed.

    Five hands times four objects, the same cells and the same scene order as
    ``FINDINGS.md`` §8p. That run drove the arm pose by pose under position
    control -- no policy, no attractor integration, no lag gate -- and placed
    **16 of 20**. It is therefore the executor-free ceiling, and this campaign
    asks how much of it survives the thing that actually ships.

    The comparison is only meaningful because the maps are now identical: the
    grasp-pose cube, the same filters, the same whole-path check and the same
    object order all reached ``pipeline.run`` in the two commits before this
    one. Any shortfall against 16/20 is therefore the executor's, which is
    precisely the split ``run_keypoint_replay``'s own docstring sets up.
    """
    return [
        {"gripper": g, "obj": obj, "slot": slot, "seed": seed,
         "objects": SCENE_OBJECTS}
        for g in REPLAY_GRIPPERS
        for obj in SCENE_OBJECTS
        for seed in seeds
    ]


CAMPAIGNS = {
    "execution": execution_settings,
    "keypoints": keypoint_settings,
    "grippers": gripper_settings,
    "objects": object_settings,
    "slots": slot_settings,
    "shelves": shelf_settings,
}


def main(names=("keypoints",), out_root="outputs/campaigns", strict: bool = False):
    # Before anything expensive runs, say whether its results will be
    # attributable to a state of the code. Checked here rather than at the
    # end so a dirty tree costs one line of output now instead of a deleted
    # directory later -- see ROBOTICS_NOTES section 7.26.
    from tpgpt.reporting.provenance import warn_if_unreproducible

    warn_if_unreproducible(strict=strict)
    out_root = Path(out_root)
    for name in names:
        print(f"\n=== {name} ===")
        campaign(name, CAMPAIGNS[name](), out_root / name)
    index = write_index("outputs")
    print(f"\nindex: {index}")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaigns", default="keypoints",
                        help="comma-separated: " + ",".join(CAMPAIGNS))
    parser.add_argument("--out", default="outputs/campaigns")
    parser.add_argument(
        "--require-clean", action="store_true",
        help="refuse to run unless the results would be reproducible: "
             "committed, with no modified or untracked code files",
    )
    args = parser.parse_args()
    main(tuple(args.campaigns.split(",")), args.out, strict=args.require_clean)
