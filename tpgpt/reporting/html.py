"""Self-contained HTML reports for a single run, and an index over all of them.

Written to be read by someone who is not a specialist: each section opens with a
plain sentence saying what is being shown and whether it went well, with the
numbers underneath for someone who wants them. Images are embedded in the file
itself, so a report can be moved, copied or attached to an email and still work.

The index exists because ``outputs/`` becomes unreadable otherwise. Every
experiment drops a ``manifest.json`` saying what it was and how it went, and the
index is built by finding those, so nothing can appear in the directory without
appearing in the index.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import json
from pathlib import Path

import numpy as np

MANIFEST = "manifest.json"

STYLE = """
:root { --ink:#1c1c1e; --muted:#6b6b70; --line:#e2e2e6; --ok:#137a3f; --bad:#b3261e;
        --card:#ffffff; --bg:#f6f6f8; --accent:#2d5fd1; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width: 960px; margin: 0 auto; padding: 32px 20px 72px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing:-.02em; }
h2 { font-size: 17px; margin: 34px 0 6px; letter-spacing:-.01em; }
.sub { color:var(--muted); margin:0 0 20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:18px 20px; margin:14px 0; }
.lead { margin:0 0 12px; }
.verdict { display:inline-block; padding:3px 11px; border-radius:999px;
           font-weight:600; font-size:13px; color:#fff; }
.verdict.ok { background:var(--ok); } .verdict.bad { background:var(--bad); }
table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
img { width:100%; border-radius:8px; border:1px solid var(--line); display:block; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }
.note { color:var(--muted); font-size:13px; margin-top:8px; }
.bar { height:9px; background:#ececf0; border-radius:5px; overflow:hidden; }
.bar > i { display:block; height:100%; background:var(--accent); }
.tag { display:inline-block; background:#eef1fa; color:var(--accent); border-radius:5px;
       padding:1px 8px; font-size:12px; margin-right:6px; }
.warn { background:#fff4e5; border-color:#f0c98a; }
code { background:#f0f0f3; border-radius:4px; padding:1px 5px; font-size:13px; }
a { color:var(--accent); }
"""


def _b64(path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _img(path, alt="") -> str:
    if path is None or not Path(path).is_file():
        return ""
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{_b64(path)}">'


def _e(text) -> str:
    return html.escape(str(text))


def _card(title, lead, body="", classes="") -> str:
    return (
        f'<div class="card {classes}"><h2>{_e(title)}</h2>'
        f'<p class="lead">{lead}</p>{body}</div>'
    )


STYLE += """
table.stages td.mark { width: 1.6em; font-weight: 700; text-align: center; }
table.stages tr.ok  td.mark { color: #137a3f; }
table.stages tr.bad td.mark { color: #b3261e; }
table.stages tr.bad td       { font-weight: 600; }
table.stages tr.skip td      { color: #8a8a8a; }
"""


STYLE += """
.banner { padding:9px 12px; border-radius:7px; font-size:13.5px; margin:0 0 14px;
          border:1px solid transparent; }
.banner.ok  { background:#e9f5ee; border-color:#b7ddc6; color:#0f5c31; }
.banner.bad { background:#fdecea; border-color:#f2b8b2; color:#8c1d16; }
"""


def _rows(pairs) -> str:
    cells = "".join(
        f"<tr><th>{_e(k)}</th><td class='num'>{_e(v)}</td></tr>" for k, v in pairs
    )
    return f"<table>{cells}</table>"


def _setting_rows(settings) -> list:
    """Flatten a campaign's settings into label/value pairs for the index.

    Marks the varied keys as such. A reader looking at a results table needs to
    know which column is the experiment's axis and which are the background
    held fixed around it, and that distinction is not recoverable from the
    numbers -- section 7.21 was withdrawn for exactly that confusion.

    Accepts the older flat shape as well, so manifests written before the
    settings/results split still render.
    """
    if not settings:
        return []
    if "varied" not in settings and "fixed" not in settings:
        return [(k, v) for k, v in settings.items()]
    rows = []
    for key, values in (settings.get("varied") or {}).items():
        shown = ", ".join(
            "+".join(str(x) for x in v) if isinstance(v, list) else str(v)
            for v in values[:6]
        )
        more = f" (+{len(values) - 6} more)" if len(values) > 6 else ""
        rows.append((f"varied: {key}", f"{shown}{more}"))
    for key, value in (settings.get("fixed") or {}).items():
        shown = "+".join(str(x) for x in value) if isinstance(value, list) else value
        rows.append((f"fixed: {key}", shown))
    for key in ("scene_objects", "default_keypoints", "max_steps", "n_settings"):
        if key in settings:
            value = settings[key]
            rows.append(
                (key, ", ".join(str(x) for x in value)
                 if isinstance(value, list) else value)
            )
    return rows


def _funnel_table(funnel) -> str:
    if funnel is None or not funnel.stages:
        return "<p class='note'>No grasp filtering ran.</p>"
    start = max(funnel.stages[0].entered, 1)
    rows = []
    for stage in funnel.stages:
        width = 100.0 * stage.survived / start
        flag = " <span class='tag'>fell back</span>" if stage.fallback else ""
        rows.append(
            f"<tr><th>{_e(stage.name)}{flag}</th>"
            f"<td class='num'>{stage.entered} &rarr; {stage.survived}</td>"
            f"<td style='width:45%'><div class='bar'><i style='width:{width:.1f}%'></i></div></td>"
            f"<td class='note'>{_e(stage.checks)}</td></tr>"
        )
    return f"<table>{''.join(rows)}</table>"


OUTCOME_TEXT = {
    "success": "The object ended up where the instruction asked.",
    "prompt_not_understood": "The instruction could not be resolved against this scene.",
    "object_not_seen": "The cameras saw too little of the object to describe it.",
    "no_grasp_generated": "The grasp generator returned no candidates at all.",
    "no_grasp_survived": "Candidates existed, but none passed the safety checks.",
    "keypoints_degenerate": "The keypoints did not span three dimensions, so no map exists.",
    "map_not_a_diffeomorphism": "The warp folded space, which is not a valid transport.",
    "placed_in_the_wrong_place": "The robot completed the motion but missed the slot.",
    "arm_could_not_hold_the_pose": "The arm could reach the place it was asked to "
        "go, but not while holding its hand the way the task needs, so it stopped "
        "making progress.",
    "policy_stalled": "The robot ran out of steps without finishing the motion.",
}


def write_report(result, out_dir, figures: dict | None = None, title: str = "") -> Path:
    """One run, as a page a non-specialist can follow top to bottom."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = figures or {}
    ok = result.success

    header = (
        f"<h1>{_e(title or result.prompt)}</h1>"
        f"<p class='sub'>{_e(result.gripper)} hand &middot; {_e(result.shelf_variant)} shelf "
        f"&middot; seed {result.seed} &middot; {result.seconds:.0f} s</p>"
    )

    tone = "ok" if ok else "bad"
    verdict = (
        f"<span class='verdict {tone}'>"
        f"{'Task completed' if ok else 'Task failed'}</span>"
        f"<p class='lead' style='margin-top:12px'>"
        f"{_e(OUTCOME_TEXT.get(result.outcome, result.outcome))}"
        + (f" <em>{_e(result.detail)}</em>" if result.detail else "")
        + "</p>"
    )

    understood = _rows([
        ("what was asked", result.prompt),
        ("object it picked", result.object_name or "-"),
        ("where it was told to put it", (result.slot or "-").replace("_", " ")),
    ])
    if result.assumptions:
        understood += (
            "<p class='note'><strong>Assumed, because the instruction did not say:</strong> "
            + "; ".join(_e(a) for a in result.assumptions) + "</p>"
        )

    metrics = result.metrics
    sections = [
        _card("Did it work?", verdict),
        _card(
            "What the instruction was understood to mean",
            "The sentence is parsed against what is actually in the scene, so it can "
            "only ever name an object that is really there.",
            understood,
        ),
        _card(
            "Choosing a grasp",
            "The generator proposes hundreds of ways to hold the object. Most are "
            "unusable &mdash; they approach from a side no camera ever saw, pass through "
            "the shelf, or need the arm to reach somewhere it cannot. Each row shows "
            "how many candidates were left after that check.",
            _funnel_table(result.funnel)
            + (f"<div style='margin-top:14px'>{_img(figures.get('grasp'), 'chosen grasp')}</div>"
               if figures.get("grasp") else "")
            + "<p class='note'>Green marks where the jaws close; red is the direction the "
              "hand comes in from; grey lines are the runners-up.</p>",
        ),
        _card(
            "The keypoints",
            "The robot has never seen this object. It is described by a box fitted to "
            "the points the cameras measured, oriented by the grasp and by the surface "
            "it is standing on &mdash; so a corner of that box means the same thing here as "
            "it did in the demonstration. Circled points touch a surface: those are what "
            "keeps the object from being dropped from too high or pushed into the shelf.",
            f"<div class='grid'>{_img(figures.get('keypoints_source'), 'source keypoints')}"
            f"{_img(figures.get('keypoints_target'), 'target keypoints')}</div>"
            + _rows([
                ("keypoints used", metrics.get("n_keypoints", "-")),
                ("how exactly they are matched", _mm(metrics.get("keypoint_residual"))),
                ("space never folded", _pct(metrics.get("fraction_positive"))),
                ("tightest squeeze of space", _round(metrics.get("min_det"))),
            ]),
        ),
        _card(
            "The transported motion",
            "The blue path is the single demonstration, recorded once on a different "
            "object in a different scene. The orange path is that same motion warped "
            "onto this object and this shelf. Nothing was taught again.",
            f"{_img(figures.get('trajectory'), 'demonstrated and transported paths')}"
            + (f"<div style='margin-top:14px'>{_img(figures.get('trajectory_scene'), 'paths in the scene')}</div>"
               if figures.get("trajectory_scene") else ""),
        ),
        _card(
            "Step by step, what the robot actually did",
            "A pick-and-place has seven things that each have to work, in order. "
            "The first one that fails is the one worth fixing &mdash; everything after "
            "it is a knock-on effect. A tick means that step worked.",
            _stage_table(result.diagnosis),
            classes="" if ok else "warn",
        ),
        _card(
            "What happened when it ran",
            "The motion above was executed on the arm in physics.",
            _rows([
                ("control steps used", metrics.get("steps", "-")),
                ("distance from the target slot", _mm(metrics.get("placement_error_xy"))),
                ("points the cameras got on the object", metrics.get("cloud_points", "-")),
                ("grasp candidates generated", metrics.get("grasps_generated", "-")),
            ]),
            classes="" if ok else "warn",
        ),
    ]

    page = (
        f"<!doctype html><meta charset='utf-8'><title>{_e(title or result.prompt)}</title>"
        f"<style>{STYLE}</style><div class='wrap'>{header}{''.join(sections)}"
        f"<p class='note'>Generated {dt.datetime.now():%Y-%m-%d %H:%M}. "
        f"TPGPT &mdash; policy transportation with keypoint parameterisation.</p></div>"
    )
    path = out_dir / "report.html"
    path.write_text(page)
    return path


def _funnel_from_runs(runs) -> dict:
    """Count how far each run got, from the rows themselves.

    Derived here rather than read from a stored summary so the index cannot
    disagree with the rows it is displaying -- including for campaigns recorded
    before the tally was corrected. A run whose outcome is ``success`` counts as
    having passed every stage, whatever its per-stage proxies said.
    """
    from tpgpt.experiments.diagnose import STAGES

    counts = {name: 0 for name in STAGES}
    for run in runs:
        stages = run.get("stages") or {}
        if run.get("outcome") == "success":
            for name in STAGES:
                counts[name] += 1
            continue
        for name in STAGES:
            if stages.get(name):
                counts[name] += 1
            else:
                break
    return counts


def _stage_funnel(stages, total: int) -> str:
    """How far the runs of a campaign got, as a funnel.

    One number -- "3 of 42 succeeded" -- says a campaign failed but not where,
    and where is the only part that tells anyone what to do next.
    """
    if not stages or not total:
        return ""
    from tpgpt.experiments.diagnose import STAGES, STAGE_TEXT

    rows = []
    for name in STAGES:
        passed = stages.get(name, 0)
        width = 100.0 * passed / total
        rows.append(
            f"<tr><th>{_e(name)}</th><td class='num'>{passed}/{total}</td>"
            f"<td style='width:45%'><div class='bar'><i style='width:{width:.1f}%'></i></div></td>"
            f"<td class='note'>{_e(STAGE_TEXT[name])}</td></tr>"
        )
    return (
        "<p class='note' style='margin-top:14px'><strong>How far the runs got.</strong> "
        "Each row counts the runs that completed that step and every step before it.</p>"
        f"<table>{''.join(rows)}</table>"
    )


def _stage_table(diagnosis) -> str:
    """The seven stages, with the measured margin beside each.

    Written so the failing row is readable on its own: a reader who stops at the
    first cross should already know what went wrong and by how much.
    """
    if diagnosis is None:
        return "<p class='note'>No step-by-step trace was recorded for this run.</p>"

    from tpgpt.experiments.diagnose import STAGE_TEXT

    rows = []
    failed = False
    for stage in diagnosis.stages:
        if stage.ok:
            mark, tone = "&#10003;", "ok"
        elif failed:
            mark, tone = "&middot;", "skip"
        else:
            mark, tone = "&#10007;", "bad"
            failed = True
        rows.append(
            f"<tr class='{tone}'><td class='mark'>{mark}</td>"
            f"<td>{_e(STAGE_TEXT.get(stage.name, stage.name))}</td>"
            f"<td class='num'>{_e(stage.detail)}</td></tr>"
        )
    verdict = (
        "<p class='note'>Every step worked.</p>" if diagnosis.first_failure is None
        else f"<p class='note'><strong>First thing that went wrong:</strong> "
             f"{_e(diagnosis.summary())}</p>"
    )
    return f"<table class='stages'>{''.join(rows)}</table>{verdict}"


def _mm(value):
    return "-" if value is None else f"{float(value) * 1000:.1f} mm"


def _pct(value):
    return "-" if value is None else f"{float(value) * 100:.0f}%"


def _round(value, digits=3):
    return "-" if value is None else f"{float(value):.{digits}f}"


def write_manifest(out_dir, *, record_provenance: bool = True, **fields) -> Path:
    """Record what an experiment was, so the index can find and describe it.

    **Provenance is recorded automatically and by default.** Every manifest
    carries the git commit, the modified files, the untracked code files and the
    package versions behind it, because a result that cannot be attributed to a
    state of the code cannot be reproduced, compared or defended. Eight
    campaigns and 140 runs were deleted for want of exactly this
    (``ROBOTICS_NOTES.md`` section 7.26), so it is opt-*out* rather than opt-in:
    the failure mode being guarded against is forgetting.

    Args:
        record_provenance: Set ``False`` only where the answer is meaningless,
            such as a unit test writing a manifest into a temporary directory.
        fields: Everything else describing the experiment. Callers should pass a
            ``settings`` dict holding **every** parameter that was varied or
            held fixed -- a manifest that records an outcome without recording
            what produced it is how a table becomes unreadable six weeks later.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields.setdefault("created", dt.datetime.now().isoformat(timespec="seconds"))
    if record_provenance and "provenance" not in fields:
        from tpgpt.reporting.provenance import provenance

        fields["provenance"] = provenance()
    path = out_dir / MANIFEST
    path.write_text(json.dumps(fields, indent=2, default=_jsonable))
    return path


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _provenance_banner(info) -> str:
    """A visible verdict on whether a campaign's numbers can be reproduced.

    Placed at the top of every card, above the results, and deliberately loud
    when the answer is no. The index this replaces showed only a title and a
    timestamp, so eight campaigns produced by four different states of the code
    looked identical and directly comparable. They were not, and the whole set
    had to be deleted (``ROBOTICS_NOTES.md`` section 7.26). A reader must be
    able to see at a glance whether a table is evidence or a scratch run.
    """
    if not info:
        return (
            "<p class='banner bad'>No provenance recorded. The code behind "
            "these numbers cannot be identified, so they cannot be reproduced "
            "or compared against any other run.</p>"
        )
    if info.get("reproducible"):
        return (
            f"<p class='banner ok'>Reproducible at "
            f"<code>{_e(info.get('commit_short'))}</code> "
            f"({_e(info.get('branch'))}), {_e(info.get('committed'))}.</p>"
        )
    bits = []
    if info.get("untracked_code"):
        files = info["untracked_code"]
        shown = ", ".join(f"<code>{_e(f)}</code>" for f in files[:4])
        bits.append(
            f"{len(files)} untracked code file{'s' if len(files) != 1 else ''} "
            f"({shown}{', ...' if len(files) > 4 else ''}) - these exist in no "
            f"commit, so this run's code cannot be recovered at all"
        )
    if info.get("dirty"):
        bits.append(
            f"{len(info['dirty'])} file"
            f"{'s' if len(info['dirty']) != 1 else ''} modified since "
            f"<code>{_e(info.get('commit_short'))}</code>"
        )
    if not info.get("commit"):
        bits.append("not a git repository")
    return (
        "<p class='banner bad'><strong>Not reproducible.</strong> "
        + "; ".join(bits)
        + ". Treat these numbers as a scratch experiment, not as evidence.</p>"
    )


def write_index(outputs_dir="outputs") -> Path:
    """Build ``outputs/index.html`` from every manifest under ``outputs/``."""
    root = Path(outputs_dir)
    manifests = sorted(root.rglob(MANIFEST))
    entries = []
    for path in manifests:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data["_dir"] = path.parent.relative_to(root).as_posix() or "."
        entries.append(data)
    entries.sort(key=lambda d: d.get("created", ""), reverse=True)

    cards = []
    for entry in entries:
        runs = entry.get("runs", [])
        successes = sum(1 for r in runs if r.get("outcome") == "success")
        rate = (
            f"<td class='num'>{successes}/{len(runs)}</td>" if runs else "<td class='num'>-</td>"
        )
        report = entry.get("report")
        link = (
            f"<a href='{_e(entry['_dir'])}/{_e(report)}'>open report</a>"
            if report else f"<a href='{_e(entry['_dir'])}/'>browse files</a>"
        )
        rows = "".join(
            f"<tr><td>{_e(r.get('label', ''))}</td>"
            f"<td>{_e(r.get('outcome', ''))}</td>"
            f"<td>{_e(r.get('blame', ''))}</td>"
            f"<td class='num'>{_mm(r.get('placement_error_xy'))}</td>"
            + (f"<td><a href='{_e(entry['_dir'])}/{_e(r['report'])}'>report</a></td>"
               if r.get("report") else "<td></td>")
            + "</tr>"
            for r in runs[:48]
        )
        detail = (
            "<table><tr><th>run</th><th>outcome</th><th>broke at</th>"
            f"<th class='num'>error</th><th></th></tr>{rows}</table>"
            if rows else ""
        )
        elided = (
            f"<p class='note'>{len(runs) - 48} further runs are in "
            f"<code>{_e(entry['_dir'])}/rows.json</code>.</p>"
            if len(runs) > 48 else ""
        )
        cards.append(
            f"<div class='card'><h2>{_e(entry.get('title', entry['_dir']))}</h2>"
            f"<p class='lead'>{_e(entry.get('description', ''))}</p>"
            + _provenance_banner(entry.get("provenance"))
            + _rows(
                [("folder", entry["_dir"]), ("when", entry.get("created", "-"))]
                + _setting_rows(entry.get("settings"))
                + [(k, v) for k, v in (entry.get("thresholds") or {}).items()]
            )
            + (f"<p class='note'>{successes} of {len(runs)} runs succeeded.</p>" if runs else "")
            + _stage_funnel(_funnel_from_runs(runs), len(runs))
            + detail
            + elided
            + f"<p style='margin-top:12px'>{link}</p></div>"
        )

    page = (
        "<!doctype html><meta charset='utf-8'><title>TPGPT experiments</title>"
        f"<style>{STYLE}</style><div class='wrap'>"
        "<h1>TPGPT experiments</h1>"
        "<p class='sub'>Everything under <code>outputs/</code>, newest first. "
        "Each entry is one experiment: what it was, how it was set up, and how it went.</p>"
        + ("".join(cards) or "<div class='card'><p>No manifests found.</p></div>")
        + f"<p class='note'>Generated {dt.datetime.now():%Y-%m-%d %H:%M} "
        f"from {len(entries)} manifest files.</p></div>"
    )
    path = root / "index.html"
    path.write_text(page)
    return path
