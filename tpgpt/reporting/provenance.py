"""What code produced a result, recorded alongside the result itself.

**Why this module exists.** On 2026-09-07, 254 MB of end-to-end campaign
results had to be deleted -- eight campaigns, 140 runs -- because there was no
way to determine what code had produced any of them. Four of the five source
files involved had never been committed in any version, three rounds of edits
had landed *between* campaign runs, and the manifests recorded only a title, a
description and a timestamp. Three findings in ``ROBOTICS_NOTES.md`` had to be
retracted as a direct consequence. The full post-mortem is section 7.26.

A result that cannot be attributed to a specific state of the code is not a
weak result, it is **not a result at all**: it cannot be reproduced, cannot be
compared against a later run, and cannot be defended. This module makes that
attribution automatic, so it cannot be forgotten again.

**The part that is easy to get wrong.** A plain ``git status`` dirty flag is not
enough, and missing this is exactly what happened. The files that made those
campaigns unidentifiable were **untracked** -- ``pipeline.py``, ``filters.py``,
``diagnose.py``, ``run_experiments.py`` -- so they never appeared in a diff, had
no history to inspect, and left no trace in the commit the run was nominally
"at". :func:`provenance` therefore lists untracked files inside the package
separately and treats them as fatal to reproducibility, because they are.

Nothing here raises. A campaign that cannot reach git still runs and still
records what it could determine, marked as unknown rather than clean -- an
absent answer must never be able to masquerade as a passing one, which is the
same failure mode as the zeroed ``contact_offset`` of section 7.13.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

#: Packages whose versions change what a run does. Recorded by version so a
#: result can be re-created, and so an unexplained change can be traced to a
#: dependency rather than hunted for in this repo.
TRACKED_PACKAGES = ("numpy", "scipy", "mujoco", "robosuite", "similaritymeasures")

#: Directories whose contents are code. An untracked file here invalidates a
#: run; an untracked file in ``outputs/`` obviously does not.
CODE_DIRS = ("tpgpt", "tests")


def _git(*args, cwd=None) -> str | None:
    """Run a git command, returning ``None`` rather than raising on any failure.

    Failure includes: git not installed, not a repository, and a repository in a
    state git refuses to answer about. All of them mean the same thing here --
    provenance is unknown -- and none of them should stop an experiment.
    """
    try:
        out = subprocess.run(
            ("git",) + args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def repo_root(start=None) -> Path | None:
    """The git working tree containing ``start``, or ``None``."""
    top = _git("rev-parse", "--show-toplevel", cwd=start or Path.cwd())
    return Path(top) if top else None


def provenance(cwd=None, code_dirs=CODE_DIRS) -> dict:
    """Everything needed to identify the code behind a result.

    Returns a dict that is always JSON-serialisable and always has the same
    keys, so a consumer never has to guess whether a field is missing because
    provenance failed or because the field did not apply.

    Keys:
        ``commit``: full SHA of ``HEAD``, or ``None``.
        ``commit_short``, ``branch``, ``committed``: readable forms of the same.
        ``dirty``: tracked files modified relative to ``HEAD``.
        ``diff_stat``: ``git diff --stat`` when dirty, so the *shape* of the
            difference survives even after the working tree moves on.
        ``untracked_code``: files under ``code_dirs`` that git has never seen.
            **This is the field that matters** -- see the module docstring.
        ``reproducible``: whether this run could be recreated from git alone.
        ``python``, ``platform``, ``packages``: the rest of the environment.
    """
    root = repo_root(cwd)
    info: dict = {
        "commit": None,
        "commit_short": None,
        "branch": None,
        "committed": None,
        "dirty": [],
        "diff_stat": None,
        "untracked_code": [],
        "reproducible": False,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    if root is None:
        info["note"] = "not a git repository, so this run cannot be reproduced"
        return info

    info["commit"] = _git("rev-parse", "HEAD", cwd=root)
    info["commit_short"] = _git("rev-parse", "--short", "HEAD", cwd=root)
    info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    info["committed"] = _git("log", "-1", "--format=%cI", cwd=root)

    dirty = _git("diff", "--name-only", "HEAD", "--", *code_dirs, cwd=root)
    info["dirty"] = sorted(dirty.splitlines()) if dirty else []
    if info["dirty"]:
        info["diff_stat"] = _git("diff", "--stat", "HEAD", "--", *code_dirs, cwd=root)

    untracked = _git(
        "ls-files", "--others", "--exclude-standard", "--", *code_dirs, cwd=root
    )
    info["untracked_code"] = sorted(untracked.splitlines()) if untracked else []

    info["reproducible"] = bool(
        info["commit"] and not info["dirty"] and not info["untracked_code"]
    )
    return info


def _package_versions() -> dict:
    """Installed versions of the packages that change what a run does."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def describe(info: dict) -> str:
    """One line saying whether this result can be trusted as evidence.

    Written to be readable in a log without cross-referencing anything: it names
    the specific reason a run is not reproducible rather than only that it is
    not.
    """
    if info.get("reproducible"):
        return f"reproducible at {info['commit_short']} ({info.get('branch')})"

    reasons = []
    if not info.get("commit"):
        reasons.append("no git repository")
    if info.get("untracked_code"):
        n = len(info["untracked_code"])
        reasons.append(
            f"{n} untracked code file{'s' if n != 1 else ''} "
            f"({', '.join(info['untracked_code'][:3])}"
            f"{', ...' if n > 3 else ''})"
        )
    if info.get("dirty"):
        n = len(info["dirty"])
        reasons.append(f"{n} modified file{'s' if n != 1 else ''}")
    at = f" at {info['commit_short']}" if info.get("commit_short") else ""
    return f"NOT REPRODUCIBLE{at}: " + "; ".join(reasons)


def warn_if_unreproducible(info: dict | None = None, *, strict: bool = False) -> dict:
    """Print the provenance verdict before a campaign runs, or refuse to run.

    Called at the *start* of a campaign rather than the end, so the cost of a
    dirty tree is one line of output now instead of a deleted directory later.

    Args:
        info: A :func:`provenance` dict; computed if omitted.
        strict: Raise instead of warning. Use in CI, or whenever the run is
            expensive enough that discovering it was worthless afterwards would
            hurt.

    Raises:
        RuntimeError: if ``strict`` and the run would not be reproducible.
    """
    info = provenance() if info is None else info
    if info.get("reproducible"):
        print(f"[provenance] {describe(info)}")
        return info

    message = (
        f"[provenance] {describe(info)}\n"
        "[provenance] Results from this run cannot be attributed to a code "
        "state. Commit first, or treat the output as a scratch experiment and "
        "not as evidence. See ROBOTICS_NOTES.md section 7.26."
    )
    if strict:
        raise RuntimeError(message)
    print(message)
    return info
