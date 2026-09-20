"""Is the fitted map blind to the object and the hand, once the grasp is fixed?

    python -m tpgpt.experiments.map_blindness --campaign outputs/campaigns/ycb/ycb \\
        --out outputs/map_blindness

The paper's second question (Sec. IV-B). Properties 1 and 2 concern a *fixed*
grasp: the keypoints of the grasp cube do not change when the object or the
hand changes, only when the grasp does. In a campaign the grasp does move --
a different hand and a different object are offered different candidates --
so the map moves with it. This asks whether it moves **only** because the
grasp moved, which is a claim about the maps themselves and not about any
scalar summary of them: two different maps can share a ``min det``.

Each cell's map is represented two ways, both saved by the campaign:

* ``map_fingerprint``: ``phi(x) - x`` on a fixed 20-cubed grid spanning the
  workspace (``reporting.record.fingerprint``), so two cells' fingerprints are
  directly comparable point for point;
* ``transported_positions``: the transported fingertip path, 200 waypoints.

The distance between two cells is the root-mean-square difference of those
arrays, in metres. The **predictor** is the distance between the two cells'
target grasp poses -- **both of them**, the pick and the release, because the
keypoints are exactly those two poses and nothing else: the root-sum-square of
the two SE(3) geodesics, each position in metres plus rotation angle in
radians scaled by ``ROTATION_SCALE`` metres per radian. (The release pose
depends on the object through its footprint height, which is how the object
legitimately reaches the map; a predictor on the pick alone would attribute
that to "the object" rather than to the grasp pair.) Two indicator
distances say whether two cells share an object (0 same, 1 different) and
whether they share a gripper.

Then, with ``PERMUTATIONS`` permutations of the cell labels:

* a **Mantel test** between map distance and grasp distance -- the map should
  track the grasp;
* **partial Mantel tests** between map distance and each indicator,
  controlling for grasp distance -- the map should *not* track the object or
  the gripper once the grasp is accounted for;
* a **PERMANOVA** share of variance in map distance attributed to the gripper,
  and to the object, *unconditioned*. The object's share is expected to be
  large -- the object sets where the grasp pair is -- and the gripper's small;
  the conditioned question ("beyond the grasp?") is the partial Mantel's.
  A PERMANOVA on residual distances was tried and dropped: a residual can be
  negative (closer than the grasp predicts) and is not a distance.

All permutation-based, so no distributional assumption; numpy and scipy only.
Two figures: map distance against grasp distance with same-object pairs
marked, and a two-dimensional embedding of map distance coloured by gripper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.reporting.html import write_manifest

#: Metres of translation counted equal to one radian of rotation in the grasp
#: distance. 0.1 m: the demonstration's own carry is ~0.35 m and its hand turn
#: 0.6 rad, so this weights the two about as the task does. Reported alongside
#: the two components so a reader can re-weight.
ROTATION_SCALE = 0.1

PERMUTATIONS = 9999


# ----------------------------------------------------------------------------
# distances

def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def pairwise(arrays: list[np.ndarray]) -> np.ndarray:
    n = len(arrays)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _rms(arrays[i], arrays[j])
    return D


def _se3(poses: list[np.ndarray]):
    n = len(poses)
    T = np.zeros((n, n)); A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dp = np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3])
            R = poses[i][:3, :3].T @ poses[j][:3, :3]
            ang = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
            T[i, j] = T[j, i] = dp
            A[i, j] = A[j, i] = ang
    return T, A


def grasp_distance(picks: list[np.ndarray], releases: list[np.ndarray],
                   scale: float = ROTATION_SCALE):
    """Distance between two cells' grasp *pairs*: ``(combined, translation, angle)``.

    Root-sum-square over the pick and the release of the SE(3) geodesic, so
    the predictor is the whole of what the grasp-cube keypoints encode. The
    translation and angle matrices are returned separately (each the
    root-sum-square over the two poses) for the component tests.
    """
    Tp, Ap = _se3(picks)
    Tr, Ar = _se3(releases)
    T = np.sqrt(Tp ** 2 + Tr ** 2)
    A = np.sqrt(Ap ** 2 + Ar ** 2)
    return np.sqrt(T ** 2 + (scale * A) ** 2), T, A


def indicator(labels: list) -> np.ndarray:
    lab = np.asarray(labels)
    return (lab[:, None] != lab[None, :]).astype(float)


# ----------------------------------------------------------------------------
# tests

def _upper(D: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(D), k=1)
    return D[i, j]


def mantel(X: np.ndarray, Y: np.ndarray, permutations: int = PERMUTATIONS, seed: int = 0):
    """Pearson correlation between two distance matrices, permutation p-value."""
    rng = np.random.default_rng(seed)
    x = _upper(X)
    r = float(np.corrcoef(x, _upper(Y))[0, 1])
    count = 0
    n = len(Y)
    for _ in range(permutations):
        p = rng.permutation(n)
        rp = np.corrcoef(x, _upper(Y[np.ix_(p, p)]))[0, 1]
        if abs(rp) >= abs(r):
            count += 1
    return r, (count + 1) / (permutations + 1)


def _residual_matrix(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """``Y`` with its linear dependence on ``Z`` removed, entrywise on the upper triangle."""
    y, z = _upper(Y), _upper(Z)
    b, a = np.polyfit(z, y, 1)
    R = np.zeros_like(Y)
    i, j = np.triu_indices(len(Y), k=1)
    R[i, j] = R[j, i] = y - (a + b * z)
    return R


def partial_mantel(X, Y, Z, permutations: int = PERMUTATIONS, seed: int = 0):
    """Correlation of ``X`` and ``Y`` controlling for ``Z`` (Smouse-Long-Sokal).

    Both ``X`` and ``Y`` are regressed on ``Z``; the residual matrices are
    correlated; the permutation shuffles the rows and columns of the residual
    of ``X`` -- the map distance -- which is the standard construction.
    """
    RX, RY = _residual_matrix(X, Z), _residual_matrix(Y, Z)
    return mantel(RX, RY, permutations, seed)


def permanova(D: np.ndarray, groups: list, permutations: int = PERMUTATIONS, seed: int = 0):
    """Pseudo-F and the share of variance a grouping explains in a distance matrix.

    Anderson's construction: total sum of squares from the distances,
    within-group sum of squares from within-group distances, the ratio
    ``SS_between / SS_total`` as the share, and a permutation p-value on the
    pseudo-F. A negative residual distance (after conditioning) is squared like
    any other, which is the usual treatment.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    n = len(D)
    D2 = D ** 2

    def stats(labels):
        ss_total = D2[np.triu_indices(n, 1)].sum() / n
        ss_within = 0.0
        for lab in np.unique(labels):
            idx = np.flatnonzero(labels == lab)
            if len(idx) > 1:
                sub = D2[np.ix_(idx, idx)]
                ss_within += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
        ss_between = ss_total - ss_within
        a = len(np.unique(labels))
        F = (ss_between / (a - 1)) / (ss_within / (n - a)) if n > a and ss_within > 0 else np.inf
        return F, ss_between / ss_total if ss_total > 0 else 0.0

    F, share = stats(g)
    count = 0
    for _ in range(permutations):
        Fp, _ = stats(rng.permutation(g))
        if Fp >= F:
            count += 1
    return float(F), float(share), (count + 1) / (permutations + 1)


# ----------------------------------------------------------------------------
# driver

def load_cells(campaign_dir: Path) -> list[dict]:
    rows = json.loads((campaign_dir / "rows.json").read_text())
    cells = []
    for r in rows:
        if (not r.get("arrays") or r.get("grasp_pose_target") is None
                or r.get("grasp_pose_release") is None):
            continue
        z = np.load(campaign_dir / r["arrays"])
        if "map_fingerprint" not in z or "transported_positions" not in z:
            continue
        cells.append({
            "gripper": r["gripper"], "object": r["object"],
            "condition": r.get("condition_id"), "placed": r.get("outcome") == "success",
            "min_det": r.get("min_det"),
            "pose": np.asarray(r["grasp_pose_target"], float).reshape(4, 4),
            "release": np.asarray(r["grasp_pose_release"], float).reshape(4, 4),
            "fingerprint": np.asarray(z["map_fingerprint"], float),
            "path": np.asarray(z["transported_positions"], float),
        })
    return cells


def analyse(cells: list[dict], permutations: int = PERMUTATIONS) -> dict:
    D_map = pairwise([c["fingerprint"] for c in cells])
    D_path = pairwise([c["path"] for c in cells])
    D_grasp, D_trans, D_rot = grasp_distance([c["pose"] for c in cells],
                                             [c["release"] for c in cells])
    I_obj = indicator([c["object"] for c in cells])
    I_grip = indicator([c["gripper"] for c in cells])
    out = {"n_cells": len(cells), "rotation_scale_m_per_rad": ROTATION_SCALE,
           "permutations": permutations}
    for name, D in (("fingerprint", D_map), ("path", D_path)):
        r, p = mantel(D, D_grasp, permutations)
        rt, pt = mantel(D, D_trans, permutations)
        rr, pr = mantel(D, D_rot, permutations)
        ro, po = partial_mantel(D, I_obj, D_grasp, permutations)
        rg, pg = partial_mantel(D, I_grip, D_grasp, permutations)
        # unconditioned, for the contrast the partial test is against
        ro0, po0 = mantel(D, I_obj, permutations)
        rg0, pg0 = mantel(D, I_grip, permutations)
        F_g, share_g, p_Fg = permanova(D, [c["gripper"] for c in cells], permutations)
        F_o, share_o, p_Fo = permanova(D, [c["object"] for c in cells], permutations)
        out[name] = {
            "mantel_vs_grasp": {"r": r, "p": p},
            "mantel_vs_grasp_translation": {"r": rt, "p": pt},
            "mantel_vs_grasp_rotation": {"r": rr, "p": pr},
            "same_object": {"r_unconditioned": ro0, "p_unconditioned": po0,
                            "r_partial": ro, "p_partial": po},
            "same_gripper": {"r_unconditioned": rg0, "p_unconditioned": pg0,
                             "r_partial": rg, "p_partial": pg},
            "permanova_gripper": {"F": F_g, "variance_share": share_g, "p": p_Fg},
            "permanova_object": {"F": F_o, "variance_share": share_o, "p": p_Fo},
            "median_distance_m": float(np.median(_upper(D))),
        }
    return out, {"map": D_map, "path": D_path, "grasp": D_grasp,
                 "same_object": I_obj, "same_gripper": I_grip}


def figures(cells, matrices, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    D, G, same = matrices["map"], matrices["grasp"], matrices["same_object"]
    i, j = np.triu_indices(len(D), 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    diff = same[i, j] > 0
    ax.scatter(G[i, j][diff] * 1000, D[i, j][diff] * 1000, s=6, alpha=0.4,
               label="different object")
    ax.scatter(G[i, j][~diff] * 1000, D[i, j][~diff] * 1000, s=6, alpha=0.6,
               label="same object")
    ax.set_xlabel("grasp distance (mm, SE(3) geodesic)")
    ax.set_ylabel("map distance (mm RMS on the grid)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out / "map_vs_grasp_distance.png"; fig.savefig(p, dpi=160); plt.close(fig)
    written.append(p.name)

    # classical MDS of the map distance, coloured by gripper
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:2]
    X = v[:, order] * np.sqrt(np.maximum(w[order], 0))
    fig, ax = plt.subplots(figsize=(5, 4))
    grippers = sorted({c["gripper"] for c in cells})
    for g in grippers:
        idx = [k for k, c in enumerate(cells) if c["gripper"] == g]
        ax.scatter(X[idx, 0] * 1000, X[idx, 1] * 1000, s=14, label=g)
    ax.set_xlabel("MDS 1 (mm)"); ax.set_ylabel("MDS 2 (mm)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    p = out / "map_embedding_by_gripper.png"; fig.savefig(p, dpi=160); plt.close(fig)
    written.append(p.name)
    return written


def _print(summary: dict) -> None:
    for name in ("fingerprint", "path"):
        s = summary[name]
        print(f"\n[{name}]  median pairwise distance {s['median_distance_m'] * 1000:.1f} mm")
        m = s["mantel_vs_grasp"]
        print(f"  Mantel map~grasp            r = {m['r']:+.3f}  p = {m['p']:.4g}"
              f"   (translation r = {s['mantel_vs_grasp_translation']['r']:+.3f}, "
              f"rotation r = {s['mantel_vs_grasp_rotation']['r']:+.3f})")
        for key in ("same_object", "same_gripper"):
            t = s[key]
            print(f"  {key:12s} unconditioned r = {t['r_unconditioned']:+.3f} (p {t['p_unconditioned']:.3g})"
                  f"   partial | grasp r = {t['r_partial']:+.3f} (p {t['p_partial']:.3g})")
        for key in ("permanova_gripper", "permanova_object"):
            t = s[key]
            print(f"  {key:18s} variance share {t['variance_share']:.1%} (p {t['p']:.3g})")


def main(campaign: str, out_dir: str, permutations: int = PERMUTATIONS) -> dict:
    cells = load_cells(Path(campaign))
    print(f"{len(cells)} cells with a fingerprint and a target grasp")
    summary, matrices = analyse(cells, permutations)
    _print(summary)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    summary["figures"] = figures(cells, matrices, out)
    np.savez_compressed(out / "distances.npz", **matrices,
                        gripper=np.array([c["gripper"] for c in cells]),
                        object=np.array([c["object"] for c in cells]))
    write_manifest(
        out,
        title="Is the map blind to the object and the hand?",
        description=__doc__.split("\n\n")[1],
        settings={"fixed": {"campaign": campaign, "rotation_scale_m_per_rad": ROTATION_SCALE,
                            "permutations": permutations,
                            "map_distance": "RMS of phi(x)-x over the 20^3 fingerprint grid",
                            "path_distance": "RMS over the 200 transported waypoints",
                            "grasp_distance": "root-sum-square over pick and release of the "
                                              "SE(3) geodesic between target grasp poses"},
                  "cells": len(cells)},
        results=summary,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--campaign", required=True, help="a campaign directory with rows.json and cell_*.npz")
    parser.add_argument("--out", default="outputs/map_blindness")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()
    main(args.campaign, args.out, args.permutations)
