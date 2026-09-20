"""Does the hand predict success? A regression of the outcome on the factors.

The paper's cross-gripper claim (Sec. IV-C) is tested in outcome space rather
than by eye: fit success on gripper, object and destination, and ask whether
the **gripper** factor is significant. If the map is indifferent to
morphology and the executor can follow what it produces, it is not.

Two fits, because they answer slightly different questions and one of them is
not always available:

* :func:`factor_effects` -- a logistic regression with **cluster-robust**
  standard errors, clustered on the condition (pick pose x destination x seed)
  the cell was run under, and a joint Wald test per factor. Frequentist,
  always fits, and its per-factor p-value is the one the table reports.
* :func:`mixed_effects` -- the mixed-effects logistic regression the draft
  names, with the seed as a random intercept. ``statsmodels`` fits it by
  variational Bayes (``BinomialBayesMixedGLM``), which gives posterior means
  and standard deviations rather than likelihood-ratio tests, and it needs at
  least two seeds to have a random effect to estimate. Reported when it can
  be, as the stated model; the Wald test above carries the p-value.

Both take the campaign's ``rows.json`` rows as they are.

Wilson score intervals for the per-hand rates are here too, since the same
table wants them.
"""
from __future__ import annotations

import numpy as np


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def _frame(rows: list[dict], factors: tuple[str, ...], cluster: str):
    import pandas as pd

    df = pd.DataFrame([
        {
            "success": int(r.get("outcome") == "success"),
            **{f: str(r.get(f)) for f in factors},
            "cluster": str(r.get(cluster)),
        }
        for r in rows if not r.get("skipped")
    ])
    return df


def factor_effects(
    rows: list[dict],
    factors: tuple[str, ...] = ("gripper", "object", "destination"),
    cluster: str = "condition_id",
) -> dict:
    """Logistic regression of success on the factors, cluster-robust, joint Wald per factor.

    A factor with a single level in the data is dropped (nothing to contrast).
    Returns per-factor Wald statistics and p-values, the per-level odds ratios
    against the first level, and the count of cells.
    """
    import statsmodels.formula.api as smf

    df = _frame(rows, factors, cluster)
    kept = [f for f in factors if df[f].nunique() > 1]
    if not kept:
        return {"n": int(len(df)), "factors": {}, "note": "no factor varies"}
    formula = "success ~ " + " + ".join(f"C({f})" for f in kept)
    n_clusters = df["cluster"].nunique()
    model = smf.glm(formula, df, family=__import__("statsmodels.api", fromlist=["families"]).families.Binomial())
    if n_clusters > 1:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": df["cluster"].astype("category").cat.codes})
        covariance = f"cluster-robust on {cluster} ({n_clusters} clusters)"
    else:
        fit = model.fit(cov_type="HC1")
        covariance = "heteroskedasticity-robust (HC1); one cluster only"
    out = {"n": int(len(df)), "covariance": covariance, "formula": formula, "factors": {}}
    for f in kept:
        names = [p for p in fit.params.index if p.startswith(f"C({f})")]
        constraint = " = 0, ".join(names) + " = 0"
        try:
            wt = fit.wald_test(constraint, scalar=True)
            chi2, p = (float(np.asarray(wt.statistic).ravel()[0]),
                       float(np.asarray(wt.pvalue).ravel()[0]))
        except ValueError as exc:  # too few cells for the covariance to be usable
            chi2, p = float("nan"), float("nan")
            out.setdefault("notes", []).append(f"{f}: {exc}")
        out["factors"][f] = {
            "levels": sorted(df[f].unique().tolist()),
            "wald_chi2": chi2,
            "df": int(len(names)),
            "p": p,
            "odds_ratio_vs_first": {
                p.split("[T.")[1].rstrip("]"): float(np.exp(fit.params[p])) for p in names
            },
        }
    return out


def mixed_effects(
    rows: list[dict],
    factors: tuple[str, ...] = ("gripper", "object", "destination"),
    group: str = "seed",
) -> dict | None:
    """Mixed-effects logistic regression with ``group`` as a random intercept.

    Returns ``None`` when there is only one group -- there is then no random
    effect to estimate and the fit degenerates to :func:`factor_effects`.
    Fitted by variational Bayes; reports posterior means and SDs of the fixed
    effects and the random-intercept SD.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    df = _frame(rows, factors, group)
    if df["cluster"].nunique() < 2:
        return None
    kept = [f for f in factors if df[f].nunique() > 1]
    formula = "success ~ " + " + ".join(f"C({f})" for f in kept)
    model = BinomialBayesMixedGLM.from_formula(formula, {"g": "0 + C(cluster)"}, df)
    fit = model.fit_vb()
    fixed = {n: {"mean": float(m), "sd": float(s)}
             for n, m, s in zip(fit.model.exog_names, fit.fe_mean, fit.fe_sd)}
    return {"n": int(len(df)), "groups": int(df["cluster"].nunique()),
            "formula": formula, "fixed_effects": fixed,
            "random_intercept_sd": float(np.exp(fit.vcp_mean[0]))}


def per_level_table(rows: list[dict], factor: str = "gripper") -> dict:
    """``{level: {placed, n, rate, wilson}}`` for one factor."""
    out = {}
    for r in rows:
        if r.get("skipped"):
            continue
        level = str(r.get(factor))
        e = out.setdefault(level, {"placed": 0, "n": 0})
        e["n"] += 1
        e["placed"] += int(r.get("outcome") == "success")
    for e in out.values():
        e["rate"] = e["placed"] / e["n"] if e["n"] else float("nan")
        e["wilson"] = wilson(e["placed"], e["n"])
    return out
