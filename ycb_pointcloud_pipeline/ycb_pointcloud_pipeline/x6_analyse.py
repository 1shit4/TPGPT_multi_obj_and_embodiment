#!/usr/bin/env python3
"""X6 analysis (brief §5). Reads only the trial records; no ROS.

    ros2 run ycb_pointcloud_pipeline x6_analyse [--sim-csv sim.csv] [--config ...]

Writes, under <root>/:
    summary.csv                  one flat row per trial (every trial directory, scratch excluded)
    analysis/table5.csv          hand, object, trials, placed, median error, dominant failure stage
    analysis/table5.tex          the same as an IEEEtran tabular, totals rows with Wilson intervals
    analysis/analysis.md         every statistic with its test and assumptions, instrument
                                 repeatability, and "What went wrong"

Denominator: every main-grid trial that reached planning (status other than
preflight_failed), unless the operator excluded it for a cause outside the method.
Excluded and pre-flight-failed attempts are listed, not counted.

--sim-csv: optional simulated results with columns hand,object,trials,placed for the
sim-to-real comparison. Without it that section says the question is not answered.
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from ycb_pointcloud_pipeline import x6_common as xc

Z95 = 1.959963984540054
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260915


# ---------------------------------------------------------------------------- statistics
def wilson(k, n, z=Z95):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return float('nan'), float('nan')
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def fisher(k1, n1, k2, n2):
    from scipy.stats import fisher_exact
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan')
    odds, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative='two-sided')
    return float(odds), float(p)


def bootstrap_median(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(resamples, len(v)), replace=True), axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def mann_whitney(a, b):
    from scipy.stats import mannwhitneyu
    if len(a) == 0 or len(b) == 0:
        return float('nan'), float('nan')
    res = mannwhitneyu(a, b, alternative='two-sided')
    return float(res.statistic), float(res.pvalue)


def point_biserial(binary, continuous):
    from scipy.stats import pointbiserialr
    b = np.asarray(binary, dtype=float)
    c = np.asarray(continuous, dtype=float)
    keep = np.isfinite(b) & np.isfinite(c)
    b, c = b[keep], c[keep]
    if len(b) < 3 or b.min() == b.max() or np.ptp(c) == 0:
        return float('nan'), float('nan'), int(len(b))
    r, p = pointbiserialr(b, c)
    return float(r), float(p), int(len(b))


# ---------------------------------------------------------------------------- records
def get(d, *path, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def flatten(rec):
    o = rec.get('outcome', {})
    t = rec.get('timing_s', {})
    comp = o.get('placement_error_components_mm') or {}
    pick = get(rec, 'conditions', 'pick_pose') or {}
    centroid = pick.get('centroid_m') or [None, None, None]
    prov = rec.get('provenance', {})
    cost = [t.get('map_fit'), t.get('label_transport'), t.get('policy_refit')]
    return {
        'trial_id': rec.get('trial_id'),
        'hand': get(rec, 'conditions', 'hand'),
        'object': get(rec, 'conditions', 'object'),
        'destination': get(rec, 'conditions', 'destination'),
        'pick_pose_index': get(rec, 'conditions', 'pick_pose_index'),
        'pick_centroid_x_m': centroid[0], 'pick_centroid_y_m': centroid[1],
        'pick_centroid_z_m': centroid[2], 'pick_yaw_deg': pick.get('yaw_deg'),
        'session_id': get(rec, 'conditions', 'session_id'),
        'order_in_session': get(rec, 'conditions', 'order_in_session'),
        'status': rec.get('status'),
        'counted': o.get('counted_in_denominator'),
        'excluded': o.get('excluded'),
        'exclusion_reason': o.get('exclusion_reason'),
        'placed': o.get('placed'),
        'placement_error_mm': o.get('placement_error_mm'),
        'placement_error_x_mm': comp.get('x'), 'placement_error_y_mm': comp.get('y'),
        'placement_error_z_mm': comp.get('z'),
        'first_failure_stage': o.get('first_failure_stage'),
        'operator_aborted': o.get('operator_aborted'),
        'abort_reason': o.get('abort_reason') or o.get('system_abort_reason'),
        'object_disturbed_before_grasp_mm': o.get('object_disturbed_before_grasp_mm'),
        'rank_executed': get(rec, 'grasp', 'rank_executed'),
        'n_candidates_generated': get(rec, 'grasp', 'n_candidates_generated'),
        'fell_back': get(rec, 'grasp', 'fell_back'),
        'generator_score': get(rec, 'grasp', 'generator_score'),
        'min_det_J': get(rec, 'map', 'min_det_J'),
        'det_J_at_grasp': get(rec, 'map', 'det_J_at_grasp'),
        'det_J_at_release': get(rec, 'map', 'det_J_at_release'),
        'keypoint_residual_mm': get(rec, 'map', 'keypoint_residual_mm'),
        'aim_error_mm': get(rec, 'map', 'aim_error_mm'),
        'orientation_error_at_grasp_deg': get(rec, 'map', 'orientation_error_at_grasp_deg'),
        'source_to_target_rotation_deg': get(rec, 'map', 'source_to_target_rotation_deg'),
        'cloud_points': get(rec, 'perception', 'cloud_points'),
        'segmentation_score': get(rec, 'perception', 'segmentation_source', 'score'),
        'timing_perception_s': t.get('perception'),
        'timing_grasp_generation_s': t.get('grasp_generation'),
        'timing_filtering_s': t.get('filtering'),
        'timing_selection_total_s': t.get('selection_total'),
        'timing_map_fit_s': t.get('map_fit'),
        'timing_label_transport_s': t.get('label_transport'),
        'timing_policy_refit_s': t.get('policy_refit'),
        'timing_total_plan_s': t.get('total_plan'),
        'timing_execution_s': t.get('execution'),
        'new_hand_cost_s': (sum(cost) if all(v is not None for v in cost) else None),
        'policy_completed': get(rec, 'execution', 'policy_completed'),
        'stalled': get(rec, 'execution', 'stalled'),
        'final_phase': get(rec, 'execution', 'final_phase'),
        'gripper_width_after_close_m': get(rec, 'execution', 'gripper_width_after_close_m'),
        'git_dirty': prov.get('dirty'),
        'git_sha': ';'.join(f'{os.path.basename(k)}={v}' for k, v in
                            (prov.get('git_sha') or {}).items()),
        'config_sha256': prov.get('config_sha256'),
        'questions_incomplete': o.get('questions_incomplete', False),
        'operator_notes': o.get('operator_notes'),
    }


def load(root):
    trials, failed_preflight = [], []
    for path in sorted(glob.glob(os.path.join(root, 'trials', 'trial_*', 'record.json'))):
        rec = xc.read_json(path)
        rec['_dir'] = os.path.dirname(path)
        (failed_preflight if rec.get('status') == 'preflight_failed' else trials).append(rec)
    for path in sorted(glob.glob(os.path.join(root, 'trials', 'trial_*.preflight_failed_*',
                                              'record.json'))):
        rec = xc.read_json(path)
        rec['_dir'] = os.path.dirname(path)
        failed_preflight.append(rec)
    return trials, failed_preflight


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—'
    return f'{x:.{nd}f}'


def fmt_p(p):
    if p is None or not math.isfinite(p):
        return '—'
    return f'{p:.3g}'


def dominant_stage(rows):
    stages = Counter(r['first_failure_stage'] for r in rows
                     if not r['placed'] and r['first_failure_stage'])
    if not stages:
        return '—'
    top = max(stages.values())
    return '/'.join(sorted(s for s, c in stages.items() if c == top)) + f' ({top})'


# ---------------------------------------------------------------------------- report
def analyse(config, sim_csv=None):
    root = config.root
    trials, failed_preflight = load(root)
    rows = [flatten(r) for r in trials]
    os.makedirs(os.path.join(root, 'analysis'), exist_ok=True)
    if rows:
        with open(os.path.join(root, 'summary.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    counted = [r for r in rows if r['counted']]
    hands = [h for h in config.raw['hands']['order'] if any(r['hand'] == h for r in counted)]
    by_hand = {h: [r for r in counted if r['hand'] == h] for h in hands}

    inst = {}
    for path in sorted(glob.glob(os.path.join(root, 'instrument', '*', 'instrument.json'))):
        inst[os.path.basename(os.path.dirname(path))] = xc.read_json(path)
    inst_text = '; '.join(f"{k}: lateral SD {fmt(v.get('sd_lateral_mm'), 2)} mm (n={v.get('repetitions_measured')})"
                          for k, v in inst.items()) or 'not measured'

    # ---- Table 5
    table = []
    for h in hands:
        for obj in config.raw['objects']:
            sub = [r for r in by_hand[h] if r['object'] == obj]
            if not sub:
                continue
            placed = [r for r in sub if r['placed']]
            errs = [r['placement_error_mm'] for r in placed if r['placement_error_mm'] is not None]
            table.append({'hand': h, 'object': obj, 'trials': len(sub), 'placed': len(placed),
                          'median_error_mm': float(np.median(errs)) if errs else None,
                          'dominant_failure_stage': dominant_stage(sub)})
    totals = []
    for label, sub in [(h, by_hand[h]) for h in hands] + [('all', counted)]:
        k, n = sum(1 for r in sub if r['placed']), len(sub)
        p, lo, hi = wilson(k, n)
        errs = [r['placement_error_mm'] for r in sub if r['placed'] and r['placement_error_mm'] is not None]
        totals.append({'hand': label, 'object': 'total', 'trials': n, 'placed': k,
                       'placed_rate': p, 'wilson_lo': lo, 'wilson_hi': hi,
                       'median_error_mm': float(np.median(errs)) if errs else None,
                       'dominant_failure_stage': dominant_stage(sub)})
    with open(os.path.join(root, 'analysis', 'table5.csv'), 'w', newline='') as f:
        fields = ['hand', 'object', 'trials', 'placed', 'placed_rate', 'wilson_lo', 'wilson_hi',
                  'median_error_mm', 'dominant_failure_stage']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in table + totals:
            w.writerow({k: r.get(k) for k in fields})
    write_latex(os.path.join(root, 'analysis', 'table5.tex'), table, totals, inst_text)

    # ---- statistics
    md = []
    add = md.append
    add('# X6 hardware experiment — analysis\n')
    add(f'Campaign `{config.run_id}`. Generated by `x6_analyse` from the trial records in '
        f'`{root}`; every number below is recomputed from `summary.csv`.\n')
    add('## What was measured, and with what instrument\n')
    add('- **Placed** is the operator\'s judgement at the end of each trial, made while watching '
        'the arm and the fixed video camera.')
    add('- **Placement error** is the centroid of the object\'s point cloud after it came to rest '
        '(n_frames fresh Grounded-SAM segmentations from the fixed RealSense, fused with the '
        "planner's filter, in the robot base frame) minus the centroid of the plan's predicted "
        'placed cloud (the pick-time cloud moved by the planned carry). The lateral error is the '
        'horizontal distance; x, y and z are in the base frame.')
    add('- **Instrument repeatability** (same object, not moved, re-segmented each time): '
        f'{inst_text}.')
    add('- The instrument has a **bias this experiment does not measure**: the camera sees one side of '
        'the object, and the object is seen from a different direction on the shelf than on the '
        'table, so the two centroids are not taken over the same surface. Repeatability bounds '
        'the noise, not this offset.\n')
    add(f'Trials counted: {len(counted)}. Scratch trials are never included. '
        f'Excluded: {sum(1 for r in rows if r["excluded"] and not r["counted"])}. '
        f'Pre-flight failures (not trials, nothing moved): {len(failed_preflight)}.\n')

    add('## Success rate\n')
    add('Test: proportion with a **Wilson 95% score interval**. Assumes trials are independent '
        'Bernoulli draws with one success probability per group; the Wilson interval is used '
        'because n is small and the normal approximation is poor near 0 or 1.\n')
    add('| group | placed / trials | rate | Wilson 95% |')
    add('|---|---|---|---|')
    for t in totals:
        add(f"| {t['hand']} | {t['placed']} / {t['trials']} | {fmt(100 * t['placed_rate'] if t['trials'] else None)}% "
            f"| [{fmt(100 * t['wilson_lo'])}, {fmt(100 * t['wilson_hi'])}]% |")
    add('')

    if len(hands) == 2:
        a, b = by_hand[hands[0]], by_hand[hands[1]]
        ka, kb = sum(r['placed'] for r in a), sum(r['placed'] for r in b)
        odds, p = fisher(ka, len(a), kb, len(b))
        add('## Difference between the two hands\n')
        add("Test: **Fisher's exact test**, two-sided, on the 2x2 table placed/not placed x hand. "
            'Unpaired, because the two hands executed different grasps on different point clouds. '
            'Assumes independent trials; conditions on the margins.\n')
        add(f'{hands[0]} {ka}/{len(a)} vs {hands[1]} {kb}/{len(b)}: odds ratio {fmt(odds, 2)}, '
            f'p = {fmt_p(p)}.')
        add('With about 30 trials per hand this test only detects large differences (27/30 against '
            '24/30 gives p = 0.47). A p above 0.05 means the experiment **cannot distinguish** the '
            'hands, not that they are equal.\n')
    else:
        add('## Difference between the two hands\n')
        add(f'Not computed: counted trials exist for {len(hands)} hand(s) ({hands}). The central '
            'cross-embodiment claim is **not tested** until both hands have run their grid.\n')

    add('## Placement error per hand\n')
    add('Median with a **percentile bootstrap 95% CI** '
        f'({BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}), over placed trials with a '
        'settled measurement. Assumes the trials are exchangeable within a hand. Between hands: '
        '**Mann-Whitney U**, two-sided; assumes independent samples and tests whether one '
        "hand's errors tend to be larger, not whether the medians differ.\n")
    add('| hand | n | median (mm) | bootstrap 95% CI |')
    add('|---|---|---|---|')
    errs_by_hand = {}
    for h in hands:
        e = [r['placement_error_mm'] for r in by_hand[h]
             if r['placed'] and r['placement_error_mm'] is not None]
        errs_by_hand[h] = e
        med, lo, hi = bootstrap_median(e)
        add(f'| {h} | {len(e)} | {fmt(med)} | [{fmt(lo)}, {fmt(hi)}] |')
    if len(hands) == 2:
        u, p = mann_whitney(errs_by_hand[hands[0]], errs_by_hand[hands[1]])
        add(f'\nMann-Whitney U = {fmt(u, 1)}, p = {fmt_p(p)}.')
    add('')

    add('## Failure attribution\n')
    add('Count of `first_failure_stage` among counted trials that did not place, with **exact '
        '(Clopper-Pearson) binomial 95% intervals** on the share of all counted trials of that '
        'hand. The stage is the operator\'s judgement from the trial and its video.\n')
    add('| hand | stage | count | of trials | exact 95% |')
    add('|---|---|---|---|---|')
    for h in hands:
        sub = by_hand[h]
        stages = Counter(r['first_failure_stage'] or 'unattributed' for r in sub if not r['placed'])
        for stage in list(xc.FAILURE_STAGES) + ['unattributed']:
            if stage in stages:
                lo, hi = clopper_pearson(stages[stage], len(sub))
                add(f'| {h} | {stage} | {stages[stage]} | {len(sub)} | '
                    f'[{fmt(100 * lo)}, {fmt(100 * hi)}]% |')
    add('')

    add('## Does map quality predict success?\n')
    add('Test: **point-biserial correlation** between `min_det_J` and `placed` (1/0). Assumes '
        'min_det_J is roughly normal within each outcome group and the groups have similar '
        'variance; with a restricted range of min_det_J (every executed map passed min det >= '
        f"{config.method['map']['min_determinant']}) the correlation is attenuated. Predicted "
        'near zero.\n')
    for label, sub in [(h, by_hand[h]) for h in hands] + [('all', counted)]:
        sub = [r for r in sub if r['min_det_J'] is not None]
        r_, p, n = point_biserial([1 if r['placed'] else 0 for r in sub],
                                  [r['min_det_J'] for r in sub])
        add(f'- {label}: r = {fmt(r_, 3)}, p = {fmt_p(p)}, n = {n}')
    add('')

    add('## How often was rank 1 executed?\n')
    add('Rank is 1-based in the generator\'s score order. "fell back" = no generated candidate was '
        'admissible and the scripted top-down grasp ran.\n')
    for h in hands:
        ranks = Counter('fell back' if r['fell_back'] else
                        ('no plan' if r['rank_executed'] is None else str(r['rank_executed']))
                        for r in by_hand[h])
        add(f"- {h}: " + ', '.join(f'rank {k}: {v}' for k, v in
                                   sorted(ranks.items(), key=lambda kv: (not kv[0].isdigit(),
                                                                         int(kv[0]) if kv[0].isdigit() else 0))))
    tags = defaultdict(Counter)
    for rec in trials:
        if not get(rec, 'outcome', 'counted_in_denominator'):
            continue
        for rej in get(rec, 'grasp', 'rejections', default=[]) or []:
            tags[get(rec, 'conditions', 'hand')][rej.get('tag')] += 1
    add('\nRejection reasons of the ranks above the executed one (all counted trials):\n')
    for h in hands:
        add(f"- {h}: " + (', '.join(f'{k} {v}' for k, v in tags[h].most_common()) or 'none'))
    add('')

    add('## Cost of a new hand\n')
    add('`map_fit + label_transport + policy_refit` per trial, seconds (map fit and label '
        'transport re-timed on the trial\'s own inputs; policy refit measured in the executor). '
        'This is the computation a new hand needs; there is no training and no new demonstration.\n')
    for h in hands:
        c = [r['new_hand_cost_s'] for r in by_hand[h] if r['new_hand_cost_s'] is not None]
        if c:
            add(f'- {h}: median {np.median(c):.3f} s, range {min(c):.3f}–{max(c):.3f} s (n={len(c)})')
        else:
            add(f'- {h}: no timings')
    add('')

    add('## Sim-to-real agreement\n')
    if sim_csv and os.path.isfile(sim_csv):
        sim = {}
        with open(sim_csv) as f:
            for r in csv.DictReader(f):
                sim[(r['hand'], r['object'])] = (int(r['placed']), int(r['trials']))
        add("Test: **Fisher's exact test** per matching (hand, object), two-sided. Assumes the "
            'simulated and real trials are independent samples of the same condition, which they '
            'are only loosely (different demonstration source, different scene).\n')
        add('| hand | object | real | sim | p | simulation |')
        add('|---|---|---|---|---|---|')
        matched = 0
        for row in table:
            key = (row['hand'], row['object'])
            if key not in sim:
                continue
            matched += 1
            ks, ns = sim[key]
            _, p = fisher(row['placed'], row['trials'], ks, ns)
            real_rate = row['placed'] / row['trials']
            direction = ('over-predicts' if ks / ns > real_rate else
                         'under-predicts' if ks / ns < real_rate else 'matches')
            add(f"| {row['hand']} | {row['object']} | {row['placed']}/{row['trials']} | {ks}/{ns} | "
                f'{fmt_p(p)} | {direction} |')
        if not matched:
            add('No (hand, object) condition of this campaign exists in the simulated results, so '
                'the comparison **cannot be made**.')
    else:
        add('**Not answered.** No simulated results for matching (hand, object) conditions were '
            'supplied (`--sim-csv`). The Inspire hand has no simulated grid at all (its simulated '
            'model does not actuate), and the simulated Panda grid used different objects and a '
            'simulated demonstration.')
    add('')

    add('## What would make these conclusions false\n')
    add('- A calibration drift between instrument checks (compare the SDs and mean centroids '
        'above; trials between two disagreeing checks are suspect).')
    add('- Operator bias in "placed" and in the failure stage: both are judgements. The video '
        'is the check.')
    add('- A difference between hands caused by something other than morphology: the order of '
        'hands was not randomised (one hand ran entirely before the other), so drift over time '
        'is confounded with hand.')
    add('- Dirty-tree trials: see `git_dirty` in summary.csv; any `true` means that trial\'s code '
        'is not identifiable from git.\n')

    add('## What went wrong\n')
    add('Every counted trial that did not complete normally, every excluded trial, and every '
        'pre-flight failure.\n')
    add('| trial | hand | object | dest | status | counted | stage | reason / notes |')
    add('|---|---|---|---|---|---|---|---|')
    for r in rows:
        if r['status'] == 'executed' and not r['excluded'] and r['placed'] and not r['questions_incomplete']:
            continue
        reason = r['exclusion_reason'] if r['excluded'] else r['abort_reason']
        notes = ' — '.join(str(x) for x in (reason, r['operator_notes']) if x)
        add(f"| {r['trial_id']} | {r['hand']} | {r['object']} | {r['destination']} | {r['status']} | "
            f"{'yes' if r['counted'] else 'no'} | {r['first_failure_stage'] or '—'} | "
            f"{notes.replace('|', '/') or '—'} |")
    for rec in failed_preflight:
        pf = os.path.join(rec['_dir'], 'preflight.json')
        failed = []
        if os.path.isfile(pf):
            failed = [c['check'] for c in xc.read_json(pf).get('checks', []) if not c['ok']]
        add(f"| {os.path.basename(rec['_dir'])} | {get(rec, 'conditions', 'hand')} | "
            f"{get(rec, 'conditions', 'object')} | {get(rec, 'conditions', 'destination')} | "
            f"preflight_failed | no (nothing moved) | — | {', '.join(failed) or '—'} |")
    dirty = [r['trial_id'] for r in counted if r['git_dirty']]
    if dirty:
        add(f'\n**Dirty git tree** on counted trials: {", ".join(dirty)}. These are scratch-grade '
            'by the brief\'s rule 3 until the tree is committed.')
    with open(os.path.join(root, 'analysis', 'analysis.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    return {'rows': rows, 'counted': counted, 'table': table, 'totals': totals}


def write_latex(path, table, totals, inst_text):
    def tex(s):
        return str(s).replace('_', r'\_').replace('%', r'\%')
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Real-robot transfer of one demonstration. Placement error is the median over '
        r'placed trials. Instrument repeatability: ' + tex(inst_text) + r'.}',
        r'\label{tab:hardware}',
        r'\begin{tabular}{llrrrl}',
        r'\hline',
        r'Hand & Object & Trials & Placed & Median err. (mm) & Dominant failure \\',
        r'\hline',
    ]
    for r in table:
        lines.append(f"{tex(r['hand'])} & {tex(r['object'])} & {r['trials']} & {r['placed']} & "
                     f"{fmt(r['median_error_mm'])} & {tex(r['dominant_failure_stage'])} \\\\")
    lines.append(r'\hline')
    for t in totals:
        rate = (f"{t['placed']} ({fmt(100 * t['placed_rate'], 0)}\\%, "
                f"[{fmt(100 * t['wilson_lo'], 0)}, {fmt(100 * t['wilson_hi'], 0)}])"
                if t['trials'] else '0')
        lines.append(f"{tex(t['hand'])} & total & {t['trials']} & {rate} & "
                     f"{fmt(t['median_error_mm'])} & {tex(t['dominant_failure_stage'])} \\\\")
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--config', default=None)
    ap.add_argument('--sim-csv', default=None)
    a = ap.parse_args(argv if argv is not None else [x for x in sys.argv[1:] if x != '--ros-args'])
    config = xc.X6Config(a.config)
    out = analyse(config, a.sim_csv)
    print(f"{len(out['rows'])} trial records, {len(out['counted'])} counted -> "
          f"{os.path.join(config.root, 'analysis')}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
