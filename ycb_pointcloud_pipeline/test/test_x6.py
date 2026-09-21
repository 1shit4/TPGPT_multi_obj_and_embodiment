"""Unit tests for the X6 recording, provenance, order sheet and analysis. No robot, no ROS graph."""

import csv
import glob
import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ycb_pointcloud_pipeline import x6_analyse as xa
from ycb_pointcloud_pipeline import x6_common as xc

CONFIG = os.path.join(os.path.dirname(__file__), '..', '..', 'x6_experiment', 'config',
                      'x6_experiment.yaml')


# ------------------------------------------------------------------------ statistics
def test_wilson_matches_reference_values():
    p, lo, hi = xa.wilson(8, 10)
    assert p == pytest.approx(0.8)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)
    _, lo0, hi0 = xa.wilson(0, 10)
    assert lo0 == 0.0 and hi0 == pytest.approx(0.2775, abs=1e-4)


def test_clopper_pearson_reference():
    lo, hi = xa.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-4)


def test_fisher_quoted_in_the_report():
    _, p = xa.fisher(27, 30, 24, 30)
    assert p == pytest.approx(0.472, abs=1e-3)


def test_bootstrap_median_brackets_the_median():
    med, lo, hi = xa.bootstrap_median([1, 2, 3, 4, 100])
    assert med == 3 and lo <= med <= hi


def test_point_biserial_degenerate_is_nan():
    r, p, n = xa.point_biserial([1, 1, 1], [0.5, 0.6, 0.7])
    assert np.isnan(r) and n == 3


# ------------------------------------------------------------------------ cli + config
def test_cli_parameter_parsing():
    argv = ['x', '--ros-args', '-p', 'experiment:=true', '-p', 'trial_id:=0007',
            '-p', 'hand:=panda', '-p', 'speed_scale:=0.9']
    assert xc.experiment_requested(argv)
    assert xc.cli_parameter_value(argv, 'trial_id') == '0007'
    names = xc.cli_parameter_names(argv)
    assert [n for n in names if n not in xc.TRIAL_CLI_PARAMETERS] == ['speed_scale']
    assert not xc.experiment_requested(['x', '--ros-args', '-p', 'experiment:=false'])
    assert not xc.experiment_requested(['x'])


def test_unfilled_detection():
    assert xc.is_unfilled('TO_FILL')
    assert xc.is_unfilled({'x': 1.0, 'y': 'TO_FILL'})
    assert xc.is_unfilled([1, None])
    assert not xc.is_unfilled({'x': 0.5, 'y': -0.2, 'surface_height': 0.3})


def test_campaign_config_loads_and_hands_resolve():
    cfg = xc.X6Config(CONFIG)
    assert cfg.run_id
    assert set(cfg.raw['hands']['files']) == {'panda', 'inspire'}
    assert cfg.hand('panda')['integrated'] is True
    assert cfg.hand('inspire')['integrated'] is True
    assert len(cfg.raw['objects']) == 5 and 'banana' not in cfg.raw['objects']
    assert cfg.raw['method']['selection']['fallback_top_down'] is True


def test_order_sheet_is_a_permutation_of_the_grid_and_reproducible(tmp_path):
    from ycb_pointcloud_pipeline.x6_session import order_sheet
    cfg = xc.X6Config(CONFIG)
    rows, seed = order_sheet(cfg, 'panda')
    again, _ = order_sheet(cfg, 'panda')
    assert rows == again
    assert len(rows) == 30
    cells = {(r['object'], r['pick_mark'], r['destination']) for r in rows}
    assert len(cells) == 30
    assert rows[0]['trial_id'] == '0001' and rows[-1]['trial_id'] == '0030'
    inspire, _ = order_sheet(cfg, 'inspire')
    assert inspire[0]['trial_id'] == '0031'
    assert [r['object'] for r in inspire] != [r['object'] for r in rows]


# ------------------------------------------------------------------------ provenance
def test_repo_provenance_sees_modified_and_untracked_code(tmp_path):
    repo = tmp_path / 'r'
    repo.mkdir()
    run = lambda *a: subprocess.run(['git', '-C', str(repo), *a], check=True,  # noqa: E731
                                    capture_output=True)
    run('init', '-q')
    run('config', 'user.email', 't@t')
    run('config', 'user.name', 't')
    (repo / 'a.py').write_text('x = 1\n')
    run('add', 'a.py')
    run('commit', '-qm', 'init')
    clean = xc.repo_provenance(str(repo))
    assert clean['dirty'] is False and clean['git_sha']
    (repo / 'a.py').write_text('x = 2\n')
    (repo / 'new.py').write_text('y = 1\n')
    (repo / 'outputs').mkdir()
    (repo / 'outputs' / 'data.json').write_text('{}')
    dirty = xc.repo_provenance(str(repo))
    assert dirty['dirty'] is True
    assert dirty['modified_files'] == ['a.py']
    assert dirty['untracked_code_files'] == ['new.py']      # outputs/ is data, not code


def test_demo_hashes_detect_a_changed_file(tmp_path):
    (tmp_path / 'tool_labels.npz').write_bytes(b'abc')
    h1 = xc.demo_hashes(tmp_path)
    (tmp_path / 'tool_labels.npz').write_bytes(b'abd')
    assert xc.demo_hashes(tmp_path)['tool_labels.npz'] != h1['tool_labels.npz']


# ------------------------------------------------------------------------ record pieces
def _candidate(index, score, rejected=None):
    from tpgpt.sim.keypoints import GraspFrame
    g = GraspFrame(tcp=[0.5, 0.0, 0.05], approach=[0, 0, -1], closing=[1, 0, 0])
    c = SimpleNamespace(index=index, score=score, grasp=g, rejected=rejected, plan_metrics=None,
                        refined_depth=0.0, refined_centre=0.0)
    c.as_dict = lambda: {'index': index, 'score': score, 'rejected': rejected}
    return c


def test_rank_executed_and_rejections_of_earlier_ranks():
    pytest.importorskip('tpgpt')
    from ycb_pointcloud_pipeline import x6_record as xr
    c1 = _candidate(4, 0.9, 'hits_support: hand -5 mm above the table')
    c2 = _candidate(0, 0.8, 'map_folds: min det(J) 0.05')
    c3 = _candidate(2, 0.7)
    c4 = _candidate(1, 0.6)
    sel = SimpleNamespace(candidates=[c1, c2, c3, c4], chosen=c3, source='graspgenx', notes=[])
    poses = np.tile(np.eye(4), (5, 1, 1))
    doc, grasp = xr.candidates_document(sel, poses, np.linspace(1, 0, 5))
    assert grasp['rank_executed'] == 3
    assert [r['rank'] for r in grasp['rejections']] == [1, 2]
    assert [r['stage'] for r in grasp['rejections']] == ['screening', 'planning']
    assert grasp['fell_back'] is False
    statuses = [c['status'] for c in doc['candidates']]
    assert statuses == ['rejected', 'rejected', 'executed', 'admissible_after_selection_not_evaluated']
    assert doc['candidates'][0]['raw_generator_pose'] == np.eye(4).tolist()


def test_fallback_has_no_rank_and_lists_every_rejection():
    pytest.importorskip('tpgpt')
    from ycb_pointcloud_pipeline import x6_record as xr
    c1 = _candidate(0, 0.9, 'from_below: approach 120 deg from vertical')
    c2 = _candidate(1, 0.8, 'misses_grasp: transported path 20 mm from the grasp')
    fb = _candidate(-1, float('nan'))
    sel = SimpleNamespace(candidates=[c1, c2], chosen=fb, source='top_down', notes=[])
    _, grasp = xr.candidates_document(sel, np.tile(np.eye(4), (2, 1, 1)), [0.9, 0.8])
    assert grasp['fell_back'] is True
    assert grasp['rank_executed'] is None
    assert len(grasp['rejections']) == 2
    assert grasp['generator_score'] is None


def test_placement_error_components():
    from ycb_pointcloud_pipeline import x6_record as xr
    settled = np.array([[0.51, 0.02, 0.30], [0.53, 0.02, 0.32]])
    predicted = np.array([[0.50, 0.00, 0.30], [0.50, 0.00, 0.30]])
    e = xr.placement_error(settled, predicted, [0.5, 0.0, 0.3])
    assert e['placement_error_components_mm']['x'] == pytest.approx(20.0)
    assert e['placement_error_components_mm']['y'] == pytest.approx(20.0)
    assert e['placement_error_mm'] == pytest.approx(np.hypot(20, 20))
    assert xr.placement_error(np.zeros((0, 3)), predicted, None) is None


def test_pick_pose_yaw_and_nearest_mark():
    from ycb_pointcloud_pipeline import x6_record as xr
    t = np.linspace(-0.05, 0.05, 50)
    pts = np.column_stack([0.6 + t * np.cos(np.radians(30)), 0.1 + t * np.sin(np.radians(30)),
                           np.zeros_like(t)])
    pts[:, :2] += np.random.default_rng(0).normal(0, 0.002, (50, 2))
    pose = xr.pick_pose_from_cloud(pts, [{'index': 0, 'x': 0.3, 'y': 0.0},
                                         {'index': 1, 'x': 0.6, 'y': 0.1}])
    assert pose['yaw_deg'] == pytest.approx(30.0, abs=3.0)
    assert pose['nearest_mark_index'] == 1


def test_trace_has_step_index_and_no_clock(tmp_path):
    from ycb_pointcloud_pipeline import x6_record as xr
    tr = xr.TraceRecorder(20)
    for k in range(5):
        tr.add('policy' if k > 1 else 'approach', [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0], [0] * 7, -1.0, 0.08, None, 0.1 * k, 1.0)
    tr.save(tmp_path / 'trace.npz')
    d = np.load(tmp_path / 'trace.npz')
    assert list(d['step_index']) == [0, 1, 2, 3, 4]
    assert float(d['control_rate_hz']) == 20.0
    assert not any('time' in k or 'stamp' in k for k in d.files)
    assert np.isnan(d['wrench']).all()
    assert tr.policy_steps() == 3


def test_map_block_recomputes_min_det_of_a_saved_plan():
    pytest.importorskip('tpgpt')
    plans = sorted(glob.glob(os.path.expanduser('~/tpgpt_plans/*/plan.json')))
    demos = sorted(glob.glob(os.path.expanduser('~/tpgpt_demos/*/source_bundle/tool_labels.npz')))
    if not plans or not demos:
        pytest.skip('no saved plan / demo on this machine')
    from tpgpt.hardware.planner import Plan
    from tpgpt.hardware.source_bundle import SourceBundle
    from ycb_pointcloud_pipeline import x6_record as xr
    for path in reversed(plans):
        meta = json.load(open(path))['metadata']
        bundle = meta.get('source_bundle_dir')
        if bundle and os.path.isdir(bundle):
            break
    else:
        pytest.skip('no saved plan whose source bundle still exists')
    plan = Plan.load(os.path.dirname(path))
    source = SourceBundle.load(bundle)
    block, timing = xr.map_block(plan, source, 0.02, 'cube_grasp_pose')
    assert block['det_J_recomputation_matches'], (block['min_det_J'], block['det_J_recomputed_min'])
    assert block['min_det_J'] <= block['det_J_at_grasp'] + 1e-9
    assert timing['map_fit'] > 0


# ------------------------------------------------------------------------ analysis end to end
def _write_record(root, tid, hand, obj, dest, placed, err, stage=None, status='executed',
                  excluded=False, min_det=0.9, rank=1):
    d = os.path.join(root, 'trials', f'trial_{tid}')
    os.makedirs(d)
    rec = {'trial_id': tid, 'status': status,
           'conditions': {'hand': hand, 'object': obj, 'destination': dest},
           'grasp': {'rank_executed': rank, 'fell_back': False, 'rejections':
                     [{'rank': 1, 'tag': 'hits_support'}] if rank and rank > 1 else []},
           'map': {'min_det_J': min_det},
           'timing_s': {'map_fit': 0.03, 'label_transport': 0.01, 'policy_refit': 0.2},
           'provenance': {'dirty': False, 'git_sha': {}},
           'outcome': {'placed': placed, 'placement_error_mm': err,
                       'placement_error_components_mm': {'x': err or 0, 'y': 0, 'z': 0},
                       'first_failure_stage': stage, 'excluded': excluded,
                       'exclusion_reason': 'network_drop' if excluded else None,
                       'counted_in_denominator': not excluded and status != 'preflight_failed'}}
    xc.write_json(os.path.join(d, 'record.json'), rec)


def test_analysis_end_to_end(tmp_path):
    raw = yaml.safe_load(open(CONFIG))
    raw['experiment']['data_root'] = str(tmp_path)
    raw['hands']['files'] = {k: os.path.join(os.path.dirname(os.path.abspath(CONFIG)), v)
                             for k, v in raw['hands']['files'].items()}
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = xc.X6Config(str(cfg_path))
    root = cfg.root
    k = 0
    for hand, n_placed in (('panda', 8), ('inspire', 4)):
        for i in range(10):
            k += 1
            placed = i < n_placed
            _write_record(root, f'{k:04d}', hand, 'apple' if i % 2 else 'can', 'slot_a', placed,
                          10.0 + i if placed else None, None if placed else 'grasp',
                          min_det=0.8 + 0.01 * i, rank=1 if i % 3 else 2)
    k += 1
    _write_record(root, f'{k:04d}', 'panda', 'mug', 'slot_b', False, None, None,
                  status='stopped_by_operator', excluded=True)
    out = xa.analyse(cfg)
    assert len(out['rows']) == 21 and len(out['counted']) == 20
    totals = {t['hand']: t for t in out['totals']}
    assert totals['panda']['placed'] == 8 and totals['inspire']['placed'] == 4
    md = open(os.path.join(root, 'analysis', 'analysis.md')).read()
    for needle in ("Fisher's exact test", 'Wilson 95%', 'bootstrap', 'Mann-Whitney U',
                   'point-biserial', 'What went wrong', 'network_drop', 'Not answered'):
        assert needle in md, needle
    tex = open(os.path.join(root, 'analysis', 'table5.tex')).read()
    assert r'\begin{tabular}' in tex and 'total' in tex
    with open(os.path.join(root, 'summary.csv')) as f:
        assert len(list(csv.DictReader(f))) == 21
