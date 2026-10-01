"""Reproduction safety and mismatch reporting, without physics."""
import json
from pathlib import Path

import pytest

from scripts import reproduce_evaluation as repro


def test_indices_reject_duplicates_and_out_of_range():
    assert repro.parse_indices('0,7,119') == [0, 7, 119]
    assert repro.parse_indices('all') == list(range(120))
    for value in ('-1', '120', '0,0', '', 'banana'):
        with pytest.raises(ValueError):
            repro.parse_indices(value)


def test_pins_preserve_normal_selection_without_modifying_original():
    row = dict(cell={'gripper': 'panda'}, metrics={
        'grasp_chosen_index': 7, 'candidate_cache': {'candidate_sha256': 'cache'},
        'effective_grasp_pose': [[1]], 'transported_labels_sha256': 'labels'})
    cells = repro.pinned_cells([row], [0])
    assert cells[0]['expected_index'] == 7
    assert cells[0]['expected_effective_pose'] == [[1]]
    assert cells[0]['expected_cache_hash'] == 'cache'
    assert cells[0]['expected_labels_hash'] == 'labels'
    assert 'forced_grasp_index' not in cells[0]
    assert row['cell'] == {'gripper': 'panda'}
    row['cell']['forced_grasp_index'] = 7
    with pytest.raises(ValueError, match='normal selection'):
        repro.pinned_cells([row], [0])


def test_resume_refuses_changed_inputs(tmp_path):
    path = tmp_path / 'reproduction.json'
    repro.write_unchanged(path, {'indices': [7]})
    repro.write_unchanged(path, {'indices': [7]})
    with pytest.raises(ValueError, match='resume inputs changed'):
        repro.write_unchanged(path, {'indices': [8]})
    assert json.loads(path.read_text()) == {'indices': [7]}


def test_asset_mapping_uses_path_boundaries():
    mappings = [(Path('/old/assets'), Path('/new/assets'))]
    assert repro.mapped_path('/old/assets/a.obj', mappings) == Path('/new/assets/a.obj')
    assert repro.mapped_path('/old/assets_other/a.obj', mappings) == Path('/old/assets_other/a.obj')


def test_mismatch_is_written_and_not_silently_accepted(tmp_path, monkeypatch):
    archive, out = tmp_path / 'archive', tmp_path / 'out'
    out.mkdir()
    metrics = dict(candidate_cache={'candidate_sha256': 'cache'},
                   effective_grasp_pose=[[1]], transported_labels_sha256='labels')
    old = dict(outcome='success', metrics=metrics, initial_scene={'x': 0},
               forced_diagnostic=False, reproduction_index_match=True)
    new = dict(old, outcome='placed_in_the_wrong_place')
    def read_run(path):
        row = old if path.parent == archive else new
        return path, {'source_sha256': 'source'}, [row], {}
    monkeypatch.setattr(repro, 'read_run', read_run)
    assert not repro.compare_reproduction(archive, out, [0], ['joint_policy'])
    record = json.loads((out / 'reproduction_verification.json').read_text())[0]
    assert not record['all_matched']
    assert not record['checks']['outcome']
    assert record['checks']['candidates']


@pytest.mark.parametrize('relation', ['same', 'child', 'parent'])
def test_runner_refuses_archive_overlap(tmp_path, monkeypatch, relation):
    archive = tmp_path / 'archive'
    out = {'same': archive, 'child': archive / 'new', 'parent': tmp_path}[relation]
    monkeypatch.setattr('sys.argv', ['reproduce_evaluation.py', 'run', '--archive', str(archive), '--out', str(out)])
    with pytest.raises(SystemExit) as exc:
        repro.main()
    assert exc.value.code == 2
