"""Verify and reproduce the frozen 120-cell study without new grasp draws.

See README.md, "Reproduce the frozen simulation results". Run from any directory;
relative CLI paths are resolved against the caller's working directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.paper_results import read_run, export, plot_results
from tpgpt.experiments.audit_artifacts import atomic_json, load_source, sha256

EXECUTORS = ('joint_policy', 'replay')


def parse_indices(value, count=120):
    if value == 'all':
        return list(range(count))
    try:
        result = [int(x) for x in value.split(',')]
    except ValueError as exc:
        raise ValueError('--cells must be all or comma-separated zero-based indices') from exc
    if not result or len(set(result)) != len(result) or any(i < 0 or i >= count for i in result):
        raise ValueError(f'cell indices must be distinct and in [0, {count-1}]')
    return result


def mapped_path(name, mappings):
    path = Path(name)
    for old, new in mappings:
        if path.is_relative_to(old):
            return new / path.relative_to(old)
    return path if path.is_absolute() else REPO / path


def verify_archive(archive, mappings=()):
    import numpy as np
    protocol = json.loads((archive / 'protocol.json').read_text())
    for name, digest in protocol['code_sha256'].items():
        if sha256(REPO / name) != digest:
            raise ValueError(f'frozen code differs: {name}; use the archived code snapshot in a separate checkout')
    for name, digest in protocol['cells_sha256'].items():
        if sha256(archive / name) != digest:
            raise ValueError(f'cell manifest changed: {name}')
    for filename in ('asset_inventory.json', 'grasp_assets_inventory.json'):
        inventory = json.loads((archive / filename).read_text())
        for name, digest in inventory.get('files', inventory).items():
            path = mapped_path(name, mappings)
            if sha256(path) != digest:
                raise ValueError(f'asset differs: {path}')
    load_source(archive / 'source')
    if sha256(archive / 'source/source.json') != protocol['source_manifest_sha256']:
        raise ValueError('source manifest differs')
    runs = {name: read_run(archive / name) for name in EXECUTORS}
    seen = set()
    for name, (_, config, rows, summary) in runs.items():
        if len(rows) != 120 or summary['scheduled'] != 120:
            raise ValueError(f'expected complete 120-cell archive: {name}')
        if config['source_sha256'] != protocol['source_manifest_sha256']:
            raise ValueError(f'source mismatch: {name}')
        for row in rows:
            if row.get('forced_diagnostic'):
                raise ValueError('headline archive contains forced execution')
            cache = row['metrics']['candidate_cache']
            key = cache['grasp_cache_key']
            pair = (key, cache['candidate_sha256'])
            if pair in seen:
                continue
            with np.load(archive / 'candidates' / (key + '.npz')) as data:
                digest = hashlib.sha256(np.asarray(data['poses'], dtype='<f8').tobytes()
                                        + np.asarray(data['scores'], dtype='<f8').tobytes()).hexdigest()
            if digest != pair[1]:
                raise ValueError(f'candidate cache differs: {key}')
            seen.add(pair)
    return runs


def pinned_cells(rows, indices):
    cells = []
    for index in indices:
        row = rows[index]
        metrics = row['metrics']
        cell = dict(row['cell'])
        cell.update(expected_index=metrics['grasp_chosen_index'],
                    expected_cache_hash=metrics['candidate_cache']['candidate_sha256'],
                    expected_effective_pose=metrics['effective_grasp_pose'],
                    expected_labels_hash=metrics['transported_labels_sha256'])
        if 'forced_grasp_index' in cell:
            raise ValueError('reproduction must use normal selection')
        cells.append(cell)
    return cells


def write_unchanged(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'resume inputs changed: {path}; use a new output directory')
    atomic_json(path, value)


def compare_reproduction(archive, out, indices, names):
    records = []
    for name in names:
        _, config, rows, _ = read_run(out / name)
        original = read_run(archive / name)
        if len(rows) != len(indices):
            raise ValueError('reproduction row count differs from requested cells')
        for index, row in zip(indices, rows):
            old = original[2][index]
            a, b = old.get('metrics', {}), row.get('metrics', {})
            checks = dict(outcome=old['outcome'] == row['outcome'],
                          index=row.get('reproduction_index_match') is True,
                          source=config['source_sha256'] == original[1]['source_sha256'],
                          candidates=a.get('candidate_cache', {}).get('candidate_sha256') == b.get('candidate_cache', {}).get('candidate_sha256'),
                          effective_pose=a.get('effective_grasp_pose') == b.get('effective_grasp_pose'),
                          labels=a.get('transported_labels_sha256') == b.get('transported_labels_sha256'),
                          initial_scene=old.get('initial_scene') == row.get('initial_scene'),
                          normal_selection=row.get('forced_diagnostic') is False)
            records.append(dict(executor=name, evaluation_index=index, checks=checks,
                                all_matched=all(checks.values())))
    atomic_json(out / 'reproduction_verification.json', records)
    print(f"Matched {sum(r['all_matched'] for r in records)}/{len(records)} reproduced cells")
    return all(r['all_matched'] for r in records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('verify', 'run', 'report'))
    p.add_argument('--archive', type=Path, default=REPO / 'outputs/paper_evaluation_v1')
    p.add_argument('--out', type=Path, help='separate reproduction directory; required for run/report')
    p.add_argument('--cells', default='all', help='all or comma-separated original indices, e.g. 0,7,67,113')
    p.add_argument('--executor', choices=('both',) + EXECUTORS, default='both')
    p.add_argument('--render', action='store_true', help='record two synchronized cameras; slower')
    p.add_argument('--asset-map', action='append', default=[], metavar='OLD=NEW',
                   help='relocate an inventory path prefix; also set runtime asset environment variables')
    args = p.parse_args()
    archive = args.archive.resolve()
    out = args.out.resolve() if args.out else None
    mappings = []
    for item in args.asset_map:
        old, sep, new = item.partition('=')
        if not sep or not old or not new:
            p.error('--asset-map requires OLD=NEW')
        mappings.append((Path(old), Path(new).resolve()))
    os.chdir(REPO)
    if args.action == 'verify':
        runs = verify_archive(archive, mappings)
        print('Verified frozen code, cells, assets, source and candidate caches.')
        for name, run in runs.items():
            print(f"{name}: {run[3]['successes']}/120 ({run[3]['overall_success']:.1%})")
        return
    if out is None:
        p.error('--out is required')
    if out == archive or out.is_relative_to(archive) or archive.is_relative_to(out):
        p.error('--out must be separate from the archive, not its ancestor or descendant')
    if args.action == 'report':
        meta = json.loads((out / 'reproduction.json').read_text())
        if meta['archive'] != str(archive):
            p.error('report archive differs from the reproduction manifest')
        names, indices = meta['executors'], meta['indices']
    else:
        runs = verify_archive(archive, mappings)
        indices = parse_indices(args.cells)
        names = list(EXECUTORS) if args.executor == 'both' else [args.executor]
        if out.exists() and not (out / 'reproduction.json').exists():
            p.error('existing output is not a reproduction; use a new output directory')
        out.mkdir(parents=True, exist_ok=True)
        meta = dict(archive=str(archive), indices=indices, executors=names, render=args.render,
                    source_sha256=sha256(archive / 'source/source.json'), variant='task',
                    cache_only=True)
        write_unchanged(out / 'reproduction.json', meta)
        env = os.environ.copy()
        env.update(TPGPT_GRASP_CACHE=str(archive / 'candidates'), PYTHONPATH=str(REPO),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        env.setdefault('MUJOCO_GL', 'egl')
        env.setdefault('NUMBA_CACHE_DIR', '/tmp/tpgpt-numba')
        env.setdefault('MPLCONFIGDIR', '/tmp/tpgpt-mpl')
        for name in names:
            cells_path = out / f'cells_{name}.json'
            write_unchanged(cells_path, pinned_cells(runs[name][2], indices))
            cmd = [sys.executable, '-m', 'tpgpt.experiments.forensics', str(cells_path),
                   str(out / name), '--source', str(archive / 'source'),
                   '--variant', 'task', '--executor', name]
            if args.render:
                cmd.append('--render')
            print(f'Running {name}: {len(indices)} cells; log {out / (name + ".log")}', flush=True)
            with (out / (name + '.log')).open('a') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    loaded = export({name: out / name for name in names}, out / 'report')
    plot_results(loaded, out / 'report')
    if not compare_reproduction(archive, out, indices, names):
        raise SystemExit('Reproduction mismatch: inspect reproduction_verification.json; outcomes were preserved.')


if __name__ == '__main__':
    main()
