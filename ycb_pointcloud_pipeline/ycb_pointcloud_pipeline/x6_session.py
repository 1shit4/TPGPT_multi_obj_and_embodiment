#!/usr/bin/env python3
"""X6 sessions and order sheets. No ROS.

    ros2 run ycb_pointcloud_pipeline x6_session start     # optional: groups trials into a session
    ros2 run ycb_pointcloud_pipeline x6_session status
    ros2 run ycb_pointcloud_pipeline x6_session end

    ros2 run ycb_pointcloud_pipeline x6_order_sheet --hand panda

`start` records the sha256 of the camera extrinsics file the planner loads; every
trial of the session checks the file still has that hash ("the extrinsics file is
the one calibrated this session"). Sessions are numbered s01, s02, ...; no dates are
written into the records.
"""

import argparse
import csv
import itertools
import os
import sys

import numpy as np

from ycb_pointcloud_pipeline import x6_common as xc


def start(config, verify_passed=None, notes=None):
    sessions = config.sessions_dir()
    os.makedirs(sessions, exist_ok=True)
    cur = xc.current_session(config)
    if cur is not None:
        print(f"Session {cur['session_id']} is still open; closing it first.")
        end(config)
    existing = sorted(d for d in os.listdir(sessions) if d.startswith('s') and d[1:].isdigit())
    sid = f's{len(existing) + 1:02d}'
    cal = xc.expand(config.raw['calibration']['camera_extrinsics_file'])
    if not os.path.isfile(cal):
        print(f'REFUSED: extrinsics file not found: {cal}')
        return 2
    # Calibration is verified manually; nothing is asked here. `verify_passed` is
    # kept only so existing callers keep working.
    if notes is None:
        try:
            notes = input('Session notes (who, what was checked; Enter = none): ').strip()
        except EOFError:
            notes = ''
    record = {'session_id': sid, 'open': True, 'extrinsics_file': cal,
              'extrinsics_sha256': xc.sha256_file(cal), 'verify_passed': None, 'notes': notes,
              'config_sha256': config.sha256}
    xc.write_json(os.path.join(sessions, sid, 'calibration.json'), record)
    xc.write_json(os.path.join(sessions, 'current.json'), {'session_id': sid})
    print(f"Session {sid} open. Extrinsics sha256 {record['extrinsics_sha256'][:16]}...")
    return 0


def end(config):
    cur = xc.current_session(config)
    if cur is None:
        print('No open session.')
        return 0
    path = os.path.join(config.sessions_dir(), cur['session_id'], 'calibration.json')
    cur['open'] = False
    xc.write_json(path, cur)
    print(f"Session {cur['session_id']} closed.")
    return 0


def status(config):
    cur = xc.current_session(config)
    print(f'Campaign root: {config.root}')
    if cur is None:
        print('No open session.')
        return 0
    cal = cur['extrinsics_file']
    same = os.path.isfile(cal) and xc.sha256_file(cal) == cur['extrinsics_sha256']
    print(f"Open session {cur['session_id']}; extrinsics file "
          f"{'unchanged' if same else 'CHANGED since session start'}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('action', choices=['start', 'end', 'status'])
    ap.add_argument('--config', default=None)
    a = ap.parse_args(argv if argv is not None else [x for x in sys.argv[1:] if x != '--ros-args'])
    config = xc.X6Config(a.config)
    return {'start': start, 'end': end, 'status': status}[a.action](config)


# ---------------------------------------------------------------------- order sheet
def order_sheet(config, hand):
    raw = config.raw
    seed = int(raw['grid']['seeds'][hand])
    lo, hi = (int(v) for v in raw['grid']['trial_ids'][hand].split('-'))
    objects = list(raw['objects'])
    marks = [int(p['index']) for p in raw['pick_poses']]
    slots = [k for k in raw['destinations'] if k != 'table_height']
    cells = list(itertools.product(objects, marks, slots))
    if len(cells) != hi - lo + 1:
        raise ValueError(f'{len(cells)} conditions but trial ids {lo}-{hi} give {hi - lo + 1}')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cells))
    width = len(str(hi)) if hi >= 1000 else 4
    rows = []
    for k, idx in enumerate(order):
        obj, mark, slot = cells[idx]
        rows.append({'trial_id': f'{lo + k:0{width}d}', 'hand': hand, 'object': obj,
                     'pick_mark': mark, 'destination': slot})
    return rows, seed


def order_sheet_main(argv=None):
    ap = argparse.ArgumentParser(description='Write the randomised order sheet for one hand.')
    ap.add_argument('--hand', required=True)
    ap.add_argument('--config', default=None)
    a = ap.parse_args(argv if argv is not None else [x for x in sys.argv[1:] if x != '--ros-args'])
    config = xc.X6Config(a.config)
    rows, seed = order_sheet(config, a.hand)
    out = os.path.join(config.root, 'orders', f'order_{a.hand}.csv')
    if os.path.exists(out):
        print(f'REFUSED: {out} exists. The order is generated once and never regenerated.')
        return 2
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} trials for {a.hand} (seed {seed}) -> {out}')
    for r in rows:
        print(f"  {r['trial_id']}  {r['object']:<11} mark {r['pick_mark']}  {r['destination']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
