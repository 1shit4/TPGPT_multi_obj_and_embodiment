#!/usr/bin/env python3
"""X6 instrument check (brief §4): how repeatable is the placement measurement?

    ros2 run --prefix python3 ycb_pointcloud_pipeline x6_instrument_check --ros-args \
        -p when:=before_first_trial          # | after_remount | campaign_end

The object named in the config (instrument_check.object, default can) sits in the
slot and is not touched. Each of the `repetitions` measurements is the SAME
procedure as a trial's settled measurement: `n_frames` fresh segmentations,
fused with the planner's filter, cropped to settled_workspace_min/max, centroid
in the robot base frame. The standard deviation of those centroids is the
instrument's repeatability. It says nothing about bias (a partial view's centroid
is not the object's centre); that is stated in the analysis.

Output: <root>/instrument/<when>/rep_00 .. rep_NN/ and instrument.json.
"""

import os
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from tf2_ros import Buffer, TransformListener

from ycb_pointcloud_pipeline import x6_common as xc


def main(args=None):
    argv = sys.argv if args is None else args
    when = xc.cli_parameter_value(argv, 'when')
    allowed = None
    if when is None:
        print('Give -p when:=before_first_trial | after_remount | campaign_end')
        raise SystemExit(2)
    config = xc.X6Config(xc.cli_parameter_value(argv, 'x6_config'))
    ic = config.raw['instrument_check']
    allowed = ic['when']
    if when not in allowed:
        print(f'when must be one of {allowed}')
        raise SystemExit(2)
    session = xc.current_session(config)   # optional
    out = os.path.join(config.root, 'instrument', when)
    k = 1
    base = out
    while os.path.exists(out):
        out = f'{base}_repeat{k}'
        k += 1

    signal.signal(signal.SIGINT, signal.default_int_handler)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    from ycb_pointcloud_pipeline.ros_utils import CalibrationProvider
    from ycb_pointcloud_pipeline.x6_capture import X6CaptureNode, fuse_frames, save_frames

    obj = ic.get('object', 'can')
    topics = config.raw['experiment']['topics']
    cap = X6CaptureNode(obj, topics)
    helper = Node('x6_instrument_tf')
    tf_buffer = Buffer()
    TransformListener(tf_buffer, helper)
    cal_file = xc.expand(config.raw['calibration']['camera_extrinsics_file'])
    calib = CalibrationProvider(helper, cal_file, tf_buffer)
    ex = SingleThreadedExecutor()
    ex.add_node(cap)
    ex.add_node(helper)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            ex.spin_once(timeout_sec=0.05)
    threading.Thread(target=spin, daemon=True).start()

    per, oc = config.method['perception'], config.method['outcome']
    n_frames, reps = int(per['n_frames']), int(ic['repetitions'])
    centroids, counts = [], []
    status = 'complete'
    try:
        print(f"Instrument check '{when}': {reps} measurements of the stationary {obj} "
              f"at {ic.get('location')}. Do not touch it.")
        time.sleep(1.0)
        for r in range(reps):
            frames, freshness = cap.capture(n_frames, float(oc.get('capture_timeout_s', 20.0)))
            rep_dir = os.path.join(out, f'rep_{r:02d}')
            save_frames(rep_dir, frames, calib)
            fused, stats = fuse_frames(frames, calib, per,
                                       [float(v) for v in oc['settled_workspace_min']],
                                       [float(v) for v in oc['settled_workspace_max']])
            np.save(os.path.join(rep_dir, 'cloud.npy'), np.asarray(fused, dtype=np.float32))
            if len(fused):
                c = np.asarray(fused).mean(axis=0)
                centroids.append(c)
                counts.append(int(len(fused)))
                print(f'  rep {r:02d}: {len(fused)} points, centroid '
                      f'[{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}] m ({freshness})')
            else:
                counts.append(0)
                print(f'  rep {r:02d}: object NOT seen ({freshness})')
    except KeyboardInterrupt:
        status = 'interrupted'
    finally:
        stop.set()

    C = np.asarray(centroids)
    result = {'when': when, 'object': obj, 'location': ic.get('location'),
              'session_id': session['session_id'] if session else None, 'status': status,
              'repetitions_requested': reps, 'repetitions_measured': int(len(C)),
              'points_per_rep': counts, 'centroids_m': C.tolist(),
              'extrinsics_sha256': xc.sha256_file(cal_file),
              'procedure': 'n_frames fresh segmentations, fused like the planner, cropped to '
                           'settled_workspace, centroid in base frame',
              'provenance': xc.provenance(config)}
    if len(C) >= 2:
        sd = C.std(axis=0, ddof=1) * 1000.0
        mean = C.mean(axis=0)
        lateral_dev = np.hypot(C[:, 0] - mean[0], C[:, 1] - mean[1]) * 1000.0
        result.update({
            'mean_centroid_m': mean.tolist(),
            'sd_mm': {'x': float(sd[0]), 'y': float(sd[1]), 'z': float(sd[2])},
            'sd_lateral_mm': float(np.hypot(sd[0], sd[1])),
            'max_lateral_deviation_from_mean_mm': float(lateral_dev.max()),
            'sd_definition': 'sample SD (ddof=1) per axis; lateral = sqrt(sd_x^2 + sd_y^2)',
        })
        print(f"\nSD x {sd[0]:.2f}  y {sd[1]:.2f}  z {sd[2]:.2f} mm;  lateral "
              f"{result['sd_lateral_mm']:.2f} mm  (n = {len(C)})")
    else:
        print('\nFewer than 2 measurements: no standard deviation.')
    xc.write_json(os.path.join(out, 'instrument.json'), result)
    print(f'Written: {out}/instrument.json')
    cap.destroy_node()
    helper.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
