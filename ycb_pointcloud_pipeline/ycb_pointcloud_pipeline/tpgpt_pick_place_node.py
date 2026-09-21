#!/usr/bin/env python3
"""
Plan and execute in one run: tpgpt_planner_node, then tpgpt_executor_node on its plan.

    wait for the first object cloud (fuses up to n_frames that have arrived)
      -> TpgptPlannerNode._plan()                 (saved to <output_dir>/<object>_<time>/)
      -> TpgptExecutorNode(plan_dir=<that plan>)  (preflight, summary)
      -> operator types 'yes'                     (the executor's confirmation prompt)
      -> execute

Planning failures, a refused prompt and aborts before anything moved offer a
re-plan (e.g. after moving the object). The planner stops spinning while the
executor runs, so its cloud callbacks do not compete with the control loop for
the GIL; its latched RViz topics stay up.

Both nodes live in this process, so `-p name:=value` reaches both; parameters only
one of them declares (object_name, destination, dry_run, gripper_mode, ...) are
ignored by the other. `workspace_min` / `workspace_max` are declared by both.

    source ~/venvs/tpgpt/bin/activate
    ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
        -p source_bundle_dir:=$HOME/tpgpt_demos/<demo>/source_bundle \
        -p object_name:=banana -p destination:="[0.60, -0.25, 0.006]" \
        -p dry_run:=false -p gripper_mode:=none

Add `-p top_down_grasp:=true` to use the scripted top-down grasp and never call
GraspGen-X.
"""

import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.parameter import Parameter

from ycb_pointcloud_pipeline.tpgpt_executor_node import AbortRun, TpgptExecutorNode
from ycb_pointcloud_pipeline.tpgpt_planner_node import TpgptPlannerNode

CLOUD_TIMEOUT = 5.0   # s to wait for the first object cloud before trying anyway


def ask(prompt):
    try:
        return input(prompt).strip().lower() in ('y', 'yes')
    except EOFError:
        return False


def wait_for_first_cloud(planner, timeout):
    """Return as soon as one object cloud has arrived; plan with whatever is there then."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with planner.lock:
            if planner.clouds:
                return
        time.sleep(0.05)
    planner.get_logger().warn(f'no object cloud after {timeout:.0f} s; trying to plan anyway')


def main(args=None):
    # experiment:=true runs one recorded X6 trial instead (x6_trial.py); without it
    # nothing below changes.
    import sys
    from ycb_pointcloud_pipeline.x6_common import experiment_requested
    if experiment_requested(sys.argv if args is None else args):
        from ycb_pointcloud_pipeline.x6_trial import main as x6_main
        return x6_main(args)
    rclpy.init(args=args)
    planner = TpgptPlannerNode()
    log = planner.get_logger()
    planner_exec = MultiThreadedExecutor(num_threads=3)
    planner_exec.add_node(planner)
    planner_spinning = threading.Event()
    stop = threading.Event()

    def spin_planner():
        while not stop.is_set():
            if planner_spinning.is_set():
                planner_exec.spin_once(timeout_sec=0.1)
            else:
                time.sleep(0.05)

    planner_spinning.set()
    planner_thread = threading.Thread(target=spin_planner, daemon=True)
    planner_thread.start()

    node = node_exec = node_thread = None
    exit_code = 0
    try:
        while True:
            planner_spinning.set()
            wait_for_first_cloud(planner, CLOUD_TIMEOUT)
            ok, text = planner._plan()
            if not ok:
                log.error(f'Planning failed: {text}')
                if ask('Re-plan? [y/N]: '):
                    continue
                exit_code = 1
                break
            print(f'\nPLAN READY: {planner.plan_dir}\nInspect it in RViz (/tpgpt/plan/*) before '
                  'confirming below.')

            planner_spinning.clear()
            node = TpgptExecutorNode(parameter_overrides=[
                Parameter('plan_dir', Parameter.Type.STRING, planner.plan_dir)])
            node_exec = SingleThreadedExecutor()
            node_exec.add_node(node)
            node_thread = threading.Thread(target=node_exec.spin, daemon=True)
            node_thread.start()
            try:
                executed = node.run()
            except AbortRun as exc:
                node.get_logger().error(f'Aborted: {exc}')
                if node.moving:
                    node.hold_current_pose()
                    exit_code = 1
                    break
                executed = False
            if executed:
                break
            time.sleep(0.2)
            node_exec.shutdown()
            node_thread.join(timeout=2.0)
            node.destroy_node()
            node = None
            if not ask('Nothing moved. Re-plan? [y/N]: '):
                break
    except KeyboardInterrupt:
        if node is not None and node.moving:
            node.hold_current_pose()
    finally:
        time.sleep(0.2)  # let a final hold command go out
        if node is not None:
            node_exec.shutdown()
            node_thread.join(timeout=2.0)
            node.destroy_node()
        stop.set()
        planner_thread.join(timeout=2.0)
        planner_exec.shutdown()
        planner.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
