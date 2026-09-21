"""Running a transported policy on a real Franka FR3 (paper Sec. V, on hardware).

This package is the bridge between the validated simulation pipeline and a
real robot. It deliberately contains **no ROS and no MuJoCo imports**, so every
piece of arithmetic that decides where the arm goes can be unit-tested offline.

The split mirrors ``tpgpt.experiments.pipeline.run``:

* :mod:`tpgpt.hardware.source_bundle` -- the single source demonstration and
  its scene description, expressed in the robot's **base** frame. Written either
  by ``export_source`` (scripted in MuJoCo) or by :mod:`tpgpt.hardware.teach`
  (taught by hand on the FR3, from the recording ``tpgpt_teach_node`` makes).
* :mod:`tpgpt.hardware.planner` -- keypoints from a **real** object point cloud,
  the transportation map and the transported labels. The same
  ``scene_keypoints`` / ``TransportMap`` / ``transport_labels`` calls the
  simulation pipeline makes.
* :mod:`tpgpt.hardware.executor` -- the per-step execution loop of
  ``tpgpt.sim.rollout.rollout_policy``, built from the same
  ``gate_step`` / ``advance_step`` / ``advance_phase`` / ``StallWatchdog``
  functions, with the measured pose coming from the robot instead of MuJoCo.
* :mod:`tpgpt.hardware.frames` -- rigid transforms and the grasp-convention to
  ``fr3_hand_tcp`` rotation.

The ROS 2 nodes that use these live in the ``ycb_pointcloud_pipeline`` package.
See ``docs/hardware_integration.md``.
"""
