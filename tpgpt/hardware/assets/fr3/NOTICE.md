# FR3 description (vendored)

`fr3.urdf` is the robot description the Thor's robot container published on `/robot_description`
on 2026-09-21 (xacro output of `franka_description/robots/fr3/fr3.urdf.xacro`, franka_description
1.0.2, with the Franka Hand). `collision/*.stl` are that package's collision meshes
(`meshes/robot_arms/fr3/collision/link0-7.stl`, `meshes/robot_ee/franka_hand_white/collision/hand.stl`),
copied unchanged from the `docker-thor:local` image.

franka_description is (c) Franka Robotics GmbH, licensed Apache-2.0.

Used by `tpgpt/hardware/fr3.py` for kinematics (joint origins, axes, limits) and collision
geometry (one convex hull per link, as MuJoCo would use for a mesh geom). If the robot's
description changes, refresh both from the container and rerun `tests/unit/test_fr3.py`.
