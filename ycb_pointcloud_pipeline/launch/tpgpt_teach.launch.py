"""
Teach the TPGPT source demonstration by hand on the FR3. Moves the arm only in
the sense that it FLOATS (zero stiffness) while teaching.

Starts
  camera_tf_publisher   base -> camera from config/camera_calibration.yaml
  tpgpt_teach_node      records the demonstration, converts it to a source bundle,
                        plots it in RViz (/tpgpt/teach/*)
  rviz2                 (rviz:=true) fixed frame `base`

Assumes, each in its own terminal:
  the robot container with FCI active
    cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot
  camera + segmentation (ycb venv, it needs torch; the TPGPT venv has none)
    ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py

The teach node imports tpgpt, so activate the TPGPT venv before launching (it is
started with python3 from PATH). Drive it from a second terminal:
  ros2 run ycb_pointcloud_pipeline tpgpt_teach_keyboard

  ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py object_name:=mustard_bottle
  ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py teach_mode:=passive

Inspire hand mounted (starts inspire_hand_node too; g = close, r = open):
  ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py object_name:=tomato_soup_can \
      end_effector:=inspire gripper_mode:=inspire
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('ycb_pointcloud_pipeline')
    arg = LaunchConfiguration
    args = [
        DeclareLaunchArgument('calibration',
                              default_value=os.path.join(share, 'config', 'camera_calibration.yaml')),
        DeclareLaunchArgument('object_name', default_value='mustard_bottle',
                              description='Object in target_objects.yaml; clouds from '
                                          '/ycb_pointclouds/<object_name>'),
        DeclareLaunchArgument('teach_mode', default_value='compliant',
                              description="'compliant' (zero-stiffness CRISP controller) or "
                                          "'passive' (nothing switched; Franka guiding mode)"),
        DeclareLaunchArgument('gripper_mode', default_value='franka_grasp',
                              description="'franka_grasp' (g closes, r opens the hand), "
                                          "'inspire' (inspire_hand_node services) or 'none'"),
        DeclareLaunchArgument('end_effector', default_value='panda',
                              description="'panda' (Franka Hand) or 'inspire' (Inspire RH56, "
                                          'palm point from config/inspire_hand.yaml)'),
        DeclareLaunchArgument('end_effector_config',
                              default_value=os.path.join(share, 'config', 'inspire_hand.yaml')),
        DeclareLaunchArgument('output_dir', default_value=os.path.expanduser('~/tpgpt_demos')),
        DeclareLaunchArgument('publish_tf', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
    ]
    return LaunchDescription(args + [
        Node(package='ycb_pointcloud_pipeline', executable='camera_tf_publisher',
             output='screen', parameters=[{'calibration_yaml': arg('calibration')}],
             condition=IfCondition(arg('publish_tf'))),
        Node(package='ycb_pointcloud_pipeline', executable='tpgpt_teach_node',
             output='screen', emulate_tty=True,
             # python3 from PATH = the activated TPGPT venv, not colcon's shebang.
             prefix='python3',
             parameters=[{
                 'calibration_yaml': arg('calibration'),
                 'object_name': arg('object_name'),
                 'teach_mode': arg('teach_mode'),
                 'gripper_mode': arg('gripper_mode'),
                 'end_effector': arg('end_effector'),
                 'end_effector_config': arg('end_effector_config'),
                 'gripper_namespace': PythonExpression([
                     "'/inspire_hand' if '", arg('gripper_mode'), "' == 'inspire' "
                     "else '/franka_gripper'"]),
                 'output_dir': arg('output_dir'),
             }]),
        Node(package='ycb_pointcloud_pipeline', executable='inspire_hand_node',
             output='screen', emulate_tty=True,
             parameters=[{'config_file': arg('end_effector_config')}],
             condition=IfCondition(PythonExpression(["'", arg('gripper_mode'), "' == 'inspire'"]))),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(share, 'rviz', 'tpgpt.rviz')],
             condition=IfCondition(arg('rviz'))),
    ])
