"""
Inspire RH56 hand driver on its own: /inspire_hand/open, /inspire_hand/close (Trigger),
/inspire_hand/joint_states, /inspire_hand/grasp_state.

  ros2 launch ycb_pointcloud_pipeline inspire_hand.launch.py
  ros2 launch ycb_pointcloud_pipeline inspire_hand.launch.py port:=/dev/ttyUSB1 dry_run:=true

Port, hand ID, speed and force default to config/inspire_hand.yaml.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('ycb_pointcloud_pipeline')
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument('config_file',
                              default_value=os.path.join(share, 'config', 'inspire_hand.yaml')),
        DeclareLaunchArgument('port', default_value='', description="'' = config serial.port"),
        DeclareLaunchArgument('hand_id', default_value='-1', description='-1 = config serial.hand_id'),
        DeclareLaunchArgument('dry_run', default_value='false'),
        Node(package='ycb_pointcloud_pipeline', executable='inspire_hand_node', output='screen',
             emulate_tty=True,
             parameters=[{'config_file': arg('config_file'), 'port': arg('port'),
                          'hand_id': ParameterValue(arg('hand_id'), value_type=int),
                          'dry_run': ParameterValue(arg('dry_run'), value_type=bool)}]),
    ])
