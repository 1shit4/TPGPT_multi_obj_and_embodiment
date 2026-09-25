"""
Two nodes:
  1. realsense2_camera  -- reads the D435 and publishes color + depth aligned to color
  2. ycb_segmentation_node -- every `period` seconds detects the YAML objects and
     publishes one PointCloud2 per object on /ycb_pointclouds/<name>

Examples:
  ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py
  ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py period:=5.0 detector_mode:=dummy
  (profile defaults to 848,480,30: camera_calibration.yaml intrinsics are for 848x480)
  ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py targets:=/path/my_objects.yaml
  ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py launch_camera:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = get_package_share_directory('ycb_pointcloud_pipeline')
    args = [
        DeclareLaunchArgument('launch_camera', default_value='true',
                              description='Start the realsense2_camera node'),
        DeclareLaunchArgument('serial_no', default_value="''",
                              description='RealSense serial number (empty = first found)'),
        DeclareLaunchArgument('profile', default_value='848,480,30',
                              description='Color and depth stream profile W,H,FPS'),
        DeclareLaunchArgument('targets',
                              default_value=os.path.join(share, 'config', 'target_objects.yaml')),
        DeclareLaunchArgument('period', default_value='3.0',
                              description='Seconds between detections / point cloud publishes'),
        DeclareLaunchArgument('detector_mode', default_value='grounded_sam',
                              description="'grounded_sam' or 'dummy' (no models, plumbing test)"),
        DeclareLaunchArgument('caption_mode', default_value='joint',
                              description="'joint' (one caption, one pass) or 'per_object' "
                                          "(a caption each, one pass each -- no phrase competes)"),
        DeclareLaunchArgument('save_frames', default_value='off',
                              description="'misses' keeps every frame where a target reached no "
                                          "topic, 'all' keeps all: replay them with prompt_bench"),
        DeclareLaunchArgument('save_frames_dir',
                              default_value=os.path.expanduser('~/ycb_detection_frames')),
    ]

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        launch_arguments={
            'serial_no': LaunchConfiguration('serial_no'),
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'false',
            'rgb_camera.color_profile': LaunchConfiguration('profile'),
            'depth_module.depth_profile': LaunchConfiguration('profile'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('launch_camera')),
    )

    segmentation = Node(
        package='ycb_pointcloud_pipeline',
        executable='segmentation_node',
        name='ycb_segmentation_node',
        output='screen',
        # Run with python3 from PATH (the activated venv, which has torch) instead
        # of the script's shebang, which is whatever Python colcon was run with.
        prefix='python3',
        parameters=[{
            'color_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'target_objects_yaml': LaunchConfiguration('targets'),
            'process_period_sec': ParameterValue(LaunchConfiguration('period'), value_type=float),
            'detector_mode': LaunchConfiguration('detector_mode'),
            # value_type=str: launch passes these through a YAML file, where a bare
            # 'off' (or 'on', 'yes', 'no') reads as a boolean and the node, which
            # declared a string, refuses to start.
            'caption_mode': ParameterValue(LaunchConfiguration('caption_mode'), value_type=str),
            'save_frames': ParameterValue(LaunchConfiguration('save_frames'), value_type=str),
            'save_frames_dir': ParameterValue(LaunchConfiguration('save_frames_dir'),
                                              value_type=str),
        }],
    )

    return LaunchDescription(args + [camera, segmentation])
