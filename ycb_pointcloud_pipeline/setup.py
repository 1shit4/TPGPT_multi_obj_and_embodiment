import os
from glob import glob
from setuptools import setup

package_name = 'ycb_pointcloud_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.sh')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='krishnapranav',
    maintainer_email='you@example.com',
    description="RealSense D435 + Grounded-SAM point clouds for YAML-listed YCB objects",
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_node = ycb_pointcloud_pipeline.segmentation_node:main',
            'prompt_bench = ycb_pointcloud_pipeline.prompt_bench:main',
            'detection_replay = ycb_pointcloud_pipeline.prompt_bench:replay_main',
            'detection_sweep = ycb_pointcloud_pipeline.prompt_bench:sweep_main',
            'joint_target_publisher = ycb_pointcloud_pipeline.joint_target_publisher:main',
            'joint_homing = ycb_pointcloud_pipeline.joint_homing:main',
            'demo_franka = ycb_pointcloud_pipeline.demo_franka:main',
            'free_drive = ycb_pointcloud_pipeline.free_drive:main',
            'cartesian_target_publisher = ycb_pointcloud_pipeline.cartesian_target_publisher:main',
            'camera_tf_publisher = ycb_pointcloud_pipeline.camera_tf_publisher:main',
            'calibration_check_node = ycb_pointcloud_pipeline.calibration_check_node:main',
            'tpgpt_planner_node = ycb_pointcloud_pipeline.tpgpt_planner_node:main',
            'tpgpt_executor_node = ycb_pointcloud_pipeline.tpgpt_executor_node:main',
            'tpgpt_pick_place = ycb_pointcloud_pipeline.tpgpt_pick_place_node:main',
            'tpgpt_teach_node = ycb_pointcloud_pipeline.tpgpt_teach_node:main',
            'tpgpt_teach_keyboard = ycb_pointcloud_pipeline.tpgpt_teach_keyboard:main',
            'shelf_fit = ycb_pointcloud_pipeline.shelf_fit:main',
            'x6_session = ycb_pointcloud_pipeline.x6_session:main',
            'x6_order_sheet = ycb_pointcloud_pipeline.x6_session:order_sheet_main',
            'x6_instrument_check = ycb_pointcloud_pipeline.x6_instrument_check:main',
            'x6_analyse = ycb_pointcloud_pipeline.x6_analyse:main',
            'inspire_hand = ycb_pointcloud_pipeline.inspire_hand:main',
            'inspire_hand_node = ycb_pointcloud_pipeline.inspire_hand_node:main',
            'inspire_touch = ycb_pointcloud_pipeline.inspire_touch:main',
            'payload_measure = ycb_pointcloud_pipeline.payload_measure:main',
        ],
    },
)
