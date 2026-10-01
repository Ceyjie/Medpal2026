#!/usr/bin/env python3
"""
bringup_scan.launch.py -- start static TF + depthimage_to_laserscan.

Publishes:
  /tf_static    base_link -> camera_link -> camera_depth_optical_frame
  /scan         sensor_msgs/LaserScan

Requires:
  /camera/depth/image_raw  (from astra_depth_publisher.py)
  /camera/depth/camera_info

Run inside the ROS 2 container:
    source /opt/ros/jazzy/setup.bash
    ros2 launch /workspace/ros2_bridge/bringup_scan.launch.py
"""

import math

from launch import LaunchDescription
from launch_ros.actions import Node


# Robot geometry -- match your physical build
CAMERA_HEIGHT_M = 0.20       # 20 cm above the base_link origin
CAMERA_TILT_DEG = 15.0       # degrees downward from horizontal


def generate_launch_description():
    # base_link -> camera_link: position + downward tilt around Y
    tilt_rad = math.radians(CAMERA_TILT_DEG)

    tf_base_to_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_camera_tf",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", str(CAMERA_HEIGHT_M),
            "--roll", "0.0",
            "--pitch", str(-tilt_rad),
            "--yaw", "0.0",
            "--frame-id", "base_link",
            "--child-frame-id", "camera_link",
        ],
    )

    # camera_link -> camera_depth_optical_frame: standard optical rotation
    tf_camera_to_optical = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_to_optical_tf",
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", "0.0",
            "--qx", "-0.5", "--qy", "0.5",
            "--qz", "-0.5", "--qw", "0.5",
            "--frame-id", "camera_link",
            "--child-frame-id", "camera_depth_optical_frame",
        ],
    )

    # depthimage_to_laserscan: depth image -> 2D LaserScan
    depth_to_scan = Node(
        package="depthimage_to_laserscan",
        executable="depthimage_to_laserscan_node",
        name="depthimage_to_laserscan",
        remappings=[
            ("depth", "/camera/depth/image_raw"),
            ("depth_camera_info", "/camera/depth/camera_info"),
            ("scan", "/scan"),
        ],
        parameters=[{
            "scan_height": 20,
            "output_frame": "camera_depth_optical_frame",
            "range_min": 0.3,
            "range_max": 5.0,
        }],
    )

    return LaunchDescription([
        tf_base_to_camera,
        tf_camera_to_optical,
        depth_to_scan,
    ])