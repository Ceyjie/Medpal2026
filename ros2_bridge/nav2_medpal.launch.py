#!/usr/bin/env python3
"""
nav2_medpal.launch.py -- full MedPal stack.
"""
import os

from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node


URDF_FILE = "/workspace/ros2_bridge/robot.urdf"
PARAMS_FILE = "/workspace/ros2_bridge/nav2_params.yaml"


def generate_launch_description():
    urdf_content = ""
    if os.path.exists(URDF_FILE):
        with open(URDF_FILE) as f:
            urdf_content = f.read()

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": urdf_content}],
        output="screen",
    )

    depth_node = ExecuteProcess(
        cmd=["python3", "/workspace/ros2_bridge/astra_depth_publisher.py"],
        output="screen",
    )

    d2l = Node(
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
        output="screen",
    )

    odom_node = ExecuteProcess(
        cmd=["python3", "/workspace/ros2_bridge/fake_odom.py"],
        output="screen",
    )

    bridge_node = ExecuteProcess(
        cmd=["python3", "/workspace/ros2_bridge/cmd_vel_bridge.py"],
        output="screen",
    )

    slam = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        parameters=[{
            "use_sim_time": False,
            "base_frame": "base_link",
            "odom_frame": "odom",
            "map_frame": "map",
            "scan_topic": "/scan",
            "mode": "mapping",
            "resolution": 0.05,
            "max_laser_range": 5.0,
            "minimum_time_interval": 0.5,
            "transform_publish_period": 0.02,
            "map_update_interval": 1.0,
        }],
        output="screen",
    )

    controller = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        parameters=[PARAMS_FILE],
        remappings=[("cmd_vel", "cmd_vel_nav")],
        output="screen",
    )

    planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        parameters=[PARAMS_FILE],
        output="screen",
    )

    behaviors = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        parameters=[PARAMS_FILE],
        output="screen",
    )

    bt_nav = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        parameters=[PARAMS_FILE],
        output="screen",
    )

    smoother = Node(
        package="nav2_smoother",
        executable="smoother_server",
        name="smoother_server",
        parameters=[PARAMS_FILE],
        output="screen",
    )

    vel_smoother = Node(
        package="nav2_velocity_smoother",
        executable="velocity_smoother",
        name="velocity_smoother",
        parameters=[PARAMS_FILE],
        remappings=[
            ("cmd_vel", "cmd_vel_nav"),
            ("cmd_vel_smoothed", "cmd_vel"),
        ],
        output="screen",
    )

    # --- Lifecycle managers: activate the Nav2 nodes ---
    lifecycle_nav = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        parameters=[{
            "use_sim_time": False,
            "autostart": True,
            "node_names": [
                "controller_server",
                "smoother_server",
                "planner_server",
                "behavior_server",
                "bt_navigator",
                "velocity_smoother",
            ],
            "bond_timeout": 0.0,
        }],
        output="screen",
    )

    lifecycle_slam = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_slam",
        parameters=[{
            "use_sim_time": False,
            "autostart": True,
            "node_names": ["slam_toolbox"],
            "bond_timeout": 0.0,
        }],
        output="screen",
    )

    ld = LaunchDescription()

    # Bring up in stages so TF and depth come up first.
    ld.add_action(rsp)
    ld.add_action(depth_node)
    ld.add_action(TimerAction(period=1.0, actions=[d2l]))
    ld.add_action(TimerAction(period=3.0, actions=[odom_node]))
    ld.add_action(TimerAction(period=3.0, actions=[bridge_node]))
    ld.add_action(TimerAction(period=4.0, actions=[slam]))
    ld.add_action(TimerAction(period=5.0, actions=[
        controller, planner, behaviors, bt_nav, smoother, vel_smoother]))
    # Lifecycle managers start after their nodes
    ld.add_action(TimerAction(period=8.0, actions=[lifecycle_nav]))
    ld.add_action(TimerAction(period=8.0, actions=[lifecycle_slam]))

    return ld
