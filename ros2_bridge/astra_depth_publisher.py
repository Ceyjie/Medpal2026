#!/usr/bin/env python3
"""
astra_depth_publisher.py -- publish Orbbec Astra depth as sensor_msgs/Image.

Topic: /camera/depth/image_raw  (encoding: 16UC1, units: mm)
Frame: camera_depth_optical_frame

Run inside the ROS 2 container:
    source /opt/ros/jazzy/setup.bash
    python3 /workspace/ros2_bridge/astra_depth_publisher.py
"""

import sys
import time

sys.path.insert(0, "/opt/sdk/pyorbbecsdk/build")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


class AstraDepthPublisher(Node):
    def __init__(self):
        super().__init__("astra_depth_publisher")

        self.pub_img = self.create_publisher(
            Image, "/camera/depth/image_raw", 10)
        self.pub_info = self.create_publisher(
            CameraInfo, "/camera/depth/camera_info", 10)

        self.pipeline = Pipeline()
        self.cfg = Config()

        profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR)
        self.profile = None
        for w, h, fps in [(640, 480, 15), (320, 240, 15),
                          (320, 240, 30), (160, 120, 15)]:
            try:
                self.profile = profiles.get_video_stream_profile(
                    w, h, OBFormat.Y16, fps)
                self.get_logger().info(
                    f"depth profile: {w}x{h}@{fps}")
                break
            except Exception:
                continue
        if self.profile is None:
            self.profile = profiles.get_default_video_stream_profile()
            self.get_logger().warn(
                f"using default depth profile: "
                f"{self.profile.get_width()}x"
                f"{self.profile.get_height()}")

        self.cfg.enable_stream(self.profile)
        self.pipeline.start(self.cfg)

        # Frame dimensions
        self.w = self.profile.get_width()
        self.h = self.profile.get_height()
        try:
            self.scale = self.pipeline.get_depth_frame_scale()
        except Exception:
            self.scale = 1.0

        # Publish at 15 Hz
        self.timer = self.create_timer(1.0 / 15.0, self.tick)

        # CameraInfo message (static; update if you ever calibrate)
        self.info_msg = CameraInfo()
        self.info_msg.header.frame_id = "camera_depth_optical_frame"
        self.info_msg.width = self.w
        self.info_msg.height = self.h
        # Rough Astra Pro intrinsics at depth resolution:
        # fx ~ 570 at 640 wide, so scale by width/640
        fx = 570.0 * self.w / 640.0
        fy = fx
        cx = self.w / 2.0
        cy = self.h / 2.0
        self.info_msg.k = [fx, 0.0, cx,
                           0.0, fy, cy,
                           0.0, 0.0, 1.0]
        # depthimage_to_laserscan requires these
        self.info_msg.distortion_model = "plumb_bob"
        self.info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.info_msg.r = [1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0]
        self.info_msg.p = [fx, 0.0, cx, 0.0,
                           0.0, fy, cy, 0.0,
                           0.0, 0.0, 1.0, 0.0]

        self.frame_count = 0
        self.get_logger().info("astra_depth_publisher started")

    def tick(self):
        try:
            frames = self.pipeline.wait_for_frames(500)
        except Exception as e:
            self.get_logger().warn(f"wait_for_frames error: {e}")
            return
        if frames is None:
            return
        depth = frames.get_depth_frame()
        if depth is None:
            return

        w = depth.get_width()
        h = depth.get_height()
        data = np.frombuffer(depth.get_data(), dtype=np.uint16)

        # Apply depth scale if not 1.0 (some SDK versions report mm via scale)
        if abs(self.scale - 1.0) > 1e-6:
            data = (data.astype(np.float32) * self.scale).astype(np.uint16)

        if data.size != w * h:
            # stride mismatch; be defensive
            stride = depth.get_stride() if hasattr(depth, "get_stride") else w * 2
            row_u16 = stride // 2
            data = data[: row_u16 * h].reshape(h, row_u16)[:, :w]
        else:
            data = data.reshape(h, w)

        stamp = self.get_clock().now().to_msg()

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_depth_optical_frame"
        msg.height = h
        msg.width = w
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = w * 2
        msg.data = data.tobytes()
        self.pub_img.publish(msg)

        self.info_msg.header.stamp = stamp
        self.pub_info.publish(self.info_msg)

        self.frame_count += 1
        if self.frame_count % 60 == 0:
            valid = data[(data > 0) & (data < 10000)]
            med = int(np.median(valid)) if valid.size else 0
            self.get_logger().info(
                f"frames={self.frame_count}  "
                f"valid={valid.size}/{data.size}  median={med}mm")


def main():
    rclpy.init()
    node = AstraDepthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pipeline.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
