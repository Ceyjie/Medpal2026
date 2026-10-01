#!/usr/bin/env python3
"""
fake_odom.py -- publish nav_msgs/Odometry by integrating /cmd_vel.

Placeholder odometry until real encoders are wired on the ESP32.
Assumes perfect velocity tracking.

Publishes:
  /odom (nav_msgs/Odometry)
  TF: odom -> base_link

Subscribes:
  /cmd_vel (geometry_msgs/Twist)
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class FakeOdom(Node):
    def __init__(self):
        super().__init__("fake_odom")
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.wz = 0.0
        self.last_time = self.get_clock().now()

        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_br = TransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_timer(1.0 / 30.0, self.tick)
        self.get_logger().info("fake_odom started")

    def on_cmd(self, msg):
        self.vx = msg.linear.x
        self.wz = msg.angular.z

    def tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0 or dt > 0.5:
            return

        self.x += self.vx * math.cos(self.theta) * dt
        self.y += self.vx * math.sin(self.theta) * dt
        self.theta += self.wz * dt

        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi

        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.angular.z = self.wz

        for i in range(36):
            odom.pose.covariance[i] = 0.0
            odom.twist.covariance[i] = 0.0
        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.10
        odom.twist.covariance[0] = 0.05
        odom.twist.covariance[35] = 0.10

        self.pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_br.sendTransform(t)


def main():
    rclpy.init()
    node = FakeOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()