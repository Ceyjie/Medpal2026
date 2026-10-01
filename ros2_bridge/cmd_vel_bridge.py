#!/usr/bin/env python3
"""
cmd_vel_bridge.py -- Subscribe to /cmd_vel, translate to SerialMotors commands.

Run inside the ROS 2 container:
    source /opt/ros/jazzy/setup.bash
    python3 /workspace/ros2_bridge/cmd_vel_bridge.py
"""

import sys
import os
import time

sys.path.insert(0, "/workspace")

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from serial_motors import SerialMotors


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__("cmd_vel_bridge")

        self.motors = SerialMotors()
        if not self.motors.available:
            self.get_logger().warn(
                "SerialMotors not available -- commands will be ignored")

        # Deadband and scaling
        self.LINEAR_DEADBAND = 0.01    # m/s below this -> treat as 0
        self.ANGULAR_DEADBAND = 0.05   # rad/s below this -> treat as 0
        self.MAX_LINEAR = 0.5          # m/s -> 100 % speed
        self.MAX_ANGULAR = 1.5         # rad/s -> full turn

        self.last_sent = None
        self.last_send_time = 0.0
        self.MIN_SEND_INTERVAL = 0.05  # 20 Hz max

        self.sub = self.create_subscription(
            Twist, "cmd_vel", self.on_cmd_vel, 10)
        self.get_logger().info("cmd_vel_bridge ready -- listening on /cmd_vel")

    def on_cmd_vel(self, msg):
        lin = msg.linear.x
        ang = msg.angular.z

        # Deadband
        if abs(lin) < self.LINEAR_DEADBAND:
            lin = 0.0
        if abs(ang) < self.ANGULAR_DEADBAND:
            ang = 0.0

        # Convert to a 3-state command: forward / backward / turn
        # Simple priority logic: turning dominates forward if both strong.
        command = self._decide(lin, ang)

        now = time.monotonic()
        if command == self.last_sent and now - self.last_send_time < 0.2:
            return  # don't spam the serial port
        self.last_sent = command
        self.last_send_time = now

        self._send(command)

    def _decide(self, lin, ang):
        """Map (lin, ang) to one of: stop/fwd/back/left/right with speed."""
        # If turning hard, turn in place
        if abs(ang) > 0.3:
            speed = int(min(100, abs(ang) / self.MAX_ANGULAR * 100))
            speed = max(30, speed)
            if ang > 0:
                return ("turn_left", speed)
            else:
                return ("turn_right", speed)

        # Pure forward / backward
        if lin > self.LINEAR_DEADBAND:
            speed = int(min(100, abs(lin) / self.MAX_LINEAR * 100))
            speed = max(30, speed)
            return ("forward", speed)
        if lin < -self.LINEAR_DEADBAND:
            speed = int(min(100, abs(lin) / self.MAX_LINEAR * 100))
            speed = max(30, speed)
            return ("backward", speed)

        # Slight turn: blend
        if abs(ang) > self.ANGULAR_DEADBAND:
            speed = 40
            return ("turn_left" if ang > 0 else "turn_right", speed)

        return ("stop", 0)

    def _send(self, command):
        kind, speed = command
        if not self.motors.available:
            return
        if kind == "stop":
            self.motors.stop()
        elif kind == "forward":
            self.motors.set_speed(speed)
            self.motors.forward()
        elif kind == "backward":
            self.motors.set_speed(speed)
            self.motors.backward()
        elif kind == "turn_left":
            self.motors.set_speed(speed)
            self.motors.turn_left()
        elif kind == "turn_right":
            self.motors.set_speed(speed)
            self.motors.turn_right()


def main():
    rclpy.init()
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.motors.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
