import math
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        self.declare_parameter('target_x', 3.0)
        self.declare_parameter('target_y', 1.0)

        self.target_x = float(self.get_parameter('target_x').value)
        self.target_y = float(self.get_parameter('target_y').value)
        self.kp_turn = 1.0
        self.kp_forward = 0.5

        self.declare_parameter('max_turn_speed', 0.3)
        self.declare_parameter('max_forward_speed', 0.10)

        self.max_turn_speed = float(
            self.get_parameter('max_turn_speed').value
        )
        self.max_forward_speed = float(
            self.get_parameter('max_forward_speed').value
        )

        if not (
            math.isfinite(self.max_turn_speed)
            and math.isfinite(self.max_forward_speed)
            and self.max_turn_speed > 0.0
            and self.max_forward_speed > 0.0
        ):
            raise ValueError('Speed limits must be finite positive numbers.')
        self.distance_tolerance = 0.05
        self.drive_angle_limit = math.radians(10.0)

        self.position = None
        self.last_odom_time = None
        self.finished = False

        self.cmd_publisher = self.create_publisher(
            TwistStamped, '/cmd_vel', 10
        )

        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f'Moving toward target: '
            f'x={self.target_x:.2f} m, y={self.target_y:.2f} m'
        )

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(sin_yaw, cos_yaw)

        self.position = (x, y, yaw)
        self.last_odom_time = time.monotonic()

    def send_velocity(self, forward, turn):
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = 'base_footprint'

        command.twist.linear.x = float(forward)
        command.twist.angular.z = float(turn)

        self.cmd_publisher.publish(command)

    def control_loop(self):
        if self.finished:
            self.send_velocity(0.0, 0.0)
            return

        if self.position is None:
            self.send_velocity(0.0, 0.0)
            self.get_logger().info(
                'Waiting for odometry...',
                throttle_duration_sec=2.0
            )
            return

        if time.monotonic() - self.last_odom_time > 1.0:
            self.send_velocity(0.0, 0.0)
            self.get_logger().warning(
                'Odometry stopped arriving. Sending stop.',
                throttle_duration_sec=2.0
            )
            return

        x, y, yaw = self.position

        dx = self.target_x - x
        dy = self.target_y - y
        distance = math.hypot(dx, dy)

        if distance <= self.distance_tolerance:
            self.send_velocity(0.0, 0.0)
            self.finished = True

            self.get_logger().info(
                f'Target reached within tolerance! '
                f'x={x:.3f} m, y={y:.3f} m, '
                f'remaining distance={distance:.3f} m'
            )
            return

        target_heading = math.atan2(dy, dx)
        error = target_heading - yaw
        error = math.atan2(math.sin(error), math.cos(error))

        turn_speed = self.kp_turn * error
        turn_speed = max(
            -self.max_turn_speed,
            min(self.max_turn_speed, turn_speed)
        )

        if abs(error) > self.drive_angle_limit:
            forward_speed = 0.0
            state = 'TURNING'
        else:
            forward_speed = min(
                self.max_forward_speed,
                self.kp_forward * distance
            )
            state = 'DRIVING'

        self.send_velocity(forward_speed, turn_speed)

        self.get_logger().info(
            f'{state}: distance={distance:.3f} m, '
            f'heading error={math.degrees(error):.1f} degrees, '
            f'forward={forward_speed:.3f} m/s',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            for _ in range(3):
                node.send_velocity(0.0, 0.0)
                time.sleep(0.05)

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
