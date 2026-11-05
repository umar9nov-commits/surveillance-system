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

        # Speed limits supplied when starting the node.
        self.declare_parameter('max_forward_speed', 0.10)
        self.declare_parameter('max_turn_speed', 0.3)

        self.max_forward_speed = float(
            self.get_parameter('max_forward_speed').value
        )
        self.max_turn_speed = float(
            self.get_parameter('max_turn_speed').value
        )

        if not (
            math.isfinite(self.max_forward_speed)
            and math.isfinite(self.max_turn_speed)
            and self.max_forward_speed > 0.0
            and self.max_turn_speed > 0.0
        ):
            raise ValueError(
                'Speed limits must be finite positive numbers.'
            )

        # Coordinates are in the odometry frame, in metres.
        self.waypoints = [
            (1.0, 1.0),  # A
            (3.0, 1.0),  # B
            (3.0, 3.0),  # C
            (1.0, 3.0),  # D
            (1.0, 1.0),  # Return to A
        ]

        self.waypoint_index = 0
        self.pause_until = None
        self.pause_duration = 2.0
        self.finished = False

        # Proportional controller settings.
        self.kp_turn = 1.0
        self.kp_forward = 0.5
        self.distance_tolerance = 0.05
        self.drive_angle_limit = math.radians(10.0)

        self.position = None
        self.last_odom_time = None

        self.cmd_publisher = self.create_publisher(
            TwistStamped, '/cmd_vel', 10
        )

        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Waypoint patrol started.')
        self.report_target()

    def report_target(self):
        target_x, target_y = self.waypoints[self.waypoint_index]

        self.get_logger().info(
            f'Waypoint {self.waypoint_index + 1}/{len(self.waypoints)}: '
            f'x={target_x:.2f} m, y={target_y:.2f} m'
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
        now = time.monotonic()

        # Keep sending stop commands after completing the patrol.
        if self.finished:
            self.send_velocity(0.0, 0.0)
            return

        # Wait until position feedback arrives.
        if self.position is None:
            self.send_velocity(0.0, 0.0)

            self.get_logger().info(
                'Waiting for odometry...',
                throttle_duration_sec=2.0
            )
            return

        # Stop if position feedback becomes stale.
        if now - self.last_odom_time > 1.0:
            self.send_velocity(0.0, 0.0)

            self.get_logger().warning(
                'Odometry stopped arriving. Sending stop.',
                throttle_duration_sec=2.0
            )
            return

        # Pause at a reached waypoint before selecting the next one.
        if self.pause_until is not None:
            self.send_velocity(0.0, 0.0)

            if now < self.pause_until:
                return

            self.pause_until = None
            self.waypoint_index += 1

            if self.waypoint_index >= len(self.waypoints):
                # Point A was already reached at the end of the route.
                # Continue toward B to start the next circuit.
                self.waypoint_index = 1

                self.get_logger().info(
                    'Patrol circuit complete. Starting the next circuit.'
                )

            self.report_target()
            return

        x, y, yaw = self.position
        target_x, target_y = self.waypoints[self.waypoint_index]

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)

        # Stop within 5 cm of the current waypoint.
        if distance <= self.distance_tolerance:
            self.send_velocity(0.0, 0.0)
            self.pause_until = now + self.pause_duration

            self.get_logger().info(
                f'Waypoint {self.waypoint_index + 1} reached! '
                f'x={x:.3f} m, y={y:.3f} m, '
                f'remaining distance={distance:.3f} m. '
                f'Pausing for {self.pause_duration:.1f} seconds.'
            )
            return

        # Find the shortest turning angle toward the waypoint.
        target_heading = math.atan2(dy, dx)
        error = target_heading - yaw
        error = math.atan2(math.sin(error), math.cos(error))

        turn_speed = self.kp_turn * error
        turn_speed = max(
            -self.max_turn_speed,
            min(self.max_turn_speed, turn_speed)
        )

        # Turn in place if the heading error exceeds 10 degrees.
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
            f'Waypoint {self.waypoint_index + 1}: {state}, '
            f'distance={distance:.3f} m, '
            f'heading error={math.degrees(error):.1f} degrees, '
            f'forward={forward_speed:.3f} m/s',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO
    )
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