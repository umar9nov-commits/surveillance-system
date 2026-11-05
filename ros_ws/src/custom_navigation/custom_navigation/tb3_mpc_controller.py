"""MPC path following for the TurtleBot3 Gazebo simulation."""

import math
from copy import deepcopy
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import (
    QoSProfile,
    DurabilityPolicy,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformException

from custom_navigation.mpc_core import solve_mpc, wrap_angle


def yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class MPCController(Node):
    def __init__(self):
        super().__init__('tb3_mpc_controller')
        self.declare_parameter('enabled', False)
        self.declare_parameter('person_exclusion_enabled', False)
        self.declare_parameter('person_x', 1.9266527891159058)
        self.declare_parameter('person_y', -1.5833185911178589)
        self.declare_parameter('person_min_distance', 1.0)

        self.goal_completion = None
        self.completion_pub = self.create_publisher(
            PoseStamped, '/navigation/goal_reached', 10
        )

        self.dt = 0.2
        self.horizon = 12
        self.speed = 0.08
        self.clearance = 0.22

        self.map_msg = None
        self.distances = None
        self.scan = None
        self.scan_received = 0.0
        self.points = None
        self.lengths = None
        self.goal_yaw = 0.0
        self.progress = 0.0
        self.previous = np.zeros(2)
        self.guess = None

        durable = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.command_pub = self.create_publisher(
            TwistStamped, '/cmd_vel', 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.on_map, durable
        )

        # Receive new plans only. Do not execute a retained old plan.
        self.path_sub = self.create_subscription(
            Path, '/plan', self.on_path, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.on_scan,
            qos_profile_sensor_data,
        )

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.timer = self.create_timer(self.dt, self.control)

        self.get_logger().info(
            'MPC ready but DISABLED. No movement is enabled.'
        )

    def send(self, forward=0.0, angular=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(forward)
        msg.twist.angular.z = float(angular)
        self.command_pub.publish(msg)
        self.previous = np.array([forward, angular])

    def stop(self, reason=None):
        self.heading_settle_count = 0
        self.send()
        self.guess = None
        if reason:
            self.get_logger().warning(
                reason, throttle_duration_sec=3.0
            )

    def on_map(self, msg):
        info = msg.info
        if (
            info.width == 0 or info.height == 0
            or info.resolution <= 0
            or len(msg.data) != info.width * info.height
            or not msg.header.frame_id
        ):
            self.map_msg = None
            self.stop('Invalid map. Stopped.')
            return

        data = np.asarray(msg.data).reshape(
            info.height, info.width
        )
        free = (data >= 0) & (data < 50)

        # Unknown space and map boundaries count as obstacles.
        padded = np.pad(free, 1, constant_values=False)
        distances = distance_transform_edt(
            padded, sampling=info.resolution
        )[1:-1, 1:-1]

        # Conservative allowance for occupied-cell area and interpolation.
        self.distances = (
            distances - math.sqrt(2.0) * info.resolution
        )
        self.map_msg = msg
        self.get_logger().info('Controller map received.')

    def on_scan(self, msg):
        self.scan = msg
        self.scan_received = time.monotonic()

    def on_path(self, msg):
        self.goal_completion = None
        self.aligning = False
        self.heading_settle_count = 0
        self.points = None
        self.stop()

        if not self.get_parameter('enabled').value:
            self.get_logger().info(
                'Plan ignored while disabled. Send a new goal after enabling.'
            )
            return

        if self.map_msg is None:
            self.get_logger().warning('Map is not ready.')
            return

        if msg.header.frame_id != self.map_msg.header.frame_id:
            self.get_logger().error('Path frame does not match map.')
            return

        if not msg.poses:
            self.get_logger().warning('Empty path. Stopped.')
            return

        points = np.array([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in msg.poses
        ])

        if not np.all(np.isfinite(points)):
            self.get_logger().error('Invalid path coordinates.')
            return

        keep = np.r_[
            True,
            np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6,
        ]
        self.points = points[keep]
        self.lengths = np.r_[
            0.0,
            np.cumsum(
                np.linalg.norm(
                    np.diff(self.points, axis=0), axis=1
                )
            ),
        ]
        self.goal_yaw = yaw(msg.poses[-1].pose.orientation)
        self.progress = 0.0
        self.goal_completion = deepcopy(msg.poses[-1])
        self.goal_completion.header = deepcopy(msg.header)

        self.get_logger().info(
            f'New path accepted: {self.lengths[-1]:.2f} m.'
        )

    def margins(self, states):
        info = self.map_msg.info
        origin = info.origin
        angle = yaw(origin.orientation)

        dx = states[:, 0] - origin.position.x
        dy = states[:, 1] - origin.position.y

        gx = (
            math.cos(angle) * dx + math.sin(angle) * dy
        ) / info.resolution - 0.5
        gy = (
            -math.sin(angle) * dx + math.cos(angle) * dy
        ) / info.resolution - 0.5

        values = map_coordinates(
            self.distances, [gy, gx],
            order=1, mode='constant', cval=-1.0,
            prefilter=False,
        )
        margins = values - self.clearance

        if self.get_parameter('person_exclusion_enabled').value:
            px = float(self.get_parameter('person_x').value)
            py = float(self.get_parameter('person_y').value)
            radius = float(
                self.get_parameter('person_min_distance').value
            )
            if (
                not all(math.isfinite(v) for v in (px, py, radius))
                or radius <= 0
            ):
                return np.full(len(states), -1.0)

            person_margins = np.hypot(
                states[:, 0] - px, states[:, 1] - py
            ) - radius
            margins = np.minimum(margins, person_margins)

        return margins

    def reference(self, state):
        if len(self.points) < 2:
            self.stop('Path is too short. Send a new goal.')
            return None, None

        starts = self.points[:-1]
        segments = np.diff(self.points, axis=0)
        squared = np.sum(segments ** 2, axis=1)

        fractions = np.clip(
            np.sum(
                (state[:2] - starts) * segments, axis=1
            ) / squared,
            0.0, 1.0,
        )
        projections = starts + fractions[:, None] * segments
        index = int(np.argmin(
            np.linalg.norm(projections - state[:2], axis=1)
        ))

        cross_track = np.linalg.norm(
            projections[index] - state[:2]
        )
        if cross_track > 0.30:
            self.stop('Too far from path. Send a new goal.')
            return None, None

        along = (
            self.lengths[index]
            + fractions[index] * math.sqrt(squared[index])
        )
        self.progress = max(self.progress, along)

        # ENDPOINT_BEARING_APPROACH
        # Face the actual endpoint near route completion. Keeping only the
        # final path tangent can leave a lateral zero-motion local optimum.
        endpoint_delta = self.points[-1] - state[:2]
        endpoint_distance = float(np.linalg.norm(endpoint_delta))
        remaining = self.lengths[-1] - self.progress
        if 1e-9 < endpoint_distance <= 0.30 and remaining <= 0.20:
            direction = endpoint_delta / endpoint_distance
            advance = np.minimum(
                np.arange(1, self.horizon + 1) * self.speed * self.dt,
                endpoint_distance,
            )
            positions = state[:2] + advance[:, None] * direction
            bearing = math.atan2(endpoint_delta[1], endpoint_delta[0])
            return np.column_stack([
                positions, np.full(self.horizon, bearing)
            ]), self.speed

        samples = np.minimum(
            self.progress
            + np.arange(1, self.horizon + 1)
            * self.speed * self.dt,
            self.lengths[-1],
        )

        x = np.interp(samples, self.lengths, self.points[:, 0])
        y = np.interp(samples, self.lengths, self.points[:, 1])

        indices = np.clip(
            np.searchsorted(self.lengths, samples, side='right') - 1,
            0, len(segments) - 1,
        )
        headings = np.arctan2(
            segments[indices, 1], segments[indices, 0]
        )
        return np.column_stack([x, y, headings]), self.speed

    def scan_is_clear(self, now):
        msg = self.scan
        if msg is None or time.monotonic() - self.scan_received > 2.0:
            return False

        age = (now - Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
        if age < -0.1 or age > 1.0:
            return False

        # This simulation uses the verified forward-facing base_scan frame.
        if msg.header.frame_id != 'base_scan':
            return False

        ranges = np.asarray(msg.ranges)
        angles = (
            msg.angle_min
            + np.arange(len(ranges)) * msg.angle_increment
        )
        front = ranges[
            np.abs(wrap_angle(angles)) < math.radians(60)
        ]

        if len(front) == 0 or np.any(np.isnan(front)):
            return False
        if np.any(front < msg.range_min):
            return False

        return bool(np.all(front > 0.30))

    def control(self):
        if not self.get_parameter('enabled').value:
            self.points = None
            self.stop()
            return

        if self.map_msg is None or self.points is None:
            self.stop()
            return

        # Stop if another node also commands the robot.
        if self.count_publishers('/cmd_vel') > 1:
            self.stop('Another /cmd_vel publisher exists. Stopped.')
            return

        now = self.get_clock().now()

        if not self.scan_is_clear(now):
            self.stop('LiDAR is stale, invalid, or an obstacle is close.')
            return

        try:
            transform = self.buffer.lookup_transform(
                self.map_msg.header.frame_id,
                'base_footprint',
                Time(),
            )
        except TransformException:
            self.stop('Robot transform unavailable.')
            return

        age = (
            now - Time.from_msg(transform.header.stamp)
        ).nanoseconds / 1e9

        if age < -0.2 or age > 1.0:
            self.stop('Robot transform is stale.')
            return

        translation = transform.transform.translation
        state = np.array([
            translation.x,
            translation.y,
            yaw(transform.transform.rotation),
        ])

        if (
            not np.all(np.isfinite(state))
            or self.margins(state[None, :])[0] < 0
        ):
            self.stop('Insufficient map clearance at robot position.')
            return

        distance = float(np.linalg.norm(self.points[-1] - state[:2]))
        signed_error = float(wrap_angle(self.goal_yaw - state[2]))

        if not getattr(self, 'aligning', False) and distance <= 0.08:
            if getattr(self, '_alignment_path', None) is not self.points:
                self._alignment_path = self.points
                self.alignment_retries = 0
            self.aligning = True
            self.stop()
            self.get_logger().info('Position reached. Aligning final heading.')
            return

        if getattr(self, 'aligning', False):
            if distance > 0.12:
                retries = getattr(self, 'alignment_retries', 0)

                if distance > 0.30 or retries >= 2:
                    self.points = None
                    self.aligning = False
                    self.stop(
                        'Final alignment recovery limit reached. '
                        'Send a new goal.'
                    )
                    return

                self.alignment_retries = retries + 1
                self.aligning = False
                self.progress = 0.0
                self.stop()
                self.get_logger().warn(
                    f'Estimated goal distance increased to {distance:.3f} m. '
                    f'Resuming MPC position correction '
                    f'({self.alignment_retries}/2).'
                )
                return

            if abs(signed_error) <= 0.12:
                self.heading_settle_count = getattr(
                    self, 'heading_settle_count', 0
                ) + 1
                self.send(0.0, 0.0)
                if self.heading_settle_count >= 3:
                    self.points = None
                    self.aligning = False
                    self.stop()
                    if self.goal_completion is not None:
                        self.completion_pub.publish(self.goal_completion)
                        self.goal_completion = None
                    self.get_logger().info('Goal reached. Robot stopped.')
                return

            self.heading_settle_count = 0
            target_turn = float(np.clip(0.8 * signed_error, -0.20, 0.20))
            turn = float(np.clip(
                target_turn,
                self.previous[1] - 0.6 * self.dt,
                self.previous[1] + 0.6 * self.dt,
            ))
            self.send(0.0, turn)
            self.get_logger().info(
                f'FINAL ALIGN: distance={distance:.3f} m, '
                f'heading error={math.degrees(signed_error):.1f} deg, '
                f'turn={turn:.3f} rad/s',
                throttle_duration_sec=1.0,
            )
            return

        reference, speed_limit = self.reference(state)
        if reference is None:
            return

        started = time.monotonic()
        try:
            controls, predicted, status = solve_mpc(
                state,
                reference,
                self.previous,
                initial_guess=self.guess,
                dt=self.dt,
                max_speed=speed_limit,
                clearance_function=self.margins,
            )
        except Exception as error:
            self.stop(f'MPC calculation failed: {error}')
            return

        if time.monotonic() - started > 0.5:
            self.stop('MPC calculation exceeded 0.5 seconds.')
            return

        if controls is None:
            self.stop(f'MPC could not find a valid command: {status}')
            return

        # Recheck sensor age after optimization.
        if not self.scan_is_clear(self.get_clock().now()):
            self.stop('LiDAR check failed after optimization.')
            return

        self.send(*controls[0])
        self.guess = np.vstack([controls[1:], controls[-1:]])

        self.get_logger().info(
            f'MPC: goal distance={distance:.2f} m, '
            f'forward={controls[0, 0]:.3f} m/s, '
            f'turn={controls[0, 1]:.3f} rad/s',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
