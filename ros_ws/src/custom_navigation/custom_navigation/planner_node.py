"""Publish an A* path without commanding robot movement."""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from tf2_ros import Buffer, TransformListener, TransformException

from custom_navigation.IAstar import build_blocked_grid, astar
from custom_navigation.start_recovery import find_recovery_start
from custom_navigation.path_processing import prune_path, smooth_path


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        self.declare_parameter('clearance', 0.22)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('person_exclusion_enabled', False)
        self.declare_parameter('person_x', 1.9266527891159058)
        self.declare_parameter('person_y', -1.5833185911178589)
        self.declare_parameter('person_min_distance', 1.0)


        self.map_msg = None
        self.blocked = None
        self.pending_goal = None

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.plan_pub = self.create_publisher(Path, '/plan', qos)
        self.pruned_pub = self.create_publisher(
            Path, '/pruned_plan', qos
        )
        self.smooth_pub = self.create_publisher(
            Path, '/smooth_path', qos
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.on_map, qos
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.on_goal, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.5, self.process_goal)

        self.get_logger().info(
            'Planner started. Waiting for /map and an RViz goal.'
        )

    def on_map(self, msg):
        if (
            not msg.header.frame_id
            or msg.info.width == 0
            or msg.info.height == 0
            or msg.info.resolution <= 0
            or len(msg.data) != msg.info.width * msg.info.height
        ):
            self.get_logger().error('Received an invalid map.')
            return

        clearance = float(self.get_parameter('clearance').value)
        if clearance < 0:
            self.get_logger().error('Clearance must be non-negative.')
            return

        self.blocked = build_blocked_grid(
            msg.data,
            msg.info.width,
            msg.info.height,
            msg.info.resolution,
            clearance,
        )
        self.map_msg = msg

        self.get_logger().info(
            f'Map received: {msg.info.width} x {msg.info.height}.'
        )

    def publish_empty(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = (
            self.map_msg.header.frame_id if self.map_msg else 'map'
        )
        self.plan_pub.publish(path)
        self.pruned_pub.publish(path)
        self.smooth_pub.publish(path)

    def on_goal(self, msg):
        self.pending_goal = msg
        self.publish_empty()
        self.get_logger().info('Goal received. Calculating a path.')

    def world_to_grid(self, x, y):
        info = self.map_msg.info
        origin = info.origin
        angle = yaw_from_quaternion(origin.orientation)

        dx = x - origin.position.x
        dy = y - origin.position.y

        local_x = math.cos(angle) * dx + math.sin(angle) * dy
        local_y = -math.sin(angle) * dx + math.cos(angle) * dy

        return (
            math.floor(local_x / info.resolution),
            math.floor(local_y / info.resolution),
        )

    def grid_to_world(self, cell):
        info = self.map_msg.info
        origin = info.origin
        angle = yaw_from_quaternion(origin.orientation)

        local_x = (cell[0] + 0.5) * info.resolution
        local_y = (cell[1] + 0.5) * info.resolution

        return (
            origin.position.x
            + math.cos(angle) * local_x
            - math.sin(angle) * local_y,
            origin.position.y
            + math.sin(angle) * local_x
            + math.cos(angle) * local_y,
        )

    def process_goal(self):
        if self.pending_goal is None or self.map_msg is None:
            return

        goal = self.pending_goal
        frame = self.map_msg.header.frame_id

        if goal.header.frame_id != frame:
            self.get_logger().error(
                f'Goal must use frame "{frame}". '
                'Set the RViz Fixed Frame to the map frame.'
            )
            self.pending_goal = None
            return

        base_frame = str(self.get_parameter('base_frame').value)

        try:
            transform = self.tf_buffer.lookup_transform(
                frame, base_frame, Time()
            )
        except TransformException:
            self.get_logger().warning(
                'Waiting for the robot transform.',
                throttle_duration_sec=3.0,
            )
            return

        robot = transform.transform.translation
        target = goal.pose.position

        if not all(math.isfinite(v) for v in (
            robot.x, robot.y, target.x, target.y
        )):
            self.get_logger().error('Position contains invalid values.')
            self.pending_goal = None
            return

        start_cell = self.world_to_grid(robot.x, robot.y)
        goal_cell = self.world_to_grid(target.x, target.y)

        info = self.map_msg.info
        planning_blocked = list(self.blocked)

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
                self.pending_goal = None
                self.publish_empty()
                self.get_logger().error('Invalid person exclusion settings.')
                return

            cell_allowance = math.sqrt(2.0) * info.resolution / 2.0
            for gy in range(info.height):
                for gx in range(info.width):
                    wx, wy = self.grid_to_world((gx, gy))
                    if math.hypot(wx-px, wy-py) <= radius + cell_allowance:
                        planning_blocked[gy*info.width+gx] = True

            self.get_logger().info(
                f'Planning with person exclusion: {radius:.2f} m.'
            )

        recovery_origin = None
        cells = astar(
            planning_blocked,
            info.width,
            info.height,
            start_cell,
            goal_cell,
        )

        if not cells:
            age = (
                self.get_clock().now()
                - Time.from_msg(transform.header.stamp)
            ).nanoseconds / 1e9

            if -0.2 <= age <= 1.0:
                angle = yaw_from_quaternion(info.origin.orientation)
                c, s = math.cos(angle), math.sin(angle)

                def continuous_grid(x, y):
                    dx = x - info.origin.position.x
                    dy = y - info.origin.position.y
                    return (
                        (c*dx + s*dy) / info.resolution - 0.5,
                        (-s*dx + c*dy) / info.resolution - 0.5,
                    )

                person = None
                if self.get_parameter('person_exclusion_enabled').value:
                    person_grid = continuous_grid(px, py)
                    person = (person_grid[0], person_grid[1], radius)

                cells = find_recovery_start(
                    self.map_msg.data,
                    info.width, info.height, info.resolution,
                    planning_blocked,
                    continuous_grid(robot.x, robot.y),
                    goal_cell,
                    float(self.get_parameter('clearance').value),
                    person=person,
                )

                if cells:
                    recovery_origin = (robot.x, robot.y)
                    wx, wy = self.grid_to_world(cells[0])
                    length = math.hypot(wx-robot.x, wy-robot.y)
                    self.get_logger().warning(
                        f'Validated start recovery connector: {length:.3f} m. '
                        'Full planning buffer retained on the remaining route.'
                    )

        self.pending_goal = None

        if not cells:
            self.publish_empty()
            self.get_logger().warning(
                'No path found. Start or goal may be blocked, '
                'outside the map, or disconnected. '
                'Choose a goal in clear white space.'
            )
            return

        try:
            pruned_cells = prune_path(
                cells, planning_blocked, info.width, info.height
            )
        except ValueError as error:
            self.publish_empty()
            self.get_logger().error(f'Path validation failed: {error}')
            return

        try:
            smooth_cells, spline_accepted = smooth_path(
                pruned_cells, planning_blocked, info.width, info.height
            )

            if not spline_accepted and len(cells) > len(pruned_cells):
                self.get_logger().info(
                    'Pruned spline rejected. Trying the original '
                    'A* waypoints with the same clearance checks.'
                )
                candidate, accepted = smooth_path(
                    cells, planning_blocked, info.width, info.height
                )
                if accepted:
                    smooth_cells = candidate
                    spline_accepted = True
        except ValueError as error:
            self.publish_empty()
            self.get_logger().error(f'Smoothing failed: {error}')
            return

        final_yaw = yaw_from_quaternion(goal.pose.orientation)
        raw_path = self.make_path(cells, frame, final_yaw)
        pruned_path = self.make_path(pruned_cells, frame, final_yaw)
        self.prepend_recovery(raw_path, recovery_origin)
        self.prepend_recovery(pruned_path, recovery_origin)

        self.plan_pub.publish(raw_path)
        self.pruned_pub.publish(pruned_path)
        smooth_message = self.make_path(smooth_cells, frame, final_yaw)
        self.prepend_recovery(smooth_message, recovery_origin)
        self.smooth_pub.publish(smooth_message)

        if spline_accepted:
            self.get_logger().info(
                f'Validated cubic spline on /smooth_path: '
                f'{len(smooth_cells)} sampled points.'
            )
        else:
            self.get_logger().info(
                'Using the pruned route on /smooth_path. '
                'Spline was unnecessary or failed clearance validation.'
            )
        self.get_logger().info(
            f'A*: {len(cells)} points on /plan; '
            f'pruned: {len(pruned_cells)} points on /pruned_plan.'
        )

    def prepend_recovery(self, path, origin):
        if origin is None or not path.poses:
            return
        first = path.poses[0].pose.position
        if math.hypot(first.x-origin[0], first.y-origin[1]) < 1e-6:
            return
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = origin[0]
        pose.pose.position.y = origin[1]
        heading = math.atan2(first.y-origin[1], first.x-origin[0])
        pose.pose.orientation.z = math.sin(heading/2.0)
        pose.pose.orientation.w = math.cos(heading/2.0)
        path.poses.insert(0, pose)

    def make_path(self, cells, frame, final_yaw):
        path = Path()
        path.header.frame_id = frame
        path.header.stamp = self.get_clock().now().to_msg()
        points = [self.grid_to_world(cell) for cell in cells]

        for index, (x, y) in enumerate(points):
            pose = PoseStamped()
            pose.header.frame_id = frame
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = x
            pose.pose.position.y = y

            if index + 1 < len(points):
                nx, ny = points[index + 1]
                heading = math.atan2(ny - y, nx - x)
            else:
                heading = final_yaw

            pose.pose.orientation.z = math.sin(heading / 2.0)
            pose.pose.orientation.w = math.cos(heading / 2.0)
            path.poses.append(pose)

        return path



def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
