"""Run one patrol circuit using matching path-completion messages."""

import json
import argparse
import hashlib
import math
import time
from pathlib import Path

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavigationPath
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters


def stamp_key(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def heading(q):
    return math.atan2(
        2.0 * (q.w*q.z + q.x*q.y),
        1.0 - 2.0 * (q.y*q.y + q.z*q.z),
    )


def endpoint_matches(pose, point):
    p = pose.position
    angle = heading(pose.orientation) - point['yaw']
    angle = math.atan2(math.sin(angle), math.cos(angle))
    return (
        math.hypot(p.x-point['x'], p.y-point['y']) <= 0.08
        and abs(angle) <= 0.01
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume the saved interrupted leg in the same simulation session.',
    )
    options = parser.parse_args()
    config = json.loads(
        (Path.home() / 'surveillance_ws/config/patrol_waypoints.json')
        .read_text()
    )
    points = config['waypoints']
    if config['frame_id'] != 'map' or len(points) != 3:
        raise ValueError('Expected three map-frame patrol points.')
    for point in points:
        if not all(math.isfinite(point[k]) for k in ('x', 'y', 'yaw')):
            raise ValueError('Invalid waypoint.')

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('patrol_once')
    node.set_parameters([
        rclpy.parameter.Parameter('use_sim_time', value=True)
    ])

    publisher = node.create_publisher(PoseStamped, '/goal_pose', 10)
    get_client = node.create_client(
        GetParameters, '/tb3_mpc_controller/get_parameters'
    )
    set_client = node.create_client(
        SetParameters, '/tb3_mpc_controller/set_parameters'
    )

    received = {'paths': [], 'done': [], 'own_stamp': None, 'external': False}

    def watch_goal(msg):
        own = received['own_stamp']
        if own is not None and stamp_key(msg.header.stamp) != own:
            received['external'] = True

    subscriptions = [
        node.create_subscription(
            NavigationPath, '/smooth_path',
            lambda msg: received['paths'].append(msg), 10,
        ),
        node.create_subscription(
            PoseStamped, '/navigation/goal_reached',
            lambda msg: received['done'].append(msg), 10,
        ),
        node.create_subscription(
            PoseStamped, '/goal_pose', watch_goal, 10,
        ),
    ]

    def await_response(future, timeout=2.0):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if not future.done():
            raise RuntimeError('Controller parameter service timed out.')
        return future.result()

    def controller_enabled():
        request = GetParameters.Request()
        request.names = ['enabled']
        response = await_response(get_client.call_async(request))
        return (
            len(response.values) == 1
            and response.values[0].type == ParameterType.PARAMETER_BOOL
            and response.values[0].bool_value
        )

    try:
        if not get_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Controller parameter service unavailable.')
        if not set_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Controller stop service unavailable.')

        # Wait for the actual planner, not just our own goal watcher.
        node.get_logger().info('Waiting for planner goal connection.')
        deadline = time.monotonic() + 25.0
        ready_since = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            endpoints = node.get_subscriptions_info_by_topic('/goal_pose')
            planners = [item for item in endpoints
                        if item.node_name == 'planner_node'
                        and item.node_namespace == '/']
            matched = publisher.get_subscription_count()
            ready = len(planners) == 1 and matched >= max(2, len(endpoints))
            if ready:
                if ready_since is None:
                    ready_since = time.monotonic()
                elif time.monotonic() - ready_since >= 2.0:
                    node.get_logger().info('Planner goal connection ready.')
                    break
            else:
                ready_since = None
        else:
            raise RuntimeError('Planner goal connection unavailable after 25 seconds.')

        if not controller_enabled():
            raise RuntimeError('Enable the MPC controller before starting patrol.')

        if publisher.get_subscription_count() == 0:
            raise RuntimeError('No goal subscriber found.')

        route = points + [points[0]]
        checkpoint = (
            Path.home() / 'surveillance_ws/config/patrol_progress.json'
        )
        route_id = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()

        def save_checkpoint(next_index):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'route_id': route_id,
                'next_index': next_index,
                'simulation_time_ns': node.get_clock().now().nanoseconds,
                'complete': next_index == len(route),
            }
            temporary = checkpoint.with_name(checkpoint.name + '.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(checkpoint)

        start_index = 0
        if options.resume:
            if not checkpoint.is_file():
                raise RuntimeError('No saved patrol progress. Start a fresh circuit.')
            saved = json.loads(checkpoint.read_text())
            if saved.get('route_id') != route_id:
                raise RuntimeError('Waypoints changed. Start a fresh circuit.')

            start_index = saved.get('next_index')
            if (
                type(start_index) is not int
                or not 0 <= start_index < len(route)
                or saved.get('complete')
            ):
                raise RuntimeError('No unfinished circuit to resume.')

            saved_time = saved.get('simulation_time_ns')
            current_time = node.get_clock().now().nanoseconds
            if (
                type(saved_time) is not int
                or current_time <= 0
                or current_time < saved_time
            ):
                raise RuntimeError(
                    'Simulation clock reset or unavailable. Start a fresh circuit.'
                )

            node.get_logger().info(
                f'Resuming patrol at leg {start_index + 1}/4.'
            )

        for leg, point in enumerate(route[start_index:], start_index + 1):
            if not controller_enabled():
                raise RuntimeError('Controller disabled. Patrol cancelled.')

            if received['external']:
                raise RuntimeError('Another goal was received. Patrol cancelled.')

            received['paths'].clear()
            received['done'].clear()

            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = node.get_clock().now().to_msg()
            sent_stamp = stamp_key(goal.header.stamp)
            if sent_stamp <= 0:
                raise RuntimeError('Simulation clock is unavailable.')

            goal.pose.position.x = float(point['x'])
            goal.pose.position.y = float(point['y'])
            goal.pose.orientation.z = math.sin(point['yaw']/2.0)
            goal.pose.orientation.w = math.cos(point['yaw']/2.0)
            received['own_stamp'] = sent_stamp

            save_checkpoint(leg - 1)
            publisher.publish(goal)
            node.get_logger().info(
                f'Patrol leg {leg}/4: x={point["x"]:.3f}, '
                f'y={point["y"]:.3f}'
            )

            started = time.monotonic()
            next_check = started + 1.0
            expected_stamp = None
            completed = False

            while not completed:
                rclpy.spin_once(node, timeout_sec=0.1)
                now = time.monotonic()

                if received['external']:
                    raise RuntimeError('External goal detected. Patrol cancelled.')

                if now >= next_check:
                    if not controller_enabled():
                        raise RuntimeError('Controller disabled. Patrol cancelled.')
                    next_check = time.monotonic() + 1.0

                for path in received['paths']:
                    key = stamp_key(path.header.stamp)
                    if key < sent_stamp:
                        continue
                    if not path.poses:
                        if expected_stamp is not None:
                            raise RuntimeError('Active path was cleared.')
                        continue
                    if (
                        path.header.frame_id != 'map'
                        or not endpoint_matches(path.poses[-1].pose, point)
                    ):
                        raise RuntimeError('Received path does not match patrol goal.')
                    if expected_stamp is not None and key != expected_stamp:
                        raise RuntimeError('Path changed unexpectedly.')
                    expected_stamp = key
                received['paths'].clear()

                if expected_stamp is not None:
                    for message in received['done']:
                        if (
                            message.header.frame_id == 'map'
                            and stamp_key(message.header.stamp) == expected_stamp
                            and endpoint_matches(message.pose, point)
                        ):
                            completed = True
                            break

                if expected_stamp is None and now-started > 15.0:
                    raise RuntimeError('No matching path within 15 wall seconds.')
                if now-started > 300.0:
                    raise RuntimeError('Goal exceeded 300 wall seconds.')

            save_checkpoint(leg)
            node.get_logger().info(f'Patrol leg {leg}/4 completed.')

            if leg < len(route):
                pause_until = node.get_clock().now().nanoseconds + 2_000_000_000
                pause_deadline = time.monotonic() + 15.0
                while node.get_clock().now().nanoseconds < pause_until:
                    rclpy.spin_once(node, timeout_sec=0.1)
                    if received['external']:
                        raise RuntimeError('External goal detected during pause.')
                    if time.monotonic() > pause_deadline:
                        raise RuntimeError('Simulation clock stalled during pause.')

        node.get_logger().info('Patrol circuit complete.')

    except KeyboardInterrupt:
        print('Patrol interrupted.')
    except Exception as error:
        node.get_logger().error(str(error))
    finally:
        try:
            request = SetParameters.Request()
            request.parameters = [
                Parameter(
                    name='enabled',
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_BOOL,
                        bool_value=False,
                    ),
                )
            ]
            if not set_client.service_is_ready():
                raise RuntimeError('Controller stop service unavailable.')
            response = await_response(set_client.call_async(request))
            if not response.results or not response.results[0].successful:
                raise RuntimeError('Controller rejected disable request.')
            print('Controller disabled. Patrol ended.')
        except Exception as error:
            print(f'Could not confirm controller disable: {error}')
            print('Run: ros2 param set /tb3_mpc_controller enabled false')

        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
