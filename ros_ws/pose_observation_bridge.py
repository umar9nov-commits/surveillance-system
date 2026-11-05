"""Trigger one fixed-person observation mission after standing detection."""

import json
import re
import math
import time
import hashlib
import signal
import subprocess
import sys
from pathlib import Path as FilePath
from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String


PERSON = 'person_1'
PERSON_X = 3.94423
PERSON_Y = -1.08
GOAL_X = 2.578
GOAL_Y = -1.406
GOAL_YAW = 0.2342329431482537


def stamp_key(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def matches(pose):
    q = pose.orientation
    heading = math.atan2(
        2.0 * (q.w*q.z + q.x*q.y),
        1.0 - 2.0 * (q.y*q.y + q.z*q.z),
    )
    error = math.atan2(
        math.sin(heading - GOAL_YAW),
        math.cos(heading - GOAL_YAW),
    )
    return (
        math.hypot(
            pose.position.x - GOAL_X,
            pose.position.y - GOAL_Y,
        ) <= 0.08
        and abs(error) <= 0.01
    )



def numeric(data, key):
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid numeric field: " + key)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite field: " + key)
    return value


def valid_track(track):
    return (
        isinstance(track, str)
        and re.fullmatch(r"[0-9a-f]{32}:[1-9][0-9]*", track) is not None
    )


def pose_motion_data(data):
    if (
        not isinstance(data, dict)
        or data.get("preview") is not True
        or data.get("method") != "normalized_hip_speed_reference"
    ):
        raise ValueError("Unexpected pose evidence format.")

    label = data.get("result")
    if label == "insufficient_data":
        return {"preview_only": True, "motion": "unknown"}

    track = data.get("track_id")
    if (
        not valid_track(track)
        or data.get("tracking_method") != "single_person_bbox_gate"
        or data.get("track_continuity") is not True
    ):
        raise ValueError("Temporary visual track required.")

    start = numeric(data, "window_start")
    end = numeric(data, "window_end")
    source = numeric(data, "source_time")
    speed = numeric(data, "speed_torso_lengths_per_second")
    gap = numeric(data, "max_source_gap")
    poses = data.get("valid_poses")

    if (
        start <= 0
        or not 1.6 <= end - start <= 2.400001
        or abs(source - end) > 1e-6
        or not 0 < gap <= 0.400001
        or type(poses) is not int or poses < 7
    ):
        raise ValueError("Pose window failed quality checks.")

    expected = "uncertain"
    if 0 <= speed <= 0.25:
        expected = "standing_reference_match"
    elif 0.55 <= speed <= 1.06:
        expected = "walking_reference_match"
    elif 1.25 <= speed <= 1.90:
        expected = "running_reference_match"

    if label != expected:
        raise ValueError("Pose label does not match speed rules.")

    motion = {
        "standing_reference_match": "stationary",
        "walking_reference_match": "moving",
        "running_reference_match": "moving",
        "uncertain": "uncertain",
    }[label]

    return {
        "preview_only": True,
        "motion": motion,
        "track_id": track,
        "source_time": source,
        "window_start": start,
    }


def pose_decision_data(data):
    if (
        not isinstance(data, dict)
        or data.get("preview") is not True
        or data.get("navigation_authorized") is not False
    ):
        raise ValueError("Unexpected decision preview.")

    if data.get("action") != "stationary_review_candidate":
        return {
            "preview_only": True, "motion": "unknown",
            "action": "hold",
        }

    track = data.get("track_id")
    if not valid_track(track):
        raise ValueError("Decision track missing.")

    source = numeric(data, "source_time")
    duration = numeric(data, "stationary_scene_seconds")
    if source <= 0 or duration < 30:
        raise ValueError("Insufficient stationary duration.")

    return {
        "preview_only": True,
        "motion": "stationary",
        "action": "request_observation_review",
        "simulation_time": source,
        "source_time": source,
        "stationary_seconds": duration,
        "track_id": track,
    }


def main():
    # Camera observation input mode
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', action='store_true')
    parser.add_argument(
        '--execute', action='store_true',
        help='Enable the fixed-person simulation observation mission.',
    )
    options, ros_args = parser.parse_known_args()
    options.camera = True
    rclpy.init(
        args=ros_args,
        signal_handler_options=SignalHandlerOptions.NO
    )
    node = rclpy.create_node('pose_observation_bridge')
    node.set_parameters([Parameter('use_sim_time', value=True)])

    state = {
        'activity': None, 'sim_seen': 0.0, 'wall_seen': 0.0,
        'trigger': False, 'abort': None, 'goal_stamp': None,
    }
    state.update({
        'camera_track': None,
        'camera_started': None,
        'locked_track': None,
        # Camera patrol interruption and resumption
        'camera_monitoring': True,
    })
    # Automatic patrol resumption
    patrol_process = None
    resume_candidate = False
    resume_snapshot = None
    workspace = FilePath.home() / 'surveillance_ws'
    checkpoint = workspace / 'config/patrol_progress.json'
    paths = deque(maxlen=20)
    done = deque(maxlen=20)

    def seconds():
        return node.get_clock().now().nanoseconds / 1e9

    def activity(msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
            person = data['person_id']
            action = data['activity']
            confidence = float(data['confidence'])
            if (
                not isinstance(person, str)
                or action not in ('standing', 'walking', 'running')
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                return
            if action == 'running' and confidence >= 0.7:
                state['abort'] = 'Running activity received.'
            if person == PERSON:
                state['activity'] = action if confidence >= 0.7 else None
                state['sim_seen'] = seconds()
                state['wall_seen'] = time.monotonic()
        except (ValueError, TypeError, KeyError):
            pass

    def decision(msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
            timestamp = float(data['simulation_time'])
            if (
                data.get('person_id') == PERSON
                and data.get('activity') == 'standing'
                and data.get('action') == 'request_observation_goal'
                and data.get('prototype_only') is True
                and math.isfinite(timestamp)
                and 0 <= seconds() - timestamp <= 2.0
            ):
                state['trigger'] = True
        except (ValueError, TypeError, KeyError):
            pass


    def invalidate_camera(reason):
        state['activity'] = None
        state['camera_started'] = None
        if state['trigger']:
            state['abort'] = reason

    def camera_motion(msg):
        # FIXED_OBSERVATION_TRIP_AFTER_30_SECONDS
        # After qualification, finish the predefined goal using navigation feedback.
        if not state['camera_monitoring'] or state['trigger']:
            return
        try:
            data = pose_motion_data(json.loads(msg.data))
            if not isinstance(data, dict):
                raise ValueError('Expected an object.')
            if data.get('preview_only') is not True:
                raise ValueError('Expected preview motion evidence.')

            motion = data['motion']
            if motion not in ('stationary', 'moving', 'uncertain', 'unknown'):
                raise ValueError('Invalid motion label.')

            if motion != 'stationary':
                invalidate_camera(
                    'Camera observation cancelled: motion is ' + motion
                )
                return

            track = data['track_id']
            stamp = float(data['source_time'])
            now = seconds()
            wall = time.monotonic()

            if not isinstance(track, str) or not track.strip():
                raise ValueError('Missing temporary track ID.')
            if (
                not math.isfinite(stamp)
                or stamp <= 0
                or not -0.2 <= now - stamp <= 2.5
            ):
                raise ValueError('Camera evidence is stale.')

            if (
                state['trigger']
                and track != state['locked_track']
            ):
                invalidate_camera('Camera track changed during observation.')
                return

            same_track = track == state['camera_track']

            if same_track and stamp == state['sim_seen']:
                return

            continuous = (
                state['activity'] == 'stationary'
                and same_track
                and abs(data['window_start'] - state['sim_seen']) <= 0.05
                and 0 < stamp - state['sim_seen'] <= 2.5
                and wall - state['wall_seen'] <= 8.0
            )

            if not continuous:
                if state['trigger']:
                    invalidate_camera(
                        'Camera evidence continuity was interrupted.'
                    )
                    return
                state['camera_started'] = data['window_start']

            state['camera_track'] = track
            state['activity'] = 'stationary'
            state['sim_seen'] = stamp
            state['wall_seen'] = wall

        except (ValueError, TypeError, KeyError) as error:
            invalidate_camera('Invalid camera evidence: ' + str(error))

    def camera_decision(msg):
        if not state['camera_monitoring'] or state['trigger']:
            return
        try:
            data = pose_decision_data(json.loads(msg.data))
            if not isinstance(data, dict):
                return
            if (
                data.get('preview_only') is not True
                or data.get('motion') != 'stationary'
                or data.get('action') != 'request_observation_review'
            ):
                return

            timestamp = float(data['simulation_time'])
            source = float(data['source_time'])
            duration = float(data['stationary_seconds'])

            valid = (
                all(math.isfinite(v) for v in
                    (timestamp, source, duration))
                and 0 <= seconds() - timestamp <= 2.0
                and -0.2 <= seconds() - source <= 2.5
                and duration >= 30.0
                and data.get('track_id') == state['camera_track']
                and state['activity'] == 'stationary'
                and state['camera_started'] is not None
                and source - state['camera_started'] >= 30.0
                and abs(source - state['sim_seen']) <= 2.5
                and time.monotonic() - state['wall_seen'] <= 8.0
            )

            if valid and not state['trigger']:
                state['locked_track'] = state['camera_track']
                state['trigger'] = True
                node.get_logger().info(
                    'Fresh camera decision accepted for fixed-person test. '
                    'Track: ' + state['locked_track']
                )

        except (ValueError, TypeError, KeyError):
            pass

    def watch_goal(msg):
        own = state['goal_stamp']
        if own is not None and stamp_key(msg.header.stamp) != own:
            state['abort'] = 'Another navigation goal was received.'

    subscriptions = [
        node.create_subscription(
            String,
            '/surveillance/pose_activity_preview'
            if options.camera else '/surveillance/activity',
            camera_motion if options.camera else activity, 10),
        node.create_subscription(
            String,
            '/surveillance/pose_decision_preview'
            if options.camera else '/surveillance/decision',
            camera_decision if options.camera else decision, 10),
        node.create_subscription(
            PoseStamped, '/goal_pose', watch_goal, 10),
        node.create_subscription(
            Path, '/smooth_path', lambda msg: paths.append(msg), 10),
        node.create_subscription(
            PoseStamped, '/navigation/goal_reached', lambda msg: done.append(msg), 10),
    ]
    publisher = node.create_publisher(PoseStamped, '/goal_pose', 10)
    setter = node.create_client(
        SetParameters, '/tb3_mpc_controller/set_parameters')
    getters = {
        name: node.create_client(
            GetParameters, f'/{name}/get_parameters')
        for name in ('planner_node', 'tb3_mpc_controller')
    }

    def spin():
        rclpy.spin_once(node, timeout_sec=0.05)

    def await_response(future):
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            spin()
        if not future.done():
            future.cancel()
            raise RuntimeError('Parameter service timed out.')
        return future.result()

    def enabled(value):
        if not setter.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('Controller service unavailable.')
        request = SetParameters.Request()
        request.parameters = [
            Parameter('enabled', value=value).to_parameter_msg()
        ]
        response = await_response(setter.call_async(request))
        if (
            len(response.results) != 1
            or not response.results[0].successful
        ):
            raise RuntimeError('Controller rejected enabled setting.')

    def get_values(name, names):
        client = getters[name]
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f'{name} service unavailable.')
        request = GetParameters.Request()
        request.names = names
        response = await_response(client.call_async(request))
        if len(response.values) != len(names):
            raise RuntimeError('Incomplete parameter response.')
        return response.values

    def check_standing():
        if options.camera:
            if state['abort']:
                raise RuntimeError(state['abort'])
            if state['trigger']:
                return
            if not (
                state['activity'] == 'stationary'
                and state['camera_track'] == state['locked_track']
                and -0.2 <= seconds() - state['sim_seen'] <= 2.5
                and time.monotonic() - state['wall_seen'] <= 8.0
            ):
                raise RuntimeError(
                    'Stationary camera evidence stopped or changed.'
                )
            return

        if state['abort']:
            raise RuntimeError(state['abort'])
        if not (
            state['activity'] == 'standing'
            and 0 <= seconds() - state['sim_seen'] <= 3.0
            and time.monotonic() - state['wall_seen'] <= 5.0
        ):
            raise RuntimeError('Standing updates stopped or activity changed.')

    def check_setup():
        names = node.get_node_names()
        if names.count('pose_observation_bridge') != 1:
            raise RuntimeError('Expected exactly one pose observation bridge.')
        if 'standing_observation_bridge' in names:
            raise RuntimeError('Stop the original observation bridge first.')
        if options.camera:
            for topic in (
                '/surveillance/pose_activity_preview',
                '/surveillance/pose_decision_preview',
            ):
                if node.count_publishers(topic) != 1:
                    raise RuntimeError(
                        'Expected exactly one publisher on ' + topic
                    )
        for name in getters:
            if names.count(name) != 1:
                raise RuntimeError(f'Expected exactly one {name}.')
            values = get_values(name, [
                'person_exclusion_enabled',
                'person_x', 'person_y', 'person_min_distance',
            ])
            if values[0].type != 1 or not values[0].bool_value:
                raise RuntimeError(f'Person exclusion disabled in {name}.')
            for value, expected in zip(
                values[1:], (PERSON_X, PERSON_Y, 1.0)
            ):
                if value.type == 3:
                    actual = value.double_value
                elif value.type == 2:
                    actual = float(value.integer_value)
                else:
                    raise RuntimeError(f'Invalid person parameter in {name}.')
                if (
                    not math.isfinite(actual)
                    or abs(actual - expected) > 1e-6
                ):
                    raise RuntimeError(
                        f'Fixed-person settings do not match in {name}.')
        endpoints = node.get_subscriptions_info_by_topic('/smooth_path')
        if not any(
            item.node_name == 'tb3_mpc_controller' for item in endpoints
        ):
            raise RuntimeError('MPC must subscribe to /smooth_path.')


    if not options.execute:
        try:
            node.get_logger().info(
                "DRY RUN: waiting for tracked stationary evidence and "
                "a 30-second pose decision. No navigation commands."
            )
            deadline = time.monotonic() + 180.0
            while not state['trigger']:
                spin()
                if time.monotonic() > deadline:
                    raise RuntimeError("Dry run timed out before a valid trigger.")

            check_standing()
            check_setup()
            check_standing()
            node.get_logger().info(
                "DRY RUN PASSED. Track: " + state['locked_track']
                + f". Proposed fixed goal: x={GOAL_X}, y={GOAL_Y}, "
                + f"yaw={GOAL_YAW:.4f}. No goal sent; MPC not enabled."
            )
        except KeyboardInterrupt:
            print("Pose bridge dry run interrupted.")
        except Exception as error:
            node.get_logger().error(str(error))
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
        return

    try:
        node.get_logger().info(
            'Waiting for a fresh 30-second camera stationary decision. '
            'Fixed-person test; start camera decision program AFTER this bridge.'
            if options.camera else
            'Waiting for a fresh 30-second standing decision. '
            'Fixed-person simulation only.')
        while not state['trigger']:
            spin()

        check_standing()
        names = node.get_node_names()
        if names.count('patrol_once') > 1:
            raise RuntimeError('More than one patrol node exists.')

        values = get_values('tb3_mpc_controller', ['enabled'])
        resume_candidate = (
            names.count('patrol_once') == 1
            and values[0].type == 1
            and values[0].bool_value
        )

        enabled(False)
        node.get_logger().info(
            'Controller disabled. Waiting for patrol to exit.')

        deadline = time.monotonic() + 20.0
        absent_since = None
        while True:
            spin()
            check_standing()
            if 'patrol_once' in node.get_node_names():
                absent_since = None
            elif absent_since is None:
                absent_since = time.monotonic()
            elif time.monotonic() - absent_since >= 1.0:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Patrol did not exit within 20 seconds.')

        enabled(False)

        if resume_candidate:
            saved = json.loads(checkpoint.read_text())
            if saved.get('complete') is True:
                resume_candidate = False
                node.get_logger().info(
                    'Patrol completed before interruption. No resume needed.')
            else:
                config = json.loads(
                    (workspace / 'config/patrol_waypoints.json').read_text()
                )
                route_id = hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()
                ).hexdigest()
                index = saved.get('next_index')
                saved_time = saved.get('simulation_time_ns')
                if (
                    saved.get('route_id') != route_id
                    or type(index) is not int
                    or not 0 <= index < 4
                    or type(saved_time) is not int
                    or not 0 < saved_time <= node.get_clock().now().nanoseconds
                ):
                    raise RuntimeError('Invalid interrupted patrol checkpoint.')
                resume_snapshot = saved
                node.get_logger().info(
                    f'Patrol leg {index + 1}/4 saved for automatic resume.')

        check_setup()
        check_standing()
        if publisher.get_subscription_count() == 0:
            raise RuntimeError('No navigation goal subscriber.')

        paths.clear()
        done.clear()
        enabled(True)
        check_standing()

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = node.get_clock().now().to_msg()
        state['goal_stamp'] = stamp_key(goal.header.stamp)
        if state['goal_stamp'] <= 0:
            raise RuntimeError('Simulation clock unavailable.')
        goal.pose.position.x = GOAL_X
        goal.pose.position.y = GOAL_Y
        goal.pose.orientation.z = math.sin(GOAL_YAW / 2)
        goal.pose.orientation.w = math.cos(GOAL_YAW / 2)
        publisher.publish(goal)
        node.get_logger().info('Observation goal sent.')

        started = time.monotonic()
        next_check = started
        expected_stamp = None

        while True:
            spin()
            check_standing()
            now = time.monotonic()

            if now >= next_check:
                values = get_values('tb3_mpc_controller', ['enabled'])
                if values[0].type != 1 or not values[0].bool_value:
                    raise RuntimeError('Controller was disabled.')
                next_check = time.monotonic() + 1.0

            while paths:
                path = paths.popleft()
                key = stamp_key(path.header.stamp)
                if key < state['goal_stamp']:
                    continue
                if not path.poses:
                    if expected_stamp is not None:
                        raise RuntimeError('Observation path was cleared.')
                    continue
                if (
                    path.header.frame_id != 'map'
                    or not matches(path.poses[-1].pose)
                ):
                    raise RuntimeError('Path does not match observation goal.')
                if expected_stamp is not None and key != expected_stamp:
                    raise RuntimeError('Observation path changed.')
                expected_stamp = key

            if expected_stamp is not None and any(
                msg.header.frame_id == 'map'
                and stamp_key(msg.header.stamp) == expected_stamp
                and matches(msg.pose)
                for msg in done
            ):
                node.get_logger().info(
                    'Observation point reached. Robot faces the fixed person.')
                break

            if expected_stamp is None and now - started > 15.0:
                raise RuntimeError('No matching observation path.')
            if now - started > 300.0:
                raise RuntimeError('Observation mission timed out.')

        enabled(False)

        if resume_candidate:
            check_standing()
            check_setup()

            if 'patrol_once' in node.get_node_names():
                raise RuntimeError('Another patrol is already running.')
            if json.loads(checkpoint.read_text()) != resume_snapshot:
                raise RuntimeError('Patrol checkpoint changed during observation.')

            config = json.loads(
                (workspace / 'config/patrol_waypoints.json').read_text()
            )
            route_id = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()
            if route_id != resume_snapshot['route_id']:
                raise RuntimeError('Patrol waypoints changed during observation.')

            # Patrol will publish its own goals and monitor external goals.
            state['goal_stamp'] = None
            enabled(True)
            check_standing()

            if options.camera:
                # Observation has succeeded. Normal patrol now takes over.
                state['camera_monitoring'] = False
                node.get_logger().info(
                    'Camera observation complete. Returning to normal patrol.'
                )

            patrol_process = subprocess.Popen(
                [sys.executable, str(workspace / 'patrol_once.py'), '--resume'],
                cwd=str(workspace),
            )
            node.get_logger().info('Automatically resuming interrupted patrol.')

            while patrol_process.poll() is None:
                spin()
                if state['abort']:
                    raise RuntimeError(state['abort'])

            if patrol_process.returncode != 0:
                raise RuntimeError('Resumed patrol process failed.')

            saved = json.loads(checkpoint.read_text())
            if (
                saved.get('route_id') != route_id
                or saved.get('complete') is not True
                or saved.get('next_index') != 4
            ):
                raise RuntimeError(
                    'Resumed patrol ended before completing its circuit.')

            node.get_logger().info('Automatic resumed patrol circuit complete.')
        else:
            node.get_logger().info(
                'No active patrol was interrupted. Remaining stopped.')

    except KeyboardInterrupt:
        print('Observation bridge interrupted.')
    except Exception as error:
        node.get_logger().error(str(error))
    finally:
        if patrol_process is not None and patrol_process.poll() is None:
            try:
                patrol_process.send_signal(signal.SIGINT)
                patrol_process.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                patrol_process.kill()
                patrol_process.wait()
            except ProcessLookupError:
                pass
        try:
            enabled(False)
            print('Controller disabled. Observation bridge ended.')
        except Exception as error:
            print(f'Could not confirm disable: {error}')
            print('Run: ros2 param set /tb3_mpc_controller enabled false')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
