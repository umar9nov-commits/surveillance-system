"""Test activity decisions without commanding robot movement."""

import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ActivityDecision(Node):
    def __init__(self):
        super().__init__('activity_decision')
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time', value=True)
        ])
        self.people = {}
        self.last_time = None
        self.publisher = self.create_publisher(
            String, '/surveillance/decision', 10
        )
        self.subscription = self.create_subscription(
            String, '/surveillance/activity', self.on_activity, 10
        )
        self.timer = self.create_timer(0.2, self.check)
        self.get_logger().info(
            'Decision prototype ready. No robot movement is commanded.'
        )

    def seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def emit(self, person_id, activity, action, reason):
        message = String()
        message.data = json.dumps({
            'person_id': person_id,
            'activity': activity,
            'action': action,
            'reason': reason,
            'simulation_time': self.seconds(),
            'prototype_only': True,
        })
        self.publisher.publish(message)
        self.get_logger().info(message.data)

    def on_activity(self, msg):
        try:
            data = json.loads(msg.data)
            person_id = data['person_id']
            activity = data['activity']
            confidence = float(data['confidence'])
            if not isinstance(person_id, str) or not person_id.strip():
                raise ValueError('A non-empty person_id is required.')
            if activity not in ('standing', 'walking', 'running'):
                raise ValueError('Unknown activity.')
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Invalid confidence.')
        except (ValueError, KeyError, TypeError) as error:
            self.get_logger().warning(f'Invalid activity message: {error}')
            return

        now = self.seconds()
        if now <= 0:
            self.get_logger().warning('Waiting for the simulation clock.')
            return

        if confidence < 0.7:
            self.people.pop(person_id, None)
            return

        previous = self.people.get(person_id)
        fresh = (
            previous is not None
            and 0 <= now - previous['seen'] <= 3.0
            and time.monotonic() - previous['wall_seen'] <= 5.0
        )
        changed = not fresh or previous['activity'] != activity

        if changed:
            self.people[person_id] = {
                'activity': activity,
                'started': now,
                'seen': now,
                'wall_seen': time.monotonic(),
                'triggered': False,
            }
            if activity == 'standing':
                self.emit(
                    person_id, activity, 'monitor',
                    'Standing timer started.',
                )
            elif activity == 'walking':
                self.emit(
                    person_id, activity, 'notify_and_continue',
                    'Walking detected. Standing timer reset.',
                )
            else:
                self.emit(
                    person_id, activity, 'notify_and_request_pause',
                    'Proposed running response; no threat classification.',
                )
        else:
            previous['seen'] = now
            previous['wall_seen'] = time.monotonic()

    def check(self):
        now = self.seconds()
        if self.last_time is not None and now < self.last_time:
            self.people.clear()
            self.get_logger().warning('Clock moved backward. Timers reset.')
        self.last_time = now

        for person_id, state in list(self.people.items()):
            if (
                now - state['seen'] > 3.0
                or time.monotonic() - state['wall_seen'] > 5.0
            ):
                self.emit(
                    person_id, 'lost', 'cancel_person_timer',
                    'Activity updates stopped.',
                )
                del self.people[person_id]
                continue

            if (
                state['activity'] == 'standing'
                and not state['triggered']
                and now - state['started'] >= 30.0
            ):
                state['triggered'] = True
                self.emit(
                    person_id, 'standing', 'request_observation_goal',
                    'Standing for 30 simulation seconds. '
                    'A valid map position is required before approach.',
                )



class CameraMotionDecision(Node):
    """Preview decisions from single-person image-motion estimates."""

    def __init__(self):
        super().__init__('camera_motion_decision')
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time', value=True)
        ])
        self.state = None
        self.last_clock = None

        self.publisher = self.create_publisher(
            String, '/surveillance/decision_preview', 10
        )
        self.subscription = self.create_subscription(
            String, '/surveillance/observer/motion_preview',
            self.on_motion, 10
        )
        self.timer = self.create_timer(0.2, self.check)
        self.get_logger().info(
            'Camera decision preview ready. '
            'Thirty source-time seconds of stationary evidence required. '
            'Output: /surveillance/decision_preview. No robot control.'
        )

    def seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def emit(self, track, motion, action, reason, source_time=None,
             detection_confidence=None, duration=0.0):
        message = String()
        message.data = json.dumps({
            'track_id': track,
            'motion': motion,
            'action': action,
            'reason': reason,
            'simulation_time': self.seconds(),
            'source_time': source_time,
            'stationary_seconds': round(duration, 3),
            'detection_confidence': detection_confidence,
            'activity_confidence': None,
            'preview_only': True,
            'map_position_available': False,
        })
        self.publisher.publish(message)
        self.get_logger().info(message.data)

    def reset(self, reason, motion='unknown'):
        if self.state is not None:
            self.emit(
                self.state['track'], motion,
                'cancel_person_timer', reason
            )
        self.state = None

    def on_motion(self, message):
        try:
            data = json.loads(message.data)
            if data.get('preview_only') is not True:
                raise ValueError('Expected a preview motion message.')

            motion = data['motion']
            if motion not in (
                'stationary', 'moving', 'uncertain', 'unknown'
            ):
                raise ValueError('Invalid motion label.')

            now = self.seconds()
            if now <= 0:
                self.reset('Simulation clock unavailable.')
                return

            if motion in ('unknown', 'uncertain'):
                self.reset(
                    'Motion evidence interrupted: '
                    + str(data.get('reason', motion)),
                    motion
                )
                return

            track = data['track_id']
            stamp = float(data['source_time'])
            confidence = float(data['detection_confidence'])

            if not isinstance(track, str) or not track.strip():
                raise ValueError('Missing temporary track ID.')
            if (
                not math.isfinite(stamp)
                or stamp <= 0
                or not -0.2 <= now - stamp <= 2.5
            ):
                raise ValueError('Motion source timestamp is not fresh.')
            if (
                not math.isfinite(confidence)
                or not 0.35 <= confidence <= 1.0
            ):
                raise ValueError('Invalid person-detection confidence.')

            wall = time.monotonic()
            previous = self.state

            if previous is not None:
                if (
                    track == previous['track']
                    and stamp == previous['seen']
                ):
                    return

                continuous = (
                    track == previous['track']
                    and 0 < stamp - previous['seen'] <= 2.5
                    and wall - previous['wall_seen'] <= 4.0
                )
                if not continuous:
                    self.reset('Track or timing continuity interrupted.')
                    previous = None

            if previous is None or previous['motion'] != motion:
                if previous is not None:
                    self.reset('Motion changed.', motion)

                self.state = {
                    'track': track,
                    'motion': motion,
                    'started': stamp,
                    'seen': stamp,
                    'wall_seen': wall,
                    'triggered': False,
                }

                if motion == 'stationary':
                    self.emit(
                        track, motion, 'monitor',
                        'Stationary image-motion timer started. '
                        'Standing posture is not established.',
                        stamp, confidence
                    )
                else:
                    self.emit(
                        track, motion, 'notify_motion',
                        'Image movement detected. '
                        'Walking versus running is not classified.',
                        stamp, confidence
                    )
            else:
                previous['seen'] = stamp
                previous['wall_seen'] = wall

            state = self.state
            duration = stamp - state['started']

            # Trigger only on a newly received stationary observation.
            # Timer callbacks never add evidence between frames.
            if (
                motion == 'stationary'
                and duration >= 30.0
                and not state['triggered']
            ):
                state['triggered'] = True
                self.emit(
                    track, motion, 'request_observation_review',
                    'Thirty seconds of stationary image-motion evidence. '
                    'A verified map position is required before approach.',
                    stamp, confidence, duration
                )

        except (ValueError, TypeError, KeyError) as error:
            self.reset('Invalid motion input: ' + str(error))
            self.get_logger().warning(str(error))

    def check(self):
        now = self.seconds()
        if self.last_clock is not None and now < self.last_clock:
            self.reset('Simulation clock moved backward.')
        self.last_clock = now

        if self.state is not None:
            if (
                now - self.state['seen'] > 2.5
                or time.monotonic() - self.state['wall_seen'] > 4.0
            ):
                self.reset('Motion updates stopped or became stale.')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', action='store_true')
    options, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = CameraMotionDecision() if options.camera else ActivityDecision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
