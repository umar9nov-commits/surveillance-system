"""Single-person image-motion preview for a fixed observer camera."""

import json
import math
import statistics
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String


def overlap(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


class MotionPreview(Node):
    def __init__(self):
        super().__init__('person_motion_preview')
        self.set_parameters([Parameter('use_sim_time', value=True)])

        self.publisher = self.create_publisher(
            String, '/surveillance/observer/motion_preview', 10
        )
        self.subscription = self.create_subscription(
            String, '/surveillance/observer/detections',
            self.receive, 10
        )
        self.timer = self.create_timer(0.5, self.check_timeout)

        self.samples = deque()
        self.previous_box = None
        self.previous_stamp = None
        self.previous_wall = None
        self.image_context = None
        self.track_number = 0
        self.activity = 'unknown'
        self.timeout_reported = False

        self.get_logger().info(
            'Motion preview ready. Fixed camera, one person only. '
            'No navigation commands or live activity messages.'
        )

    def clear_track(self):
        self.samples.clear()
        self.previous_box = None
        self.activity = 'unknown'

    def publish(self, activity, reason, stamp=None, confidence=None,
                speed=None):
        message = String()
        message.data = json.dumps({
            'track_id': (
                f'preview_person_{self.track_number}'
                if self.previous_box is not None else None
            ),
            'motion': activity,
            'reason': reason,
            'source_time': stamp,
            'detection_confidence': confidence,
            'image_speed_body_heights_per_second': speed,
            'preview_only': True,
        })
        self.publisher.publish(message)
        self.get_logger().info(message.data)

    def check_timeout(self):
        if self.previous_wall is None or self.timeout_reported:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        wall_stale = time.monotonic() - self.previous_wall > 4.0
        sim_stale = (
            self.previous_stamp is not None
            and now - self.previous_stamp > 2.5
        )

        if wall_stale or sim_stale:
            self.clear_track()
            self.timeout_reported = True
            self.publish('unknown', 'Detection updates stopped.')

    def receive(self, message):
        try:
            data = json.loads(message.data)
            stamp = (
                int(data['source_stamp']['sec'])
                + int(data['source_stamp']['nanosec']) / 1e9
            )
            now = self.get_clock().now().nanoseconds / 1e9
            context = (
                data['frame_id'],
                int(data['image_width']),
                int(data['image_height']),
            )
            if stamp <= 0 or now <= 0:
                return

            if not -0.2 <= now - stamp <= 2.5:
                self.clear_track()
                self.publish('unknown', 'Image timestamp is not fresh.')
                return

            if (
                context != self.image_context
                or (
                    self.previous_stamp is not None
                    and stamp < self.previous_stamp
                )
            ):
                self.clear_track()

            self.image_context = context

            if stamp == self.previous_stamp:
                return

            gap = (
                self.previous_stamp is not None
                and (
                    stamp - self.previous_stamp > 2.5
                    or time.monotonic() - self.previous_wall > 4.0
                )
            )
            if gap:
                self.clear_track()

            self.previous_stamp = stamp
            self.previous_wall = time.monotonic()
            self.timeout_reported = False

            people = []
            for person in data['people']:
                box = [float(v) for v in person['bbox_xyxy']]
                confidence = float(person['confidence'])
                valid = (
                    len(box) == 4
                    and all(math.isfinite(v) for v in box)
                    and math.isfinite(confidence)
                    and 0.35 <= confidence <= 1.0
                    and 0 <= box[0] < box[2] <= context[1]
                    and 0 <= box[1] < box[3] <= context[2]
                )
                if valid:
                    people.append((box, confidence))

            if len(people) != 1:
                self.clear_track()
                self.publish(
                    'unknown',
                    'Need exactly one valid person detection.',
                    stamp,
                )
                return

            box, confidence = people[0]
            x = (box[0] + box[2]) / 2
            y = (box[1] + box[3]) / 2
            height = box[3] - box[1]

            if self.previous_box is not None:
                old = self.previous_box
                old_height = old[3] - old[1]
                displacement = math.hypot(
                    x - (old[0] + old[2]) / 2,
                    y - (old[1] + old[3]) / 2,
                )
                associated = (
                    0.5 <= height / old_height <= 2.0
                    and (
                        overlap(box, old) >= 0.2
                        or displacement <= 0.75 * old_height
                    )
                )
                if not associated:
                    self.clear_track()

            if self.previous_box is None:
                self.track_number += 1

            self.previous_box = box
            self.samples.append((stamp, x, y, height))

            # Retain four samples and the window boundary sample.
            # Freshness and track-continuity checks still apply separately.
            while (
                len(self.samples) > 4
                and stamp - self.samples[1][0] >= 4.0
            ):
                self.samples.popleft()

            duration = stamp - self.samples[0][0]
            if len(self.samples) < 4 or duration < 2.5:
                self.publish(
                    'unknown', 'Collecting motion history.',
                    stamp, confidence,
                )
                return

            speeds = []
            for a, b in zip(self.samples, list(self.samples)[1:]):
                dt = b[0] - a[0]
                if dt > 0:
                    distance = math.hypot(b[1] - a[1], b[2] - a[2])
                    speeds.append(distance / ((a[3] + b[3]) / 2) / dt)

            speed = statistics.median(speeds)

            # Preliminary thresholds, requiring scene calibration.
            if speed <= 0.04:
                self.activity = 'stationary'
            elif speed >= 0.10:
                self.activity = 'moving'
            else:
                self.activity = 'uncertain'

            self.publish(
                self.activity,
                'Preliminary image-motion estimate; not an action classifier.',
                stamp, confidence, round(speed, 4),
            )

        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.clear_track()
            self.publish('unknown', f'Invalid detection message: {error}')


def main():
    rclpy.init()
    node = MotionPreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
