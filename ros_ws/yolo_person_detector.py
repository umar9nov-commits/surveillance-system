"""Detect people in the observer camera. No robot movement commands."""

import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.clock_type import ClockType
from rclpy.parameter import Parameter
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy,
)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class PersonDetector(Node):
    def __init__(self):
        super().__init__('yolo_person_detector')
        self.set_parameters([Parameter('use_sim_time', value=True)])

        # Limit CPU use while Gazebo is also running.
        torch.set_num_threads(1)

        directory = Path.home() / 'surveillance_ws/models'
        directory.mkdir(parents=True, exist_ok=True)
        weights = directory / 'yolo11n.pt'

        self.get_logger().info(
            'Loading YOLO11n. First run may download model weights.'
        )
        self.model = YOLO(str(weights))

        if self.model.names.get(0) != 'person':
            raise RuntimeError('Expected a COCO model with class 0 = person.')

        self.bridge = CvBridge()
        self.latest = None
        self.first_image = True

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.subscription = self.create_subscription(
            Image,
            '/surveillance/observer/image_raw',
            self.receive,
            qos,
        )
        self.image_pub = self.create_publisher(
            Image, '/surveillance/observer/yolo_image', 1
        )
        self.detection_pub = self.create_publisher(
            String, '/surveillance/observer/detections', 10
        )

        # Process selected frames instead of running on every camera frame.
        # Schedule processing independently of simulation speed.
        # Published images retain their original simulation timestamps.
        self.processing_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            1.0, self.detect, clock=self.processing_clock
        )
        self.get_logger().info(
            'YOLO ready. Waiting for observer camera images. '
            'No activity or movement commands are published.'
        )

    def receive(self, msg):
        self.latest = (msg, time.monotonic())

        if self.first_image:
            self.first_image = False
            self.get_logger().info(
                f'Camera received: {msg.width} x {msg.height}, '
                f'encoding={msg.encoding}'
            )

    def detect(self):
        if self.latest is None:
            return

        msg, received = self.latest
        self.latest = None

        if time.monotonic() - received > 2.0:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
            started = time.monotonic()

            result = self.model.predict(
                source=image,
                classes=[0],
                conf=0.35,
                imgsz=416,
                device='cpu',
                verbose=False,
            )[0]

            elapsed = time.monotonic() - started
            annotated = result.plot()

            output = self.bridge.cv2_to_imgmsg(
                annotated, encoding='bgr8'
            )
            output.header = msg.header
            self.image_pub.publish(output)

            people = []
            if result.boxes is not None:
                coordinates = result.boxes.xyxy.cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                for box, confidence in zip(coordinates, confidences):
                    people.append({
                        'class_name': 'person',
                        'confidence': float(confidence),
                        'bbox_xyxy': [float(value) for value in box],
                    })

            detections = String()
            detections.data = json.dumps({
                'source_stamp': {
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec,
                },
                'frame_id': msg.header.frame_id,
                'image_width': msg.width,
                'image_height': msg.height,
                'people': people,
                'inference_seconds': elapsed,
            })
            self.detection_pub.publish(detections)

            best = max(
                (person['confidence'] for person in people),
                default=0.0,
            )
            self.get_logger().info(
                f'People detected: {len(people)}, '
                f'best confidence={best:.2f}, '
                f'inference={elapsed:.2f} s',
                throttle_duration_sec=2.0,
            )

        except Exception as error:
            self.get_logger().error(
                f'Detection failed: {error}',
                throttle_duration_sec=3.0,
            )


def main():
    rclpy.init()
    node = None
    try:
        node = PersonDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
