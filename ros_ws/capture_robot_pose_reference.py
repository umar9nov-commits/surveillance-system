"""Record observer images, then analyse poses offline. No robot control."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy
qos_profile_sensor_data = QoSProfile(
    depth=10, reliability=ReliabilityPolicy.RELIABLE
)
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--label', required=True,
        choices=('standing', 'scripted_walk', 'scripted_run'),
    )
    parser.add_argument('--seconds', type=float, default=20.0)
    options = parser.parse_args()

    if not 5 <= options.seconds <= 30:
        parser.error('--seconds must be between 5 and 30.')

    workspace = Path.home() / 'surveillance_ws'
    weights = workspace / 'models/yolo11n-pose.pt'
    if not weights.is_file():
        raise SystemExit('Pose model weights are missing.')

    suffix = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = workspace / 'robot_activity_samples' / f'{options.label}_{suffix}'
    directory.mkdir(parents=True)
    images = directory / 'images'
    images.mkdir()

    rclpy.init()
    node = rclpy.create_node('robot_pose_reference_recorder')
    bridge = CvBridge()
    frames = []
    errors = []

    def receive(message):
        stamp = (
            message.header.stamp.sec
            + message.header.stamp.nanosec / 1e9
        )
        if stamp <= 0:
            return
        if frames:
            if stamp < frames[-1]['source_time']:
                errors.append('Camera timestamp moved backward.')
                return
            if stamp == frames[-1]['source_time']:
                return
            if stamp - frames[0]['source_time'] > options.seconds + 1.0:
                errors.append('Camera timestamp jumped beyond recording duration.')
                return

        try:
            image = bridge.imgmsg_to_cv2(
                message, desired_encoding='bgr8'
            )
            name = f'frame_{len(frames):05d}.jpg'
            if not cv2.imwrite(str(images / name), image):
                raise RuntimeError('Image could not be saved.')

            frames.append({
                'source_time': stamp,
                'image_file': 'images/' + name,
            })

            if len(frames) % 25 == 0:
                elapsed = stamp - frames[0]['source_time']
                print(
                    f'Recorded {len(frames)} frames; '
                    f'{elapsed:.1f}/{options.seconds:.1f} simulation seconds',
                    flush=True,
                )
        except Exception as error:
            errors.append(str(error))

    subscription = node.create_subscription(
        Image, '/surveillance/robot/image_raw',
        receive, qos_profile_sensor_data,
    )

    completed = False
    interrupted = False
    print('Recording camera frames. Pose inference starts afterward.', flush=True)

    try:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline and not errors:
            rclpy.spin_once(node, timeout_sec=0.1)
            if len(frames) >= 2:
                elapsed = frames[-1]['source_time'] - frames[0]['source_time']
                if elapsed >= options.seconds:
                    completed = True
                    break
            if len(frames) >= 1000:
                errors.append('Frame limit reached.')
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.try_shutdown()

    gaps = [
        b['source_time'] - a['source_time']
        for a, b in zip(frames, frames[1:])
    ]
    metadata = {
        'reference_label': options.label,
        'label_source': 'User-selected simulation test mode',
        'activity_classified': False,
        'recording_complete': completed,
        'errors': errors,
        'records': frames,
    }

    output = directory / 'poses.json'
    output.write_text(json.dumps(metadata, indent=2))

    if len(frames) < 2:
        raise SystemExit(f'Too few frames. Recording saved in: {directory}')

    print('\nCAPTURE RESULTS')
    print('Frames:', len(frames))
    print('Source duration:', round(
        frames[-1]['source_time'] - frames[0]['source_time'], 3
    ), 'simulation seconds')
    print('Median source gap:', round(float(np.median(gaps)), 3), 's')
    print('Largest source gap:', round(max(gaps), 3), 's')
    print('Source gaps over 0.4 s:', sum(gap > 0.400001 for gap in gaps))
    print('Recording complete:', completed)

    if interrupted or errors:
        raise SystemExit(
            f'Recording interrupted or invalid. Images preserved: {directory}'
        )

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(1)
    model = YOLO(str(weights))
    print('\nAnalysing saved images. This may take a few minutes.', flush=True)

    for index, record in enumerate(frames, 1):
        image = cv2.imread(str(directory / record['image_file']))
        if image is None:
            record['error'] = 'Saved image could not be read.'
            continue

        result = model.predict(
            image, device='cpu', imgsz=640,
            conf=0.35, verbose=False,
        )[0]

        count = len(result.boxes) if result.boxes is not None else 0
        record['people'] = count

        if (
            count == 1 and result.keypoints is not None
            and result.keypoints.conf is not None
        ):
            record['bbox_xyxy'] = result.boxes.xyxy[0].cpu().tolist()
            record['keypoints_xy'] = result.keypoints.xy[0].cpu().tolist()
            record['keypoint_confidences'] = (
                result.keypoints.conf[0].cpu().tolist()
            )

        if index % 25 == 0 or index == len(frames):
            output.write_text(json.dumps(metadata, indent=2))
            print(f'Pose analysis: {index}/{len(frames)}', flush=True)

    output.write_text(json.dumps(metadata, indent=2))
    print('\nSaved measurements:', output)
    print('No activity decisions or navigation commands published.')


if __name__ == '__main__':
    main()
