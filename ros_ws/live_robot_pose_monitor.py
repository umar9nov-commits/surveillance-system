"""Live fixed-camera speed-reference monitor. No robot control."""
import argparse
import json
import queue
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
import torch
from cv_bridge import CvBridge
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy,
)
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from ultralytics import YOLO


def classify(speed):
    if 0.55 <= speed <= 1.15:
        return "walking_reference_match"
    if 1.40 <= speed <= 2.10:
        return "running_reference_match"
    return "uncertain"


class BoxTrack:
    """Temporary single-person association; not identity recognition."""

    def __init__(self):
        self.session = uuid.uuid4().hex
        self.counter = 0
        self.clear()

    def clear(self):
        self.box = None
        self.track_id = None

    def matches(self, box):
        if self.box is None:
            return False
        a, b = self.box, box
        aw, ah = a[2] - a[0], a[3] - a[1]
        bw, bh = b[2] - b[0], b[3] - b[1]

        intersection = (
            max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        )
        iou = intersection / (aw * ah + bw * bh - intersection)
        distance = float(np.hypot(
            (a[0] + a[2] - b[0] - b[2]) / 2,
            (a[1] + a[3] - b[1] - b[3]) / 2,
        ))
        area_ratio = bw * bh / (aw * ah)
        return (
            iou >= 0.2
            and distance <= 0.5 * max(ah, bh)
            and 0.5 <= area_ratio <= 2.0
        )

    def update(self, box):
        if self.box is None:
            self.counter += 1
            self.track_id = f"{self.session}:{self.counter}"
            print("New temporary track:", self.track_id, flush=True)
        self.box = tuple(float(v) for v in box)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive.")

    weights = Path.home() / "surveillance_ws/models/yolo11n-pose.pt"
    if not weights.is_file():
        raise SystemExit("Pose model weights are missing.")

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    model = YOLO(str(weights))
    print("Warming up pose model...", flush=True)
    model.predict(
        np.zeros((480, 640, 3), dtype=np.uint8),
        device="cpu", imgsz=640, conf=0.35, verbose=False,
    )

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("live_robot_pose_reference_monitor")
    publisher = node.create_publisher(
        String, "/surveillance/robot_pose_activity_preview", 10
    )
    bridge = CvBridge()
    pending = queue.Queue(maxsize=2)
    stop = threading.Event()
    state = {
        "arrival": time.monotonic(), "error": None,
        "camera_previous": None, "camera_frames": 0,
        "camera_large_gaps": 0, "camera_max_gap": 0.0,
        "queue_drops": 0, "main_skips": 0,
    }


    robot = {
        "stamp": None, "arrival": None,
        "stationary_since": None, "anchor": None,
    }

    def receive_odometry(message):
        stamp = (
            message.header.stamp.sec
            + message.header.stamp.nanosec / 1e9
        )
        position = message.pose.pose.position
        q = message.pose.pose.orientation
        yaw = float(np.arctan2(
            2 * (q.w*q.z + q.x*q.y),
            1 - 2 * (q.y*q.y + q.z*q.z),
        ))
        pose = np.array([position.x, position.y, yaw])
        velocity = message.twist.twist
        linear = float(np.linalg.norm([
            velocity.linear.x, velocity.linear.y, velocity.linear.z,
        ]))
        angular = float(np.linalg.norm([
            velocity.angular.x, velocity.angular.y, velocity.angular.z,
        ]))

        if (
            stamp <= 0 or not np.all(np.isfinite(pose))
            or not np.isfinite(linear) or not np.isfinite(angular)
        ):
            robot["stationary_since"] = None
            robot["anchor"] = None
            return

        backward = (
            robot["stamp"] is not None and stamp < robot["stamp"]
        )
        robot["stamp"] = stamp
        robot["arrival"] = time.monotonic()

        if backward or linear > 0.01 or angular > 0.02:
            robot["stationary_since"] = None
            robot["anchor"] = None
            return

        if robot["anchor"] is not None:
            delta = pose - robot["anchor"]
            angle = abs(float(np.arctan2(
                np.sin(delta[2]), np.cos(delta[2])
            )))
            if np.linalg.norm(delta[:2]) > 0.01 or angle > 0.02:
                robot["stationary_since"] = None
                robot["anchor"] = None

        if robot["stationary_since"] is None:
            robot["stationary_since"] = stamp
            robot["anchor"] = pose.copy()

    odometry_subscription = node.create_subscription(
        Odometry, "/odom", receive_odometry,
        QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        ),
    )

    def receive(message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if stamp <= 0:
            return
        try:
            frame = bridge.imgmsg_to_cv2(
                message, desired_encoding="bgr8"
            ).copy()
            arrival = time.monotonic()
            state["arrival"] = arrival
            state["camera_frames"] += 1
            previous = state["camera_previous"]
            if previous is not None and stamp > previous:
                camera_gap = stamp - previous
                state["camera_max_gap"] = max(
                    state["camera_max_gap"], camera_gap
                )
                if camera_gap > 0.400001:
                    state["camera_large_gaps"] += 1
            state["camera_previous"] = stamp
            while True:
                try:
                    pending.put_nowait((stamp, arrival, frame))
                    break
                except queue.Full:
                    try:
                        pending.get_nowait()
                        state["queue_drops"] += 1
                    except queue.Empty:
                        pass
        except Exception as error:
            state["error"] = str(error)

    subscription = node.create_subscription(
        Image, "/surveillance/robot/image_raw",
        receive, QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        ),
    )

    def spin():
        try:
            while not stop.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
        except Exception as error:
            state["error"] = str(error)
            stop.set()

    worker = threading.Thread(target=spin)
    worker.start()

    tracker = BoxTrack()
    rows = []
    previous_stamp = None
    last_status = None
    accepted = 0
    infer_times = []

    def publish_result(label, reason=None, speed=None,
                       window_start=None, window_end=None,
                       poses=0, max_gap=None):
        payload = {
            "preview": True,
            "camera_source": "robot",
            "navigation_authorized": False,
            "method": "normalized_hip_speed_reference",
            "result": label,
            "reason": reason,
            "speed_torso_lengths_per_second": speed,
            "source_time": window_end,
            "window_start": window_start,
            "window_end": window_end,
            "valid_poses": poses,
            "max_source_gap": max_gap,
            "activity_confidence": None,
            "track_id": (
                tracker.track_id if label != "insufficient_data" else None
            ),
            "tracking_method": "single_person_bbox_gate",
            "track_continuity": (
                label != "insufficient_data" and tracker.track_id is not None
            ),
            "identity_verified": False,
        }
        message = String()
        message.data = json.dumps(payload, allow_nan=False)
        publisher.publish(message)

    def status(message):
        nonlocal last_status
        if message.startswith("insufficient_data:"):
            tracker.clear()
            rows.clear()
            publish_result(
                "insufficient_data", reason=message,
                window_end=previous_stamp,
            )
        if message != last_status:
            print(message, flush=True)
            last_status = message

    print("Monitoring for", args.seconds, "real seconds.", flush=True)
    print("Publishing /surveillance/robot_pose_activity_preview", flush=True)
    print("No navigation commands published.", flush=True)
    publish_result("insufficient_data", reason="Monitor starting")
    deadline = time.monotonic() + args.seconds

    try:
        while time.monotonic() < deadline and not stop.is_set():
            if state["error"]:
                raise RuntimeError(state["error"])

            try:
                stamp, arrival, frame = pending.get(timeout=0.2)
            except queue.Empty:
                if time.monotonic() - state["arrival"] > 4:
                    rows.clear()
                    previous_stamp = None
                    status("insufficient_data: camera arrival timeout")
                continue

            # Process the bounded queue in arrival order.
            # Existing queue limits and image-age checks remain active.

            if previous_stamp is not None:
                gap = stamp - previous_stamp
                if gap == 0:
                    continue
                if gap < 0 or gap > 0.400001:
                    rows.clear()
                    status(
                        f"insufficient_data: processed-frame gap={gap:.3f} s"
                        f" | camera_max_gap={state['camera_max_gap']:.3f} s"
                        f" | queue_drops={state['queue_drops']}"
                        f" | newest_skips={state['main_skips']}"
                    )
            previous_stamp = stamp

            odom_stamp = robot["stamp"]
            stationary_since = robot["stationary_since"]
            odom_arrival = robot["arrival"]

            if (
                odom_stamp is None or odom_arrival is None
                or time.monotonic() - odom_arrival > 2.0
                or abs(stamp - odom_stamp) > 0.7
            ):
                status("insufficient_data: fresh robot odometry required")
                continue

            if (
                stationary_since is None
                or stamp - stationary_since < 2.0
            ):
                status(
                    "insufficient_data: robot moving or settling; "
                    "camera-motion compensation pending"
                )
                continue

            began = time.monotonic()
            result = model.predict(
                frame, device="cpu", imgsz=640,
                conf=0.35, verbose=False,
            )[0]
            infer_times.append(time.monotonic() - began)

            if time.monotonic() - arrival > 4:
                rows.clear()
                status("insufficient_data: image too old after inference")
                continue

            count = len(result.boxes) if result.boxes is not None else 0
            if (
                count != 1 or result.keypoints is None
                or result.keypoints.conf is None
            ):
                rows.clear()
                status("insufficient_data: exactly one pose required")
                continue

            p = result.keypoints.xy[0].cpu().numpy()
            c = result.keypoints.conf[0].cpu().numpy()
            selected = [5, 6, 11, 12]
            if (
                p.shape != (17, 2) or c.shape != (17,)
                or not np.all(np.isfinite(p[selected]))
                or not np.all(np.isfinite(c[selected]))
                or np.min(c[selected]) < 0.5
            ):
                rows.clear()
                status("insufficient_data: unreliable shoulders or hips")
                continue

            hip = p[[11, 12]].mean(axis=0)
            shoulder = p[[5, 6]].mean(axis=0)
            scale = float(np.linalg.norm(hip - shoulder))
            if scale <= 1:
                rows.clear()
                status("insufficient_data: invalid torso size")
                continue

            box = result.boxes.xyxy[0].cpu().numpy()
            if (
                box.shape != (4,) or not np.all(np.isfinite(box))
                or box[2] <= box[0] or box[3] <= box[1]
            ):
                status("insufficient_data: invalid bounding box")
                continue

            if tracker.box is not None and not tracker.matches(box):
                status("insufficient_data: bounding-box track break")
            tracker.update(box)

            rows.append((stamp, hip, scale))
            if stamp - rows[0][0] < 2.0 - 1e-6:
                status("Collecting a two-second window...")
                continue

            t = np.array([r[0] for r in rows])
            hips = np.stack([r[1] for r in rows])
            scales = np.array([r[2] for r in rows])
            dt = np.diff(t)

            if (
                len(rows) >= 5 and t[-1] - t[0] <= 2.400001
                and np.all(dt > 0) and np.max(dt) <= 0.400001
            ):
                displacement = np.linalg.norm(np.diff(hips, axis=0), axis=1)
                pair_scale = (scales[:-1] + scales[1:]) / 2
                speed = float(np.median(displacement / dt / pair_scale))
                accepted += 1
                publish_result(
                    classify(speed), speed=speed,
                    window_start=float(t[0]),
                    window_end=float(t[-1]),
                    poses=len(rows), max_gap=float(np.max(dt)),
                )
                print(
                    f"source={stamp:.1f} s | speed={speed:.4f} torso lengths/s"
                    f" | {classify(speed)} | poses={len(rows)}"
                    f" | track={tracker.track_id}",
                    flush=True,
                )
                last_status = None
            else:
                status("insufficient_data: window timing or coverage")

            # Keep the endpoint as the start of the next window.
            rows = rows[-1:]

    except KeyboardInterrupt:
        pass
    finally:
        publish_result("insufficient_data", reason="Monitor stopped")
        stop.set()
        worker.join()
        node.destroy_subscription(odometry_subscription)
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.try_shutdown()

    print("\nAccepted windows:", accepted)
    print("Camera timing counters:")
    print("Camera frames received:", state["camera_frames"])
    print("Camera source gaps above 0.4 s:", state["camera_large_gaps"])
    print("Largest camera source gap:", round(state["camera_max_gap"], 3))
    print("Frames dropped from full queue:", state["queue_drops"])
    print("Frames skipped to select newest:", state["main_skips"])
    if infer_times:
        print(f"Median inference time: {np.median(infer_times):.3f} real seconds")
    if state["error"]:
        print("Monitor error:", state["error"])


if __name__ == "__main__":
    main()
