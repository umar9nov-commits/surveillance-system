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
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.clock import ClockType
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from ultralytics import YOLO


def classify(speed):
    return "measurement_only"


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
    node = rclpy.create_node(
        "robot_camera_motion_probe",
        parameter_overrides=[Parameter("use_sim_time", value=True)],
    )
    tf_buffer = Buffer(node=node)
    tf_listener = TransformListener(tf_buffer, node)

    publisher = node.create_publisher(
        String, "/surveillance/robot_motion_probe", 10
    )
    bridge = CvBridge()
    debug_publisher = node.create_publisher(
        Image, "/surveillance/robot/pose_debug_image", 2
    )
    last_debug = 0.0
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
            "method": "fixed_actor_plane_world_speed",
            "result": label,
            "reason": reason,
            "speed_metres_per_second": speed,
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
    print("Publishing /surveillance/robot_motion_probe", flush=True)
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

            if time.monotonic() - last_debug >= 1.0:
                annotated = result.plot()
                detail = f"People: {count} | Hip confidence required: 0.50"

                if (
                    count == 1
                    and result.keypoints is not None
                    and result.keypoints.conf is not None
                ):
                    scores = result.keypoints.conf[0].cpu().numpy()
                    detail = (
                        f"People: 1 | Hips: "
                        f"{scores[11]:.2f}, {scores[12]:.2f}"
                    )

                cv2.putText(
                    annotated, detail, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3,
                )
                cv2.putText(
                    annotated, detail, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1,
                )

                debug_message = bridge.cv2_to_imgmsg(
                    annotated, encoding="bgr8"
                )
                debug_message.header.stamp = Time(
                    nanoseconds=int(round(stamp * 1e9)),
                    clock_type=ClockType.ROS_TIME,
                ).to_msg()
                debug_message.header.frame_id = "robot_camera_optical_frame"
                debug_publisher.publish(debug_message)
                last_debug = time.monotonic()
            if (
                count != 1 or result.keypoints is None
                or result.keypoints.conf is None
            ):
                rows.clear()
                status("insufficient_data: exactly one pose required")
                continue

            p = result.keypoints.xy[0].cpu().numpy()
            c = result.keypoints.conf[0].cpu().numpy()
            selected = [11, 12]
            if (
                p.shape != (17, 2) or c.shape != (17,)
                or not np.all(np.isfinite(p[selected]))
                or not np.all(np.isfinite(c[selected]))
                or np.min(c[selected]) < 0.5
            ):
                rows.clear()
                status("insufficient_data: unreliable hips")
                continue


            try:
                image_time = Time(
                    nanoseconds=int(round(stamp * 1e9)),
                    clock_type=ClockType.ROS_TIME,
                )
                deadline_tf = time.monotonic() + 0.6
                while True:
                    try:
                        transform = tf_buffer.lookup_transform(
                            "map", "robot_camera_optical_frame", image_time
                        )
                        break
                    except Exception:
                        if time.monotonic() >= deadline_tf:
                            raise RuntimeError("matching image-time TF unavailable")
                        time.sleep(0.01)

                if time.monotonic() - arrival > 4:
                    raise RuntimeError("image expired while waiting for TF")

                height, width = frame.shape[:2]
                focal = width / (2 * np.tan(1.5708 / 2))
                pixel = p[[11, 12]].mean(axis=0)
                ray_camera = np.array([
                    (pixel[0] - (width-1)/2) / focal,
                    (pixel[1] - (height-1)/2) / focal,
                    1.0,
                ])

                q = transform.transform.rotation
                quaternion = np.array([q.x, q.y, q.z, q.w])
                norm = float(np.linalg.norm(quaternion))
                if not np.isfinite(norm) or norm < 1e-8:
                    raise RuntimeError("invalid camera rotation")
                x, y, z, w = quaternion / norm

                rotation = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
                ])
                translation = transform.transform.translation
                origin = np.array([
                    translation.x, translation.y, translation.z,
                ])
                ray_map = rotation @ ray_camera

                # Known travel plane for this simulation's fixed actor route.
                plane_x = 3.94423
                if abs(ray_map[0]) < 0.15:
                    raise RuntimeError("view too parallel to actor travel plane")

                depth = (plane_x - origin[0]) / ray_map[0]
                if not np.isfinite(depth) or not 0.1 < depth < 12:
                    raise RuntimeError("invalid projected person distance")

                world_point = origin + depth * ray_map
                if not np.all(np.isfinite(world_point)):
                    raise RuntimeError("invalid projected position")

                hip = world_point[:2]
                scale = 1.0
            except Exception as error:
                status("insufficient_data: projection: " + str(error))
                continue

            box = result.boxes.xyxy[0].cpu().numpy()
            if (
                box.shape != (4,) or not np.all(np.isfinite(box))
                or box[2] <= box[0] or box[3] <= box[1]
            ):
                status("insufficient_data: invalid bounding box")
                continue

            # Map-position association for the moving-camera probe.
            if rows:
                interval = stamp - rows[-1][0]
                jump = float(np.linalg.norm(hip - rows[-1][1]))
                if interval <= 0 or jump > 1.5 * interval + 0.10:
                    status("insufficient_data: projected-position track break")
                    continue
            tracker.update(box)

            camera_yaw = float(np.arctan2(
                rotation[1, 2], rotation[0, 2]
            ))
            rows.append((
                stamp, hip, scale, origin[:2].copy(), camera_yaw
            ))
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
                camera_positions = np.stack([r[3] for r in rows])
                camera_angles = np.unwrap([r[4] for r in rows])
                camera_travel = float(np.sum(np.linalg.norm(
                    np.diff(camera_positions, axis=0), axis=1
                )))
                camera_turn = float(np.sum(np.abs(
                    np.diff(camera_angles)
                )))
                accepted += 1
                publish_result(
                    classify(speed), speed=speed,
                    window_start=float(t[0]),
                    window_end=float(t[-1]),
                    poses=len(rows), max_gap=float(np.max(dt)),
                )
                print(
                    f"source={stamp:.1f} s | speed={speed:.4f} m/s"
                    f" | {classify(speed)} | poses={len(rows)}"
                    f" | track={tracker.track_id}"
                    f" | camera_travel={camera_travel:.3f} m"
                    f" | camera_turn={camera_turn:.3f} rad",
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
