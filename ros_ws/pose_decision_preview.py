"""Consume pose previews and publish decision previews. No robot control."""
import json
import math
import re
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


def main():
    rclpy.init()
    node = rclpy.create_node("pose_decision_preview")
    input_topic = "/surveillance/pose_activity_preview"
    output = node.create_publisher(
        String, "/surveillance/pose_decision_preview", 10
    )

    state = {
        "clock": None, "clock_arrival": None,
        "arrival": None, "last_end": None,
        "standing_start": None, "last_output": None,
        "track_id": None,
    }

    def emit(action, reason, duration=0.0, source_time=None):
        payload = {
            "preview": True,
            "action": action,
            "reason": reason,
            "stationary_scene_seconds": round(duration, 3),
            "source_time": source_time,
            "track_id": state["track_id"],
            "identity_verified": False,
            "navigation_authorized": False,
        }
        encoded = json.dumps(payload, allow_nan=False)
        if encoded != state["last_output"]:
            message = String()
            message.data = encoded
            output.publish(message)
            print(encoded, flush=True)
            state["last_output"] = encoded

    def reset(reason):
        state["standing_start"] = None
        state["track_id"] = None
        emit("hold_insufficient_evidence", reason)

    def clock_received(message):
        stamp = message.clock.sec + message.clock.nanosec / 1e9
        if state["clock"] is not None and stamp < state["clock"]:
            state["last_end"] = None
            state["arrival"] = None
            reset("Simulation clock moved backward")
        state["clock"] = stamp
        state["clock_arrival"] = time.monotonic()

    def number(value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Invalid numeric field")
        return float(value)

    def expected_label(speed):
        if 0 <= speed <= 0.25:
            return "standing_reference_match"
        if 0.55 <= speed <= 1.06:
            return "walking_reference_match"
        if 1.25 <= speed <= 1.90:
            return "running_reference_match"
        return "uncertain"

    def received(message):
        try:
            if node.count_publishers(input_topic) != 1:
                raise ValueError("Exactly one pose publisher required")

            data = json.loads(message.data)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            if (
                data.get("preview") is not True
                or data.get("method") != "normalized_hip_speed_reference"
            ):
                raise ValueError("Unexpected preview format")

            if data.get("result") == "insufficient_data":
                reset(data.get("reason") or "Pose evidence unavailable")
                return

            start = number(data.get("window_start"))
            end = number(data.get("window_end"))
            source = number(data.get("source_time"))
            speed = number(data.get("speed_torso_lengths_per_second"))
            gap = number(data.get("max_source_gap"))
            poses = number(data.get("valid_poses"))

            if (
                abs(source - end) > 1e-6
                or not 1.6 <= end - start <= 2.400001
                or not 0 < gap <= 0.400001
                or poses < 7 or poses != int(poses)
            ):
                raise ValueError("Window failed quality checks")

            clock = state["clock"]
            if (
                clock is None or state["clock_arrival"] is None
                or time.monotonic() - state["clock_arrival"] > 4
                or not -0.2 <= clock - end <= 2.5
            ):
                raise ValueError("Window is stale or simulation clock unavailable")

            track_id = data.get("track_id")
            if (
                not isinstance(track_id, str)
                or re.fullmatch(r"[0-9a-f]{32}:[1-9][0-9]*", track_id) is None
                or data.get("tracking_method") != "single_person_bbox_gate"
                or data.get("track_continuity") is not True
            ):
                raise ValueError("Valid temporary track required")

            previous = state["last_end"]
            if previous is not None and end <= previous:
                raise ValueError("Duplicate or out-of-order window")

            label = expected_label(speed)
            if label != data.get("result"):
                raise ValueError("Label does not match the speed rules")

            contiguous = (
                previous is not None
                and abs(start - previous) <= 0.05
                and state["track_id"] == track_id
            )
            state["track_id"] = track_id
            state["last_end"] = end
            state["arrival"] = time.monotonic()

            if label == "standing_reference_match":
                if state["standing_start"] is None or not contiguous:
                    state["standing_start"] = start
                duration = end - state["standing_start"]
                action = (
                    "stationary_review_candidate"
                    if duration >= 30 - 1e-6
                    else "collect_stationary_evidence"
                )
                emit(
                    action, "Stationary scene evidence; identity unverified",
                    duration, end,
                )
            else:
                state["standing_start"] = None
                actions = {
                    "walking_reference_match": "walking_alert_candidate",
                    "running_reference_match": "running_alert_candidate",
                    "uncertain": "hold_insufficient_evidence",
                }
                emit(actions[label], label, source_time=end)

        except (ValueError, TypeError, KeyError) as error:
            reset(str(error))

    clock_sub = node.create_subscription(
        Clock, "/clock", clock_received, qos_profile_sensor_data
    )
    pose_sub = node.create_subscription(
        String, input_topic, received, 10
    )

    print("Decision preview running. Press Ctrl+C to stop.", flush=True)
    print("No robot-control topics are published.", flush=True)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if node.count_publishers(input_topic) != 1:
                reset("Exactly one pose publisher required")
            elif (
                state["clock_arrival"] is None
                or now - state["clock_arrival"] > 4
            ):
                reset("Simulation clock unavailable")
            elif state["arrival"] is None:
                reset("Waiting for a valid pose window")
            elif now - state["arrival"] > 8:
                reset("Pose message timeout")
            elif (
                state["last_end"] is not None
                and state["clock"] - state["last_end"] > 2.5
            ):
                reset("Pose evidence expired")
    except KeyboardInterrupt:
        pass
    finally:
        reset("Decision preview stopped")
        node.destroy_subscription(pose_sub)
        node.destroy_subscription(clock_sub)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
