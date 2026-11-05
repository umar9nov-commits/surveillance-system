"""Display independent camera results. No robot control."""
import json
import time
import tkinter as tk
from datetime import datetime

import rclpy
from std_msgs.msg import String


def main():
    rclpy.init()
    node = rclpy.create_node("dual_camera_notifications")

    root = tk.Tk()
    root.title("External and Robot Camera Notifications")
    root.geometry("760x480")

    tk.Label(
        root, text="SURVEILLANCE CAMERA NOTIFICATIONS",
        font=("Arial", 18, "bold"),
    ).pack(pady=12)

    panels = {}
    state = {}
    history = []

    for source in ("EXTERNAL CAMERA", "ROBOT CAMERA"):
        frame = tk.LabelFrame(
            root, text=source, font=("Arial", 12, "bold")
        )
        frame.pack(fill="x", padx=16, pady=7)

        label = tk.Label(
            frame, text="Waiting for results...",
            font=("Arial", 16, "bold"), fg="gray",
        )
        label.pack(pady=5)

        detail = tk.Label(
            frame, text="", wraplength=710, font=("Arial", 10)
        )
        detail.pack(pady=4)

        panels[source] = (label, detail)
        state[source] = {
            "arrival": None, "key": None, "last_bell": -100,
        }

    history_label = tk.Label(
        root, text="", justify="left", anchor="w",
        font=("Arial", 11), wraplength=720,
    )
    history_label.pack(fill="x", padx=20, pady=12)

    def make_callback(source):
        def receive(message):
            try:
                data = json.loads(message.data)
                if not isinstance(data, dict) or data.get("preview") is not True:
                    return

                now = time.monotonic()
                current = state[source]
                current["arrival"] = now
                label, detail = panels[source]

                if source == "EXTERNAL CAMERA":
                    result = {
                        "walking_alert_candidate": "walking_reference_match",
                        "running_alert_candidate": "running_reference_match",
                    }.get(data.get("action"))
                else:
                    result = data.get("result")

                if result in (
                    "walking_reference_match", "running_reference_match"
                ):
                    running = result == "running_reference_match"
                    title = "RUNNING DETECTED" if running else "WALKING DETECTED"
                    label.config(
                        text=title, fg="red" if running else "#b36b00"
                    )
                    detail.config(
                        text="Activity notification. No patrol interruption requested."
                    )
                    key = (result, data.get("track_id"))

                    if (
                        key != current["key"]
                        and now - current["last_bell"] >= 15
                    ):
                        stamp = datetime.now().strftime("%H:%M:%S")
                        history.insert(0, f"{stamp}  |  {source}  |  {title}")
                        del history[6:]
                        history_label.config(text="\n".join(history))
                        root.bell()
                        current["last_bell"] = now

                    current["key"] = key
                    return

                current["key"] = None
                action = data.get("action")

                if action == "collect_stationary_evidence":
                    duration = float(data.get("stationary_scene_seconds", 0))
                    label.config(text="STATIONARY PERSON", fg="#186a3b")
                    detail.config(
                        text=f"Continuous evidence: {duration:.1f} / 30 seconds"
                    )
                elif action == "stationary_review_candidate":
                    label.config(
                        text="30-SECOND CONDITION REACHED", fg="#186a3b"
                    )
                    detail.config(
                        text="The observation bridge manages the robot response."
                    )
                elif result == "uncertain":
                    label.config(text="ACTIVITY UNCERTAIN", fg="gray")
                    detail.config(text="No confirmed walking or running result.")
                else:
                    label.config(text="EVIDENCE UNAVAILABLE", fg="gray")
                    detail.config(text=str(data.get("reason") or "Waiting"))
            except (ValueError, TypeError, AttributeError):
                pass
        return receive

    subscriptions = [
        node.create_subscription(
            String, "/surveillance/pose_decision_preview",
            make_callback("EXTERNAL CAMERA"), 10,
        ),
        node.create_subscription(
            String, "/surveillance/robot_pose_activity_preview",
            make_callback("ROBOT CAMERA"), 10,
        ),
    ]

    def tick():
        for _ in range(8):
            rclpy.spin_once(node, timeout_sec=0.0)

        now = time.monotonic()
        for source, current in state.items():
            if current["arrival"] is not None and now - current["arrival"] > 12:
                panels[source][0].config(text="NO RECENT RESULTS", fg="gray")
                panels[source][1].config(text="Check this camera's monitor.")
                current["key"] = None

        root.after(100, tick)

    try:
        root.after(100, tick)
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
