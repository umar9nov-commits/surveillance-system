"""Display ROS activity decisions. Does not control the robot."""
import json
import time
import tkinter as tk
from datetime import datetime

import rclpy
from std_msgs.msg import String


def main():
    rclpy.init()
    node = rclpy.create_node("activity_notification_window")

    root = tk.Tk()
    root.title("Surveillance Activity Notifications")
    root.geometry("620x360")

    tk.Label(
        root, text="SURVEILLANCE ACTIVITY",
        font=("Arial", 18, "bold"),
    ).pack(pady=12)

    status = tk.Label(
        root, text="Waiting for activity decisions...",
        font=("Arial", 17, "bold"), fg="gray",
        wraplength=590,
    )
    status.pack(pady=10)

    detail = tk.Label(
        root, text="", font=("Arial", 11), wraplength=590
    )
    detail.pack(pady=5)

    tk.Label(
        root, text="Recent walking / running notifications",
        font=("Arial", 11, "bold"),
    ).pack(pady=(15, 5))

    history_label = tk.Label(
        root, text="", font=("Arial", 11),
        justify="left", anchor="w", wraplength=590,
    )
    history_label.pack(fill="x", padx=20)

    state = {
        "last_alert": None,
        "last_received": None,
        "history": [],
    }

    def receive(message):
        try:
            data = json.loads(message.data)
            if not isinstance(data, dict) or data.get("preview") is not True:
                return

            action = data.get("action")
            track = data.get("track_id")
            state["last_received"] = time.monotonic()

            if action in ("walking_alert_candidate", "running_alert_candidate"):
                running = action == "running_alert_candidate"
                label = "RUNNING DETECTED" if running else "WALKING DETECTED"
                status.config(
                    text=label,
                    fg="red" if running else "#b36b00",
                )
                detail.config(
                    text="Activity alert. No patrol interruption requested."
                )

                key = (action, track)
                if key != state["last_alert"]:
                    stamp = datetime.now().strftime("%H:%M:%S")
                    state["history"].insert(0, f"{stamp}   {label}")
                    state["history"] = state["history"][:6]
                    history_label.config(
                        text="\n".join(state["history"])
                    )
                    root.bell()
                state["last_alert"] = key

            elif action == "collect_stationary_evidence":
                state["last_alert"] = None
                duration = float(data.get("stationary_scene_seconds", 0))
                status.config(text="STATIONARY PERSON", fg="#186a3b")
                detail.config(
                    text=f"Continuous evidence: {duration:.1f} / 30 seconds"
                )

            elif action == "stationary_review_candidate":
                state["last_alert"] = None
                status.config(text="30-SECOND CONDITION REACHED", fg="#186a3b")
                detail.config(
                    text="Observation candidate issued. "
                         "The observation bridge manages the robot response."
                )

            else:
                state["last_alert"] = None
                status.config(text="ACTIVITY EVIDENCE UNAVAILABLE", fg="gray")
                detail.config(text=str(data.get("reason") or "Waiting"))

        except (ValueError, TypeError, AttributeError):
            pass

    subscription = node.create_subscription(
        String, "/surveillance/pose_decision_preview", receive, 10
    )

    def tick():
        rclpy.spin_once(node, timeout_sec=0.0)
        last = state["last_received"]
        if last is not None and time.monotonic() - last > 10:
            state["last_alert"] = None
            status.config(text="NO RECENT DECISION UPDATES", fg="gray")
            detail.config(text="Check the pose monitor and decision component.")
        root.after(100, tick)

    try:
        root.after(100, tick)
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
