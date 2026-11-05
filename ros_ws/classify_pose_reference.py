import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--file", type=Path)
source.add_argument("--latest", choices=[
    "standing", "scripted_walk", "scripted_run"
])
args = parser.parse_args()

root = Path.home() / "surveillance_ws/activity_samples"

if args.file:
    path = args.file.expanduser()
else:
    matches = sorted(root.glob(f"{args.latest}_*/poses.json"))
    if not matches:
        raise SystemExit("No matching recording found.")
    path = matches[-1]

data = json.loads(path.read_text())
if not data.get("recording_complete") or data.get("errors"):
    raise SystemExit("Recording is incomplete or contains errors.")

records = data.get("records", [])
if len(records) < 2:
    raise SystemExit("Not enough records.")

times = np.array([r["source_time"] for r in records], dtype=float)
if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
    raise SystemExit("Source timestamps must be finite and increasing.")

def classify(speed):
    if 0 <= speed <= 0.25:
        return "standing_reference_match"
    if 0.55 <= speed <= 1.06:
        return "walking_reference_match"
    if 1.25 <= speed <= 1.90:
        return "running_reference_match"
    return "uncertain"

print("File:", path)
print("Reference label:", data.get("reference_label"))
print("Method: two-second median normalized hip speed")
print("No ROS messages or navigation commands published.")
print()

counts = {}
start, end = times[0], times[-1]

for boundary in np.arange(start, end - 2.0 + 1e-6, 2.0):
    rows = []
    for r in records:
        if not boundary - 1e-6 <= r["source_time"] <= boundary + 2 + 1e-6:
            continue
        if r.get("people") != 1:
            continue
        if "keypoints_xy" not in r or "keypoint_confidences" not in r:
            continue

        p = np.asarray(r["keypoints_xy"], dtype=float)
        c = np.asarray(r["keypoint_confidences"], dtype=float)
        if p.shape != (17, 2) or c.shape != (17,):
            continue
        selected = [5, 6, 11, 12]
        if not np.all(np.isfinite(p[selected])):
            continue
        if not np.all(np.isfinite(c[selected])) or np.min(c[selected]) < 0.5:
            continue

        hip = p[[11, 12]].mean(axis=0)
        shoulder = p[[5, 6]].mean(axis=0)
        scale = np.linalg.norm(hip - shoulder)
        if scale > 1:
            rows.append((r["source_time"], hip, scale))

    result = "insufficient_data"
    speed = None

    if len(rows) >= 7:
        t = np.array([r[0] for r in rows])
        hips = np.stack([r[1] for r in rows])
        scales = np.array([r[2] for r in rows])
        dt = np.diff(t)

        if t[-1] - t[0] >= 1.6 and np.all(dt > 0) and np.max(dt) <= 0.400001:
            displacement = np.linalg.norm(np.diff(hips, axis=0), axis=1)
            pair_scale = (scales[:-1] + scales[1:]) / 2
            speed = float(np.median(displacement / dt / pair_scale))
            result = classify(speed)

    counts[result] = counts.get(result, 0) + 1
    value = "unavailable" if speed is None else f"{speed:.4f}"
    elapsed = boundary - start
    print(f"{elapsed:4.1f}-{elapsed + 2:4.1f} s | {value:>11} | {result}")

print("\nWindow counts:", json.dumps(counts, sort_keys=True))
