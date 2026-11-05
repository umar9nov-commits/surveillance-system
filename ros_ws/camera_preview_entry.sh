#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
source "$HOME/surveillance_ws/install/setup.bash"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec "$HOME/surveillance_ws/.venv_yolo/bin/python" -u \
    "$HOME/surveillance_ws/camera_preview_stdout.py" "$@"
