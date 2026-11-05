#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger

python3 - <<'PY'
import os
import subprocess
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess

config = os.path.expanduser('~/surveillance_ws/mapping.rviz')

if not os.path.isfile(config):
    raise SystemExit('Save your RViz configuration as mapping.rviz first.')

pattern = r'gz sim .* -s|async_slam_toolbox_node|/rviz2/rviz2'
running = subprocess.run(
    ['pgrep', '-af', pattern],
    capture_output=True,
    text=True,
)
if running.returncode == 0:
    print('Existing processes found:\n' + running.stdout)
    raise SystemExit(
        'Close the existing Gazebo, SLAM, and RViz sessions first.'
    )

description = LaunchDescription([
    ExecuteProcess(
        name='gazebo',
        cmd=['ros2', 'launch', 'turtlebot3_gazebo',
             'turtlebot3_house.launch.py'],
        output='screen',
    ),
    ExecuteProcess(
        name='slam',
        cmd=['ros2', 'launch', 'slam_toolbox',
             'online_async_launch.py', 'use_sim_time:=true', 'slam_params_file:=/home/umaryasir/surveillance_ws/config/slam_mapping.yaml'],
        output='screen',
    ),
    ExecuteProcess(
        name='rviz',
        cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', config,
             '--ros-args', '-p', 'use_sim_time:=true'],
        additional_env={
            'QT_QPA_PLATFORM': 'xcb',
            'LIBGL_ALWAYS_SOFTWARE': 'true',
            'QT_SCALE_FACTOR': '1',
        },
        output='screen',
    ),
])

service = LaunchService()
service.include_launch_description(description)
raise SystemExit(service.run())
PY
