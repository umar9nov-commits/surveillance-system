"""Launch the exact localization/navigation commands from the validated demo."""
import argparse
import os
from pathlib import Path
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess

parser = argparse.ArgumentParser()
parser.add_argument('component', choices=('localization', 'navigation', 'rviz'))
args = parser.parse_args()
root = Path.home() / 'surveillance_ws'
commands = {
    'localization': [[
        'ros2', 'launch', 'nav2_bringup', 'localization_launch.py',
        f'params_file:={root / "config/demo_localization.yaml"}',
        f'map:={root / "maps/surveillance_map.yaml"}',
        'use_sim_time:=true', 'autostart:=true',
    ]],
    'navigation': [
        ['ros2', 'run', 'custom_navigation', 'planner_node', '--ros-args',
         '-p', 'use_sim_time:=true', '-p', 'person_exclusion_enabled:=true',
         '-p', 'person_x:=3.94423', '-p', 'person_y:=-1.08', '-p', 'person_min_distance:=1.0'],
        ['ros2', 'run', 'custom_navigation', 'tb3_mpc_controller', '--ros-args',
         '-p', 'use_sim_time:=true', '-p', 'enabled:=false',
         '-p', 'person_exclusion_enabled:=true', '-p', 'person_x:=3.94423',
         '-p', 'person_y:=-1.08', '-p', 'person_min_distance:=1.0',
         '-r', '/plan:=/smooth_path'],
    ],
    'rviz': [['ros2', 'run', 'rviz2', 'rviz2', '-d', str(root / 'mapping.rviz'),
              '--ros-args', '-p', 'use_sim_time:=true']],
}
environment = dict(os.environ)
if args.component == 'rviz':
    environment.update(QT_QPA_PLATFORM='xcb', LIBGL_ALWAYS_SOFTWARE='true', QT_SCALE_FACTOR='1')
service = LaunchService()
service.include_launch_description(LaunchDescription([
    ExecuteProcess(cmd=command, output='screen', additional_env=environment)
    for command in commands[args.component]
]))
raise SystemExit(service.run())
