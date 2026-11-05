#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source "$HOME/surveillance_ws/install/setup.bash"
export TURTLEBOT3_MODEL=burger

# Robot-mounted camera overlay
export AMENT_PREFIX_PATH="$HOME/surveillance_ws/simulation/robot_camera_overlay${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"

/usr/bin/python3 - "$@" <<'PY'
import argparse
import os
import re
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import (
    EmitEvent, ExecuteProcess, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

parser = argparse.ArgumentParser()
parser.add_argument('--gui', action='store_true',
                    help='Open the Gazebo 3D window.')
parser.add_argument('--rviz', action='store_true',
                    help='Open RViz.')
parser.add_argument('--viewer', action='store_true',
                    help='Open the camera image viewer.')
# Optional standing-animation baseline
actor_modes = parser.add_mutually_exclusive_group()
actor_modes.add_argument(
    '--standing', action='store_true',
    help='Use the standing actor animation for pose testing.',
)
actor_modes.add_argument(
    '--walking', action='store_true',
    help='Use a scripted walking actor for pose testing.',
)
actor_modes.add_argument(
    '--running', action='store_true',
    help='Use a scripted running actor for pose testing.',
)
options = parser.parse_args()

workspace = Path.home() / 'surveillance_ws'
map_file = workspace / 'maps/surveillance_map.yaml'
rviz_file = workspace / 'mapping.rviz'
person_file = workspace / 'simulation/surveillance_person.sdf'
camera_file = workspace / 'simulation/observer_camera.sdf'
detector = workspace / 'yolo_person_detector.py'
vision_python = workspace / '.venv_yolo/bin/python'

required = [
    map_file, person_file, camera_file, detector, vision_python,
]
if options.rviz:
    required.append(rviz_file)

for file in required:
    if not file.is_file():
        raise SystemExit(f'Required file missing: {file}')

person_xml = ET.parse(person_file)
for element in person_xml.findall('.//filename'):
    value = (element.text or '').strip()
    if value.startswith('/') and not Path(value).is_file():
        raise SystemExit(f'Human model asset missing: {value}')

pattern = (
    r'gz sim |gz-sim-server|/rviz2/rviz2'
    r'|async_slam_toolbox_node'
    r'|/nav2_amcl/amcl|/nav2_map_server/map_server'
    r'|/custom_navigation/(planner_node|tb3_mpc_controller)'
    r'|yolo_person_detector[.]py'
    r'|__node:=observer_camera_bridge'
    r'|__node:=robot_camera_bridge'
    r'|__node:=robot_camera_mount_tf'
    r'|__node:=robot_camera_optical_tf'
    r'|__node:=simulation_clock_bridge'
    r'|__node:=lifecycle_manager_localization'
)
result = subprocess.run(
    ['pgrep', '-af', pattern], capture_output=True, text=True,
)
if result.returncode not in (0, 1):
    raise SystemExit('Could not check existing processes.')

matches = [
    line for line in result.stdout.splitlines()
    if int(line.split()[0]) != os.getpid()
]
if matches:
    raise SystemExit(
        'Existing sessions remain. Close their launch terminals first:\n'
        + '\n'.join(matches)
    )

share = Path(get_package_share_directory('turtlebot3_gazebo'))
world_launch = share / 'launch/turtlebot3_world.launch.py'
world_description = runpy.run_path(str(world_launch))[
    'generate_launch_description'
]()

def static_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ''.join(static_text(item) for item in value)
    return getattr(value, 'text', '')

def is_gui_action(action):
    source = getattr(action, 'launch_description_source', None)
    location = static_text(getattr(source, 'location', ''))
    if 'gzclient' in location or 'gz_sim_gui' in location:
        return True

    arguments = getattr(action, 'launch_arguments', [])
    for name, value in arguments:
        if static_text(name) == 'gz_args':
            if re.search(r'(^|\s)-g(\s|$)', static_text(value)):
                return True
    return False

world_actions = list(world_description.entities)
if not options.gui:
    gui_actions = [
        action for action in world_actions if is_gui_action(action)
    ]
    if not gui_actions:
        raise SystemExit(
            'Could not identify the Gazebo GUI action safely.\n'
            f'Send the contents of: {world_launch}\n'
            'No simulation was started.'
        )
    world_actions = [
        action for action in world_actions if not is_gui_action(action)
    ]

def process(command, **kwargs):
    return ExecuteProcess(cmd=command, output='screen', **kwargs)

spawn_code = r'''
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

workspace = Path(sys.argv[1])
standing_mode = sys.argv[2] == 'standing'
walking_mode = sys.argv[2] == 'walking'
running_mode = sys.argv[2] == 'running'

def service(name, request_type, reply_type, request, timeout=2000):
    return subprocess.run([
        'gz', 'service', '-s', name,
        '--reqtype', request_type,
        '--reptype', reply_type,
        '--timeout', str(timeout),
        '--req', request,
    ], capture_output=True, text=True, timeout=timeout/1000 + 3)

def scene():
    result = service(
        '/world/default/scene/info',
        'gz.msgs.Empty', 'gz.msgs.Scene', '',
    )
    return result.stdout if result.returncode == 0 else ''

def has_entity(text, name):
    return re.search(
        r'\bname:\s*"' + re.escape(name) + r'"', text
    ) is not None

print('Waiting for Gazebo world and TurtleBot3...', flush=True)
deadline = time.monotonic() + 90.0
while time.monotonic() < deadline:
    if has_entity(scene(), 'burger'):
        break
    time.sleep(0.5)
else:
    raise SystemExit('Gazebo world/robot did not become ready.')

entities = [
    ('surveillance_person',
     'person_standing_test.sdf' if standing_mode
     else 'person_walking_test.sdf' if walking_mode
     else 'person_running_test.sdf' if running_mode
     else 'surveillance_person.sdf'),
    ('surveillance_observer_camera', 'observer_camera.sdf'),
]

for name, filename in entities:
    if has_entity(scene(), name):
        print('Already present:', name, flush=True)
        continue

    root = ET.parse(workspace / 'simulation' / filename).getroot()

    if name == 'surveillance_observer_camera':
        for rate in root.findall('.//sensor/update_rate'):
            rate.text = '5'

    sdf = ET.tostring(root, encoding='unicode')
    result = service(
        '/world/default/create',
        'gz.msgs.EntityFactory',
        'gz.msgs.Boolean',
        'sdf: ' + json.dumps(sdf) + ' allow_renaming: false',
        timeout=10000,
    )
    if (
        result.returncode != 0
        or not re.search(r'\bdata:\s*true\b', result.stdout)
    ):
        print(result.stdout, result.stderr, flush=True)
        raise SystemExit('Could not insert: ' + name)

    # Actors may not appear in the model list returned by scene/info.
    # The successful creation response confirms acceptance by Gazebo.
    if name == 'surveillance_person':
        print('Human creation accepted by Gazebo.', flush=True)
        continue

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if has_entity(scene(), name):
            print('Inserted:', name, flush=True)
            break
        time.sleep(0.2)
    else:
        raise SystemExit('Inserted entity not found: ' + name)

print('Human and observer camera ready.', flush=True)
'''

spawner = process([
    '/usr/bin/python3', '-c', spawn_code, str(workspace),
    'running' if options.running else
    'walking' if options.walking else
    'standing' if options.standing else 'demo',
])

def after_spawn(event, context):
    if event.returncode != 0:
        return [EmitEvent(event=Shutdown(
            reason='Human/camera setup failed.'
        ))]

    actions = []

    import rclpy
    import time

    rclpy.init()
    check = rclpy.create_node('surveillance_startup_check')
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rclpy.spin_once(check, timeout_sec=0.1)
    clock_publishers = check.count_publishers('/clock')
    check.destroy_node()
    rclpy.try_shutdown()

    if clock_publishers > 1:
        return [EmitEvent(event=Shutdown(
            reason='Multiple ROS clock publishers detected.'
        ))]

    if clock_publishers == 0:
        actions.append(process([
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '--ros-args', '-r', '__node:=simulation_clock_bridge',
        ]))

    actions.extend([
        process([
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/surveillance/robot/image_raw'
            '@sensor_msgs/msg/Image[gz.msgs.Image',
            '--ros-args', '-r', '__node:=robot_camera_bridge',
        ]),
        process([
            'ros2', 'run', 'tf2_ros', 'static_transform_publisher',
            '--x', '0.04', '--y', '0', '--z', '0.25',
            '--roll', '0', '--pitch', '-0.30', '--yaw', '0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'robot_camera_link',
            '--ros-args', '-r', '__node:=robot_camera_mount_tf',
        ]),
        process([
            'ros2', 'run', 'tf2_ros', 'static_transform_publisher',
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '-1.5707963267948966',
            '--pitch', '0',
            '--yaw', '-1.5707963267948966',
            '--frame-id', 'robot_camera_link',
            '--child-frame-id', 'robot_camera_optical_frame',
            '--ros-args', '-r', '__node:=robot_camera_optical_tf',
        ]),
        process([
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/surveillance/observer/image_raw'
            '@sensor_msgs/msg/Image[gz.msgs.Image',
            '--ros-args', '-r', '__node:=observer_camera_bridge',
        ]),

    ])

    gui_environment = {
        'QT_QPA_PLATFORM': 'xcb',
        'LIBGL_ALWAYS_SOFTWARE': 'true',
        'QT_SCALE_FACTOR': '1',
    }

    if options.rviz:
        actions.append(process([
            'ros2', 'run', 'rviz2', 'rviz2',
            '-d', str(rviz_file),
            '--ros-args', '-p', 'use_sim_time:=true',
        ], additional_env=gui_environment))

    if options.viewer:
        actions.append(process([
            'ros2', 'run', 'rqt_image_view', 'rqt_image_view',
        ], additional_env=gui_environment))

    print(
        '\nGazebo and cameras ready. Start AMCL Localisation, Custom Navigation and YOLO from Flutter.\n'
        'MPC remains DISABLED.\n'
        'Observer view: /surveillance/observer/image_raw\n'
        'Robot view: /surveillance/robot/image_raw\n',
        flush=True,
    )
    return actions

description = LaunchDescription([
    *world_actions,
    RegisterEventHandler(OnProcessExit(
        target_action=spawner,
        on_exit=after_spawn,
    )),
    spawner,
])

service = LaunchService()
service.include_launch_description(description)
sys.exit(service.run())
PY
