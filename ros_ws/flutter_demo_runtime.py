"""ROS readiness and ownership checks for the fixed-person Gazebo demo."""
import math
import os
from pathlib import Path
import signal
import time


class Cancelled(Exception):
    pass


class ProcessTree:
    """Record PID/start-time pairs, including ROS-created process groups."""
    def __init__(self):
        self.tokens = {}

    @staticmethod
    def snapshot():
        result = {}
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
                if fields[0] != 'Z':
                    result[int(path.parent.name)] = (int(fields[1]), fields[19])
            except (OSError, ValueError, IndexError):
                pass
        return result

    def remember(self, roots, current=None):
        if current is None:
            current = self.snapshot()
        for root in roots:
            if root in current and root not in self.tokens:
                self.tokens[root] = current[root][1]
        changed = True
        while changed:
            changed = False
            for pid, (parent, token) in current.items():
                if (pid not in self.tokens and parent in self.tokens
                        and current.get(parent, (None, None))[1] == self.tokens[parent]):
                    self.tokens[pid] = token
                    changed = True

    def alive(self):
        current = self.snapshot()
        return any(current.get(pid, (None, None))[1] == token
                   for pid, token in self.tokens.items())

    def send(self, sig):
        current = self.snapshot()
        for pid, token in list(self.tokens.items()):
            if current.get(pid, (None, None))[1] == token:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass


class RosProbe:
    def __init__(self, host):
        import rclpy
        from lifecycle_msgs.srv import GetState
        from nav_msgs.msg import OccupancyGrid
        from rcl_interfaces.srv import GetParameters, SetParameters
        from rclpy.parameter import Parameter
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image
        from tf2_ros import Buffer, TransformListener
        self.rclpy = rclpy
        self.host = host
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node('flutter_demo_startup')
        self.node.set_parameters([Parameter('use_sim_time', value=True)])
        self.clock = 0.0
        self.first_clock = None
        self.clock_arrival = 0.0
        self.map_received = False
        self.camera_arrival = 0.0
        self.map_subscription = self.node.create_subscription(
            OccupancyGrid, '/map', self.on_map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        self.clock_subscription = self.node.create_subscription(
            Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.camera_subscription = self.node.create_subscription(
            Image, '/surveillance/observer/image_raw', self.on_camera,
            qos_profile_sensor_data)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.node)
        self.lifecycle = {
            name: self.node.create_client(GetState, '/' + name + '/get_state')
            for name in ('map_server', 'amcl')}
        self.lifecycle_futures = {}
        self.lifecycle_values = {}
        self.setter = self.node.create_client(
            SetParameters, '/tb3_mpc_controller/set_parameters')
        self.getters = {name: self.node.create_client(GetParameters, '/' + name + '/get_parameters')
                        for name in ('planner_node', 'tb3_mpc_controller')}

    def on_clock(self, msg):
        value = msg.clock.sec + msg.clock.nanosec / 1e9
        if self.first_clock is None:
            self.first_clock = value
        self.clock = value
        self.clock_arrival = time.monotonic()

    def on_map(self, msg):
        self.map_received = (msg.info.width > 0 and msg.info.height > 0
                             and msg.info.resolution > 0 and len(msg.data) > 0)

    def on_camera(self, msg):
        if msg.width > 0 and msg.height > 0 and msg.data:
            self.camera_arrival = time.monotonic()

    def spin(self, timeout=0.05):
        self.host.refresh()
        self.rclpy.spin_once(self.node, timeout_sec=timeout)

    def request(self, client, request, timeout=5, allow_stopping=False):
        until = time.monotonic() + timeout
        while not client.service_is_ready():
            if not allow_stopping and self.host.stop.is_set():
                raise Cancelled()
            if time.monotonic() >= until:
                raise RuntimeError('ROS service unavailable: ' + client.srv_name)
            self.spin()
        future = client.call_async(request)
        while not future.done():
            if not allow_stopping and self.host.stop.is_set():
                future.cancel()
                raise Cancelled()
            if time.monotonic() >= until:
                future.cancel()
                raise RuntimeError('ROS service timed out: ' + client.srv_name)
            self.spin()
        response = future.result()
        if response is None:
            raise RuntimeError('Empty ROS service response: ' + client.srv_name)
        return response

    def enabled(self, value, allow_stopping=False):
        from rcl_interfaces.srv import SetParameters
        from rclpy.parameter import Parameter
        request = SetParameters.Request()
        request.parameters = [Parameter('enabled', value=value).to_parameter_msg()]
        response = self.request(self.setter, request, allow_stopping=allow_stopping)
        if len(response.results) != 1 or not response.results[0].successful:
            raise RuntimeError('Controller rejected enabled=' + str(value))
        if not allow_stopping:
            actual = self.values('tb3_mpc_controller', ['enabled'])[0]
            if actual.type != 1 or actual.bool_value != value:
                raise RuntimeError('Controller enabled setting was not confirmed.')

    def values(self, name, names):
        from rcl_interfaces.srv import GetParameters
        request = GetParameters.Request()
        request.names = names
        response = self.request(self.getters[name], request)
        if len(response.values) != len(names):
            raise RuntimeError('Incomplete parameter response from ' + name)
        return response.values

    def lifecycle_ready(self):
        from lifecycle_msgs.srv import GetState
        for name, client in self.lifecycle.items():
            if name not in self.lifecycle_futures and client.service_is_ready():
                self.lifecycle_futures[name] = (client.call_async(GetState.Request()), time.monotonic())
            pending = self.lifecycle_futures.get(name)
            if pending is not None:
                future, started = pending
                if future.done():
                    result = future.result()
                    self.lifecycle_values[name] = result.current_state.id if result else None
                    del self.lifecycle_futures[name]
                elif time.monotonic() - started > 3:
                    future.cancel()
                    del self.lifecycle_futures[name]
        return all(self.lifecycle_values.get(name) == 3 for name in self.lifecycle)

    def ready_problem(self):
        from rclpy.time import Time
        from tf2_ros import TransformException
        names = self.node.get_node_names()
        for name in ('map_server', 'amcl', 'planner_node', 'tb3_mpc_controller',
                     'lifecycle_manager_localization'):
            if names.count(name) != 1:
                return 'Waiting for exactly one ' + name
        if self.node.count_publishers('/clock') != 1:
            return 'Waiting for one simulation clock publisher'
        if (self.first_clock is None or self.clock <= self.first_clock
                or time.monotonic() - self.clock_arrival > 4):
            return 'Waiting for an advancing simulation clock'
        if not self.lifecycle_ready():
            return 'Waiting for active map server and AMCL'
        if not self.map_received:
            return 'Waiting for occupancy map'
        if time.monotonic() - self.camera_arrival > 4:
            return 'Waiting for external-camera frames'
        if not self.setter.service_is_ready() or not all(c.service_is_ready() for c in self.getters.values()):
            return 'Waiting for planner/controller services'
        if not any(item.node_name == 'planner_node'
                   for item in self.node.get_subscriptions_info_by_topic('/goal_pose')):
            return 'Waiting for planner goal subscription'
        if not any(item.node_name == 'tb3_mpc_controller'
                   for item in self.node.get_subscriptions_info_by_topic('/smooth_path')):
            return 'Waiting for controller path subscription'
        try:
            transform = self.buffer.lookup_transform('map', 'base_link', Time())
            stamp = transform.header.stamp.sec + transform.header.stamp.nanosec / 1e9
            if not -0.5 <= self.clock - stamp <= 3:
                return 'Waiting for fresh map-to-robot transform'
        except TransformException:
            return 'Waiting for map-to-robot transform'
        return None

    def check_parameters(self):
        for name in self.getters:
            values = self.values(name, ['person_exclusion_enabled', 'person_x',
                                       'person_y', 'person_min_distance', 'use_sim_time'])
            if values[0].type != 1 or not values[0].bool_value:
                raise RuntimeError('Person exclusion must be enabled in ' + name)
            for item, expected in zip(values[1:4], (3.94423, -1.08, 1.0)):
                value = item.double_value if item.type == 3 else item.integer_value if item.type == 2 else math.nan
                if not math.isfinite(value) or abs(value - expected) > 1e-6:
                    raise RuntimeError('Fixed-person configuration mismatch in ' + name)
            if values[4].type != 1 or not values[4].bool_value:
                raise RuntimeError('Simulation time must be enabled in ' + name)
        if self.values('tb3_mpc_controller', ['enabled'])[0].bool_value:
            raise RuntimeError('Controller must start disabled before automatic patrol setup.')

    def close(self):
        self.node.destroy_node()
        self.rclpy.try_shutdown()
