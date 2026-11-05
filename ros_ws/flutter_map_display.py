"""Read-only map transport. No camera subscriptions or image processing."""
import json
import math
import time

def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

def grid_json(message):
    info = message.info
    if (message.header.frame_id != 'map'
            or info.width <= 0 or info.height <= 0
            or info.width*info.height > 262144
            or info.resolution <= 0 or not math.isfinite(info.resolution)
            or len(message.data) != info.width*info.height):
        raise ValueError('Invalid map-frame grid.')
    origin = [info.origin.position.x, info.origin.position.y, yaw(info.origin.orientation)]
    if not all(math.isfinite(v) for v in origin):
        raise ValueError('Invalid map origin.')
    return json.dumps({'width': info.width, 'height': info.height,
                       'resolution': info.resolution, 'origin': origin,
                       'data': list(message.data)},
                      allow_nan=False, separators=(',', ':'))

def path_json(message):
    if message.header.frame_id != 'map':
        return '[]'
    points = [[p.pose.position.x, p.pose.position.y] for p in message.poses]
    if not all(math.isfinite(v) for point in points for v in point):
        return '[]'
    if len(points) > 2000:
        points = points[::math.ceil(len(points)/1999)] + [points[-1]]
    return json.dumps(points, allow_nan=False, separators=(',', ':'))

def fresh(clock, arrived, stamp, now):
    return clock > 0 and now-arrived < 8 and -0.5 <= clock-stamp <= 2.0

def packet(grid, path, robot, clock, running):
    state = json.dumps({'robot': robot, 'source_time': clock, 'clock_running': running},
                       allow_nan=False, separators=(',', ':'))
    # Grid/path are validated, cached JSON. Do not re-encode all map cells on ticks.
    return state[:-1] + ',"grid":' + grid + ',"path":' + path + '}'

def main():
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Path
    from rclpy.clock import Clock
    from rclpy.clock_type import ClockType
    from rclpy.executors import ExternalShutdownException
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
    from rclpy.time import Time
    from rosgraph_msgs.msg import Clock as ClockMessage
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = rclpy.create_node('flutter_map_display',
        parameter_overrides=[Parameter('use_sim_time', value=True)])
    state = {'grid': 'null', 'path': '[]', 'clock': 0.0, 'arrival': 0.0}
    publisher = node.create_publisher(String, '/surveillance/flutter/map', 1)
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    def on_map(message):
        try:
            state['grid'] = grid_json(message)
        except ValueError as error:
            node.get_logger().warning(str(error))
    def on_path(message):
        state['path'] = path_json(message)
    def on_clock(message):
        current = message.clock.sec + message.clock.nanosec/1e9
        if current != state['clock']:
            if current < state['clock']:
                state['path'] = '[]'
            state['clock'] = current
            state['arrival'] = time.monotonic()
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    subscriptions = [
        node.create_subscription(OccupancyGrid, '/map', on_map, latched),
        node.create_subscription(Path, '/smooth_path', on_path, latched),
        node.create_subscription(ClockMessage, '/clock', on_clock, qos_profile_sensor_data),
    ]
    def update():
        # No packets/TF lookups while the Map tab has no subscriber.
        if publisher.get_subscription_count() == 0:
            return
        now = time.monotonic()
        clock = state['clock']
        running = clock > 0 and now-state['arrival'] < 8
        robot = None
        try:
            transform = buffer.lookup_transform('map', 'base_link', Time())
            stamp = transform.header.stamp.sec + transform.header.stamp.nanosec/1e9
            if fresh(clock, state['arrival'], stamp, now):
                t = transform.transform.translation
                values = [t.x, t.y, yaw(transform.transform.rotation)]
                if all(math.isfinite(v) for v in values):
                    robot = values
        except Exception:
            pass
        msg = String()
        msg.data = packet(state['grid'], state['path'], robot, clock, running)
        publisher.publish(msg)
    steady = Clock(clock_type=ClockType.STEADY_TIME)
    timer = node.create_timer(2.0, update, clock=steady)
    node.get_logger().info(
        'MAP-ONLY display ready. Two-second refresh; no camera processing or robot commands.')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
