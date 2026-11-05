"""Read-only camera/map transport for Flutter. Never publishes robot commands."""
import json
import math
import time

def yaw(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))

def map_payload(message):
    info = message.info
    if (message.header.frame_id != 'map' or not 0 < info.width*info.height <= 262144
            or info.resolution <= 0 or not math.isfinite(info.resolution)
            or len(message.data) != info.width*info.height):
        raise ValueError('Expected a valid map-frame occupancy grid.')
    origin = [info.origin.position.x, info.origin.position.y, yaw(info.origin.orientation)]
    if not all(math.isfinite(v) for v in origin):
        raise ValueError('Invalid map origin.')
    return {'width': info.width, 'height': info.height,
            'resolution': info.resolution, 'origin': origin, 'data': list(message.data)}

def path_payload(message):
    if message.header.frame_id != 'map':
        return []
    points = [[p.pose.position.x, p.pose.position.y] for p in message.poses]
    if not all(math.isfinite(v) for point in points for v in point):
        return []
    if len(points) > 2000:
        points = points[::math.ceil(len(points)/1999)] + [points[-1]]
    return points

def pose_is_fresh(clock, clock_arrival, stamp, now):
    return clock > 0 and now-clock_arrival < 8 and -0.5 <= clock-stamp <= 2.0

def main():
    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from nav_msgs.msg import OccupancyGrid, Path as NavPath
    from rclpy.clock import Clock
    from rclpy.executors import ExternalShutdownException
    from rclpy.clock_type import ClockType
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock as ClockMessage
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener
    from rclpy.time import Time

    rclpy.init()
    node = rclpy.create_node('flutter_display_bridge',
        parameter_overrides=[Parameter('use_sim_time', value=True)])
    cv2.setNumThreads(1)
    bridge = CvBridge()
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    latest = {}
    sent = {}
    state = {'map': None, 'path': [], 'clock': 0.0, 'clock_arrival': 0.0}
    pub = {key: node.create_publisher(String, '/surveillance/flutter/' + key, 1)
           for key in ('external_camera', 'robot_camera', 'map')}
    subscriptions = []
    def publish(key, payload):
        msg = String()
        msg.data = json.dumps(payload, allow_nan=False, separators=(',', ':'))
        pub[key].publish(msg)

    def receive_camera(key, message):
        latest[key] = message

    for key, topic in (('external_camera', '/surveillance/observer/image_raw'),
                       ('robot_camera', '/surveillance/robot/image_raw')):
        def receive(message, camera_key=key):
            receive_camera(camera_key, message)
        subscriptions.append(node.create_subscription(Image, topic, receive, qos_profile_sensor_data))

    def receive_map(msg):
        try:
            state['map'] = map_payload(msg)
        except ValueError as error:
            node.get_logger().warning(str(error))
    def receive_path(msg):
        state['path'] = path_payload(msg)
    def receive_clock(msg):
        value = msg.clock.sec + msg.clock.nanosec/1e9
        if value != state['clock']:
            if value < state['clock']:
                state['path'] = []
            state['clock'] = value
            state['clock_arrival'] = time.monotonic()
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    subscriptions += [
        node.create_subscription(OccupancyGrid, '/map', receive_map, latched),
        node.create_subscription(NavPath, '/smooth_path', receive_path, latched),
        node.create_subscription(ClockMessage, '/clock', receive_clock, qos_profile_sensor_data),
    ]
    last_map = 0.0
    last_warning = {}
    def warn(key, error):
        now = time.monotonic()
        if now-last_warning.get(key, -100) > 10:
            node.get_logger().warning(key + ': ' + str(error))
            last_warning[key] = now

    def update():
        nonlocal last_map
        import base64
        for key, msg in list(latest.items()):
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            if sent.get(key) == stamp:
                continue
            sent[key] = stamp
            try:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                height, width = frame.shape[:2]
                if width > 640:
                    frame = cv2.resize(frame, (640, max(1, round(height*640/width))),
                                       interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    publish(key, {'jpeg': base64.b64encode(encoded).decode('ascii'),
                                  'source_time': stamp[0] + stamp[1]/1e9})
            except Exception as error:
                warn(key, error)

        now = time.monotonic()
        if now-last_map < 1.0:
            return
        last_map = now
        robot = None
        clock_running = state['clock'] > 0 and now-state['clock_arrival'] < 8
        try:
            tf = buffer.lookup_transform('map', 'base_link', Time())
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec/1e9
            if pose_is_fresh(state['clock'], state['clock_arrival'], stamp, now):
                t, q = tf.transform.translation, tf.transform.rotation
                values = [t.x, t.y, yaw(q)]
                if all(math.isfinite(v) for v in values):
                    robot = values
        except Exception:
            pass
        publish('map', {'grid': state['map'], 'path': state['path'],
                        'robot': robot, 'clock_running': clock_running,
                        'source_time': state['clock']})

    steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
    timer = node.create_timer(0.5, update, clock=steady_clock)
    node.get_logger().info('Flutter camera and map display bridge ready. Read-only; no robot commands.')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
