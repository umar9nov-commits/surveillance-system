"""Connect a simulated running decision to controller disable."""

import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


class ActivityStopBridge(Node):
    def __init__(self):
        super().__init__('activity_stop_bridge')
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time', value=True)
        ])
        self.client = self.create_client(
            SetParameters, '/tb3_mpc_controller/set_parameters'
        )
        self.subscription = self.create_subscription(
            String, '/surveillance/decision', self.on_decision, 10
        )
        self.pending = False
        self.latched = False
        self.future = None
        self.attempts = 0
        self.sent_at = 0.0
        self.next_attempt = 0.0
        self.timer = self.create_timer(0.2, self.check)
        self.get_logger().info(
            'Stop bridge ready. Waiting for a fresh running decision.'
        )

    def on_decision(self, msg):
        if self.latched:
            return

        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
            if (
                data.get('activity') != 'running'
                or data.get('action') != 'notify_and_request_pause'
                or data.get('prototype_only') is not True
            ):
                return

            timestamp = float(data['simulation_time'])
            now = self.get_clock().now().nanoseconds / 1e9
            age = now - timestamp
            if not math.isfinite(timestamp) or not 0 <= age <= 2.0:
                self.get_logger().warning('Old or invalid decision ignored.')
                return
        except (ValueError, TypeError, KeyError):
            self.get_logger().warning('Invalid decision ignored.')
            return

        self.latched = True
        self.pending = True
        self.get_logger().warning(
            'Running decision received. Requesting controller disable.'
        )

    def check(self):
        if not self.pending:
            return

        now = time.monotonic()

        if self.future is not None:
            if self.future.done():
                try:
                    response = self.future.result()
                    success = (
                        len(response.results) == 1
                        and response.results[0].successful
                    )
                except Exception:
                    success = False

                self.future = None
                if success:
                    self.pending = False
                    self.get_logger().info(
                        'Controller disable confirmed. '
                        'Patrol should cancel at its next enabled check.'
                    )
                    return
                self.next_attempt = now + 1.0

            elif now - self.sent_at <= 2.0:
                return
            else:
                self.future.cancel()
                self.future = None
                self.next_attempt = now + 1.0

        if now < self.next_attempt:
            return

        if self.attempts >= 3:
            self.pending = False
            self.get_logger().error(
                'Disable could not be confirmed. Run: '
                'ros2 param set /tb3_mpc_controller enabled false'
            )
            return

        self.attempts += 1
        if not self.client.service_is_ready():
            self.next_attempt = now + 1.0
            return

        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name='enabled',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=False,
                ),
            )
        ]
        self.future = self.client.call_async(request)
        self.sent_at = now


def main():
    rclpy.init()
    node = ActivityStopBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
