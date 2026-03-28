import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from cf_control_msgs.msg import ThrustAndTorque
from uav_model.params import UAVParameters
from uav_model.state import UAVState
from uav_model.model import dynamics
from uav_model.integrator import RK4Integrator


class UAVModelNode(Node):
    def __init__(self):
        super().__init__('uav_model_node')

        self._declare_parameters()
        params = self._load_parameters()

        self._params = params
        self._dyn_params = {
            'mass': params.mass,
            'J': params.J,
            'gravity': params.gravity,
        }

        self._state = UAVState()
        self._integrator = RK4Integrator(dynamics)
        self._control = np.zeros(4)  # [T, tau_x, tau_y, tau_z]

        self._sub = self.create_subscription(
            ThrustAndTorque,
            'thrust_and_torque',
            self._cmd_callback,
            10,
        )
        self._pub = self.create_publisher(Odometry, 'state', 10)
        self._timer = self.create_timer(params.dt, self._step)

        self.get_logger().info(
            f'UAV model node started | mass={params.mass} kg  '
            f'dt={params.dt} s  hover_thrust={params.hover_thrust():.4f} N'
        )

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('mass', 1.5)
        self.declare_parameter('gravity', 9.81)
        self.declare_parameter('arm_length', 0.25)
        self.declare_parameter('kf', 1.0)
        self.declare_parameter('km', 0.02)
        self.declare_parameter('thrust_min', 0.0)
        self.declare_parameter('thrust_max', 15.0)
        self.declare_parameter('dt', 0.002)
        # J stored as flat row-major 3x3, e.g. [Jxx, Jxy, Jxz, ...]
        self.declare_parameter(
            'J',
            [0.02, 0.0, 0.0,
             0.0,  0.02, 0.0,
             0.0,  0.0,  0.04],
        )

    def _load_parameters(self) -> UAVParameters:
        p = UAVParameters()
        p.mass = self.get_parameter('mass').value
        p.gravity = self.get_parameter('gravity').value
        p.arm_length = self.get_parameter('arm_length').value
        p.kf = self.get_parameter('kf').value
        p.km = self.get_parameter('km').value
        p.thrust_min = self.get_parameter('thrust_min').value
        p.thrust_max = self.get_parameter('thrust_max').value
        p.dt = self.get_parameter('dt').value
        J_flat = list(self.get_parameter('J').value)
        p.J = np.array(J_flat).reshape(3, 3)
        return p

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cmd_callback(self, msg: ThrustAndTorque):
        self._control = np.array([
            msg.collective_thrust,
            msg.torque.x,
            msg.torque.y,
            msg.torque.z,
        ])

    def _step(self):
        x = self._state.as_vector()
        x_next = self._integrator.step(
            x, self._control, self._dyn_params, self._params.dt
        )
        self._state.from_vector(x_next)
        self._publish_state()

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_state(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        r = self._state.r
        v = self._state.v
        q = self._state.q  # model convention: [w, x, y, z]
        w = self._state.w

        msg.pose.pose.position.x = float(r[0])
        msg.pose.pose.position.y = float(r[1])
        msg.pose.pose.position.z = float(r[2])

        # nav_msgs/Odometry uses ROS convention: [x, y, z, w]
        msg.pose.pose.orientation.w = float(q[0])
        msg.pose.pose.orientation.x = float(q[1])
        msg.pose.pose.orientation.y = float(q[2])
        msg.pose.pose.orientation.z = float(q[3])

        msg.twist.twist.linear.x = float(v[0])
        msg.twist.twist.linear.y = float(v[1])
        msg.twist.twist.linear.z = float(v[2])

        msg.twist.twist.angular.x = float(w[0])
        msg.twist.twist.angular.y = float(w[1])
        msg.twist.twist.angular.z = float(w[2])

        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = UAVModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
