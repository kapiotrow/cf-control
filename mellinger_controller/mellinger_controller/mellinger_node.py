"""
ROS2 node wrapping the Mellinger geometric controller.

Subscribes to the UAV state (nav_msgs/Odometry) and a desired trajectory
(nav_msgs/Path), then publishes cf_control_msgs/ThrustAndTorque commands
at a fixed control rate.

Topic remappings (set in the launch file):
    state              → /uav_model/state   (or /crazyflie/odom)
    thrust_and_torque  → /cf_control/control_command
    waypoints          → /mellinger/waypoints
"""

import math

import numpy as np
import rclpy
from cf_control_msgs.msg import ThrustAndTorque
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from uav_model.params import UAVParameters
from uav_model.state import UAVState

from mellinger_controller.controller import ControllerGains, MellingerController
from mellinger_controller.trajectory_server import TrajectoryServer


class MellingerControllerNode(Node):
    """
    Mellinger controller ROS2 node.

    Runs a position-attitude control loop at control_rate Hz.  Trajectory
    waypoints are received on the 'waypoints' topic (nav_msgs/Path); each
    PoseStamped provides position, yaw (from orientation quaternion), and
    arrival time (from header.stamp).

    """

    def __init__(self):
        super().__init__('mellinger_controller')

        self._declare_parameters()
        self._uav_params = self._load_uav_parameters()
        gains = self._load_gains()

        self._controller = MellingerController(self._uav_params, gains)
        self._traj_server = TrajectoryServer(self._uav_params)

        self._current_state: UAVState | None = None

        self._sub_state = self.create_subscription(
            Odometry, 'state', self._state_callback, 10)
        self._sub_waypoints = self.create_subscription(
            Path, 'waypoints', self._waypoints_callback, 10)

        self._pub_cmd = self.create_publisher(ThrustAndTorque, 'thrust_and_torque', 10)
        self._pub_e_r = self.create_publisher(Vector3, '~/e_r', 10)
        self._pub_e_R = self.create_publisher(Vector3, '~/e_R', 10)

        rate = self.get_parameter('control_rate').value
        self._timer = self.create_timer(1.0 / rate, self._control_loop)

        self.get_logger().info(
            f'Mellinger controller started | rate={rate:.0f} Hz  '
            f'mass={self._uav_params.mass} kg'
        )

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('control_rate', 200.0)
        self.declare_parameter('mass', 1.2)
        self.declare_parameter('gravity', 9.81)
        self.declare_parameter('thrust_min', 0.0)
        self.declare_parameter('thrust_max', 12.0)
        self.declare_parameter(
            'J',
            [0.018, 0.0,   0.0,
             0.0,   0.018, 0.0,
             0.0,   0.0,   0.035],
        )
        self.declare_parameter('kp_xy', 6.0)
        self.declare_parameter('kp_z', 10.0)
        self.declare_parameter('kv_xy', 4.0)
        self.declare_parameter('kv_z', 6.0)
        self.declare_parameter('kR_xy', 0.04)
        self.declare_parameter('kR_z', 0.01)
        self.declare_parameter('komega_xy', 0.004)
        self.declare_parameter('komega_z', 0.001)

    def _load_uav_parameters(self) -> UAVParameters:
        p = UAVParameters()
        p.mass = self.get_parameter('mass').value
        p.gravity = self.get_parameter('gravity').value
        p.thrust_min = self.get_parameter('thrust_min').value
        p.thrust_max = self.get_parameter('thrust_max').value
        J_flat = list(self.get_parameter('J').value)
        p.J = np.array(J_flat).reshape(3, 3)
        return p

    def _load_gains(self) -> ControllerGains:
        keys = ['kp_xy', 'kp_z', 'kv_xy', 'kv_z', 'kR_xy', 'kR_z',
                'komega_xy', 'komega_z']
        d = {k: self.get_parameter(k).value for k in keys}
        return ControllerGains.from_dict(d)

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _state_callback(self, msg: Odometry):
        s = UAVState()
        p = msg.pose.pose.position
        s.r = np.array([p.x, p.y, p.z])
        v = msg.twist.twist.linear
        s.v = np.array([v.x, v.y, v.z])
        # uav_model convention: q = [w, x, y, z]
        o = msg.pose.pose.orientation
        s.q = np.array([o.w, o.x, o.y, o.z])
        w = msg.twist.twist.angular
        # uav_model_node publishes body-frame angular velocity
        s.w = np.array([w.x, w.y, w.z])
        self._current_state = s

    def _waypoints_callback(self, msg: Path):
        from uav_model.trajectory import Waypoint

        if len(msg.poses) < 2:
            self.get_logger().warn(
                'Received Path with fewer than 2 poses — ignoring.')
            return

        # Parse waypoint times as relative durations from the first stamp,
        # then offset so the trajectory starts at the current ROS clock time.
        t_now = self.get_clock().now().nanoseconds * 1e-9
        stamps = [
            ps.header.stamp.sec + ps.header.stamp.nanosec * 1e-9
            for ps in msg.poses
        ]
        t_offset = t_now - stamps[0]

        waypoints = []
        for ps, stamp in zip(msg.poses, stamps):
            pos = ps.pose.position
            q = ps.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            waypoints.append(Waypoint(
                position=np.array([pos.x, pos.y, pos.z]),
                yaw=yaw,
                t=stamp + t_offset,
            ))

        try:
            self._traj_server.load_waypoints(waypoints)
            self.get_logger().info(
                f'Loaded trajectory with {len(waypoints)} waypoints, '
                f't=[{waypoints[0].t:.2f}, {waypoints[-1].t:.2f}] s'
            )
        except ValueError as e:
            self.get_logger().error(f'Trajectory planning failed: {e}')

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        if self._current_state is None:
            return

        t = self.get_clock().now().nanoseconds * 1e-9
        ref = self._traj_server.get_reference(t)
        if ref is None:
            return

        ref_state_vec, T_ff, tau_ff = ref
        ref_state = UAVState()
        ref_state.from_vector(ref_state_vec)

        # Back-compute feedforward angular acceleration from tau_ff
        J = self._uav_params.J
        ref_omega = ref_state.w
        J_inv = np.linalg.inv(J)
        ref_alpha = J_inv @ (tau_ff - np.cross(ref_omega, J @ ref_omega))

        thrust, tau = self._controller.compute(
            self._current_state, ref_state, T_ff, ref_omega, ref_alpha
        )

        self._publish_command(thrust, tau)
        self._publish_diagnostics(self._current_state, ref_state, tau)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_command(self, thrust: float, tau: np.ndarray):
        p = self._uav_params
        thrust_clamped = float(np.clip(thrust, p.thrust_min, p.thrust_max))

        msg = ThrustAndTorque()
        msg.collective_thrust = thrust_clamped
        msg.torque = Vector3(x=float(tau[0]), y=float(tau[1]), z=float(tau[2]))
        self._pub_cmd.publish(msg)

    def _publish_diagnostics(
        self, state: UAVState, ref_state: UAVState, tau: np.ndarray
    ):
        from uav_model.model import quat_to_rot
        from mellinger_controller.controller import _vee

        e_r = state.r - ref_state.r
        self._pub_e_r.publish(
            Vector3(x=float(e_r[0]), y=float(e_r[1]), z=float(e_r[2])))

        R = quat_to_rot(state.q)
        R_des = quat_to_rot(ref_state.q)
        eR_mat = R_des.T @ R - R.T @ R_des
        e_R = 0.5 * _vee(eR_mat)
        self._pub_e_R.publish(
            Vector3(x=float(e_R[0]), y=float(e_R[1]), z=float(e_R[2])))


def main(args=None):
    rclpy.init(args=args)
    node = MellingerControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
