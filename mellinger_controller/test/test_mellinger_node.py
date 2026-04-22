"""
ROS2 integration tests for MellingerControllerNode.

Follow the same fixture pattern as uav_model/test/test_uav_model.py.
"""

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor

from uav_model.state import UAVState
from uav_model.trajectory import Waypoint

from mellinger_controller.mellinger_node import MellingerControllerNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(rclpy_context):
    n = MellingerControllerNode()
    yield n
    n.destroy_node()


# ---------------------------------------------------------------------------
# Node creation and parameter loading
# ---------------------------------------------------------------------------

class TestNodeCreation:

    def test_node_created(self, node):
        assert node is not None
        assert node.get_name() == 'mellinger_controller'

    def test_parameters_loaded(self, node):
        p = node._uav_params
        assert p.mass > 0.0
        assert p.gravity > 0.0
        assert p.J.shape == (3, 3)
        assert p.thrust_min >= 0.0
        assert p.thrust_max > p.thrust_min

    def test_gains_nonzero(self, node):
        g = node._controller._gains
        for arr in (g.kp, g.kv, g.kR, g.komega):
            assert np.all(arr > 0.0)

    def test_no_trajectory_loaded(self, node):
        assert not node._traj_server.is_loaded

    def test_initial_state_is_none(self, node):
        assert node._current_state is None


# ---------------------------------------------------------------------------
# State callback
# ---------------------------------------------------------------------------

class TestStateCallback:

    def test_state_callback_parses_position(self, node):
        from nav_msgs.msg import Odometry
        msg = Odometry()
        msg.pose.pose.position.x = 1.0
        msg.pose.pose.position.y = 2.0
        msg.pose.pose.position.z = 3.0
        msg.pose.pose.orientation.w = 1.0
        node._state_callback(msg)
        assert np.allclose(node._current_state.r, [1.0, 2.0, 3.0])

    def test_state_callback_parses_velocity(self, node):
        from nav_msgs.msg import Odometry
        msg = Odometry()
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = 0.5
        msg.twist.twist.linear.y = -0.3
        msg.twist.twist.linear.z = 0.1
        node._state_callback(msg)
        assert np.allclose(node._current_state.v, [0.5, -0.3, 0.1])

    def test_state_callback_parses_quaternion(self, node):
        from nav_msgs.msg import Odometry
        msg = Odometry()
        msg.pose.pose.orientation.w = 0.707
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.707
        msg.pose.pose.orientation.z = 0.0
        node._state_callback(msg)
        q = node._current_state.q
        assert np.isclose(q[0], 0.707)
        assert np.isclose(q[2], 0.707)

    def test_state_callback_parses_angular_velocity(self, node):
        from nav_msgs.msg import Odometry
        msg = Odometry()
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.angular.x = 0.1
        msg.twist.twist.angular.y = 0.2
        msg.twist.twist.angular.z = 0.3
        node._state_callback(msg)
        assert np.allclose(node._current_state.w, [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Waypoints callback
# ---------------------------------------------------------------------------

class TestWaypointsCallback:

    def test_single_pose_ignored(self, node):
        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped
        msg = Path()
        ps = PoseStamped()
        ps.pose.orientation.w = 1.0
        msg.poses = [ps]
        # Should not raise, should not load
        node._waypoints_callback(msg)
        assert not node._traj_server.is_loaded

    def test_two_poses_loads_trajectory(self, node):
        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped
        from builtin_interfaces.msg import Time

        msg = Path()
        for t_sec, x in [(0, 0.0), (2, 1.0)]:
            ps = PoseStamped()
            ps.header.stamp = Time(sec=t_sec, nanosec=0)
            ps.pose.position.x = x
            ps.pose.position.z = 1.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        node._waypoints_callback(msg)
        assert node._traj_server.is_loaded


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------

class TestControlLoop:

    def test_no_publish_without_state(self, node):
        """Timer fires but state is None → no command published."""
        node._current_state = None
        received = []
        from cf_control_msgs.msg import ThrustAndTorque
        sub = node.create_subscription(
            ThrustAndTorque, 'thrust_and_torque',
            lambda msg: received.append(msg), 10)

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin_once(timeout_sec=0.05)
        executor.remove_node(node)
        node.destroy_subscription(sub)

        assert len(received) == 0

    def test_no_publish_without_trajectory(self, node):
        """State present but no trajectory → no command published."""
        node._current_state = UAVState()
        node._traj_server._traj = None   # clear any loaded trajectory
        received = []
        from cf_control_msgs.msg import ThrustAndTorque
        sub = node.create_subscription(
            ThrustAndTorque, 'thrust_and_torque',
            lambda msg: received.append(msg), 10)

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin_once(timeout_sec=0.05)
        executor.remove_node(node)
        node.destroy_subscription(sub)

        assert len(received) == 0

    def test_hover_publishes_near_hover_thrust(self, node):
        """Hover state + hover reference → thrust ≈ m*g within 10%."""
        # Set current state to level hover at origin
        node._current_state = UAVState()

        # Build a hover trajectory (two waypoints at same position, 1 m up)
        wps = [
            Waypoint(position=[0, 0, 1], yaw=0.0, t=0.0),
            Waypoint(position=[0, 0, 1], yaw=0.0, t=4.0),
        ]
        node._traj_server.load_waypoints(wps)

        received = []
        from cf_control_msgs.msg import ThrustAndTorque
        sub = node.create_subscription(
            ThrustAndTorque, 'thrust_and_torque',
            lambda msg: received.append(msg), 10)

        # Manually trigger one control step at t=2 s (mid-trajectory)
        node._traj_server.get_reference(2.0)   # warm-up / verify no error

        # Force the control loop to run once
        node._control_loop()

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin_once(timeout_sec=0.1)
        executor.remove_node(node)
        node.destroy_subscription(sub)

        assert len(received) >= 1
        T_expected = node._uav_params.mass * node._uav_params.gravity
        assert abs(received[0].collective_thrust - T_expected) / T_expected < 0.10

    def test_thrust_is_clamped(self, node):
        """Thrust must never exceed thrust_max in the published message."""
        from cf_control_msgs.msg import ThrustAndTorque

        # Drive a huge position error to force extreme thrust demand
        extreme_state = UAVState()
        extreme_state.r = np.array([0.0, 0.0, -100.0])   # 100 m below ref
        node._current_state = extreme_state

        wps = [
            Waypoint(position=[0, 0, 0], yaw=0.0, t=0.0),
            Waypoint(position=[0, 0, 0], yaw=0.0, t=4.0),
        ]
        node._traj_server.load_waypoints(wps)

        received = []
        sub = node.create_subscription(
            ThrustAndTorque, 'thrust_and_torque',
            lambda msg: received.append(msg), 10)

        node._control_loop()

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin_once(timeout_sec=0.1)
        executor.remove_node(node)
        node.destroy_subscription(sub)

        assert len(received) >= 1
        assert received[0].collective_thrust <= node._uav_params.thrust_max
