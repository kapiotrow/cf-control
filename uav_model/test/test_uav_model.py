import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor

from uav_model.state import UAVState
from uav_model.params import UAVParameters
from uav_model.model import dynamics, quat_to_rot
from uav_model.integrator import RK4Integrator
from uav_model.motor_model import QuadMotorModel
from uav_model.uav_model_node import UAVModelNode


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
    n = UAVModelNode()
    yield n
    n.destroy_node()


@pytest.fixture
def default_params():
    return UAVParameters()


@pytest.fixture
def dyn_params(default_params):
    p = default_params
    return {'mass': p.mass, 'J': p.J, 'gravity': p.gravity}


# ---------------------------------------------------------------------------
# UAVState
# ---------------------------------------------------------------------------

class TestUAVState:
    def test_default_is_identity(self):
        s = UAVState()
        assert np.allclose(s.r, [0, 0, 0])
        assert np.allclose(s.v, [0, 0, 0])
        assert np.allclose(s.q, [1, 0, 0, 0])
        assert np.allclose(s.w, [0, 0, 0])

    def test_roundtrip(self):
        s = UAVState()
        s.r = np.array([1.0, 2.0, 3.0])
        s.v = np.array([0.1, 0.2, 0.3])
        s.q = np.array([0.707, 0.0, 0.707, 0.0])
        s.normalize()
        s.w = np.array([0.01, 0.02, 0.03])

        vec = s.as_vector()
        s2 = UAVState()
        s2.from_vector(vec)

        assert np.allclose(s2.r, s.r)
        assert np.allclose(s2.v, s.v)
        assert np.allclose(s2.w, s.w)

    def test_normalize_produces_unit_quaternion(self):
        s = UAVState()
        s.q = np.array([2.0, 0.0, 0.0, 0.0])
        s.normalize()
        assert np.isclose(np.linalg.norm(s.q), 1.0)

    def test_copy_is_independent(self):
        s = UAVState()
        s.r = np.array([1.0, 2.0, 3.0])
        s2 = s.copy()
        s2.r[0] = 99.0
        assert s.r[0] == 1.0


# ---------------------------------------------------------------------------
# UAVParameters
# ---------------------------------------------------------------------------

class TestUAVParameters:
    def test_hover_thrust(self):
        p = UAVParameters()
        assert np.isclose(p.hover_thrust(), p.mass * p.gravity / 4.0)

    def test_load_from_dict_scalars(self):
        p = UAVParameters()
        p.load_from_dict({'mass': 2.0, 'gravity': 9.0, 'dt': 0.005})
        assert p.mass == 2.0
        assert p.gravity == 9.0
        assert p.dt == 0.005

    def test_load_from_dict_flat_J(self):
        p = UAVParameters()
        flat = [1, 0, 0, 0, 2, 0, 0, 0, 3]
        p.load_from_dict({'J': flat})
        assert p.J.shape == (3, 3)
        assert np.isclose(p.J[0, 0], 1.0)
        assert np.isclose(p.J[1, 1], 2.0)
        assert np.isclose(p.J[2, 2], 3.0)

    def test_load_from_dict_nested_J(self):
        p = UAVParameters()
        nested = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        p.load_from_dict({'J': nested})
        assert p.J.shape == (3, 3)
        assert np.isclose(p.J[2, 2], 3.0)

    def test_missing_keys_keep_defaults(self):
        p = UAVParameters()
        original_mass = p.mass
        p.load_from_dict({})
        assert p.mass == original_mass


# ---------------------------------------------------------------------------
# dynamics()
# ---------------------------------------------------------------------------

class TestDynamics:
    def test_gravity_only(self, dyn_params):
        """Zero thrust, level attitude → vertical acceleration = -g."""
        x = UAVState().as_vector()
        u = np.array([0.0, 0.0, 0.0, 0.0])
        xdot = dynamics(x, u, dyn_params)
        # v_dot = [0, 0, -g]
        assert np.allclose(xdot[3:6], [0.0, 0.0, -dyn_params['gravity']])

    def test_hover_zero_acceleration(self, default_params, dyn_params):
        """Hover thrust + level attitude → zero net vertical acceleration."""
        x = UAVState().as_vector()
        T_hover = default_params.mass * default_params.gravity
        u = np.array([T_hover, 0.0, 0.0, 0.0])
        xdot = dynamics(x, u, dyn_params)
        assert np.allclose(xdot[3:6], [0.0, 0.0, 0.0], atol=1e-9)

    def test_quaternion_norm_preserved(self, dyn_params):
        """q_dot is orthogonal to q, so |q| is conserved by RK4."""
        integrator = RK4Integrator(dynamics)
        x = UAVState().as_vector()
        u = np.zeros(4)
        for _ in range(500):
            x = integrator.step(x, u, dyn_params, dt=0.002)
        q = x[6:10]
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)

    def test_torque_causes_angular_acceleration(self, dyn_params):
        """Non-zero roll torque → non-zero roll rate derivative."""
        x = UAVState().as_vector()
        u = np.array([0.0, 1.0, 0.0, 0.0])  # roll torque
        xdot = dynamics(x, u, dyn_params)
        assert xdot[10] != 0.0   # w_x_dot ≠ 0

    def test_quat_to_rot_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quat_to_rot(q)
        assert np.allclose(R, np.eye(3))


# ---------------------------------------------------------------------------
# RK4Integrator
# ---------------------------------------------------------------------------

class TestRK4Integrator:
    def test_free_fall_matches_analytical(self, dyn_params):
        """z(t) = -½ g t² for zero thrust, level attitude."""
        integrator = RK4Integrator(dynamics)
        x = UAVState().as_vector()
        u = np.zeros(4)
        dt = 0.001
        n_steps = 1000  # 1 second
        for _ in range(n_steps):
            x = integrator.step(x, u, dyn_params, dt)

        z_numerical = x[2]
        z_analytical = -0.5 * dyn_params['gravity'] * (n_steps * dt) ** 2
        assert np.isclose(z_numerical, z_analytical, rtol=1e-4)

    def test_hover_altitude_stable(self, default_params, dyn_params):
        """Hover thrust → altitude stays within ±1 mm after 1 s."""
        integrator = RK4Integrator(dynamics)
        x = UAVState().as_vector()
        T_hover = default_params.mass * default_params.gravity
        u = np.array([T_hover, 0.0, 0.0, 0.0])
        dt = 0.002
        for _ in range(500):
            x = integrator.step(x, u, dyn_params, dt)
        assert np.isclose(x[2], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# QuadMotorModel
# ---------------------------------------------------------------------------

class TestQuadMotorModel:
    def test_equal_speeds_zero_torque(self):
        model = QuadMotorModel(arm_length=0.25, kf=1.0, km=0.02)
        T, tau = model.map_to_forces([1.0, 1.0, 1.0, 1.0])
        assert np.allclose(tau, [0.0, 0.0, 0.0])

    def test_total_thrust_scales_with_kf(self):
        model = QuadMotorModel(arm_length=0.25, kf=2.0, km=0.02)
        T, _ = model.map_to_forces([1.0, 1.0, 1.0, 1.0])
        assert np.isclose(T, 2.0 * 4.0)

    def test_clamp_below_min(self):
        model = QuadMotorModel(arm_length=0.25, kf=1.0, km=0.02, thrust_min=0.5)
        T, _ = model.map_to_forces([-1.0, -1.0, -1.0, -1.0])
        assert np.isclose(T, 1.0 * 4 * 0.5)

    def test_roll_torque_sign(self):
        """Motor 2 faster than motor 4 → positive roll torque."""
        model = QuadMotorModel(arm_length=0.25, kf=1.0, km=0.02)
        _, tau = model.map_to_forces([1.0, 2.0, 1.0, 1.0])
        assert tau[0] > 0.0  # tau_x > 0


# ---------------------------------------------------------------------------
# UAVModelNode (integration)
# ---------------------------------------------------------------------------

class TestUAVModelNode:
    def test_node_created(self, node):
        assert node is not None
        assert node.get_name() == 'uav_model_node'

    def test_initial_state_is_zero(self, node):
        s = node._state
        assert np.allclose(s.r, [0, 0, 0])
        assert np.allclose(s.v, [0, 0, 0])
        assert np.allclose(s.q, [1, 0, 0, 0])
        assert np.allclose(s.w, [0, 0, 0])

    def test_default_control_is_zero(self, node):
        assert np.allclose(node._control, [0, 0, 0, 0])

    def test_parameters_loaded(self, node):
        p = node._params
        assert p.mass > 0.0
        assert p.gravity > 0.0
        assert p.dt > 0.0
        assert p.J.shape == (3, 3)

    def test_free_fall_step(self, node):
        """Zero thrust → drone falls (z decreases)."""
        node._control = np.array([0.0, 0.0, 0.0, 0.0])
        initial_z = node._state.r[2]
        for _ in range(200):
            node._step()
        assert node._state.r[2] < initial_z

    def test_hover_thrust_step(self, node):
        """Hover thrust → altitude stays within ±1 mm after 200 steps."""
        # Reset state
        node._state = UAVState()
        T_hover = node._params.mass * node._params.gravity
        node._control = np.array([T_hover, 0.0, 0.0, 0.0])
        for _ in range(200):
            node._step()
        assert np.isclose(node._state.r[2], 0.0, atol=1e-3)

    def test_cmd_callback_updates_control(self, node):
        from cf_control_msgs.msg import ThrustAndTorque
        from geometry_msgs.msg import Vector3
        msg = ThrustAndTorque()
        msg.collective_thrust = 5.0
        msg.torque = Vector3(x=0.1, y=0.2, z=0.3)
        node._cmd_callback(msg)
        assert np.isclose(node._control[0], 5.0)
        assert np.isclose(node._control[1], 0.1)
        assert np.isclose(node._control[2], 0.2)
        assert np.isclose(node._control[3], 0.3)

    def test_publish_state_produces_odometry(self, node):
        from nav_msgs.msg import Odometry
        received = []

        sub = node.create_subscription(
            Odometry, 'state', lambda msg: received.append(msg), 10
        )
        node._state.r = np.array([1.0, 2.0, 3.0])
        node._state.v = np.array([0.1, 0.2, 0.3])
        node._publish_state()

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin_once(timeout_sec=0.1)
        executor.remove_node(node)
        node.destroy_subscription(sub)

        assert len(received) == 1
        msg = received[0]
        assert np.isclose(msg.pose.pose.position.x, 1.0)
        assert np.isclose(msg.pose.pose.position.y, 2.0)
        assert np.isclose(msg.pose.pose.position.z, 3.0)
        assert np.isclose(msg.twist.twist.linear.x, 0.1)

    def test_quaternion_stays_unit_after_steps(self, node):
        """Quaternion norm must stay 1.0 over long integration."""
        node._state = UAVState()
        node._control = np.array([0.0, 0.1, 0.05, 0.02])
        for _ in range(1000):
            node._step()
        q = node._state.q
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)
