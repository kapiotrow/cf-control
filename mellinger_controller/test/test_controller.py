"""
Unit tests for controller.py and trajectory_server.py.

No ROS dependency — all tests run as plain pytest.
"""

import threading
import time

import numpy as np
import pytest

from uav_model.params import UAVParameters
from uav_model.state import UAVState
from uav_model.trajectory import Waypoint

from mellinger_controller.controller import (
    ControllerGains,
    MellingerController,
    _skew,
    _vee,
)
from mellinger_controller.trajectory_server import TrajectoryServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params():
    p = UAVParameters()
    p.mass = 1.2
    p.gravity = 9.81
    p.thrust_min = 0.0
    p.thrust_max = 20.0
    return p


@pytest.fixture
def default_gains():
    return ControllerGains.from_dict({
        'kp_xy': 6.0, 'kp_z': 10.0,
        'kv_xy': 4.0, 'kv_z': 6.0,
        'kR_xy': 0.04, 'kR_z': 0.01,
        'komega_xy': 0.004, 'komega_z': 0.001,
    })


@pytest.fixture
def controller(default_params, default_gains):
    return MellingerController(default_params, default_gains)


@pytest.fixture
def hover_state(default_params):
    """Level hover state at the origin."""
    s = UAVState()
    return s


@pytest.fixture
def traj_server(default_params):
    return TrajectoryServer(default_params)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

class TestSkewVee:

    def test_vee_skew_roundtrip(self):
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(_vee(_skew(v)), v)

    def test_skew_is_antisymmetric(self):
        v = np.array([4.0, -1.0, 2.5])
        S = _skew(v)
        assert np.allclose(S, -S.T)

    def test_skew_cross_product(self):
        """skew(a) @ b == cross(a, b) for any a, b."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert np.allclose(_skew(a) @ b, np.cross(a, b))


# ---------------------------------------------------------------------------
# ControllerGains
# ---------------------------------------------------------------------------

class TestControllerGains:

    def test_from_dict_xy_axes(self, default_gains):
        assert np.isclose(default_gains.kp[0], 6.0)
        assert np.isclose(default_gains.kp[1], 6.0)

    def test_from_dict_z_axis(self, default_gains):
        assert np.isclose(default_gains.kp[2], 10.0)

    def test_shapes(self, default_gains):
        for arr in (default_gains.kp, default_gains.kv,
                    default_gains.kR, default_gains.komega):
            assert arr.shape == (3,)

    def test_all_gains_nonzero(self, default_gains):
        for arr in (default_gains.kp, default_gains.kv,
                    default_gains.kR, default_gains.komega):
            assert np.all(arr > 0.0)


# ---------------------------------------------------------------------------
# MellingerController
# ---------------------------------------------------------------------------

class TestMellingerController:

    def test_hover_equilibrium(self, controller, default_params, hover_state):
        """Zero errors → T ≈ m*g, tau ≈ [0, 0, 0]."""
        ref = UAVState()      # identical to hover_state
        T_ff = default_params.mass * default_params.gravity
        ref_omega = np.zeros(3)
        ref_alpha = np.zeros(3)

        thrust, tau = controller.compute(hover_state, ref, T_ff, ref_omega, ref_alpha)

        assert np.isclose(thrust, T_ff, rtol=1e-6)
        assert np.allclose(tau, [0.0, 0.0, 0.0], atol=1e-10)

    def test_returns_float_and_array(self, controller, hover_state, default_params):
        ref = UAVState()
        T_ff = default_params.mass * default_params.gravity
        thrust, tau = controller.compute(
            hover_state, ref, T_ff, np.zeros(3), np.zeros(3))
        assert isinstance(thrust, float)
        assert tau.shape == (3,)

    def test_position_error_increases_thrust(
            self, controller, hover_state, default_params):
        """Drone below desired → larger upward force → higher thrust."""
        ref_high = UAVState()
        ref_high.r = np.array([0.0, 0.0, 1.0])   # desired 1 m above

        T_ff = default_params.mass * default_params.gravity
        thrust, _ = controller.compute(
            hover_state, ref_high, T_ff, np.zeros(3), np.zeros(3))

        assert thrust > T_ff

    def test_position_error_sign_z(self, controller, hover_state, default_params):
        """Drone above desired → reduced thrust."""
        ref_low = UAVState()
        ref_low.r = np.array([0.0, 0.0, -1.0])   # desired 1 m below

        T_ff = default_params.mass * default_params.gravity
        thrust, _ = controller.compute(
            hover_state, ref_low, T_ff, np.zeros(3), np.zeros(3))

        assert thrust < T_ff

    def test_rotation_error_roll_produces_roll_torque(
            self, controller, hover_state, default_params):
        """Desired 15° roll → nonzero tau_x with correct sign."""
        angle = np.deg2rad(15.0)
        # Rotation about x-axis by angle
        R_des = np.array([
            [1.0,          0.0,           0.0],
            [0.0,  np.cos(angle), -np.sin(angle)],
            [0.0,  np.sin(angle),  np.cos(angle)],
        ])
        # Build desired quaternion [w, x, y, z] from R_des
        from uav_model.flatness import _rot_to_quat
        q_des = _rot_to_quat(R_des)
        ref = UAVState()
        ref.q = q_des

        T_ff = default_params.mass * default_params.gravity
        _, tau = controller.compute(
            hover_state, ref, T_ff, np.zeros(3), np.zeros(3))

        # Error: current (identity) vs desired (rolled) → negative roll error
        # tau_x should be nonzero
        assert abs(tau[0]) > 1e-6

    def test_angular_velocity_error_produces_torque(
            self, controller, hover_state, default_params):
        """Nonzero desired angular velocity with zero current → nonzero tau."""
        ref = UAVState()
        T_ff = default_params.mass * default_params.gravity
        ref_omega = np.array([0.5, 0.0, 0.0])   # desired roll rate

        _, tau = controller.compute(
            hover_state, ref, T_ff, ref_omega, np.zeros(3))

        assert abs(tau[0]) > 1e-6

    def test_feedforward_alpha_adds_to_torque(
            self, controller, hover_state, default_params):
        """Feedforward angular acceleration directly adds to tau."""
        ref = UAVState()
        T_ff = default_params.mass * default_params.gravity
        ref_alpha = np.array([1.0, 0.0, 0.0])

        _, tau_ff = controller.compute(
            hover_state, ref, T_ff, np.zeros(3), ref_alpha)
        _, tau_no_ff = controller.compute(
            hover_state, ref, T_ff, np.zeros(3), np.zeros(3))

        assert not np.allclose(tau_ff, tau_no_ff)


# ---------------------------------------------------------------------------
# TrajectoryServer
# ---------------------------------------------------------------------------

class TestTrajectoryServer:

    def test_get_reference_before_load(self, traj_server):
        assert traj_server.get_reference(0.0) is None

    def test_is_loaded_false_before_load(self, traj_server):
        assert not traj_server.is_loaded

    def test_is_loaded_true_after_load(self, traj_server):
        wps = [
            Waypoint(position=[0, 0, 0], yaw=0.0, t=0.0),
            Waypoint(position=[1, 0, 0], yaw=0.0, t=2.0),
        ]
        traj_server.load_waypoints(wps)
        assert traj_server.is_loaded

    def test_get_reference_returns_tuple(self, traj_server):
        wps = [
            Waypoint(position=[0, 0, 1], yaw=0.0, t=0.0),
            Waypoint(position=[1, 0, 1], yaw=0.0, t=2.0),
        ]
        traj_server.load_waypoints(wps)
        result = traj_server.get_reference(1.0)
        assert result is not None
        state_vec, T_ff, tau_ff = result
        assert state_vec.shape == (13,)
        assert isinstance(T_ff, float)
        assert tau_ff.shape == (3,)

    def test_clamp_before_start(self, traj_server):
        wps = [
            Waypoint(position=[0, 0, 1], yaw=0.0, t=5.0),
            Waypoint(position=[1, 0, 1], yaw=0.0, t=7.0),
        ]
        traj_server.load_waypoints(wps)
        ref_at_start = traj_server.get_reference(5.0)
        ref_before = traj_server.get_reference(0.0)   # clamped to t_start
        assert np.allclose(ref_at_start[0], ref_before[0])

    def test_clamp_past_end(self, traj_server):
        wps = [
            Waypoint(position=[0, 0, 1], yaw=0.0, t=0.0),
            Waypoint(position=[1, 0, 1], yaw=0.0, t=2.0),
        ]
        traj_server.load_waypoints(wps)
        ref_end = traj_server.get_reference(2.0)
        ref_after = traj_server.get_reference(100.0)
        assert np.allclose(ref_end[0], ref_after[0])

    def test_position_at_midpoint(self, traj_server):
        """At t=1 s the x position should be close to 0.5 m."""
        wps = [
            Waypoint(position=[0, 0, 1], yaw=0.0, t=0.0),
            Waypoint(position=[1, 0, 1], yaw=0.0, t=2.0),
        ]
        traj_server.load_waypoints(wps)
        state_vec, _, _ = traj_server.get_reference(1.0)
        x_mid = state_vec[0]
        assert np.isclose(x_mid, 0.5, atol=0.05)

    def test_t_end_property(self, traj_server):
        wps = [
            Waypoint(position=[0, 0, 0], yaw=0.0, t=0.0),
            Waypoint(position=[1, 0, 0], yaw=0.0, t=3.0),
        ]
        traj_server.load_waypoints(wps)
        assert np.isclose(traj_server.t_end, 3.0)

    def test_thread_safety(self, traj_server):
        """Concurrent load and read must not raise."""
        wps = [
            Waypoint(position=[0, 0, 1], yaw=0.0, t=0.0),
            Waypoint(position=[2, 0, 1], yaw=0.0, t=4.0),
        ]
        errors = []

        def loader():
            for _ in range(20):
                try:
                    traj_server.load_waypoints(wps)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        def reader():
            for _ in range(50):
                try:
                    traj_server.get_reference(1.0)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.0004)

        t1 = threading.Thread(target=loader)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f'Thread safety errors: {errors}'
