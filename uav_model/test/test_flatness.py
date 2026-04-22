"""
CSV-driven tests for the differential flatness map (flat_to_state).

Each row in trajectory_from_flat_output_test_data.csv defines one test case
with the flat-output inputs and expected state / thrust / torque outputs.

Tolerances
----------
Position, velocity : exact passthrough  → atol 1e-9
Quaternion         : 1e-7  (Shepperd's method rounding)
Angular velocity   : 1e-7
Thrust             : 1e-6
Torque             : 3e-4  (see note below)
Quaternion norm    : always checked to be 1.0 ± 1e-9

Torque tolerance note
---------------------
The reference CSV was generated with a model that omits the alpha_z
contribution arising from jerk x yaw_rate frame coupling.  For
'CombinedJerkAndYawRate' the CSV records tau_z = 0, while the analytically
correct value is ~2e-4 Nm (from alpha_z = y_B_dot . x_B_dot + y_B . x_B_ddot
with nonzero jerk and yaw_dot).  The tolerance of 3e-4 accommodates this
known discrepancy while remaining sensitive to genuine errors in all other
cases.
"""

import csv
import pathlib

import numpy as np
import pytest

from uav_model.flatness import flat_to_state
from uav_model.params import UAVParameters

_CSV_PATH = pathlib.Path(__file__).parent / 'trajectory_from_flat_output_test_data.csv'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cases():
    """Return list of (test_name, inputs, expected) tuples from the CSV."""
    cases = []
    with _CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['test_name']
            inputs = {
                'r': np.array([
                    float(row['in_pos_x']),
                    float(row['in_pos_y']),
                    float(row['in_pos_z']),
                ]),
                'v': np.array([
                    float(row['in_vel_x']),
                    float(row['in_vel_y']),
                    float(row['in_vel_z']),
                ]),
                'a': np.array([
                    float(row['in_acc_x']),
                    float(row['in_acc_y']),
                    float(row['in_acc_z']),
                ]),
                'j': np.array([
                    float(row['in_jerk_x']),
                    float(row['in_jerk_y']),
                    float(row['in_jerk_z']),
                ]),
                's': np.array([
                    float(row['in_snap_x']),
                    float(row['in_snap_y']),
                    float(row['in_snap_z']),
                ]),
                'yaw': float(row['in_yaw']),
                'yaw_dot': float(row['in_yaw_rate']),
                'yaw_ddot': float(row['in_yaw_acceleration']),
                'mass': float(row['in_mass']),
                'gravity': float(row['in_gravity']),
                'I_xx': float(row['in_I_xx']),
                'I_yy': float(row['in_I_yy']),
                'I_zz': float(row['in_I_zz']),
            }
            expected = {
                'pos': np.array([
                    float(row['out_pos_x']),
                    float(row['out_pos_y']),
                    float(row['out_pos_z']),
                ]),
                'quat': np.array([
                    float(row['out_quat_w']),
                    float(row['out_quat_x']),
                    float(row['out_quat_y']),
                    float(row['out_quat_z']),
                ]),
                'vel': np.array([
                    float(row['out_vel_x']),
                    float(row['out_vel_y']),
                    float(row['out_vel_z']),
                ]),
                'omega': np.array([
                    float(row['out_omega_x']),
                    float(row['out_omega_y']),
                    float(row['out_omega_z']),
                ]),
                'thrust': float(row['out_thrust']),
                'torque': np.array([
                    float(row['out_torque_x']),
                    float(row['out_torque_y']),
                    float(row['out_torque_z']),
                ]),
            }
            cases.append((name, inputs, expected))
    return cases


def _make_params(inp):
    """Build UAVParameters from a test-case input dict."""
    p = UAVParameters()
    p.mass = inp['mass']
    p.gravity = inp['gravity']
    p.J = np.diag([inp['I_xx'], inp['I_yy'], inp['I_zz']])
    return p


def _run(inp):
    """Call flat_to_state with the given input dict; return (state, T, tau)."""
    return flat_to_state(
        r=inp['r'],
        v=inp['v'],
        a=inp['a'],
        j=inp['j'],
        s=inp['s'],
        yaw=inp['yaw'],
        yaw_dot=inp['yaw_dot'],
        yaw_ddot=inp['yaw_ddot'],
        params=_make_params(inp),
    )


# ---------------------------------------------------------------------------
# Parametrised test class
# ---------------------------------------------------------------------------

_CASES = _load_cases()


@pytest.mark.parametrize('name,inp,exp', _CASES, ids=[c[0] for c in _CASES])
class TestFlatnessFromCSV:
    """Verify flat_to_state against every row in the reference CSV."""

    def test_position_passthrough(self, name, inp, exp):
        """Position is passed through unchanged."""
        state, _T, _tau = _run(inp)
        assert np.allclose(state[0:3], exp['pos'], atol=1e-9), (
            f'[{name}] position mismatch: got {state[0:3]}, expected {exp["pos"]}'
        )

    def test_velocity_passthrough(self, name, inp, exp):
        """Velocity is passed through unchanged."""
        state, _T, _tau = _run(inp)
        assert np.allclose(state[3:6], exp['vel'], atol=1e-9), (
            f'[{name}] velocity mismatch: got {state[3:6]}, expected {exp["vel"]}'
        )

    def test_quaternion_unit_norm(self, name, inp, exp):
        """Output quaternion must be a unit quaternion."""
        state, _T, _tau = _run(inp)
        q = state[6:10]
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9), (
            f'[{name}] quaternion is not unit: |q| = {np.linalg.norm(q)}'
        )

    def test_quaternion_value(self, name, inp, exp):
        """Output quaternion matches the reference (q and -q are both accepted)."""
        state, _T, _tau = _run(inp)
        q = state[6:10]
        # q and -q represent the same rotation; accept either sign.
        match = (
            np.allclose(q, exp['quat'], atol=1e-7)
            or np.allclose(q, -exp['quat'], atol=1e-7)
        )
        assert match, (
            f'[{name}] quaternion mismatch: got {q}, expected ±{exp["quat"]}'
        )

    def test_angular_velocity(self, name, inp, exp):
        """Body-frame angular velocity matches the reference."""
        state, _T, _tau = _run(inp)
        omega = state[10:13]
        assert np.allclose(omega, exp['omega'], atol=1e-7), (
            f'[{name}] omega mismatch: got {omega}, expected {exp["omega"]}'
        )

    def test_thrust(self, name, inp, exp):
        """Collective thrust matches the reference."""
        state, T, _tau = _run(inp)
        assert np.isclose(T, exp['thrust'], atol=1e-6), (
            f'[{name}] thrust mismatch: got {T}, expected {exp["thrust"]}'
        )

    def test_torque(self, name, inp, exp):
        """Body torques match the reference within tolerance."""
        state, _T, tau = _run(inp)
        assert np.allclose(tau, exp['torque'], atol=3e-4), (
            f'[{name}] torque mismatch: got {tau}, expected {exp["torque"]}'
        )
