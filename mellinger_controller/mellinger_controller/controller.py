"""
Mellinger geometric controller for quadrotor UAVs.

Implements the position and attitude controller from:
    Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control
    for Quadrotors", ICRA 2011.

The controller is split into two loops:

Position loop
    Computes the desired thrust magnitude by comparing actual and reference
    position/velocity and adding a feedforward acceleration term.

Attitude loop
    Drives the body frame toward the desired rotation using the geometric
    error on SO(3) and the angular-velocity error, plus feedforward torque.

All math is pure NumPy — no ROS dependency.
"""

from dataclasses import dataclass

import numpy as np

from uav_model.model import quat_to_rot
from uav_model.params import UAVParameters
from uav_model.state import UAVState


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of vector v."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def _vee(S: np.ndarray) -> np.ndarray:
    """Extract the axial vector from a skew-symmetric matrix (inverse of _skew)."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# ---------------------------------------------------------------------------
# Controller gains
# ---------------------------------------------------------------------------

@dataclass
class ControllerGains:
    """
    Diagonal gain matrices for the Mellinger controller.

    All gains are stored as 1-D arrays of shape (3,) for element-wise
    multiplication with error vectors.  Use from_dict to construct from
    a parameter dictionary that provides per-axis scalar values.

    """

    kp: np.ndarray      # position gains  [kpx, kpy, kpz]
    kv: np.ndarray      # velocity gains  [kvx, kvy, kvz]
    kR: np.ndarray      # rotation-error gains  [kRx, kRy, kRz]
    komega: np.ndarray  # angular-rate-error gains  [komx, komy, komz]

    @classmethod
    def from_dict(cls, d: dict) -> 'ControllerGains':
        """
        Build gains from a flat parameter dictionary.

        Expected keys: kp_xy, kp_z, kv_xy, kv_z, kR_xy, kR_z,
        komega_xy, komega_z.  xy values are used for both x and y axes.

        """
        def _vec(xy_key, z_key):
            return np.array([d[xy_key], d[xy_key], d[z_key]], dtype=float)

        return cls(
            kp=_vec('kp_xy', 'kp_z'),
            kv=_vec('kv_xy', 'kv_z'),
            kR=_vec('kR_xy', 'kR_z'),
            komega=_vec('komega_xy', 'komega_z'),
        )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MellingerController:
    """
    Mellinger geometric controller.

    Computes collective thrust and body torques given current and desired
    UAV states plus feedforward angular velocity and acceleration.

    """

    _GRAVITY_DIR = np.array([0.0, 0.0, 1.0])   # world z (up)

    def __init__(self, params: UAVParameters, gains: ControllerGains):
        self._m = params.mass
        self._g = params.gravity
        self._J = params.J
        self._gains = gains

    def compute(
        self,
        state: UAVState,
        ref_state: UAVState,
        ref_thrust: float,
        ref_omega: np.ndarray,
        ref_alpha: np.ndarray,
    ) -> tuple:
        """
        Compute thrust and body torques for one control step.

        The feedforward angular velocity ref_omega and angular acceleration
        ref_alpha are expressed in the desired body frame and come from the
        differential flatness map.  Returns (collective_thrust [N], tau [3 Nm]).

        """
        g = self._gains
        m = self._m
        gravity = self._g
        J = self._J

        # Current rotation matrix and body z-axis
        R = quat_to_rot(state.q)
        z_B = R[:, 2]

        # Desired rotation matrix
        R_des = quat_to_rot(ref_state.q)

        # ------------------------------------------------------------------
        # Position loop
        # ------------------------------------------------------------------
        e_r = state.r - ref_state.r
        e_v = state.v - ref_state.v

        # Feedforward acceleration: back-compute from ref_thrust and z_B_des
        a_ref = ref_thrust / m * R_des[:, 2] - gravity * self._GRAVITY_DIR

        f_des = -g.kp * e_r - g.kv * e_v + m * gravity * self._GRAVITY_DIR + m * a_ref

        # Thrust is the projection of f_des onto the current body z-axis
        thrust = float(np.dot(f_des, z_B))

        # ------------------------------------------------------------------
        # Attitude loop
        # ------------------------------------------------------------------
        # Rotation error: e_R = 0.5 * vee(R_des^T R - R^T R_des)
        eR_mat = R_des.T @ R - R.T @ R_des
        e_R = 0.5 * _vee(eR_mat)

        # Angular-velocity error in body frame
        e_omega = state.w - R.T @ R_des @ ref_omega

        # Torque: feedback + gyroscopic compensation + feedforward
        J_omega = J @ state.w
        tau = (
            -g.kR * e_R
            - g.komega * e_omega
            + np.cross(state.w, J_omega)
            + J @ ref_alpha
        )

        return thrust, tau
