"""
Differential flatness map for a quadrotor UAV.

A quadrotor is differentially flat with flat outputs sigma = [x, y, z, psi].
Given sigma and its derivatives, the full state, collective thrust, and body
torques can be recovered without integration.

Derivative order convention
----------------------------
order 0 : position / yaw
order 1 : velocity (v) / yaw rate (psi_dot)
order 2 : acceleration (a) / yaw acceleration (psi_ddot)
order 3 : jerk (j)
order 4 : snap (s)                  ← needed for angular acceleration / torque

Reference: Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control
           for Quadrotors", ICRA 2011.
"""

import numpy as np

from uav_model.params import UAVParameters
from uav_model.state import UAVState


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to unit quaternion [w, x, y, z] (Shepperd's method, model.py convention)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _desired_rotation(z_B: np.ndarray, yaw: float):
    """
    Build the desired body-frame rotation matrix from the thrust direction and yaw.

    Returns
    -------
    R     : 3×3 rotation matrix with columns [x_B, y_B, z_B]
    n     : cross product z_B × x_C  (kept for derivative computation)
    n_norm: ‖n‖
    x_C   : heading unit vector [cos ψ, sin ψ, 0]

    """
    x_C = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    n = np.cross(z_B, x_C)
    n_norm = np.linalg.norm(n)

    if n_norm < 1e-6:
        # z_B nearly parallel to x_C: fall back to global y-axis
        alt = np.array([0.0, 1.0, 0.0])
        n = np.cross(z_B, alt)
        n_norm = np.linalg.norm(n)

    y_B = n / n_norm
    x_B = np.cross(y_B, z_B)
    R = np.column_stack([x_B, y_B, z_B])
    return R, n, n_norm, x_C


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flat_to_state(
    r: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    j: np.ndarray,
    s: np.ndarray,
    yaw: float,
    yaw_dot: float,
    yaw_ddot: float,
    params: UAVParameters,
) -> tuple:
    """
    Convert flat outputs and time derivatives to UAV state, thrust, and body torques.

    Implements the differential flatness map for a quadrotor (Mellinger & Kumar 2011).
    Inputs are position r, velocity v, acceleration a, jerk j, snap s (all in m and
    their derivatives), plus yaw and its first two time derivatives.  Returns a tuple
    of (state_vec [13], collective thrust T [N], body torques tau [3 Nm]).
    Yaw must be unwrapped (continuous) across calls for correct derivatives.

    """
    m = params.mass
    g = params.gravity
    J = params.J

    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    j = np.asarray(j, dtype=float)
    s = np.asarray(s, dtype=float)

    # ---------------------------------------------------------------
    # Thrust and z_B (body z-axis = thrust direction in world frame)
    # ---------------------------------------------------------------
    f = a + np.array([0.0, 0.0, g])    # specific force
    f_norm = np.linalg.norm(f)
    T = m * f_norm

    z_B = f / f_norm if f_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    # ---------------------------------------------------------------
    # Desired rotation matrix R = [x_B | y_B | z_B]
    # ---------------------------------------------------------------
    R, n, n_norm, x_C = _desired_rotation(z_B, yaw)
    x_B = R[:, 0]
    y_B = R[:, 1]

    # ---------------------------------------------------------------
    # First derivatives of body-frame vectors  (from jerk, yaw_dot)
    # ---------------------------------------------------------------
    # ż_B = j⊥ / ‖f‖  where j⊥ = j − (j·z_B) z_B
    j_z = np.dot(j, z_B)
    j_perp = j - j_z * z_B
    z_B_dot = j_perp / f_norm

    # ẋ_C, needed for ṅ
    x_C_dot = np.array([-np.sin(yaw) * yaw_dot, np.cos(yaw) * yaw_dot, 0.0])

    # ṅ = ż_B × x_C + z_B × ẋ_C
    n_dot = np.cross(z_B_dot, x_C) + np.cross(z_B, x_C_dot)

    # ẏ_B = (ṅ − (y_B·ṅ) y_B) / ‖n‖
    y_B_dot = (n_dot - np.dot(n_dot, y_B) * y_B) / n_norm

    # ẋ_B = ẏ_B × z_B + y_B × ż_B
    x_B_dot = np.cross(y_B_dot, z_B) + np.cross(y_B, z_B_dot)

    # ---------------------------------------------------------------
    # Angular velocity in body frame  ω = vee(R^T Ṙ)
    #   Omega = R^T Ṙ  →  ω = [Omega[2,1], Omega[0,2], Omega[1,0]]
    # ---------------------------------------------------------------
    omega_B = np.array([
        np.dot(z_B, y_B_dot),    # ω_x = z_B · ẏ_B
        np.dot(x_B, z_B_dot),    # ω_y = x_B · ż_B
        np.dot(y_B, x_B_dot),    # ω_z = y_B · ẋ_B
    ])

    # ---------------------------------------------------------------
    # Second derivatives of body-frame vectors  (from snap, yaw_ddot)
    # ---------------------------------------------------------------
    # z̈_B from snap:
    #   Let N = j⊥,  ż_B = N/‖f‖
    #   Ṅ = s − (s·z_B + ‖j⊥‖²/‖f‖) z_B − j_z ż_B
    #   z̈_B = Ṅ/‖f‖ − (z_B·j/‖f‖²) N
    j_perp_sq = np.dot(j_perp, j_perp)
    s_z = np.dot(s, z_B)
    N_dot = s - (s_z + j_perp_sq / f_norm) * z_B - j_z * z_B_dot
    z_B_ddot = N_dot / f_norm - (j_z / f_norm ** 2) * j_perp

    # ẍ_C from yaw_dot and yaw_ddot
    x_C_ddot = np.array([
        -np.cos(yaw) * yaw_dot ** 2 - np.sin(yaw) * yaw_ddot,
        -np.sin(yaw) * yaw_dot ** 2 + np.cos(yaw) * yaw_ddot,
        0.0,
    ])

    # n̈ = z̈_B × x_C + 2 ż_B × ẋ_C + z_B × ẍ_C
    n_ddot = (
        np.cross(z_B_ddot, x_C)
        + 2.0 * np.cross(z_B_dot, x_C_dot)
        + np.cross(z_B, x_C_ddot)
    )

    # ÿ_B = [n̈ − (ẏ_B·ṅ) y_B − (y_B·n̈) y_B] / ‖n‖ − 2 (y_B·ṅ) ẏ_B / ‖n‖
    y_B_dot_dot_n = np.dot(y_B_dot, n_dot)    # ẏ_B · ṅ
    y_B_ddot_n = np.dot(y_B, n_ddot)          # y_B · n̈
    y_B_n = np.dot(y_B, n_dot)                # y_B · ṅ
    y_B_ddot = (
        (n_ddot - y_B_dot_dot_n * y_B - y_B_ddot_n * y_B) / n_norm
        - 2.0 * y_B_n * y_B_dot / n_norm
    )

    # ẍ_B = ÿ_B × z_B + 2 ẏ_B × ż_B + y_B × z̈_B
    x_B_ddot = (
        np.cross(y_B_ddot, z_B)
        + 2.0 * np.cross(y_B_dot, z_B_dot)
        + np.cross(y_B, z_B_ddot)
    )

    # ---------------------------------------------------------------
    # Angular acceleration in body frame  α = d/dt vee(R^T Ṙ)
    #   α_x = d/dt(z_B · ẏ_B) = ż_B · ẏ_B + z_B · ÿ_B
    #   α_y = d/dt(x_B · ż_B) = ẋ_B · ż_B + x_B · z̈_B
    #   α_z = d/dt(y_B · ẋ_B) = ẏ_B · ẋ_B + y_B · ẍ_B
    # ---------------------------------------------------------------
    alpha_B = np.array([
        np.dot(z_B_dot, y_B_dot) + np.dot(z_B, y_B_ddot),
        np.dot(x_B_dot, z_B_dot) + np.dot(x_B, z_B_ddot),
        np.dot(y_B_dot, x_B_dot) + np.dot(y_B, x_B_ddot),
    ])

    # ---------------------------------------------------------------
    # Body torques  τ = J α + ω × (J ω)
    # ---------------------------------------------------------------
    J_omega = J @ omega_B
    tau = J @ alpha_B + np.cross(omega_B, J_omega)

    # ---------------------------------------------------------------
    # Assemble state vector [r, v, q, omega]
    # ---------------------------------------------------------------
    q = _rot_to_quat(R)
    state = UAVState()
    state.r = r
    state.v = v
    state.q = q
    state.w = omega_B

    return state.as_vector(), float(T), tau


def trajectory_to_state(traj, t: float, params: UAVParameters) -> tuple:
    """
    Evaluate a PiecewiseTrajectory at time t, returning state, thrust, and torques.

    Returns a tuple of (state_vec [13], thrust T [N], body torques tau [3]).

    """
    sigma0 = traj.evaluate(t, derivative=0)
    sigma1 = traj.evaluate(t, derivative=1)
    sigma2 = traj.evaluate(t, derivative=2)
    sigma3 = traj.evaluate(t, derivative=3)
    sigma4 = traj.evaluate(t, derivative=4)

    return flat_to_state(
        r=sigma0[:3],
        v=sigma1[:3],
        a=sigma2[:3],
        j=sigma3[:3],
        s=sigma4[:3],
        yaw=float(sigma0[3]),
        yaw_dot=float(sigma1[3]),
        yaw_ddot=float(sigma2[3]),
        params=params,
    )
