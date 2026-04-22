"""
Minimum-snap trajectory planner for flat-output UAV control.

Generates a piecewise degree-7 polynomial trajectory through a sequence of
waypoints by minimising the integral of squared snap (4th derivative) for
each flat axis independently.

Formulation
-----------
For N waypoints there are N-1 segments, each parameterised by 8 coefficients
(p_0 … p_7) on the local time interval [0, T_k].

Cost (per axis):
    J = sum_k integral_0^{T_k} (d^4 p_k / dt^4)^2 dt

Constraints:
    • Position at every waypoint endpoint (2*(N-1) equalities)
    • Continuity of derivatives 1-3 at interior junctions (3*(N-2) equalities)
    • User-specified velocity, acceleration, and jerk = 0 at start/end

The resulting QP (min c^T Q c  s.t. A c = b) is solved via the KKT system.

Reference: Mellinger & Kumar, ICRA 2011; Richter et al., ISRR 2013.
"""

import numpy as np
from scipy import linalg
from typing import List

from uav_model.trajectory import (
    PolynomialSegment,
    PiecewiseTrajectory,
    Waypoint,
)

_N_COEFFS = PolynomialSegment.N_COEFFS    # 8
_SNAP_ORDER = 4                            # derivative order to minimise


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def _cost_matrix(duration: float) -> np.ndarray:
    """
    8×8 snap-cost matrix Q for one segment of given duration T.

    Q[i,j] = integral_0^T (d^4/dt^4 t^i)(d^4/dt^4 t^j) dt
            = fi * fj * T^(i+j-7) / (i+j-7),  for i,j >= 4
    where fi = i*(i-1)*(i-2)*(i-3) is the falling factorial.
    """
    T = float(duration)
    Q = np.zeros((_N_COEFFS, _N_COEFFS))
    r = _SNAP_ORDER
    for i in range(r, _N_COEFFS):
        for j in range(r, _N_COEFFS):
            fi = float(np.prod([i - k for k in range(r)]))
            fj = float(np.prod([j - k for k in range(r)]))
            exp = i + j - 2 * r + 1
            Q[i, j] = fi * fj * (T ** exp) / exp
    return Q


def _basis_row(t: float, derivative: int) -> np.ndarray:
    """Row vector b such that b @ c = p^(derivative)(t) for coefficient vector c."""
    b = np.zeros(_N_COEFFS)
    for i in range(derivative, _N_COEFFS):
        coeff = 1.0
        for k in range(derivative):
            coeff *= (i - k)
        b[i] = coeff * (t ** (i - derivative))
    return b


# ---------------------------------------------------------------------------
# Per-axis QP solver
# ---------------------------------------------------------------------------

def _solve_axis(
    wp_vals: np.ndarray,
    durations: np.ndarray,
    start_constraints: dict,
    end_constraints: dict,
) -> List[PolynomialSegment]:
    """
    Solve the minimum-snap QP for one flat axis.

    Returns a list of PolynomialSegment, one per segment, whose coefficients
    minimise the integral of squared snap subject to position, continuity,
    and boundary constraints.

    """
    N_seg = len(durations)
    N_c = N_seg * _N_COEFFS

    # ---- Block-diagonal cost matrix ----
    Q_full = np.zeros((N_c, N_c))
    for k, T_k in enumerate(durations):
        s = k * _N_COEFFS
        Q_full[s:s + _N_COEFFS, s:s + _N_COEFFS] = _cost_matrix(T_k)

    # ---- Equality constraints A c = b ----
    rows: List[np.ndarray] = []
    vals: List[float] = []

    def add(row, val):
        rows.append(row)
        vals.append(float(val))

    # 1. Position at start and end of every segment
    for k in range(N_seg):
        s = k * _N_COEFFS
        row = np.zeros(N_c)
        row[s:s + _N_COEFFS] = _basis_row(0.0, 0)
        add(row, wp_vals[k])

        row = np.zeros(N_c)
        row[s:s + _N_COEFFS] = _basis_row(durations[k], 0)
        add(row, wp_vals[k + 1])

    # 2. Derivative continuity at interior junctions (orders 1 … _SNAP_ORDER-1)
    for k in range(N_seg - 1):
        s_cur = k * _N_COEFFS
        s_nxt = (k + 1) * _N_COEFFS
        for d in range(1, _SNAP_ORDER):
            row = np.zeros(N_c)
            row[s_cur:s_cur + _N_COEFFS] = _basis_row(durations[k], d)
            row[s_nxt:s_nxt + _N_COEFFS] = -_basis_row(0.0, d)
            add(row, 0.0)

    # 3. Boundary conditions at the start (segment 0, t = 0)
    for d, v in sorted(start_constraints.items()):
        row = np.zeros(N_c)
        row[:_N_COEFFS] = _basis_row(0.0, d)
        add(row, v)

    # 4. Boundary conditions at the end (last segment, t = T_last)
    for d, v in sorted(end_constraints.items()):
        row = np.zeros(N_c)
        s = (N_seg - 1) * _N_COEFFS
        row[s:s + _N_COEFFS] = _basis_row(durations[-1], d)
        add(row, v)

    A = np.array(rows)      # (n_con, N_c)
    b = np.array(vals)      # (n_con,)
    n_con = A.shape[0]

    # ---- Solve KKT system  [Q A^T; A 0] [c; λ] = [0; b] ----
    KKT = np.zeros((N_c + n_con, N_c + n_con))
    KKT[:N_c, :N_c] = Q_full + np.eye(N_c) * 1e-10   # small regularisation
    KKT[:N_c, N_c:] = A.T
    KKT[N_c:, :N_c] = A

    rhs = np.concatenate([np.zeros(N_c), b])
    sol = linalg.solve(KKT, rhs, assume_a='sym')
    c_all = sol[:N_c]

    return [
        PolynomialSegment(c_all[k * _N_COEFFS:(k + 1) * _N_COEFFS], durations[k])
        for k in range(N_seg)
    ]


# ---------------------------------------------------------------------------
# Public planner class
# ---------------------------------------------------------------------------

class MinimumSnapPlanner:
    """
    Minimum-snap trajectory planner for flat outputs [x, y, z, yaw].

    Snap is minimised independently for each axis. Boundary conditions at the
    first and last waypoint default to zero velocity, acceleration, and jerk;
    user-specified ``velocity`` and ``acceleration`` fields on those waypoints
    override the defaults. Interior waypoint higher-order derivatives are free
    (optimised). Continuity up to jerk is enforced at every junction.

    Parameters
    ----------
    waypoints : list of Waypoint (at least 2, with strictly increasing times)

    Example
    -------
    >>> wps = [
    ...     Waypoint(position=[0, 0, 0], yaw=0.0, t=0.0),
    ...     Waypoint(position=[1, 0, 1], yaw=0.0, t=2.0),
    ...     Waypoint(position=[1, 1, 1], yaw=1.57, t=4.0),
    ... ]
    >>> traj = MinimumSnapPlanner(wps).plan()
    >>> sigma = traj.evaluate(1.0)          # flat outputs at t = 1 s
    >>> sigma_dot = traj.evaluate(1.0, derivative=1)

    """

    def __init__(self, waypoints: List[Waypoint]):
        if len(waypoints) < 2:
            raise ValueError('At least 2 waypoints are required.')
        self._waypoints = waypoints

    def plan(self) -> PiecewiseTrajectory:
        """Solve and return the minimum-snap PiecewiseTrajectory."""
        wps = self._waypoints
        times = np.array([w.t for w in wps], dtype=float)
        durations = np.diff(times)

        if np.any(durations <= 0.0):
            raise ValueError('Waypoint times must be strictly increasing.')

        # Flat output values: shape (N_wp, 4)  — [x, y, z, yaw]
        positions = np.array([
            [w.position[0], w.position[1], w.position[2], w.yaw]
            for w in wps
        ])

        all_segments: List[List[PolynomialSegment]] = []
        for ax in range(PiecewiseTrajectory.N_AXES):
            start_con, end_con = self._boundary_conditions(wps, ax)
            segs = _solve_axis(positions[:, ax], durations, start_con, end_con)
            all_segments.append(segs)

        return PiecewiseTrajectory(all_segments, times)

    @staticmethod
    def _boundary_conditions(wps: List[Waypoint], ax: int):
        """
        Build start and end boundary condition dicts for one axis.

        Axes 0-2 are x/y/z; axis 3 is yaw.
        """
        start_wp, end_wp = wps[0], wps[-1]

        def _vel(wp):
            if ax < 3 and wp.velocity is not None:
                return float(wp.velocity[ax])
            return 0.0

        def _acc(wp):
            if ax < 3 and wp.acceleration is not None:
                return float(wp.acceleration[ax])
            return 0.0

        start_con = {1: _vel(start_wp), 2: _acc(start_wp), 3: 0.0}
        end_con = {1: _vel(end_wp), 2: _acc(end_wp), 3: 0.0}
        return start_con, end_con
