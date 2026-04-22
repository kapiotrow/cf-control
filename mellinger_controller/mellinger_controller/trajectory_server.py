"""
Thread-safe trajectory storage and reference evaluation.

Wraps a PiecewiseTrajectory together with the differential flatness map so
that the control loop can fetch a (state, thrust, torque) reference tuple
at any time with a single call, regardless of whether the trajectory was
planned internally or supplied externally.
"""

import threading

from uav_model.flatness import trajectory_to_state
from uav_model.minimum_snap import MinimumSnapPlanner
from uav_model.params import UAVParameters
from uav_model.trajectory import PiecewiseTrajectory


class TrajectoryServer:
    """
    Thread-safe store for an active PiecewiseTrajectory.

    The trajectory can be loaded from either a list of Waypoint objects
    (which are planned into a minimum-snap trajectory) or a pre-built
    PiecewiseTrajectory.  A threading.Lock guards all accesses so that a
    subscriber thread may update the trajectory while the control timer
    reads from it concurrently.

    """

    def __init__(self, params: UAVParameters):
        self._params = params
        self._traj: PiecewiseTrajectory | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_waypoints(self, waypoints: list) -> None:
        """
        Plan a minimum-snap trajectory through waypoints and store it.

        waypoints must be a list of at least 2 Waypoint objects with
        strictly increasing arrival times.

        """
        traj = MinimumSnapPlanner(waypoints).plan()
        self.load_trajectory(traj)

    def load_trajectory(self, traj: PiecewiseTrajectory) -> None:
        """Replace the active trajectory with traj. Thread-safe."""
        with self._lock:
            self._traj = traj

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_reference(self, t: float):
        """
        Return the feedforward reference at global time t.

        Returns a tuple (state_vec [13], T_ff [float], tau_ff [3]) by
        evaluating the stored trajectory at t via the differential flatness
        map.  t is clamped to [t_start, t_end] so the last pose is held
        after the trajectory ends.  Returns None if no trajectory is loaded.

        """
        with self._lock:
            traj = self._traj

        if traj is None:
            return None

        t_clamped = float(max(traj.t_start, min(t, traj.t_end)))
        return trajectory_to_state(traj, t_clamped, self._params)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return True if a trajectory has been loaded."""
        with self._lock:
            return self._traj is not None

    @property
    def t_end(self) -> float:
        """Return the end time of the active trajectory, or 0.0 if none."""
        with self._lock:
            return float(self._traj.t_end) if self._traj is not None else 0.0

    @property
    def t_start(self) -> float:
        """Return the start time of the active trajectory, or 0.0 if none."""
        with self._lock:
            return float(self._traj.t_start) if self._traj is not None else 0.0
