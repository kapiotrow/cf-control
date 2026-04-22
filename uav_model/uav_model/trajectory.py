"""
Trajectory data structures for flat-output UAV planning.

Flat outputs: sigma = [x, y, z, yaw]

A trajectory is a piecewise degree-7 polynomial in each flat axis.
Evaluating at derivative order k gives the k-th time derivative of sigma.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Waypoint:
    """
    Desired flat-output state at a specific time.

    position     : [x, y, z] in metres
    yaw          : heading angle in radians (must be unwrapped across waypoints)
    t            : arrival time in seconds
    velocity     : [vx, vy, vz] at this waypoint, or None (free)
    acceleration : [ax, ay, az] at this waypoint, or None (free)
    """

    position: np.ndarray
    yaw: float = 0.0
    t: float = 0.0
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=float)
        if self.acceleration is not None:
            self.acceleration = np.asarray(self.acceleration, dtype=float)


class PolynomialSegment:
    """
    Degree-7 polynomial for one flat-output axis over a fixed duration.

        p(t) = c[0] + c[1]*t + ... + c[7]*t^7,   t in [0, duration]

    Parameters
    ----------
    coeffs   : array of shape (8,), low-to-high degree order
    duration : segment duration in seconds

    """

    DEGREE = 7
    N_COEFFS = 8

    def __init__(self, coeffs: np.ndarray, duration: float):
        if coeffs.shape != (self.N_COEFFS,):
            raise ValueError(f'Expected {self.N_COEFFS} coefficients, got {coeffs.shape}')
        self.c = np.array(coeffs, dtype=float)
        self.T = float(duration)

    def evaluate(self, t: float, derivative: int = 0) -> float:
        """Evaluate the polynomial (or derivative) at local time t."""
        t = float(np.clip(t, 0.0, self.T))
        result = 0.0
        for i in range(derivative, self.N_COEFFS):
            coeff = self.c[i]
            for k in range(derivative):
                coeff *= (i - k)
            result += coeff * t ** (i - derivative)
        return result


class PiecewiseTrajectory:
    """
    Piecewise polynomial trajectory over flat outputs [x, y, z, yaw].

    Parameters
    ----------
    segments : list of 4 lists (one per axis), each with N_segments PolynomialSegment
    times    : breakpoint times of shape (N_waypoints,), times[0] is the start time

    """

    N_AXES = 4   # x, y, z, yaw
    AXIS_NAMES = ('x', 'y', 'z', 'yaw')

    def __init__(
        self,
        segments: List[List[PolynomialSegment]],
        times: np.ndarray,
    ):
        times = np.asarray(times, dtype=float)
        if len(segments) != self.N_AXES:
            raise ValueError(f'Expected {self.N_AXES} axes, got {len(segments)}')
        n_seg = len(times) - 1
        for ax, segs in enumerate(segments):
            if len(segs) != n_seg:
                raise ValueError(
                    f'Axis {ax}: expected {n_seg} segments, got {len(segs)}'
                )
        self.segments = segments
        self.times = times

    def evaluate(self, t: float, derivative: int = 0) -> np.ndarray:
        """
        Evaluate flat outputs (or their derivative) at global time t.

        Returns array [x, y, z, yaw] (or their derivatives).
        """
        t = float(np.clip(t, self.times[0], self.times[-1]))
        # Find active segment index
        seg_idx = int(np.clip(
            np.searchsorted(self.times[1:], t, side='right'),
            0, len(self.times) - 2,
        ))
        t_local = t - self.times[seg_idx]
        return np.array([
            self.segments[ax][seg_idx].evaluate(t_local, derivative)
            for ax in range(self.N_AXES)
        ])

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def t_start(self) -> float:
        return float(self.times[0])

    @property
    def t_end(self) -> float:
        return float(self.times[-1])
