"""Constant-velocity Kalman filter plus RTS smoother over pitch positions.

Tracking output is noisy and gappy: a detector jitters the box by a pixel or
two every frame, which is tens of centimeters after the homography, and it
loses objects entirely for short stretches. Differentiating raw positions to
get velocity amplifies that jitter into nonsense. A constant-velocity filter
run forward and then smoothed backwards (Rauch-Tung-Striebel) gives both a
denoised position and a velocity that is consistent with it, and it fills gaps
by prediction instead of dropping frames.

The state is ``[x, y, vx, vy]`` in meters and meters/second. Timestamps may be
irregular, so the transition and process-noise matrices are rebuilt per step
from the actual ``dt``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["smooth_track"]

# Prior covariance on the initial state. Deliberately diffuse: the first
# measurement should dominate the initial position, and the backward pass
# fixes the initial velocity from the rest of the track.
_INITIAL_VAR = 1.0e6

# Timestamps that do not advance would make the transition singular. Treat a
# non-increasing step as an infinitesimal one rather than raising, so a
# duplicated frame timestamp does not kill a whole track.
_MIN_DT = 1.0e-9


def _transition(dt: float) -> np.ndarray:
    """Constant-velocity state transition for a step of ``dt`` seconds."""
    f = np.eye(4, dtype=np.float64)
    f[0, 2] = dt
    f[1, 3] = dt
    return f


def _process_noise(dt: float, q: float) -> np.ndarray:
    """Continuous white-noise-acceleration process covariance for one step.

    ``q`` is the acceleration power spectral density in m^2/s^3; integrating it
    over the step gives the usual ``[[dt^3/3, dt^2/2], [dt^2/2, dt]]`` block per
    axis. Using this rather than a fixed diagonal is what makes irregular
    sampling behave: a long gap earns proportionally more uncertainty.
    """
    cov = np.zeros((4, 4), dtype=np.float64)
    pos_pos = q * dt**3 / 3.0
    pos_vel = q * dt**2 / 2.0
    vel_vel = q * dt
    for pos, vel in ((0, 2), (1, 3)):
        cov[pos, pos] = pos_pos
        cov[pos, vel] = pos_vel
        cov[vel, pos] = pos_vel
        cov[vel, vel] = vel_vel
    return cov


def _initial_state(t: np.ndarray, xy: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Seed the filter at ``t[0]`` from the first one or two observed points.

    Seeding velocity from a finite difference rather than zero matters for the
    zero-noise case: a zero-velocity prior would bend the start of an otherwise
    exactly-linear track.
    """
    idx = np.flatnonzero(finite)
    first = int(idx[0])
    pos = xy[first]
    vel = np.zeros(2, dtype=np.float64)
    if idx.size >= 2:
        second = int(idx[1])
        span = t[second] - t[first]
        if span > 0.0:
            vel = (xy[second] - xy[first]) / span
    # Walk the seed back to the first timestamp so index 0 of the output is on
    # the same clock as every other index.
    pos = pos + vel * (t[0] - t[first])
    return np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float64)


def smooth_track(
    t: np.ndarray,
    xy: np.ndarray,
    *,
    process_var: float = 1.0,
    measurement_var: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a single object's pitch trajectory and recover its velocity.

    ``t`` is ``(N,)`` seconds, possibly irregularly spaced. ``xy`` is ``(N, 2)``
    meters; rows containing NaN are treated as missing measurements and are
    predicted through, so the returned arrays are finite everywhere as long as
    at least one row is observed.

    Returns ``(xy_smoothed, velocity)``, both ``(N, 2)`` float64.
    """
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    xy_arr = np.asarray(xy, dtype=np.float64)
    if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
        raise ValueError(f"xy must be (N, 2), got {xy_arr.shape}")
    if t_arr.shape[0] != xy_arr.shape[0]:
        raise ValueError(f"t has {t_arr.shape[0]} rows but xy has {xy_arr.shape[0]}")

    n = t_arr.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    finite = np.isfinite(xy_arr).all(axis=1) & np.isfinite(t_arr)
    if not finite.any():
        raise ValueError("smooth_track needs at least one finite (t, x, y) row")
    if n == 1:
        return xy_arr.astype(np.float64, copy=True), np.zeros((1, 2), dtype=np.float64)

    meas_cov = np.eye(2, dtype=np.float64) * float(measurement_var)
    obs = np.zeros((2, 4), dtype=np.float64)
    obs[0, 0] = 1.0
    obs[1, 1] = 1.0

    x_pred = np.zeros((n, 4), dtype=np.float64)
    p_pred = np.zeros((n, 4, 4), dtype=np.float64)
    x_filt = np.zeros((n, 4), dtype=np.float64)
    p_filt = np.zeros((n, 4, 4), dtype=np.float64)
    trans = np.zeros((n, 4, 4), dtype=np.float64)
    trans[0] = np.eye(4, dtype=np.float64)

    x_pred[0] = _initial_state(t_arr, xy_arr, finite)
    p_pred[0] = np.eye(4, dtype=np.float64) * _INITIAL_VAR

    for k in range(n):
        if k > 0:
            dt = max(float(t_arr[k] - t_arr[k - 1]), _MIN_DT)
            trans[k] = _transition(dt)
            x_pred[k] = trans[k] @ x_filt[k - 1]
            p_pred[k] = trans[k] @ p_filt[k - 1] @ trans[k].T + _process_noise(dt, process_var)

        if finite[k]:
            innovation = xy_arr[k] - obs @ x_pred[k]
            innovation_cov = obs @ p_pred[k] @ obs.T + meas_cov
            gain = np.linalg.solve(innovation_cov.T, (p_pred[k] @ obs.T).T).T
            x_filt[k] = x_pred[k] + gain @ innovation
            eye_minus = np.eye(4, dtype=np.float64) - gain @ obs
            # Joseph form: stays symmetric positive-definite even with the very
            # diffuse initial covariance used above.
            p_filt[k] = eye_minus @ p_pred[k] @ eye_minus.T + gain @ meas_cov @ gain.T
        else:
            x_filt[k] = x_pred[k]
            p_filt[k] = p_pred[k]

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    for k in range(n - 2, -1, -1):
        gain = np.linalg.solve(p_pred[k + 1].T, (p_filt[k] @ trans[k + 1].T).T).T
        x_smooth[k] = x_filt[k] + gain @ (x_smooth[k + 1] - x_pred[k + 1])
        p_smooth[k] = p_filt[k] + gain @ (p_smooth[k + 1] - p_pred[k + 1]) @ gain.T

    return np.ascontiguousarray(x_smooth[:, :2]), np.ascontiguousarray(x_smooth[:, 2:])
