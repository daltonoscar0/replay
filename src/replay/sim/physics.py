"""Motion model for the sandbox.

Deliberately small: players are point masses with a bounded acceleration and a
bounded speed, the ball is a point mass with first-order drag. The point of the
sandbox is to compare tactical policies, and a heavier model would add
parameters nobody can calibrate from broadcast tracking.

Every function here is pure. Inputs are never modified in place, there are no
globals beyond the constants, and nothing is random, so a rollout is fully
determined by its initial state and its policies.
"""

from __future__ import annotations

import numpy as np

from replay.schema import PITCH_LENGTH, PITCH_WIDTH

MAX_SPEED = 8.0  # m/s, roughly a fast professional sprint
MAX_ACCEL = 6.0  # m/s^2
BALL_DRAG = 0.4  # 1/s, first-order: dv/dt = -drag * v
DT = 0.1  # s, 10 Hz to match the tracking rate


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale rows of ``v`` so no row's Euclidean norm exceeds ``max_norm``.

    Clipping each component separately would be wrong: a vector at 45 degrees
    with both components at the cap has norm ``cap * sqrt(2)``, so diagonal
    motion would be 41% faster than motion along an axis. Rows that are already
    inside the cap are returned untouched, and NaN rows stay NaN.
    """
    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")
    arr = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    scale = np.ones_like(norm)
    # ``where`` leaves NaN norms (unknown objects) at scale 1 instead of
    # propagating a comparison warning.
    np.divide(max_norm, norm, out=scale, where=norm > max_norm)
    return arr * scale


def step_players(
    pos: np.ndarray,
    vel: np.ndarray,
    desired_vel: np.ndarray,
    *,
    dt: float = DT,
    max_speed: float = MAX_SPEED,
    max_accel: float = MAX_ACCEL,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance players one tick towards ``desired_vel``.

    A policy states the velocity it wants; the body delivers what the
    acceleration cap allows. Acceleration is clipped by magnitude first, then
    the resulting speed is clipped by magnitude, so a player already at
    ``max_speed`` who is asked to go faster simply does not.

    Position is integrated with the post-step velocity (semi-implicit Euler),
    which keeps displacement per tick bounded by ``max_speed * dt`` — with the
    pre-step velocity a player could out-run the speed cap on the tick they
    decelerate.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    p = np.asarray(pos, dtype=np.float64)
    v = np.asarray(vel, dtype=np.float64)
    want = np.asarray(desired_vel, dtype=np.float64)
    if p.shape != v.shape or p.shape != want.shape:
        raise ValueError(
            f"pos {p.shape}, vel {v.shape} and desired_vel {want.shape} must have the same shape"
        )

    accel = clip_norm((want - v) / dt, max_accel)
    new_vel = clip_norm(v + accel * dt, max_speed)
    new_pos = p + new_vel * dt
    return new_pos, new_vel


def step_ball(
    pos: np.ndarray, vel: np.ndarray, *, dt: float = DT, drag: float = BALL_DRAG
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the ball one tick under first-order drag.

    Solved exactly rather than with an Euler step, so the result does not
    depend on ``dt``: ``v(t+dt) = v * exp(-drag*dt)`` and the position is the
    integral of that, ``x + v * (1 - exp(-drag*dt)) / drag``. An Euler step
    with a large ``dt`` and a large drag can reverse the velocity, which shows
    up as the ball visibly bouncing backwards.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if drag < 0:
        raise ValueError(f"drag must be non-negative, got {drag}")
    p = np.asarray(pos, dtype=np.float64)
    v = np.asarray(vel, dtype=np.float64)

    if drag == 0.0:
        return p + v * dt, v.copy()

    decay = float(np.exp(-drag * dt))
    new_vel = v * decay
    new_pos = p + v * ((1.0 - decay) / drag)
    return new_pos, new_vel


def clamp_to_pitch(pos: np.ndarray) -> np.ndarray:
    """Clamp positions into ``[0, PITCH_LENGTH] x [0, PITCH_WIDTH]``.

    Applied to every row it is given, the ball row included. The sandbox has no
    model for a throw-in, a goal kick or a corner, so a ball allowed to leave
    the pitch would keep drifting into space that no policy reasons about;
    holding it on the touchline is the less wrong of the two failures. Callers
    that want the ball to leave should slice it out before calling.

    NaN rows stay NaN.
    """
    arr = np.asarray(pos, dtype=np.float64)
    out = arr.copy()
    out[..., 0] = np.clip(arr[..., 0], 0.0, PITCH_LENGTH)
    out[..., 1] = np.clip(arr[..., 1], 0.0, PITCH_WIDTH)
    return out
