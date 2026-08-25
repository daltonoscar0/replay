#!/usr/bin/env python3
"""Generate the synthetic broadcast demo clip and its ground truth.

`replay demo` has to work on a fresh clone with no network, so the demo footage
is manufactured here rather than downloaded. World-space trajectories come
first: 22 players and a ball moving on a 105x68 pitch under a scripted
possession sequence with realistic speed and acceleration caps. Those world
positions are then pushed through a fixed pinhole camera looking down at the
pitch from a stand near the halfway line, which is what gives the rendered clip
its perspective -- players further from the camera are drawn smaller, exactly
as they would be in real footage. Because the world positions are known by
construction they double as ground truth for the accuracy table.

The camera is static. A slow pan would mean a per-frame homography and a
camera file that carries a sequence rather than a matrix; the accuracy story is
cleaner with one matrix that is exactly right.

Usage:

    python scripts/make_demo_clip.py --out demo/assets --seed 0 --duration 30
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:  # running the script straight out of a clone
    sys.path.insert(0, str(_SRC))

from replay.schema import (  # noqa: E402
    PITCH_LENGTH,
    PITCH_WIDTH,
    TRACKING_COLUMNS,
    coerce_tracking_frame,
)

# ---------------------------------------------------------------------------
# clip parameters
# ---------------------------------------------------------------------------

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
VIDEO_FPS = 25
#: Trajectories are integrated at 50 Hz because that divides evenly into both
#: the 25 fps video and the 10 Hz ground truth, so every ground-truth row is a
#: sample of exactly the state that was rendered.
SIM_FPS = 50
GT_FPS = 10
DEFAULT_DURATION_S = 30.0
#: libx264 quality. 18 puts a 30 s clip at roughly 1.3 MB, small enough to
#: commit and clean enough that colour blobs survive the encode.
DEFAULT_CRF = 18

N_PLAYERS_PER_TEAM = 11
N_OBJECTS = 2 * N_PLAYERS_PER_TEAM + 1
BALL_INDEX = 2 * N_PLAYERS_PER_TEAM

MAX_SPEED = 8.0  # m/s
MAX_ACCEL = 6.0  # m/s^2
SEEK_GAIN = 1.6  # 1/s, converts target offset into a desired velocity

PASS_SPEED = 14.0  # m/s, average over the flight
MIN_PASS_M = 7.0
MAX_PASS_M = 32.0
#: Ball flight eases out, so it leaves the foot fast and slows as it arrives.
#: The exponent sets the launch speed at ``BALL_EASE * travel / flight_time``.
BALL_EASE = 1.35
#: Time for the ball to go from where a pass landed to the receiver's foot.
BALL_SETTLE_S = 0.4

# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------

#: Elevated position in a stand on the near side, level with the halfway line.
CAMERA_POSITION = (52.5, -30.0, 30.0)
CAMERA_TARGET = (52.5, 34.0, 0.0)
FOCAL_PX = 589.0

#: Players are drawn taller than life. At the far touchline a real 1.85 m
#: player is about 11 px tall in this framing, which is below what a colour
#: blob detector can hold on to; 2.4 m keeps the far side detectable without
#: distorting the geometry, which only ever uses the foot point.
PLAYER_DRAW_HEIGHT_M = 2.4
PLAYER_WIDTH_RATIO = 0.45
BALL_DRAW_RADIUS_M = 0.32

# BGR
COLOR_STAND = (46, 42, 40)
COLOR_SURROUND = (44, 92, 42)
COLOR_PITCH_A = (58, 124, 56)
COLOR_PITCH_B = (48, 110, 46)
COLOR_LINE = (238, 240, 238)
COLOR_HOME = (48, 48, 214)  # red
COLOR_HOME_SHORTS = (236, 236, 236)
COLOR_AWAY = (214, 128, 40)  # blue
COLOR_AWAY_SHORTS = (58, 52, 46)
COLOR_SKIN = (150, 176, 208)
COLOR_BALL = (246, 246, 246)
COLOR_SHADOW = (30, 66, 30)
COLOR_BOARDS = (72, 68, 64)

LINE_WIDTH_M = 0.12
GOAL_HEIGHT_M = 2.44

# ---------------------------------------------------------------------------
# formations
# ---------------------------------------------------------------------------

#: 4-4-2 resting shape for the home team, attacking left to right, with the
#: ball on the halfway line. Row 0 is the keeper.
HOME_FORMATION: np.ndarray = np.array(
    [
        (6.0, 34.0),
        (24.0, 12.0),
        (22.0, 26.0),
        (22.0, 42.0),
        (24.0, 56.0),
        (43.0, 13.0),
        (40.0, 28.0),
        (40.0, 40.0),
        (43.0, 55.0),
        (60.0, 27.0),
        (60.0, 41.0),
    ],
    dtype=np.float64,
)
AWAY_FORMATION: np.ndarray = np.column_stack(
    [PITCH_LENGTH - HOME_FORMATION[:, 0], PITCH_WIDTH - HOME_FORMATION[:, 1]]
)

#: How far the whole block slides with the ball, up the pitch and across it.
BLOCK_SHIFT_X = 0.30
BLOCK_SHIFT_Y = 0.28

#: Play is kept in the middle third so that every object stays inside the
#: camera's trapezoid for the whole clip.
PLAY_X_MIN = 30.0
PLAY_X_MAX = 78.0


# ---------------------------------------------------------------------------
# camera model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraModel:
    """A fixed pinhole camera restricted to the pitch plane (z = 0).

    ``H_pitch_to_image`` is what the renderer uses; ``H_image_to_pitch`` is its
    inverse and is the matrix written to camera.json, because
    ``replay.track.homography.Homography.H`` is defined image -> pitch.
    """

    H_pitch_to_image: np.ndarray
    H_image_to_pitch: np.ndarray
    focal_px: float

    def project(self, pts_xy: np.ndarray) -> np.ndarray:
        """Pitch meters -> image pixels, ``(N, 2)`` in, ``(N, 2)`` out."""
        pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
        homo = np.column_stack([pts, np.ones(len(pts))]) @ self.H_pitch_to_image.T
        w = np.where(np.abs(homo[:, 2]) < 1e-9, 1e-9, homo[:, 2])
        return homo[:, :2] / w[:, None]

    def unproject(self, pts_uv: np.ndarray) -> np.ndarray:
        """Image pixels -> pitch meters."""
        pts = np.atleast_2d(np.asarray(pts_uv, dtype=np.float64))
        homo = np.column_stack([pts, np.ones(len(pts))]) @ self.H_image_to_pitch.T
        w = np.where(np.abs(homo[:, 2]) < 1e-9, 1e-9, homo[:, 2])
        return homo[:, :2] / w[:, None]

    def depth(self, pts_xy: np.ndarray) -> np.ndarray:
        """Distance along the optical axis, in meters, for points on the pitch.

        This is the third homogeneous coordinate, which is what sets the
        apparent size of anything standing at that point.
        """
        pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
        homo = np.column_stack([pts, np.ones(len(pts))]) @ self.H_pitch_to_image.T
        return np.maximum(homo[:, 2], 1e-3)


def build_camera(
    *,
    position: tuple[float, float, float] = CAMERA_POSITION,
    target: tuple[float, float, float] = CAMERA_TARGET,
    focal_px: float = FOCAL_PX,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> CameraModel:
    """Build the ground-plane homography of a look-at pinhole camera."""
    c = np.asarray(position, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    forward = t - c
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward])  # world -> camera

    k = np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # On the plane z = 0 only the first two rotation columns matter. The matrix
    # is deliberately left unscaled: its third homogeneous coordinate is then
    # the true depth in meters, which is what sizes the drawn players.
    plane = np.column_stack([rot[:, 0], rot[:, 1], -rot @ c])
    h_p2i = k @ plane
    h_i2p = np.linalg.inv(h_p2i)
    return CameraModel(
        H_pitch_to_image=h_p2i,
        H_image_to_pitch=h_i2p / h_i2p[2, 2],
        focal_px=focal_px,
    )


# ---------------------------------------------------------------------------
# world trajectories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldTrack:
    """Per-frame world-space state of all 23 objects."""

    t: np.ndarray  # (F,)
    pos: np.ndarray  # (F, 23, 2) meters
    vel: np.ndarray  # (F, 23, 2) m/s
    fps: float

    def subsample(self, fps: float) -> WorldTrack:
        """Take every ``self.fps / fps`` th frame. Rates must divide evenly."""
        ratio = self.fps / fps
        step = int(round(ratio))
        if step < 1 or abs(ratio - step) > 1e-9:
            raise ValueError(f"{self.fps} Hz does not divide evenly into {fps} Hz")
        return WorldTrack(
            t=self.t[::step].copy(),
            pos=self.pos[::step].copy(),
            vel=self.vel[::step].copy(),
            fps=fps,
        )


def _clip_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    """Scale rows of ``vec`` so no row exceeds ``limit`` in magnitude.

    Clipping the magnitude rather than each component is what makes the speed
    cap hold on diagonals too.
    """
    mag = np.linalg.norm(vec, axis=-1, keepdims=True)
    scale = np.where(mag > limit, limit / np.maximum(mag, 1e-12), 1.0)
    return vec * scale


def clamp_to_pitch(pos: np.ndarray, margin: float = 0.5) -> np.ndarray:
    out = pos.copy()
    out[..., 0] = np.clip(out[..., 0], margin, PITCH_LENGTH - margin)
    out[..., 1] = np.clip(out[..., 1], margin, PITCH_WIDTH - margin)
    return out


def _step_players(
    pos: np.ndarray, vel: np.ndarray, target: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    desired = _clip_norm((target - pos) * SEEK_GAIN, MAX_SPEED)
    accel = _clip_norm((desired - vel) / dt, MAX_ACCEL)
    new_vel = _clip_norm(vel + accel * dt, MAX_SPEED)
    new_pos = clamp_to_pitch(pos + new_vel * dt)
    return new_pos, new_vel


def _team_targets(formation: np.ndarray, ball_xy: np.ndarray, attacking_right: bool) -> np.ndarray:
    """Slide the resting shape with the ball."""
    shift_x = BLOCK_SHIFT_X * (ball_xy[0] - PITCH_LENGTH / 2.0)
    shift_y = BLOCK_SHIFT_Y * (ball_xy[1] - PITCH_WIDTH / 2.0)
    out = formation.copy()
    out[:, 0] += shift_x
    out[:, 1] += shift_y
    # The keeper does not travel with the block; it shuffles along its line.
    goal_x = 6.0 if attacking_right else PITCH_LENGTH - 6.0
    out[0, 0] = goal_x + 0.05 * (ball_xy[0] - PITCH_LENGTH / 2.0)
    out[0, 1] = PITCH_WIDTH / 2.0 + 0.22 * (ball_xy[1] - PITCH_WIDTH / 2.0)
    return out


def _separation(pos: np.ndarray, radius: float = 3.5, strength: float = 2.0) -> np.ndarray:
    """Push players apart so the block does not collapse into one blob."""
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dist, np.inf)
    weight = np.clip((radius - dist) / radius, 0.0, 1.0)
    unit = delta / np.maximum(dist, 1e-6)[:, :, None]
    return strength * (unit * weight[:, :, None]).sum(axis=1)


def _pick_receiver(
    pos: np.ndarray, passer: int, candidates: np.ndarray, rng: np.random.Generator
) -> int:
    """Pick a pass target: nearby teammates preferred, chosen stochastically."""
    dist = np.linalg.norm(pos[candidates] - pos[passer], axis=1)
    ok = (dist >= MIN_PASS_M) & (dist <= MAX_PASS_M)
    pool = candidates[ok]
    pool_dist = dist[ok]
    if pool.size == 0:
        return int(candidates[int(np.argmin(dist))])
    weight = 1.0 / (pool_dist + 4.0)
    weight /= weight.sum()
    return int(rng.choice(pool, p=weight))


def generate_world_trajectories(
    *, duration_s: float = DEFAULT_DURATION_S, fps: float = SIM_FPS, seed: int = 0
) -> WorldTrack:
    """Integrate a scripted possession sequence in pitch coordinates.

    Deterministic given ``seed``: every random draw comes from the generator
    created here and the order of draws is fixed by the simulation loop.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    n_frames = int(round(duration_s * fps)) + 1

    home_idx = np.arange(0, N_PLAYERS_PER_TEAM)
    away_idx = np.arange(N_PLAYERS_PER_TEAM, 2 * N_PLAYERS_PER_TEAM)

    pos = np.zeros((n_frames, N_OBJECTS, 2), dtype=np.float64)
    vel = np.zeros((n_frames, N_OBJECTS, 2), dtype=np.float64)

    ball_xy = np.array([42.0, 34.0])
    players = np.vstack(
        [
            _team_targets(HOME_FORMATION, ball_xy, attacking_right=True),
            _team_targets(AWAY_FORMATION, ball_xy, attacking_right=False),
        ]
    )
    players += rng.normal(0.0, 0.8, size=players.shape)
    players = clamp_to_pitch(players)
    player_vel = np.zeros_like(players)

    # Low-frequency wander so nobody stands perfectly still off the ball.
    wander_w = rng.uniform(0.18, 0.5, size=2 * N_PLAYERS_PER_TEAM)
    wander_phase = rng.uniform(0.0, 2.0 * np.pi, size=(2 * N_PLAYERS_PER_TEAM, 2))
    wander_amp = rng.uniform(0.6, 1.8, size=2 * N_PLAYERS_PER_TEAM)

    carrier = 2  # a home centre-back starts with it
    ball_xy = players[carrier] + np.array([1.0, 0.0])
    hold_until = 0.8
    flight: dict[str, float | int | np.ndarray] | None = None
    # Where and when the last pass landed, used to ease the ball onto the
    # receiver's foot instead of teleporting it there the frame they touch it.
    settle_from = ball_xy.copy()
    settle_t0 = -1.0
    turnover_fracs = [0.45, 0.83]
    turnovers = [f * duration_s for f in turnover_fracs]
    next_turnover = 0

    pos[0, : 2 * N_PLAYERS_PER_TEAM] = players
    pos[0, BALL_INDEX] = ball_xy

    for k in range(n_frames - 1):
        t = k * dt

        # --- where everyone wants to be, given the ball right now ------------
        target = np.vstack(
            [
                _team_targets(HOME_FORMATION, ball_xy, attacking_right=True),
                _team_targets(AWAY_FORMATION, ball_xy, attacking_right=False),
            ]
        )
        target[:, 0] += wander_amp * np.sin(wander_w * t + wander_phase[:, 0])
        target[:, 1] += wander_amp * np.cos(wander_w * t + wander_phase[:, 1])
        target += _separation(players)

        if flight is None:
            forward = 1.0 if carrier < N_PLAYERS_PER_TEAM else -1.0
            target[carrier] = players[carrier] + np.array([forward * 5.0, 0.0])
        else:
            receiver = int(flight["receiver"])
            target[receiver] = np.asarray(flight["p1"])

        # The nearest defender closes the ball down.
        ball_team_home = carrier < N_PLAYERS_PER_TEAM
        opponents = away_idx if ball_team_home else home_idx
        press = opponents[int(np.argmin(np.linalg.norm(players[opponents] - ball_xy, axis=1)))]
        target[press] = ball_xy

        target[:, 0] = np.clip(target[:, 0], 1.0, PITCH_LENGTH - 1.0)
        target[:, 1] = np.clip(target[:, 1], 1.0, PITCH_WIDTH - 1.0)

        players, player_vel = _step_players(players, player_vel, target, dt)

        # --- ball ------------------------------------------------------------
        t_next = t + dt
        if flight is not None:
            if t_next >= float(flight["t_end"]):
                carrier = int(flight["receiver"])
                settle_from = np.asarray(flight["p1"]).copy()
                settle_t0 = t_next
                flight = None
                ball_xy = settle_from
                hold_until = t_next + float(rng.uniform(0.7, 1.8))
            else:
                u = (t_next - float(flight["t0"])) / (float(flight["t_end"]) - float(flight["t0"]))
                ease = 1.0 - (1.0 - u) ** BALL_EASE
                ball_xy = (1.0 - ease) * np.asarray(flight["p0"]) + ease * np.asarray(flight["p1"])
        elif t_next >= hold_until:
            same_team = home_idx if carrier < N_PLAYERS_PER_TEAM else away_idx
            other_team = away_idx if carrier < N_PLAYERS_PER_TEAM else home_idx
            intercept = next_turnover < len(turnovers) and t_next >= turnovers[next_turnover]
            if intercept:
                next_turnover += 1
                pool = other_team[other_team != carrier]
            else:
                pool = same_team[same_team != carrier]
            receiver = _pick_receiver(players, carrier, pool, rng)
            p0 = ball_xy.copy()
            travel = float(np.linalg.norm(players[receiver] - p0))
            t_flight = float(np.clip(travel / PASS_SPEED, 0.3, 2.2))
            lead = _clip_norm(player_vel[receiver] * t_flight * 0.5, 3.0)
            p1 = players[receiver] + lead
            p1[0] = np.clip(p1[0], PLAY_X_MIN, PLAY_X_MAX)
            p1[1] = np.clip(p1[1], 2.0, PITCH_WIDTH - 2.0)
            flight = {
                "t0": t_next,
                "t_end": t_next + t_flight,
                "p0": p0,
                "p1": p1,
                "receiver": receiver,
            }
            ball_xy = p0
        else:
            lead_dir = player_vel[carrier]
            norm = float(np.linalg.norm(lead_dir))
            if norm < 0.3:
                lead_dir = np.array([1.0 if carrier < N_PLAYERS_PER_TEAM else -1.0, 0.0])
                norm = 1.0
            attached = players[carrier] + lead_dir / norm * 0.7
            settle = float(np.clip((t_next - settle_t0) / BALL_SETTLE_S, 0.0, 1.0))
            ease = settle * settle * (3.0 - 2.0 * settle)
            ball_xy = (1.0 - ease) * settle_from + ease * attached

        ball_xy = clamp_to_pitch(ball_xy, margin=0.3)
        pos[k + 1, : 2 * N_PLAYERS_PER_TEAM] = players
        pos[k + 1, BALL_INDEX] = ball_xy

    vel[1:] = (pos[1:] - pos[:-1]) / dt
    vel[0] = vel[1]
    return WorldTrack(t=np.arange(n_frames) * dt, pos=pos, vel=vel, fps=float(fps))


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------


def object_identities() -> list[tuple[str, str, int | None]]:
    """``(team, player_id, jersey)`` for rows 0..22, in canonical row order."""
    out: list[tuple[str, str, int | None]] = []
    for team in ("home", "away"):
        for n in range(1, N_PLAYERS_PER_TEAM + 1):
            out.append((team, f"{team}_{n:02d}", n))
    out.append(("ball", "ball", None))
    return out


def build_ground_truth(track: WorldTrack, *, match_id: str = "demo") -> pd.DataFrame:
    """Canonical tracking frame holding the exact world state of ``track``.

    ``frame`` indexes the ground-truth sample rate, not the video: at 10 Hz
    against 25 fps footage there is no integer frame correspondence, so ``t``
    is the column that ties a row to a video frame (video frame = t * fps).
    """
    n_frames = len(track.t)
    identities = object_identities()
    dt = 1.0 / track.fps
    vel = np.gradient(track.pos, dt, axis=0) if n_frames > 1 else np.zeros_like(track.pos)

    frames = np.repeat(np.arange(n_frames, dtype=np.int64), N_OBJECTS)
    times = np.repeat(np.round(track.t, 6), N_OBJECTS)
    flat_pos = track.pos.reshape(-1, 2)
    flat_vel = vel.reshape(-1, 2)

    df = pd.DataFrame(
        {
            "match_id": match_id,
            "t": times,
            "period": 1,
            "frame": frames,
            "team": [identities[i][0] for i in range(N_OBJECTS)] * n_frames,
            "player_id": [identities[i][1] for i in range(N_OBJECTS)] * n_frames,
            "jersey": pd.array(
                [identities[i][2] for i in range(N_OBJECTS)] * n_frames, dtype="Int64"
            ),
            "x": flat_pos[:, 0],
            "y": flat_pos[:, 1],
            "vx": flat_vel[:, 0],
            "vy": flat_vel[:, 1],
            "confidence": 1.0,
        }
    )
    return coerce_tracking_frame(df)


def write_ground_truth(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the long-format canonical CSV. Byte-stable for a fixed seed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = df.loc[:, list(TRACKING_COLUMNS)]
    frame.to_csv(out, index=False, float_format="%.6f", lineterminator="\n")
    return out


def write_camera_json(camera: CameraModel, path: str | Path) -> Path:
    """Write the image -> pitch homography in Homography.to_json's format.

    ``replay.track.homography`` does not exist yet at the time this was
    written, so the layout follows the documented shape: an ``H`` nested list
    and a ``reprojection_error`` in meters, which is exactly zero here because
    the matrix is the one used to render.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "H": [[float(v) for v in row] for row in camera.H_image_to_pitch],
        "reprojection_error": 0.0,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_SHIFT = 4
_SCALE = 1 << _SHIFT


def _fill_quad(
    img: np.ndarray, camera: CameraModel, corners_xy: np.ndarray, color: tuple[int, int, int]
) -> None:
    uv = camera.project(corners_xy)
    pts = np.round(uv * _SCALE).astype(np.int32)
    cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA, shift=_SHIFT)


def _draw_strip(
    img: np.ndarray,
    camera: CameraModel,
    pts_xy: np.ndarray,
    *,
    width_m: float = LINE_WIDTH_M,
    closed: bool = False,
    color: tuple[int, int, int] = COLOR_LINE,
) -> None:
    """Draw a pitch-space polyline as a ribbon of constant width in meters.

    Drawing in meters rather than pixels is what makes near lines thicker than
    far ones, which is a large part of why the render reads as perspective.
    """
    pts = np.asarray(pts_xy, dtype=np.float64)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = pts[1:] - pts[:-1]
    length = np.linalg.norm(seg, axis=1, keepdims=True)
    perp = np.column_stack([-seg[:, 1], seg[:, 0]]) / np.maximum(length, 1e-9)
    half = width_m / 2.0
    for i in range(len(seg)):
        quad = np.array(
            [
                pts[i] + perp[i] * half,
                pts[i + 1] + perp[i] * half,
                pts[i + 1] - perp[i] * half,
                pts[i] - perp[i] * half,
            ]
        )
        _fill_quad(img, camera, quad, color)


def _arc_points(center: np.ndarray, radius: float, a0: float, a1: float, n: int = 40) -> np.ndarray:
    ang = np.linspace(a0, a1, n)
    return center + radius * np.column_stack([np.cos(ang), np.sin(ang)])


def _fill_disc(
    img: np.ndarray,
    camera: CameraModel,
    center_xy: np.ndarray,
    radius_m: float,
    color: tuple[int, int, int] = COLOR_LINE,
) -> None:
    """Fill a pitch-space disc. Used for the spots, which are solid, not rings."""
    ring = _arc_points(np.asarray(center_xy, dtype=np.float64), radius_m, 0.0, 2 * np.pi, 20)
    pts = np.round(camera.project(ring) * _SCALE).astype(np.int32)
    cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA, shift=_SHIFT)


def draw_pitch(img: np.ndarray, camera: CameraModel) -> None:
    """Paint stands, grass, mowing stripes and the full set of white markings."""
    img[:] = COLOR_STAND
    _fill_quad(
        img,
        camera,
        np.array([[-40.0, -8.0], [145.0, -8.0], [145.0, 76.0], [-40.0, 76.0]]),
        COLOR_SURROUND,
    )
    # Advertising boards along the far side, which give the top edge of the
    # grass a hard horizontal line instead of grass meeting stand directly.
    _fill_quad(
        img,
        camera,
        np.array([[-40.0, 73.5], [145.0, 73.5], [145.0, 76.0], [-40.0, 76.0]]),
        COLOR_BOARDS,
    )

    n_stripes = 12
    edges = np.linspace(0.0, PITCH_LENGTH, n_stripes + 1)
    for i in range(n_stripes):
        color = COLOR_PITCH_A if i % 2 == 0 else COLOR_PITCH_B
        _fill_quad(
            img,
            camera,
            np.array(
                [
                    [edges[i], 0.0],
                    [edges[i + 1], 0.0],
                    [edges[i + 1], PITCH_WIDTH],
                    [edges[i], PITCH_WIDTH],
                ]
            ),
            color,
        )

    cy = PITCH_WIDTH / 2.0
    cx = PITCH_LENGTH / 2.0

    # boundary and halfway
    _draw_strip(
        img,
        camera,
        np.array(
            [[0.0, 0.0], [PITCH_LENGTH, 0.0], [PITCH_LENGTH, PITCH_WIDTH], [0.0, PITCH_WIDTH]]
        ),
        closed=True,
    )
    _draw_strip(img, camera, np.array([[cx, 0.0], [cx, PITCH_WIDTH]]))

    # centre circle and spot
    _draw_strip(
        img, camera, _arc_points(np.array([cx, cy]), 9.15, 0.0, 2 * np.pi, 72), closed=False
    )
    _fill_disc(img, camera, np.array([cx, cy]), 0.22)

    for side in (0.0, PITCH_LENGTH):
        sign = 1.0 if side == 0.0 else -1.0
        box_x = side + sign * 16.5
        _draw_strip(
            img,
            camera,
            np.array(
                [
                    [side, cy - 20.16],
                    [box_x, cy - 20.16],
                    [box_x, cy + 20.16],
                    [side, cy + 20.16],
                ]
            ),
        )
        six_x = side + sign * 5.5
        _draw_strip(
            img,
            camera,
            np.array(
                [
                    [side, cy - 9.16],
                    [six_x, cy - 9.16],
                    [six_x, cy + 9.16],
                    [side, cy + 9.16],
                ]
            ),
        )
        spot = np.array([side + sign * 11.0, cy])
        _fill_disc(img, camera, spot, 0.22)
        # the D: the part of the 9.15 m arc that falls outside the box
        theta = np.arccos(5.5 / 9.15)
        if sign > 0:
            arc = _arc_points(spot, 9.15, -theta, theta, 32)
        else:
            arc = _arc_points(spot, 9.15, np.pi - theta, np.pi + theta, 32)
        _draw_strip(img, camera, arc)
        _draw_goal(img, camera, side)


def _draw_goal(img: np.ndarray, camera: CameraModel, goal_line_x: float) -> None:
    """Posts and crossbar, sized off the same focal length as the players."""
    cy = PITCH_WIDTH / 2.0
    bases = np.array([[goal_line_x, cy - 3.66], [goal_line_x, cy + 3.66]])
    uv = camera.project(bases)
    depth = camera.depth(bases)
    height = camera.focal_px * GOAL_HEIGHT_M / depth
    half_w = np.maximum(1.0, camera.focal_px * 0.12 / depth) / 2.0
    tops = uv.copy()
    tops[:, 1] -= height
    for i in range(2):
        cv2.rectangle(
            img,
            (int(round(uv[i, 0] - half_w[i])), int(round(tops[i, 1]))),
            (int(round(uv[i, 0] + half_w[i])), int(round(uv[i, 1]))),
            COLOR_LINE,
            -1,
            cv2.LINE_AA,
        )
    bar = np.array(
        [
            [tops[0, 0] - half_w[0], tops[0, 1] - half_w[0]],
            [tops[1, 0] + half_w[1], tops[1, 1] - half_w[1]],
            [tops[1, 0] + half_w[1], tops[1, 1] + half_w[1]],
            [tops[0, 0] - half_w[0], tops[0, 1] + half_w[0]],
        ]
    )
    cv2.fillConvexPoly(img, np.round(bar).astype(np.int32), COLOR_LINE, lineType=cv2.LINE_AA)


def _draw_player(
    img: np.ndarray,
    camera: CameraModel,
    xy: np.ndarray,
    shirt: tuple[int, int, int],
    shorts: tuple[int, int, int],
) -> None:
    uv = camera.project(xy[None, :])[0]
    depth = float(camera.depth(xy[None, :])[0])
    height = camera.focal_px * PLAYER_DRAW_HEIGHT_M / depth
    width = max(2.0, height * PLAYER_WIDTH_RATIO)
    u, v = float(uv[0]), float(uv[1])
    if u < -80 or u > FRAME_WIDTH + 80 or v < -80 or v > FRAME_HEIGHT + 80:
        return

    cv2.ellipse(
        img,
        (int(round(u)), int(round(v))),
        (max(2, int(round(width * 0.55))), max(1, int(round(width * 0.22)))),
        0,
        0,
        360,
        COLOR_SHADOW,
        -1,
        cv2.LINE_AA,
    )
    head_r = max(1, int(round(height * 0.13)))
    torso_top = v - height + 2 * head_r
    torso_bot = v - height * 0.36
    cv2.rectangle(
        img,
        (int(round(u - width / 2)), int(round(torso_top))),
        (int(round(u + width / 2)), int(round(torso_bot))),
        shirt,
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        img,
        (int(round(u - width * 0.42)), int(round(torso_bot))),
        (int(round(u + width * 0.42)), int(round(v))),
        shorts,
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        img,
        (int(round(u)), int(round(torso_top - head_r))),
        head_r,
        COLOR_SKIN,
        -1,
        cv2.LINE_AA,
    )


def _draw_ball(img: np.ndarray, camera: CameraModel, xy: np.ndarray) -> None:
    uv = camera.project(xy[None, :])[0]
    depth = float(camera.depth(xy[None, :])[0])
    radius = max(2, int(round(camera.focal_px * BALL_DRAW_RADIUS_M / depth)))
    center = (int(round(uv[0])), int(round(uv[1])))
    cv2.circle(img, center, radius + 1, (40, 60, 40), -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, COLOR_BALL, -1, cv2.LINE_AA)


def render_frame(
    pos_xy: np.ndarray,
    camera: CameraModel,
    *,
    rng: np.random.Generator | None = None,
    noise_sigma: float = 1.6,
) -> np.ndarray:
    """Render one frame from a ``(23, 2)`` array of pitch positions.

    Objects are painted far-to-near so that closer players occlude further
    ones, which is the same ordering a real camera produces.
    """
    pos = np.asarray(pos_xy, dtype=np.float64).reshape(N_OBJECTS, 2)
    img = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    draw_pitch(img, camera)

    depths = camera.depth(pos)
    order = np.argsort(-depths)
    for idx in order:
        i = int(idx)
        if i == BALL_INDEX:
            _draw_ball(img, camera, pos[i])
        elif i < N_PLAYERS_PER_TEAM:
            _draw_player(img, camera, pos[i], COLOR_HOME, COLOR_HOME_SHORTS)
        else:
            _draw_player(img, camera, pos[i], COLOR_AWAY, COLOR_AWAY_SHORTS)

    if rng is not None and noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, size=img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def write_video(
    track: WorldTrack,
    camera: CameraModel,
    path: str | Path,
    *,
    fps: int = VIDEO_FPS,
    crf: int = DEFAULT_CRF,
    seed: int = 0,
    noise_sigma: float = 1.6,
) -> Path:
    """Render every frame of ``track`` and pipe raw BGR into ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH; install it to build the demo clip")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    rng = np.random.default_rng(seed + 991)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for k in range(len(track.t)):
            frame = render_frame(track.pos[k], camera, rng=rng, noise_sigma=noise_sigma)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with status {code}")
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def make_demo(
    out_dir: str | Path,
    *,
    seed: int = 0,
    duration_s: float = DEFAULT_DURATION_S,
    crf: int = DEFAULT_CRF,
    match_id: str = "demo",
    render: bool = True,
) -> dict[str, Path]:
    """Build demo_clip.mp4, ground_truth.csv and camera.json in ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    camera = build_camera()
    track = generate_world_trajectories(duration_s=duration_s, fps=SIM_FPS, seed=seed)

    gt = build_ground_truth(track.subsample(GT_FPS), match_id=match_id)
    paths = {
        "ground_truth": write_ground_truth(gt, out / "ground_truth.csv"),
        "camera": write_camera_json(camera, out / "camera.json"),
    }
    if render:
        paths["video"] = write_video(
            track.subsample(VIDEO_FPS),
            camera,
            out / "demo_clip.mp4",
            fps=VIDEO_FPS,
            crf=crf,
            seed=seed,
        )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="demo/assets", help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="seconds")
    parser.add_argument(
        "--crf", type=int, default=DEFAULT_CRF, help="libx264 quality, lower is bigger"
    )
    parser.add_argument("--match-id", default="demo")
    parser.add_argument("--no-video", action="store_true", help="write only csv and camera")
    args = parser.parse_args(argv)

    paths = make_demo(
        args.out,
        seed=args.seed,
        duration_s=args.duration,
        crf=args.crf,
        match_id=args.match_id,
        render=not args.no_video,
    )
    for name, path in paths.items():
        size = path.stat().st_size
        print(f"{name:13s} {path}  ({size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
