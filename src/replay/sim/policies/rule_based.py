"""A hand-written tactical policy: compact defensive block, direct attack.

The point of this policy is not realism, it is legibility. Every knob in the
constructor maps to something a coach would say out loud ("press higher",
"squeeze the block", "look for the pass"), so a scenario file that changes one
number produces a change in behaviour you can point at on the pitch.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from replay.schema import PITCH_LENGTH, PITCH_WIDTH
from replay.sim.physics import MAX_SPEED
from replay.sim.state import N_PLAYERS_PER_TEAM, MatchState

TeamSide = Literal["home", "away"]

#: 4-3-3 resting shape in meters for a team attacking towards +x. Row 0 is the
#: keeper. These are mid-block numbers, not kick-off numbers.
BASE_FORMATION: np.ndarray = np.array(
    [
        [6.0, 34.0],  # GK
        [22.0, 12.0],  # RB
        [20.0, 26.0],  # RCB
        [20.0, 42.0],  # LCB
        [22.0, 56.0],  # LB
        [38.0, 18.0],  # RCM
        [36.0, 34.0],  # CDM
        [38.0, 50.0],  # LCM
        [58.0, 12.0],  # RW
        [60.0, 34.0],  # ST
        [58.0, 56.0],  # LW
    ],
    dtype=np.float64,
)

# How much of the ball's displacement from the centre spot the block copies.
# Sideways tracking is weaker than depth tracking: blocks shuffle across less
# than they drop off.
BLOCK_TRACK_X = 0.55
BLOCK_TRACK_Y = 0.45
ATTACK_TRACK_X = 0.70
ATTACK_TRACK_Y = 0.50

# Fraction of the distance to the ball that a defender's resting position is
# pulled in at ``line_compactness == 1``.
COMPACT_BASE = 0.15
MAX_SQUEEZE = 0.60
#: The keeper squeezes towards the ball far less than an outfield player.
GK_SQUEEZE_WEIGHT = 0.25

# Velocity gains, 1/s: desired speed is gain * distance-to-target, capped.
APPROACH_GAIN = 1.6
PRESS_GAIN = 4.0

#: Standard deviation of the per-tick velocity noise, m/s. Small enough that a
#: player sitting on his resting position reads as stationary.
JITTER_MS = 0.05

# Off-ball search: nine candidate positions per player (stay, plus a ring).
OFFBALL_STEP_M = 6.0
LANE_CAP_M = 8.0
FORWARD_WEIGHT = 0.35
CROWD_WEIGHT = 6.0
SHAPE_WEIGHT = 0.12
CARRIER_AVOID_M = 6.0

_RING_ANGLES = np.arange(8, dtype=np.float64) * (np.pi / 4.0)
#: (9, 2) candidate offsets in meters, first row is "stay where you are".
CANDIDATE_OFFSETS: np.ndarray = np.vstack(
    [
        np.zeros((1, 2)),
        OFFBALL_STEP_M * np.stack([np.cos(_RING_ANGLES), np.sin(_RING_ANGLES)], axis=1),
    ]
)


def point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each of ``points`` (N, 2) to the segment ``a``-``b``.

    Segment, not infinite line: the projection parameter is clamped to [0, 1],
    so a player standing behind the passer does not count as blocking the lane.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ ab / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def lane_clearance(
    origin: np.ndarray, targets: np.ndarray, opponents: np.ndarray
) -> np.ndarray:
    """Minimum opponent distance to each lane ``origin`` -> ``targets[i]``.

    Returns (M,). Larger means a cleaner passing lane. With no opponents on the
    pitch every lane is reported as fully clear (``inf`` would poison the
    scoring arithmetic, so it saturates at a large finite number).
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(2)
    tgt = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    opp = np.asarray(opponents, dtype=np.float64).reshape(-1, 2)
    if opp.shape[0] == 0:
        return np.full(tgt.shape[0], PITCH_LENGTH, dtype=np.float64)
    ab = tgt - origin  # (M, 2)
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)  # (M,)
    ap = opp[None, :, :] - origin[None, None, :]  # (1, N, 2)
    t = (ap * ab[:, None, :]).sum(axis=-1) / denom[:, None]  # (M, N)
    t = np.clip(t, 0.0, 1.0)
    proj = origin[None, None, :] + t[..., None] * ab[:, None, :]  # (M, N, 2)
    return np.linalg.norm(opp[None, :, :] - proj, axis=-1).min(axis=1)


def team_slice(team: TeamSide) -> slice:
    if team == "home":
        return slice(0, N_PLAYERS_PER_TEAM)
    return slice(N_PLAYERS_PER_TEAM, 2 * N_PLAYERS_PER_TEAM)


def attack_direction(team: TeamSide) -> float:
    """+1 if the team attacks towards +x, -1 otherwise."""
    return 1.0 if team == "home" else -1.0


def opponent_goal(team: TeamSide) -> np.ndarray:
    x = PITCH_LENGTH if team == "home" else 0.0
    return np.array([x, PITCH_WIDTH / 2.0], dtype=np.float64)


def formation_template(team: TeamSide) -> np.ndarray:
    """The resting shape in pitch coordinates for ``team`` (11, 2)."""
    if team == "home":
        return BASE_FORMATION.copy()
    mirrored = BASE_FORMATION.copy()
    mirrored[:, 0] = PITCH_LENGTH - mirrored[:, 0]
    mirrored[:, 1] = PITCH_WIDTH - mirrored[:, 1]
    return mirrored


def _clamp_to_pitch(pos: np.ndarray) -> np.ndarray:
    out = pos.copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, PITCH_LENGTH)
    out[:, 1] = np.clip(out[:, 1], 0.0, PITCH_WIDTH)
    return out


def cap_speed(vel: np.ndarray, max_speed: float = MAX_SPEED) -> np.ndarray:
    """Scale down any row whose magnitude exceeds ``max_speed``."""
    speed = np.linalg.norm(vel, axis=1)
    scale = np.ones_like(speed)
    hot = speed > max_speed
    scale[hot] = max_speed / speed[hot]
    return vel * scale[:, None]


class RuleBasedPolicy:
    """Compact-block defending and direct attacking, fully deterministic.

    ``defensive_line`` is a translation of the whole defending block up the
    pitch in meters, applied after the compactness squeeze so that a scenario
    asking for "+1.5 m higher" gets exactly that and not 1.5 m attenuated by
    whatever the ball happens to be doing.
    """

    def __init__(
        self,
        *,
        press_trigger_m: float = 12.0,
        defensive_line: float = 0.0,
        line_compactness: float = 1.0,
        pass_lane_weight: float = 1.0,
    ) -> None:
        self.press_trigger_m = float(press_trigger_m)
        self.defensive_line = float(defensive_line)
        self.line_compactness = float(line_compactness)
        self.pass_lane_weight = float(pass_lane_weight)

    def __repr__(self) -> str:
        return (
            f"RuleBasedPolicy(press_trigger_m={self.press_trigger_m}, "
            f"defensive_line={self.defensive_line}, "
            f"line_compactness={self.line_compactness}, "
            f"pass_lane_weight={self.pass_lane_weight})"
        )

    # -- shape ------------------------------------------------------------

    def resting_positions(
        self, state: MatchState, team: TeamSide, *, defending: bool = True
    ) -> np.ndarray:
        """Where each of ``team``'s players wants to stand right now, (11, 2)."""
        template = formation_template(team)
        ball = np.asarray(state.ball, dtype=np.float64).reshape(2)
        track_x = BLOCK_TRACK_X if defending else ATTACK_TRACK_X
        track_y = BLOCK_TRACK_Y if defending else ATTACK_TRACK_Y
        shift = np.array(
            [
                (ball[0] - PITCH_LENGTH / 2.0) * track_x,
                (ball[1] - PITCH_WIDTH / 2.0) * track_y,
            ]
        )
        rest = template + shift

        if defending:
            squeeze = float(np.clip(COMPACT_BASE * self.line_compactness, 0.0, MAX_SQUEEZE))
            weights = np.ones(N_PLAYERS_PER_TEAM, dtype=np.float64)
            weights[0] = GK_SQUEEZE_WEIGHT
            rest = rest + (ball - rest) * (squeeze * weights)[:, None]
            rest[:, 0] += self.defensive_line * attack_direction(team)

        return _clamp_to_pitch(rest)

    # -- policy -----------------------------------------------------------

    def desired_velocity(
        self, state: MatchState, team: TeamSide, rng: np.random.Generator
    ) -> np.ndarray:
        """Desired velocity for each of ``team``'s eleven players, (11, 2).

        The rng is consumed exactly once per call, with a fixed shape, whatever
        the parameters are. Paired baseline/scenario rollouts share a seed, and
        that only cancels variance if both arms draw the same numbers.
        """
        jitter = rng.normal(scale=JITTER_MS, size=(N_PLAYERS_PER_TEAM, 2))

        own = np.asarray(state.pos[team_slice(team)], dtype=np.float64)
        other: TeamSide = "away" if team == "home" else "home"
        opp = np.asarray(state.pos[team_slice(other)], dtype=np.float64)
        ball = np.asarray(state.ball, dtype=np.float64).reshape(2)

        carrier_team = state.ball_carrier_team
        if carrier_team == team:
            vel = self._attack(state, team, own, opp, ball)
        elif carrier_team is None:
            vel = self._contest(state, team, own, opp, ball)
        else:
            vel = self._defend(state, team, own, opp, ball)

        return cap_speed(vel + jitter)

    def _press_target(self, state: MatchState, ball: np.ndarray) -> np.ndarray:
        """The point defenders converge on: the carrier if there is one."""
        if state.possession is not None and state.possession >= 0:
            return np.asarray(state.pos[state.possession], dtype=np.float64).reshape(2)
        return ball

    def _defend(
        self,
        state: MatchState,
        team: TeamSide,
        own: np.ndarray,
        opp: np.ndarray,
        ball: np.ndarray,
    ) -> np.ndarray:
        rest = self.resting_positions(state, team, defending=True)
        vel = (rest - own) * APPROACH_GAIN

        target = self._press_target(state, ball)
        to_target = target - own
        dist = np.linalg.norm(to_target, axis=1)
        pressing = dist <= self.press_trigger_m
        if pressing.any():
            vel[pressing] = to_target[pressing] * PRESS_GAIN
        return vel

    def _contest(
        self,
        state: MatchState,
        team: TeamSide,
        own: np.ndarray,
        opp: np.ndarray,
        ball: np.ndarray,
    ) -> np.ndarray:
        """Loose ball: the two closest players go for it, the rest hold shape."""
        rest = self.resting_positions(state, team, defending=True)
        vel = (rest - own) * APPROACH_GAIN
        dist = np.linalg.norm(ball - own, axis=1)
        chasers = np.argsort(dist, kind="stable")[:2]
        vel[chasers] = (ball - own[chasers]) * PRESS_GAIN
        return vel

    def _attack(
        self,
        state: MatchState,
        team: TeamSide,
        own: np.ndarray,
        opp: np.ndarray,
        ball: np.ndarray,
    ) -> np.ndarray:
        sl = team_slice(team)
        rest = self.resting_positions(state, team, defending=False)
        goal = opponent_goal(team)
        adir = attack_direction(team)

        carrier_local = -1
        if state.possession is not None and sl.start <= state.possession < sl.stop:
            carrier_local = int(state.possession) - sl.start
        carrier_pos = own[carrier_local] if carrier_local >= 0 else ball

        vel = np.zeros_like(own)
        for i in range(N_PLAYERS_PER_TEAM):
            if i == carrier_local:
                vel[i] = self._carry(own[i], opp, goal)
                continue
            vel[i] = self._off_ball(own[i], rest[i], carrier_pos, opp, adir)
        return vel

    def _carry(self, pos: np.ndarray, opp: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Drive at goal, bending away from the nearest opponent."""
        to_goal = goal - pos
        norm = float(np.linalg.norm(to_goal))
        direction = to_goal / norm if norm > 1e-9 else np.zeros(2)

        if opp.shape[0]:
            offsets = pos - opp
            dists = np.linalg.norm(offsets, axis=1)
            j = int(np.argmin(dists))
            d = float(dists[j])
            if d < CARRIER_AVOID_M and d > 1e-9:
                away = offsets[j] / d
                weight = (CARRIER_AVOID_M - d) / CARRIER_AVOID_M
                direction = direction + away * weight
                n2 = float(np.linalg.norm(direction))
                if n2 > 1e-9:
                    direction = direction / n2
        return direction * MAX_SPEED

    def _off_ball(
        self,
        pos: np.ndarray,
        rest: np.ndarray,
        carrier_pos: np.ndarray,
        opp: np.ndarray,
        adir: float,
    ) -> np.ndarray:
        """Pick the best of nine candidate spots and head for it.

        Score trades three things off: how open the lane from the carrier is
        (scaled by ``pass_lane_weight``), how much ground the move gains, and
        how far it drags the player out of the team's shape.
        """
        cands = _clamp_to_pitch(pos[None, :] + CANDIDATE_OFFSETS)
        clear = np.minimum(lane_clearance(carrier_pos, cands, opp), LANE_CAP_M)
        forward = adir * (cands[:, 0] - pos[0])
        shape = np.linalg.norm(cands - rest[None, :], axis=1)

        if opp.shape[0]:
            crowd = np.linalg.norm(cands[:, None, :] - opp[None, :, :], axis=-1).min(axis=1)
        else:
            crowd = np.full(cands.shape[0], PITCH_LENGTH)

        score = (
            self.pass_lane_weight * clear
            + FORWARD_WEIGHT * forward
            - SHAPE_WEIGHT * shape
            - CROWD_WEIGHT / (1.0 + crowd)
        )
        best = cands[int(np.argmax(score))]
        return (best - pos) * APPROACH_GAIN
