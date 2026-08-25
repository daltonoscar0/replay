"""Rollout driver: run a scenario, twice, and report what changed.

Every scenario is run as a paired experiment. The baseline arm is the policy on
its stock settings; the scenario arm is the same policy with the overrides from
the YAML applied. Rollout *i* of both arms is driven by the same random stream,
so the difference between them is the effect of the override and not the effect
of a different sequence of coin flips. With thirty-two rollouts that pairing is
worth roughly an order of magnitude in the standard error of the delta, which is
the difference between "the block sat 1.4 m higher" and "we cannot tell".

Possession lives here rather than in ``physics``: the physics layer is pure
kinematics, and who has the ball is a decision the driver makes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from replay.sim.calibrate import pitch_control_surface
from replay.sim.physics import DT, clamp_to_pitch, step_ball, step_players
from replay.sim.policies import BCPolicy, Policy, RuleBasedPolicy
from replay.sim.policies.rule_based import (
    formation_template,
    lane_clearance,
    team_slice,
)
from replay.sim.state import (
    BALL_INDEX,
    N_PLAYERS_PER_TEAM,
    MatchState,
    infer_possession,
    state_from_tracking,
    states_to_tracking,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from replay.sim.scenario import Scenario

log = logging.getLogger(__name__)

TeamSide = Literal["home", "away"]

# --- possession model -------------------------------------------------------
#: Hazard rate of losing the ball with an opponent right on top of you, per
#: second: about a one-in-three chance of losing a one-second duel, which is
#: roughly where tackle-success rates sit. Decays with the distance to the
#: nearest opponent over TACKLE_SCALE_M.
TURNOVER_RATE_HZ = 0.35
TACKLE_SCALE_M = 2.5
#: The carrier looks for a pass once an opponent is this close, and commits with
#: this probability per tick, i.e. roughly once every 0.4 s of being pressed.
PASS_TRIGGER_M = 4.0
PASS_PROB_PER_TICK = 0.25
PASS_MIN_M = 6.0
PASS_MAX_M = 35.0
#: Passes are struck to arrive in about this long, within a sane speed band.
PASS_FLIGHT_S = 0.8
PASS_SPEED_MIN = 8.0
PASS_SPEED_MAX = 25.0
#: A loose ball is collected by the nearest player inside this radius.
CONTROL_RADIUS_M = 1.5
#: Ticks a pass must be airborne before anyone can collect it, and how long the
#: passer himself is locked out. Without these the passer instantly re-collects.
MIN_FLIGHT_TICKS = 2
PASSER_LOCKOUT_TICKS = 6

#: Pitch control is sampled rather than computed every tick; it moves on a
#: timescale of seconds, and the surface is the expensive part of a rollout.
CONTROL_SAMPLE_EVERY = 5

METRIC_NAMES: tuple[str, ...] = (
    "possession_retained",
    "possession_losses",
    "control_share_home",
    "control_share_away",
    "mean_ball_x",
    "distance_home_m",
    "distance_away_m",
)

METRICS_COLUMNS: tuple[str, ...] = ("arm", "rollout", *METRIC_NAMES)
DELTA_COLUMNS: tuple[str, ...] = (
    "metric",
    "baseline",
    "scenario",
    "delta",
    "paired_sd",
    "paired_se",
    "n_rollouts",
)

_ARMS: tuple[str, str] = ("baseline", "scenario")


@dataclass(slots=True)
class Rollout:
    """One trajectory plus the metrics computed from it."""

    states: list[MatchState]
    metrics: dict[str, float]


@dataclass
class SimResult:
    """Everything a scenario run produces.

    ``states`` is rollout 0 of the scenario arm, which is what gets rendered;
    ``baseline_states`` is its paired twin, useful for a side-by-side.
    """

    metrics: pd.DataFrame
    deltas: pd.DataFrame
    states: list[MatchState] = field(default_factory=list)
    baseline_states: list[MatchState] = field(default_factory=list)
    scenario: Any = None

    def to_tracking(self, *, match_id: str = "sim", arm: str = "scenario") -> pd.DataFrame:
        """Rollout 0 of one arm as a canonical tracking frame."""
        states = self.states if arm == "scenario" else self.baseline_states
        return states_to_tracking(states, match_id=match_id)

    def delta(self, metric: str) -> float:
        row = self.deltas.loc[self.deltas["metric"] == metric, "delta"]
        if row.empty:
            raise KeyError(f"no delta for metric {metric!r}")
        return float(row.iloc[0])


# --- scenario plumbing ------------------------------------------------------


def _policy_kind(scenario: Any) -> str:
    cfg = getattr(scenario, "policy", None)
    kind = getattr(cfg, "kind", None)
    if kind is None and isinstance(cfg, dict):
        kind = cfg.get("kind")
    return str(kind or "rule_based").lower().replace("-", "_")


def _policy_overrides(scenario: Any) -> dict[str, float]:
    cfg = getattr(scenario, "policy", None)
    if cfg is None:
        return {}
    raw = getattr(cfg, "overrides", None)
    if raw is None and isinstance(cfg, dict):
        raw = cfg.get("overrides")
    if raw is None:
        return {}
    return {k: float(v) for k, v in dict(raw).items() if v is not None}


def build_policy(scenario: Any, *, apply_overrides: bool) -> Policy:
    """The policy for one arm. The baseline arm ignores the scenario's knobs."""
    kind = _policy_kind(scenario)
    if kind in {"rule_based", "rule", "heuristic"}:
        kwargs = _policy_overrides(scenario) if apply_overrides else {}
        return RuleBasedPolicy(**kwargs)
    if kind == "bc":
        cfg = getattr(scenario, "policy", None)
        weights = getattr(cfg, "weights", None)
        if weights is None and isinstance(cfg, dict):
            weights = cfg.get("weights")
        if not weights:
            raise ValueError("policy kind 'bc' needs a 'weights' path to a saved .npz model")
        return BCPolicy.load(weights)
    raise ValueError(f"unknown policy kind {kind!r}; expected 'rule_based' or 'bc'")


def _resolve_frame(tracking: pd.DataFrame, scenario: Any) -> int:
    init = getattr(scenario, "initial_state", None)
    frame = getattr(init, "frame", None)
    if frame is None and isinstance(init, dict):
        frame = init.get("frame")
    if frame is not None:
        return int(frame)

    frames = tracking["frame"].to_numpy()
    if frames.size == 0:
        raise ValueError("tracking frame is empty; nothing to start a rollout from")

    # "auto" means "pick a sensible frame", and the first frame is rarely one:
    # a tracker needs a few frames to confirm anything, so the opening of a file
    # is exactly where the roster is thinnest and the ball most often missing.
    # Score each frame by whether it has a ball and how many players it has, and
    # take the earliest of the best — earliest so the choice stays reproducible
    # and leaves the most footage ahead of the rollout.
    have_xy = tracking[np.isfinite(tracking["x"]) & np.isfinite(tracking["y"])]
    if have_xy.empty:
        return int(np.min(frames))
    is_ball = have_xy["team"].astype("string") == "ball"
    per_frame = have_xy.assign(_ball=is_ball.astype(int), _player=(~is_ball).astype(int))
    grouped = per_frame.groupby("frame")[["_ball", "_player"]].sum()
    if grouped.empty:
        return int(np.min(frames))
    score = grouped["_ball"].clip(upper=1) * 1000 + grouped["_player"]
    best = int(score.max())
    chosen = int(score.index[np.flatnonzero(score.to_numpy() == best)[0]])
    if best < 1000:
        log.warning(
            "no frame in this tracking file has a ball; starting from frame %d, "
            "which has the fullest roster",
            chosen,
        )
    return chosen


def _fill_missing_slots(state: MatchState) -> MatchState:
    """Give absent players a formation position so the rollout is 11 v 11.

    ``state_from_tracking`` leaves a slot NaN when the detector never saw that
    player. The sandbox has no notion of a team playing with ten, and a NaN row
    poisons every array operation downstream, so the slot is filled from the
    formation template and the substitution is logged. Metrics from a rollout
    with filled slots are about a hypothetical full team, not about the footage.
    """
    out = state.copy()
    sides: tuple[tuple[TeamSide, int], ...] = (("home", 0), ("away", N_PLAYERS_PER_TEAM))
    for team, offset in sides:
        template = formation_template(team)
        block = out.pos[offset : offset + N_PLAYERS_PER_TEAM]
        bad = ~np.isfinite(block).all(axis=1)
        if bad.any():
            log.warning(
                "%s slots %s are missing at the initial frame; filling from the formation",
                team,
                np.flatnonzero(bad).tolist(),
            )
            block[bad] = template[bad]
            out.vel[offset : offset + N_PLAYERS_PER_TEAM][bad] = 0.0
    if not np.isfinite(out.pos[BALL_INDEX]).all():
        raise ValueError("the initial frame has no ball position; cannot start a rollout")
    out.vel = np.nan_to_num(out.vel, nan=0.0)
    if out.possession < 0:
        possession, carrier = infer_possession(out.pos)
        out.possession = possession
        out.ball_carrier_team = carrier
    return out


def initial_state(tracking: pd.DataFrame, scenario: Any) -> MatchState:
    """The state a scenario starts from, with any gaps filled in."""
    return _fill_missing_slots(state_from_tracking(tracking, _resolve_frame(tracking, scenario)))


# --- the rollout ------------------------------------------------------------


def _carrier_team(index: int) -> TeamSide | None:
    if index < 0:
        return None
    return "home" if index < N_PLAYERS_PER_TEAM else "away"


def _choose_receiver(
    own_pos: np.ndarray, opp_pos: np.ndarray, carrier_local: int, adir: float
) -> int:
    """Best teammate to pass to: open lane first, ground gained second.

    Returns the local index within the team, or -1 when nobody is available.
    """
    dist = np.linalg.norm(own_pos - own_pos[carrier_local], axis=1)
    eligible = (dist >= PASS_MIN_M) & (dist <= PASS_MAX_M)
    eligible[carrier_local] = False
    if not eligible.any():
        return -1
    clear = lane_clearance(own_pos[carrier_local], own_pos, opp_pos)
    forward = adir * (own_pos[:, 0] - own_pos[carrier_local, 0])
    score = np.where(eligible, clear + 0.2 * forward, -np.inf)
    return int(np.argmax(score))


def rollout(
    state0: MatchState,
    policy: Policy,
    rng: np.random.Generator,
    *,
    horizon_s: float,
    dt: float = DT,
) -> Rollout:
    """Simulate forward from ``state0`` and return the trajectory and metrics.

    The rng is consumed in a fixed pattern every tick — two policy calls, then
    one draw of two uniforms — so two arms with the same seed stay in lockstep
    even when their policies are configured differently.
    """
    n_ticks = max(1, int(round(float(horizon_s) / dt)))
    state = state0.copy()
    states = [state.copy()]

    start_team = state.ball_carrier_team
    if start_team is None:
        _, start_team = infer_possession(state.pos, radius=np.inf)

    losses = 0
    retained = 0
    ball_x_sum = 0.0
    control_home: list[float] = []
    distance = np.zeros(2 * N_PLAYERS_PER_TEAM)
    flight_ticks = 0
    passer = -1
    last_team = start_team

    for tick in range(n_ticks):
        desired = np.vstack(
            [
                policy.desired_velocity(state, "home", rng),
                policy.desired_velocity(state, "away", rng),
            ]
        )
        draws = rng.random(2)

        players = slice(0, 2 * N_PLAYERS_PER_TEAM)
        new_pos, new_vel = step_players(state.pos[players], state.vel[players], desired, dt=dt)
        new_pos = clamp_to_pitch(new_pos)
        distance += np.linalg.norm(new_pos - state.pos[players], axis=1)

        possession = state.possession
        if possession >= 0:
            ball_pos = new_pos[possession].copy()
            ball_vel = new_vel[possession].copy()
            flight_ticks = 0
        else:
            ball_pos, ball_vel = step_ball(state.pos[BALL_INDEX], state.vel[BALL_INDEX], dt=dt)
            ball_pos = clamp_to_pitch(ball_pos)
            flight_ticks += 1

        possession, ball_pos, ball_vel, passer, flight_ticks = _update_possession(
            possession, new_pos, ball_pos, ball_vel, draws, passer, flight_ticks
        )

        pos = np.vstack([new_pos, ball_pos.reshape(1, 2)])
        vel = np.vstack([new_vel, ball_vel.reshape(1, 2)])
        team = _carrier_team(possession)
        state = MatchState(
            t=state.t + dt,
            pos=pos,
            vel=vel,
            possession=possession,
            ball_carrier_team=team,
        )
        states.append(state.copy())

        if team is not None:
            if last_team is not None and team != last_team:
                losses += 1
            last_team = team
            if team == start_team:
                retained += 1
        ball_x_sum += float(ball_pos[0])
        if tick % CONTROL_SAMPLE_EVERY == 0:
            surface = pitch_control_surface(pos[team_slice("home")], pos[team_slice("away")])
            control_home.append(surface.home_share)

    share_home = float(np.mean(control_home)) if control_home else float("nan")
    metrics = {
        "possession_retained": retained / n_ticks,
        "possession_losses": float(losses),
        "control_share_home": share_home,
        "control_share_away": 1.0 - share_home,
        "mean_ball_x": ball_x_sum / n_ticks,
        "distance_home_m": float(distance[:N_PLAYERS_PER_TEAM].mean()),
        "distance_away_m": float(distance[N_PLAYERS_PER_TEAM:].mean()),
    }
    return Rollout(states=states, metrics=metrics)


def _update_possession(
    possession: int,
    player_pos: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    draws: np.ndarray,
    passer: int,
    flight_ticks: int,
) -> tuple[int, np.ndarray, np.ndarray, int, int]:
    """Tackles, passes and loose-ball pickups for one tick."""
    if possession >= 0:
        team: TeamSide = "home" if possession < N_PLAYERS_PER_TEAM else "away"
        other: TeamSide = "away" if team == "home" else "home"
        own = player_pos[team_slice(team)]
        opp = player_pos[team_slice(other)]
        opp_offset = 0 if other == "home" else N_PLAYERS_PER_TEAM

        d_opp = np.linalg.norm(opp - player_pos[possession], axis=1)
        nearest = int(np.argmin(d_opp))
        pressure = float(d_opp[nearest])

        hazard = TURNOVER_RATE_HZ * DT * float(np.exp(-pressure / TACKLE_SCALE_M))
        if draws[0] < hazard:
            new_carrier = opp_offset + nearest
            return new_carrier, player_pos[new_carrier].copy(), np.zeros(2), -1, 0

        if pressure < PASS_TRIGGER_M and draws[1] < PASS_PROB_PER_TICK:
            carrier_local = possession - (0 if team == "home" else N_PLAYERS_PER_TEAM)
            adir = 1.0 if team == "home" else -1.0
            target = _choose_receiver(own, opp, carrier_local, adir)
            if target >= 0:
                offset = own[target] - own[carrier_local]
                dist = float(np.linalg.norm(offset))
                if dist > 1e-6:
                    speed = float(np.clip(dist / PASS_FLIGHT_S, PASS_SPEED_MIN, PASS_SPEED_MAX))
                    return -1, ball_pos, offset / dist * speed, possession, 0
        return possession, ball_pos, ball_vel, passer, flight_ticks

    if flight_ticks < MIN_FLIGHT_TICKS:
        return -1, ball_pos, ball_vel, passer, flight_ticks

    dist = np.linalg.norm(player_pos - ball_pos, axis=1)
    if passer >= 0 and flight_ticks < PASSER_LOCKOUT_TICKS:
        dist[passer] = np.inf
    winner = int(np.argmin(dist))
    if dist[winner] <= CONTROL_RADIUS_M:
        return winner, player_pos[winner].copy(), np.zeros(2), -1, 0
    return -1, ball_pos, ball_vel, passer, flight_ticks


# --- driver -----------------------------------------------------------------


def _metrics_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(rows, columns=list(METRICS_COLUMNS))
    df["arm"] = df["arm"].astype("string")
    df["rollout"] = df["rollout"].astype("int64")
    for name in METRIC_NAMES:
        df[name] = df[name].astype("float64")
    return df.reset_index(drop=True)


def _delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics.loc[metrics["arm"] == "baseline"].sort_values("rollout")
    scen = metrics.loc[metrics["arm"] == "scenario"].sort_values("rollout")
    n = int(min(len(base), len(scen)))
    rows: list[dict[str, float | str]] = []
    for name in METRIC_NAMES:
        b = base[name].to_numpy(dtype=np.float64)[:n]
        s = scen[name].to_numpy(dtype=np.float64)[:n]
        diff = s - b
        sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        rows.append(
            {
                "metric": name,
                "baseline": float(np.mean(b)) if n else float("nan"),
                "scenario": float(np.mean(s)) if n else float("nan"),
                "delta": float(np.mean(diff)) if n else float("nan"),
                "paired_sd": sd,
                "paired_se": sd / np.sqrt(n) if n else float("nan"),
                "n_rollouts": float(n),
            }
        )
    df = pd.DataFrame.from_records(rows, columns=list(DELTA_COLUMNS))
    df["metric"] = df["metric"].astype("string")
    for col in DELTA_COLUMNS[1:]:
        df[col] = df[col].astype("float64")
    return df.reset_index(drop=True)


def rollout_seeds(seed: int, n_rollouts: int) -> list[np.random.SeedSequence]:
    """Independent, reproducible per-rollout streams from one scenario seed."""
    return list(np.random.SeedSequence(int(seed)).spawn(int(n_rollouts)))


def run_scenario(tracking: pd.DataFrame, scenario: Scenario) -> SimResult:
    """Run the baseline and scenario arms and return the comparison.

    The two arms share per-rollout seeds on purpose: paired differences remove
    the rollout-to-rollout variance that both arms have in common, which is most
    of it. Reusing the seed across arms is what makes a delta of a metre
    readable out of thirty-two ten-second rollouts.
    """
    state0 = initial_state(tracking, scenario)
    horizon = float(getattr(scenario, "horizon_s", 10.0))
    n_rollouts = int(getattr(scenario, "n_rollouts", 1))
    seed = int(getattr(scenario, "seed", 0))
    seeds = rollout_seeds(seed, n_rollouts)

    policies: dict[str, Policy] = {
        "baseline": build_policy(scenario, apply_overrides=False),
        "scenario": build_policy(scenario, apply_overrides=True),
    }

    rows: list[dict[str, float | str]] = []
    first: dict[str, list[MatchState]] = {}
    for arm in _ARMS:
        for i, child in enumerate(seeds):
            result = rollout(
                state0,
                policies[arm],
                np.random.default_rng(child),
                horizon_s=horizon,
            )
            if i == 0:
                first[arm] = result.states
            rows.append({"arm": arm, "rollout": float(i), **result.metrics})

    metrics = _metrics_frame(rows)
    return SimResult(
        metrics=metrics,
        deltas=_delta_table(metrics),
        states=first.get("scenario", []),
        baseline_states=first.get("baseline", []),
        scenario=scenario,
    )


def render_result(result: SimResult, out_path: Any, *, fps: int = 10) -> Any:
    """Convenience wrapper: render rollout 0 of the scenario arm.

    The renderer is imported here rather than at module scope so that running a
    scenario headless does not pull matplotlib in.
    """
    from replay.sim.render import render_rollout_gif

    return render_rollout_gif(result.states, out_path, fps=fps)


__all__ = [
    "DELTA_COLUMNS",
    "METRICS_COLUMNS",
    "METRIC_NAMES",
    "Rollout",
    "SimResult",
    "build_policy",
    "initial_state",
    "render_result",
    "rollout",
    "rollout_seeds",
    "run_scenario",
]
