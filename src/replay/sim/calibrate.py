"""Does the simulator behave like the footage it was initialised from?

Two questions, deliberately cheap ones. How often does the ball change hands,
and how is space divided up. Both are computable from a canonical tracking
frame alone, which means the same code runs over the source clip and over a
rollout without either side having to cooperate.

The tolerances below are stated constants, not fitted ones. If a rollout misses
them the report says so; that is a finding about the simulator, not a reason to
widen the number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from replay.schema import PITCH_LENGTH, PITCH_WIDTH

#: Grid resolution for the pitch-control surface, meters. 2.5 m gives a 42x27
#: grid: fine enough that a single player's zone is several cells across, coarse
#: enough that a 100-tick rollout costs milliseconds.
GRID_CELL_M = 2.5

#: Time-to-arrive model. Deliberately the same numbers the physics layer uses
#: for players, so the surface is consistent with what the sim can actually do.
CONTROL_MAX_SPEED = 8.0
CONTROL_ACCEL = 6.0
#: Logistic width in seconds. At a 0.45 s advantage a team holds ~73% of a cell;
#: this is the softening that stops the surface being a hard Voronoi diagram.
CONTROL_TAU_S = 0.45

#: A player within this distance of the ball, and closer than anyone else, is
#: treated as being in possession.
POSSESSION_RADIUS_M = 2.0

#: Consecutive frames a team must hold the ball before it counts as a
#: possession. At 10 Hz that is 0.3 s. Without it a pass that flies within two
#: meters of an opponent registers as a turnover and back again, and the
#: measured churn is dominated by that artefact rather than by real play.
MIN_POSSESSION_FRAMES = 3

#: Frames used for the pitch-control summary. Control changes slowly, so a few
#: dozen evenly spaced frames describe the clip as well as all of them.
CONTROL_SAMPLE_FRAMES = 40

#: Fraction of frames that must actually contain a ball before a possession rate
#: inferred from the source means anything. Below this the sequence is mostly
#: "loose", the count comes out near zero, and comparing to it would flatter or
#: damn the simulator for no reason.
BALL_COVERAGE_FOR_POSSESSION = 0.5

# --- tolerances -------------------------------------------------------------
# Event-data studies of open play put turnovers at roughly three to four per
# minute across both teams. We allow the sim to be wrong by that whole amount:
# the possession model here is proximity plus a duel coin-flip, not a contested
# ball model, so demanding better would be demanding something it cannot give.
POSSESSION_LOSS_TOL_PER_MIN = 4.0

# Mean home share of the pitch-control surface. Across a match this sits near
# 0.5 and moves by around 0.1 between a team camped in its own half and one
# pinning the opponent in. Being outside 0.10 means the sim has put the teams in
# a materially different shape from the source.
CONTROL_SHARE_TOL = 0.10

# Spatial standard deviation of the same surface: how sharply divided the pitch
# is. A normal frame sits around 0.35-0.45; 0.10 is the same order of slack as
# the share tolerance.
CONTROL_SPREAD_TOL = 0.10

METRIC_TOLERANCES: dict[str, float] = {
    "possession_loss_per_min": POSSESSION_LOSS_TOL_PER_MIN,
    "control_share_home": CONTROL_SHARE_TOL,
    "control_spread": CONTROL_SPREAD_TOL,
}

METRIC_UNITS: dict[str, str] = {
    "possession_loss_per_min": "per min",
    "control_share_home": "share",
    "control_spread": "sd",
}


@dataclass(slots=True)
class PitchControl:
    """Home team's share of each grid cell, plus the grid it sits on."""

    home: np.ndarray  # (ny, nx) in [0, 1]
    xs: np.ndarray  # (nx,) cell centers in meters
    ys: np.ndarray  # (ny,)

    @property
    def away(self) -> np.ndarray:
        return 1.0 - self.home

    @property
    def home_share(self) -> float:
        return float(self.home.mean())

    @property
    def spread(self) -> float:
        return float(self.home.std())


def time_to_arrive(dist: np.ndarray, *, max_speed: float = CONTROL_MAX_SPEED,
                   accel: float = CONTROL_ACCEL) -> np.ndarray:
    """Seconds to cover ``dist``: accelerate flat out, then hold top speed.

    Below the switch distance the sprint never reaches top speed, so the time is
    the pure constant-acceleration solution; above it, top speed plus the time
    it took to get there.
    """
    d = np.asarray(dist, dtype=np.float64)
    switch = max_speed * max_speed / (2.0 * accel)
    accel_phase = np.sqrt(np.maximum(2.0 * d, 0.0) / accel)
    cruise_phase = max_speed / accel + (d - switch) / max_speed
    return np.where(d <= switch, accel_phase, cruise_phase)


def pitch_control_surface(
    home_xy: np.ndarray,
    away_xy: np.ndarray,
    *,
    cell_m: float = GRID_CELL_M,
    tau: float = CONTROL_TAU_S,
    max_speed: float = CONTROL_MAX_SPEED,
    accel: float = CONTROL_ACCEL,
) -> PitchControl:
    """Logistic of the two teams' time-to-arrive advantage over a pitch grid.

    Home and away control sum to exactly 1 in every cell by construction. A
    player standing on a cell arrives in zero seconds and takes essentially all
    of it. Fully vectorized over the grid because this runs once per rollout
    frame sample.
    """
    home = np.asarray(home_xy, dtype=np.float64).reshape(-1, 2)
    away = np.asarray(away_xy, dtype=np.float64).reshape(-1, 2)
    home = home[np.isfinite(home).all(axis=1)]
    away = away[np.isfinite(away).all(axis=1)]

    nx = max(2, int(round(PITCH_LENGTH / cell_m)))
    ny = max(2, int(round(PITCH_WIDTH / cell_m)))
    xs = (np.arange(nx) + 0.5) * (PITCH_LENGTH / nx)
    ys = (np.arange(ny) + 0.5) * (PITCH_WIDTH / ny)
    gx, gy = np.meshgrid(xs, ys)
    cells = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)  # (C, 2)

    t_home = _team_arrival(home, cells, max_speed, accel)
    t_away = _team_arrival(away, cells, max_speed, accel)

    advantage = np.clip((t_away - t_home) / tau, -60.0, 60.0)
    control = 1.0 / (1.0 + np.exp(-advantage))
    return PitchControl(home=control.reshape(ny, nx), xs=xs, ys=ys)


#: Backwards-friendly alias; the surface is the thing, the function name is not.
pitch_control = pitch_control_surface


def _team_arrival(
    players: np.ndarray, cells: np.ndarray, max_speed: float, accel: float
) -> np.ndarray:
    if players.shape[0] == 0:
        return np.full(cells.shape[0], 1e6)
    dist = np.linalg.norm(cells[None, :, :] - players[:, None, :], axis=-1)
    return time_to_arrive(dist, max_speed=max_speed, accel=accel).min(axis=0)


# --- tracking-frame summaries ----------------------------------------------


def _frame_positions(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    home = group.loc[group["team"] == "home", ["x", "y"]].to_numpy(dtype=np.float64)
    away = group.loc[group["team"] == "away", ["x", "y"]].to_numpy(dtype=np.float64)
    ball = group.loc[group["team"] == "ball", ["x", "y"]].to_numpy(dtype=np.float64)
    ball_xy = ball[0] if ball.shape[0] else np.array([np.nan, np.nan])
    return home, away, ball_xy


def possession_sequence(tracking: pd.DataFrame) -> list[str | None]:
    """Per frame, which team is in possession (or None when the ball is loose)."""
    out: list[str | None] = []
    for _, group in tracking.groupby("frame", sort=True):
        home, away, ball = _frame_positions(group)
        if not np.isfinite(ball).all():
            out.append(None)
            continue
        d_home = np.linalg.norm(home - ball, axis=1) if home.shape[0] else np.array([np.inf])
        d_away = np.linalg.norm(away - ball, axis=1) if away.shape[0] else np.array([np.inf])
        best_home = float(np.nanmin(d_home)) if d_home.size else np.inf
        best_away = float(np.nanmin(d_away)) if d_away.size else np.inf
        best = min(best_home, best_away)
        if not np.isfinite(best) or best > POSSESSION_RADIUS_M:
            out.append(None)
        else:
            out.append("home" if best_home <= best_away else "away")
    return out


def count_possession_losses(
    sequence: list[str | None], *, min_hold: int = MIN_POSSESSION_FRAMES
) -> int:
    """Changes of possessing team, ignoring loose-ball gaps in between.

    A team has to hold the ball for ``min_hold`` consecutive frames before the
    change counts, so a ball passing close to an opponent is not scored as a
    turnover followed by an immediate one back.
    """
    losses = 0
    confirmed: str | None = None
    run_team: str | None = None
    run_len = 0
    for team in sequence:
        if team is None:
            continue
        if team == run_team:
            run_len += 1
        else:
            run_team, run_len = team, 1
        if run_len >= min_hold:
            if confirmed is not None and team != confirmed:
                losses += 1
            confirmed = team
    return losses


def _duration_minutes(tracking: pd.DataFrame) -> float:
    t = tracking["t"].to_numpy(dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 0.0
    return float(t.max() - t.min()) / 60.0


def summarize_tracking(
    tracking: pd.DataFrame, *, sample_frames: int = CONTROL_SAMPLE_FRAMES
) -> dict[str, float]:
    """The three calibration metrics for one canonical tracking frame."""
    required = {"frame", "team", "x", "y", "t"}
    missing = required - set(tracking.columns)
    if missing:
        raise ValueError(
            f"tracking frame is missing columns needed to calibrate: {sorted(missing)}"
        )

    seq = possession_sequence(tracking)
    losses = count_possession_losses(seq)
    minutes = _duration_minutes(tracking)
    loss_rate = losses / minutes if minutes > 0 else float(losses)

    frames = np.array(sorted(tracking["frame"].unique()))
    if frames.size == 0:
        return {
            "possession_loss_per_min": loss_rate,
            "control_share_home": float("nan"),
            "control_spread": float("nan"),
        }
    n_take = min(sample_frames, frames.size)
    take = frames[np.unique(np.linspace(0, frames.size - 1, n_take).astype(int))]
    groups = {int(f): g for f, g in tracking.groupby("frame", sort=True)}

    shares: list[float] = []
    spreads: list[float] = []
    for f in take:
        home, away, _ = _frame_positions(groups[int(f)])
        if home.shape[0] == 0 and away.shape[0] == 0:
            continue
        surf = pitch_control_surface(home, away)
        shares.append(surf.home_share)
        spreads.append(surf.spread)

    return {
        "possession_loss_per_min": float(loss_rate),
        "control_share_home": float(np.mean(shares)) if shares else float("nan"),
        "control_spread": float(np.mean(spreads)) if spreads else float("nan"),
    }


# --- report -----------------------------------------------------------------


@dataclass(slots=True)
class CalibrationMetric:
    name: str
    sim: float
    source: float
    tolerance: float
    unit: str = ""
    #: False when the source cannot support this comparison at all. An
    #: unmeasurable metric is reported as n/a and is neither a pass nor a
    #: failure: scoring it either way invents a verdict out of missing data.
    measurable: bool = True

    @property
    def abs_delta(self) -> float:
        return float(abs(self.sim - self.source))

    @property
    def rel_delta(self) -> float:
        """Absolute delta over the source value; NaN when the source is ~0."""
        denom = abs(self.source)
        if denom < 1e-9:
            return float("nan")
        return float(self.abs_delta / denom)

    @property
    def passed(self) -> bool:
        if not self.measurable:
            return True
        if not (np.isfinite(self.sim) and np.isfinite(self.source)):
            return False
        return self.abs_delta <= self.tolerance

    @property
    def status(self) -> str:
        if not self.measurable:
            return "n/a"
        return "PASS" if self.passed else "FAIL"

    def to_row(self) -> dict[str, float | str | bool]:
        return {
            "metric": self.name,
            "unit": self.unit,
            "sim": self.sim,
            "source": self.source,
            "abs_delta": self.abs_delta,
            "rel_delta": self.rel_delta,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


@dataclass(slots=True)
class CalibrationReport:
    """Sim-versus-source comparison on possession churn and pitch control."""

    metrics: list[CalibrationMetric] = field(default_factory=list)
    n_frames_sim: int = 0
    n_frames_source: int = 0
    notes: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.metrics) and all(m.passed for m in self.metrics)

    @property
    def failures(self) -> list[CalibrationMetric]:
        return [m for m in self.metrics if not m.passed]

    @property
    def unmeasured(self) -> list[CalibrationMetric]:
        return [m for m in self.metrics if not m.measurable]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([m.to_row() for m in self.metrics])

    def format(self) -> str:
        header = (
            f"Calibration: sim {self.n_frames_sim} frames vs source "
            f"{self.n_frames_source} frames"
        )
        cols = ("metric", "unit", "sim", "source", "abs delta", "rel delta", "tol", "")
        widths = (26, 8, 10, 10, 10, 10, 8, 6)
        lines = [header, "-" * len(header), _row(cols, widths)]
        for m in self.metrics:
            rel = "n/a" if not np.isfinite(m.rel_delta) else f"{m.rel_delta * 100:.1f}%"
            lines.append(
                _row(
                    (
                        m.name,
                        m.unit,
                        _num(m.sim),
                        _num(m.source),
                        _num(m.abs_delta),
                        rel,
                        _num(m.tolerance),
                        m.status,
                    ),
                    widths,
                )
            )
        verdict = "PASS" if self.passed else "FAIL"
        lines.append("")
        lines.append(f"overall: {verdict}")
        if not self.passed:
            names = ", ".join(m.name for m in self.failures)
            lines.append(f"out of tolerance: {names}")
        if self.unmeasured:
            names = ", ".join(m.name for m in self.unmeasured)
            lines.append(f"not measurable from this source: {names}")
        if self.notes:
            lines.append(f"note: {self.notes}")
        return "\n".join(lines)


def _row(values: tuple[str, ...], widths: tuple[int, ...]) -> str:
    return "".join(str(v).ljust(w) for v, w in zip(values, widths, strict=True)).rstrip()


def _num(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


def calibration_report(
    sim: pd.DataFrame,
    source: pd.DataFrame,
    *,
    sim_loss_rate_per_min: float | None = None,
    n_rollouts: int = 1,
) -> CalibrationReport:
    """Compare a simulated tracking frame against the source it came from.

    ``sim`` is one rollout, which is a small sample to judge a stochastic model
    on. A ten-second rollout holds one or two turnovers, so its implied rate
    moves in steps of six per minute and a single unlucky draw can look like a
    badly miscalibrated simulator. Pass ``sim_loss_rate_per_min`` — the mean the
    driver measured across every rollout it ran — to be judged on the full
    sample instead. The driver's count and the one inferred here agree on the
    same rollout; averaging is about variance, not about a different definition.
    """
    sim_stats = summarize_tracking(sim)
    src_stats = summarize_tracking(source)
    if sim_loss_rate_per_min is not None:
        sim_stats = {**sim_stats, "possession_loss_per_min": float(sim_loss_rate_per_min)}

    # Possession can only be inferred from a source that actually shows a ball.
    # Below the coverage floor the inferred sequence is almost all "loose", the
    # rate collapses to zero, and comparing against it would hand out a verdict
    # the data cannot support — a confident PASS being every bit as wrong as a
    # confident FAIL.
    # Count frames where the ball has a position, not rows labelled "ball". The
    # Metrica layout is a dense grid, so a round trip through it emits a ball
    # row for every frame whether or not the ball was ever seen; counting rows
    # reads 9% coverage as 100% and turns "unmeasurable" back into a confident
    # comparison against zero.
    is_ball = source["team"].astype("string") == "ball"
    located = np.isfinite(source["x"]) & np.isfinite(source["y"])
    ball_frames = int(source.loc[is_ball & located, "frame"].nunique())
    src_frames = int(source["frame"].nunique())
    ball_coverage = ball_frames / src_frames if src_frames else 0.0
    possession_measurable = ball_coverage >= BALL_COVERAGE_FOR_POSSESSION

    metrics = [
        CalibrationMetric(
            name=name,
            sim=sim_stats[name],
            source=src_stats[name],
            tolerance=tol,
            unit=METRIC_UNITS.get(name, ""),
            measurable=possession_measurable or name != "possession_loss_per_min",
        )
        for name, tol in METRIC_TOLERANCES.items()
    ]
    notes = ""
    if _duration_minutes(source) <= 0.0:
        notes = "source clip has no measurable duration; loss rate reported as a raw count"
    if sim_loss_rate_per_min is not None and n_rollouts > 1:
        notes = (
            f"possession rate averaged over {n_rollouts} rollouts; "
            "the other metrics describe rollout 0"
            + (f". {notes}" if notes else "")
        )
    if not possession_measurable:
        notes = (
            f"source has a ball in only {ball_coverage * 100:.0f}% of frames, too few "
            "to infer possession from"
            + (f". {notes}" if notes else "")
        )
    return CalibrationReport(
        metrics=metrics,
        n_frames_sim=int(sim["frame"].nunique()),
        n_frames_source=src_frames,
        notes=notes,
    )
