"""Tests for the pitch-control surface and the calibration report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from replay.schema import PITCH_LENGTH, PITCH_WIDTH, coerce_tracking_frame
from replay.sim.calibrate import (
    CONTROL_ACCEL,
    CONTROL_MAX_SPEED,
    CalibrationMetric,
    calibration_report,
    count_possession_losses,
    pitch_control_surface,
    possession_sequence,
    summarize_tracking,
    time_to_arrive,
)


def tracking_from_layout(
    home: np.ndarray,
    away: np.ndarray,
    ball_path: np.ndarray,
    *,
    match_id: str = "m",
    fps: float = 10.0,
) -> pd.DataFrame:
    """Static players, a moving ball, over ``len(ball_path)`` frames."""
    rows = []
    for f, ball in enumerate(ball_path):
        t = f / fps
        for team, xy in (("home", home), ("away", away)):
            for j in range(xy.shape[0]):
                rows.append(
                    {
                        "match_id": match_id,
                        "t": t,
                        "period": 1,
                        "frame": f,
                        "team": team,
                        "player_id": f"{team}_{j + 1}",
                        "jersey": j + 1,
                        "x": float(xy[j, 0]),
                        "y": float(xy[j, 1]),
                        "vx": 0.0,
                        "vy": 0.0,
                        "confidence": 1.0,
                    }
                )
        rows.append(
            {
                "match_id": match_id,
                "t": t,
                "period": 1,
                "frame": f,
                "team": "ball",
                "player_id": "ball",
                "jersey": pd.NA,
                "x": float(ball[0]),
                "y": float(ball[1]),
                "vx": 0.0,
                "vy": 0.0,
                "confidence": 1.0,
            }
        )
    return coerce_tracking_frame(pd.DataFrame(rows))


def spread_team(x_offset: float, *, rows: int = 11) -> np.ndarray:
    xs = np.linspace(10.0, 45.0, rows) + x_offset
    ys = np.linspace(6.0, 62.0, rows)
    return np.stack([xs, ys], axis=1)


# --- time to arrive ---------------------------------------------------------


def test_time_to_arrive_matches_the_two_phase_solution():
    switch = CONTROL_MAX_SPEED**2 / (2 * CONTROL_ACCEL)  # 5.333 m
    short = 3.0
    long = 20.0
    got = time_to_arrive(np.array([0.0, short, switch, long]))
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(np.sqrt(2 * short / CONTROL_ACCEL))
    assert got[2] == pytest.approx(CONTROL_MAX_SPEED / CONTROL_ACCEL)
    expected_long = CONTROL_MAX_SPEED / CONTROL_ACCEL + (long - switch) / CONTROL_MAX_SPEED
    assert got[3] == pytest.approx(expected_long)


def test_time_to_arrive_is_monotone():
    d = np.linspace(0.0, 60.0, 200)
    t = time_to_arrive(d)
    assert np.all(np.diff(t) > 0)


# --- pitch control ----------------------------------------------------------


def test_control_sums_to_one_in_every_cell():
    rng = np.random.default_rng(0)
    home = rng.uniform([0, 0], [PITCH_LENGTH, PITCH_WIDTH], size=(11, 2))
    away = rng.uniform([0, 0], [PITCH_LENGTH, PITCH_WIDTH], size=(11, 2))
    surf = pitch_control_surface(home, away)
    total = surf.home + surf.away
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    assert ((surf.home >= 0.0) & (surf.home <= 1.0)).all()


def test_a_player_standing_on_a_cell_owns_it():
    surf_probe = pitch_control_surface(np.array([[50.0, 30.0]]), np.array([[90.0, 60.0]]))
    ix = int(np.argmin(np.abs(surf_probe.xs - 50.0)))
    iy = int(np.argmin(np.abs(surf_probe.ys - 30.0)))
    home = np.array([[surf_probe.xs[ix], surf_probe.ys[iy]]])
    away = np.array([[90.0, 60.0]])
    surf = pitch_control_surface(home, away)
    assert surf.home[iy, ix] > 0.95
    assert surf.away[iy, ix] < 0.05


def test_symmetric_layouts_split_the_pitch():
    home = spread_team(0.0)
    away = np.stack([PITCH_LENGTH - home[:, 0], PITCH_WIDTH - home[:, 1]], axis=1)
    surf = pitch_control_surface(home, away)
    assert surf.home_share == pytest.approx(0.5, abs=0.02)


def test_a_team_with_nobody_on_the_pitch_controls_nothing():
    surf = pitch_control_surface(np.zeros((0, 2)), spread_team(30.0))
    assert surf.home_share < 1e-6


# --- possession -------------------------------------------------------------


def test_possession_sequence_and_losses():
    home = np.array([[20.0, 34.0]])
    away = np.array([[80.0, 34.0]])
    ball_path = np.array([[20.0, 34.0], [50.0, 34.0], [80.0, 34.0], [20.0, 34.0]])
    df = tracking_from_layout(home, away, ball_path)
    seq = possession_sequence(df)
    assert seq == ["home", None, "away", "home"]
    # Loose frames are bridged, so this is two changes of hands, not four.
    assert count_possession_losses(seq, min_hold=1) == 2


def test_short_touches_do_not_count_as_possession_changes():
    seq = ["home"] * 5 + ["away"] + ["home"] * 5
    # One frame near an opponent is a ball passing by, not a turnover.
    assert count_possession_losses(seq) == 0
    assert count_possession_losses(seq, min_hold=1) == 2


def test_a_held_possession_change_is_counted():
    seq = ["home"] * 5 + ["away"] * 5
    assert count_possession_losses(seq) == 1


def test_summarize_reports_all_three_metrics():
    home = spread_team(0.0)
    away = spread_team(45.0)
    ball_path = np.tile(np.array([[52.5, 34.0]]), (20, 1))
    stats = summarize_tracking(tracking_from_layout(home, away, ball_path))
    assert set(stats) == {"possession_loss_per_min", "control_share_home", "control_spread"}
    assert 0.0 <= stats["control_share_home"] <= 1.0
    assert stats["control_spread"] >= 0.0


# --- the report -------------------------------------------------------------


def _mirrored_pair(n_frames: int = 20) -> pd.DataFrame:
    home = spread_team(0.0)
    away = np.stack([PITCH_LENGTH - home[:, 0], PITCH_WIDTH - home[:, 1]], axis=1)
    ball = np.stack(
        [np.linspace(40.0, 65.0, n_frames), np.full(n_frames, 34.0)], axis=1
    )
    return tracking_from_layout(home, away, ball)


def test_report_on_an_identical_clip_passes_and_formats():
    df = _mirrored_pair()
    report = calibration_report(df, df)
    assert report.passed
    text = report.format()
    assert "Calibration:" in text
    assert "possession_loss_per_min" in text
    assert "overall: PASS" in text
    assert all(m.abs_delta == 0.0 for m in report.metrics)


def test_report_flags_an_obviously_miscalibrated_sim():
    source = _mirrored_pair()
    # The "sim" has the whole away team stacked in one corner, so home owns
    # essentially the entire pitch. Nothing about that resembles the source.
    home = spread_team(0.0)
    away = np.tile(np.array([[2.0, 2.0]]), (11, 1))
    ball = np.tile(np.array([[50.0, 34.0]]), (20, 1))
    sim = tracking_from_layout(home, away, ball, match_id="sim")

    report = calibration_report(sim, source)
    assert not report.passed
    names = {m.name for m in report.failures}
    assert "control_share_home" in names
    text = report.format()
    assert "overall: FAIL" in text
    assert "out of tolerance" in text


def test_report_frame_has_one_row_per_metric():
    df = _mirrored_pair()
    report = calibration_report(df, df)
    table = report.to_frame()
    assert len(table) == len(report.metrics)
    assert set(table.columns) >= {"metric", "sim", "source", "abs_delta", "tolerance", "passed"}


def test_metric_relative_delta_is_nan_against_a_zero_source():
    m = CalibrationMetric(name="x", sim=0.4, source=0.0, tolerance=1.0)
    assert np.isnan(m.rel_delta)
    assert m.abs_delta == pytest.approx(0.4)
    assert m.passed


def test_non_finite_metric_never_passes():
    m = CalibrationMetric(name="x", sim=float("nan"), source=0.5, tolerance=10.0)
    assert not m.passed
