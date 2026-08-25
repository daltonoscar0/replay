"""Tests for the rollout driver, the delta table and the GIF renderer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from replay.schema import coerce_tracking_frame
from replay.sim.physics import DT
from replay.sim.policies import RuleBasedPolicy
from replay.sim.policies.rule_based import formation_template
from replay.sim.render import render_rollout_gif
from replay.sim.run import (
    DELTA_COLUMNS,
    METRIC_NAMES,
    METRICS_COLUMNS,
    build_policy,
    initial_state,
    rollout,
    rollout_seeds,
    run_scenario,
)
from replay.sim.scenario import InitialState, PolicyConfig, Scenario
from replay.sim.state import MatchState

N_FRAMES = 4


def demo_tracking(*, drop_home_slots: tuple[int, ...] = ()) -> pd.DataFrame:
    """A short static clip: both teams in shape, ball at the home number 7.

    ``drop_home_slots`` removes those home players from frame 0 only, which is
    what a detector miss looks like: the player is still on the roster.
    """
    home = formation_template("home")
    away = formation_template("away")
    ball = home[6]
    rows = []
    for f in range(N_FRAMES):
        t = f * DT
        for team, xy in (("home", home), ("away", away)):
            for j in range(11):
                if f == 0 and team == "home" and j in drop_home_slots:
                    continue
                rows.append(
                    {
                        "match_id": "demo",
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
                "match_id": "demo",
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


def make_scenario(*, seed: int = 0, defensive_line: float = 1.5, n_rollouts: int = 2) -> Scenario:
    return Scenario(
        name="test",
        initial_state=InitialState(),
        policy=PolicyConfig(kind="rule_based", defensive_line=defensive_line),
        horizon_s=1.0,
        n_rollouts=n_rollouts,
        seed=seed,
    )


# --- initial state ----------------------------------------------------------


def test_initial_state_uses_the_first_frame_when_source_is_auto():
    state = initial_state(demo_tracking(), make_scenario())
    assert isinstance(state, MatchState)
    assert np.isfinite(state.pos).all()
    assert state.ball_carrier_team == "home"


def test_missing_player_slots_are_filled_from_the_formation():
    state = initial_state(demo_tracking(drop_home_slots=(2, 9)), make_scenario())
    assert np.isfinite(state.pos).all()
    template = formation_template("home")
    np.testing.assert_allclose(state.pos[2], template[2])
    np.testing.assert_allclose(state.pos[9], template[9])


def test_explicit_frame_is_honoured():
    scenario = Scenario(
        initial_state=InitialState(source="tracking.csv", frame=2),
        horizon_s=1.0,
        n_rollouts=1,
    )
    state = initial_state(demo_tracking(), scenario)
    assert state.t == pytest.approx(2 * DT)


# --- policy construction ----------------------------------------------------


def test_baseline_arm_ignores_the_scenario_overrides():
    scenario = make_scenario(defensive_line=4.0)
    base = build_policy(scenario, apply_overrides=False)
    scen = build_policy(scenario, apply_overrides=True)
    assert isinstance(base, RuleBasedPolicy)
    assert base.defensive_line == 0.0
    assert scen.defensive_line == 4.0


def test_bc_policy_without_weights_is_a_clear_error():
    class _Cfg:
        kind = "bc"
        weights = None

    class _Scenario:
        policy = _Cfg()

    with pytest.raises(ValueError, match="weights"):
        build_policy(_Scenario(), apply_overrides=True)


def test_unknown_policy_kind_is_a_clear_error():
    class _Cfg:
        kind = "transformer"

    class _Scenario:
        policy = _Cfg()

    with pytest.raises(ValueError, match="unknown policy kind"):
        build_policy(_Scenario(), apply_overrides=True)


# --- one rollout ------------------------------------------------------------


def test_rollout_length_and_metric_keys():
    state = initial_state(demo_tracking(), make_scenario())
    result = rollout(state, RuleBasedPolicy(), np.random.default_rng(0), horizon_s=1.0)
    assert len(result.states) == int(round(1.0 / DT)) + 1
    assert set(result.metrics) == set(METRIC_NAMES)
    assert 0.0 <= result.metrics["possession_retained"] <= 1.0
    assert result.metrics["distance_home_m"] >= 0.0


def test_rollout_positions_stay_on_the_pitch():
    state = initial_state(demo_tracking(), make_scenario())
    result = rollout(state, RuleBasedPolicy(), np.random.default_rng(4), horizon_s=2.0)
    for s in result.states:
        assert np.isfinite(s.pos).all()
        assert (s.pos[:, 0] >= -1e-9).all()
        assert (s.pos[:, 0] <= 105.0 + 1e-9).all()
        assert (s.pos[:, 1] >= -1e-9).all()
        assert (s.pos[:, 1] <= 68.0 + 1e-9).all()


def test_rollout_is_reproducible_from_the_generator():
    state = initial_state(demo_tracking(), make_scenario())
    policy = RuleBasedPolicy()
    a = rollout(state, policy, np.random.default_rng(9), horizon_s=1.0)
    b = rollout(state, policy, np.random.default_rng(9), horizon_s=1.0)
    assert a.metrics == b.metrics
    np.testing.assert_array_equal(a.states[-1].pos, b.states[-1].pos)


# --- the driver -------------------------------------------------------------


def test_seeds_are_derived_deterministically():
    a = rollout_seeds(7, 3)
    b = rollout_seeds(7, 3)
    assert len(a) == 3
    assert [s.generate_state(2).tolist() for s in a] == [s.generate_state(2).tolist() for s in b]
    c = rollout_seeds(8, 3)
    assert [s.generate_state(2).tolist() for s in a] != [s.generate_state(2).tolist() for s in c]


def test_same_seed_gives_a_bit_identical_metrics_table():
    tracking = demo_tracking()
    first = run_scenario(tracking, make_scenario(seed=42))
    second = run_scenario(tracking, make_scenario(seed=42))
    pdt.assert_frame_equal(first.metrics, second.metrics, check_exact=True)
    pdt.assert_frame_equal(first.deltas, second.deltas, check_exact=True)


def test_a_different_seed_gives_different_metrics():
    tracking = demo_tracking()
    a = run_scenario(tracking, make_scenario(seed=1))
    b = run_scenario(tracking, make_scenario(seed=2))
    numeric = list(METRIC_NAMES)
    assert not a.metrics[numeric].equals(b.metrics[numeric])


def test_metrics_table_shape_and_columns():
    result = run_scenario(demo_tracking(), make_scenario(n_rollouts=3))
    assert list(result.metrics.columns) == list(METRICS_COLUMNS)
    assert len(result.metrics) == 6  # two arms x three rollouts
    assert set(result.metrics["arm"]) == {"baseline", "scenario"}


def test_delta_table_reports_paired_spread():
    result = run_scenario(demo_tracking(), make_scenario(n_rollouts=3))
    assert list(result.deltas.columns) == list(DELTA_COLUMNS)
    assert list(result.deltas["metric"]) == list(METRIC_NAMES)
    assert (result.deltas["n_rollouts"] == 3).all()
    assert (result.deltas["paired_sd"] >= 0).all()
    row = result.deltas.set_index("metric").loc["mean_ball_x"]
    assert row["delta"] == pytest.approx(row["scenario"] - row["baseline"], abs=1e-9)


def test_paired_arms_share_seeds_so_the_baseline_is_stable():
    tracking = demo_tracking()
    mild = run_scenario(tracking, make_scenario(seed=5, defensive_line=0.5))
    strong = run_scenario(tracking, make_scenario(seed=5, defensive_line=6.0))
    base_a = mild.metrics.loc[mild.metrics["arm"] == "baseline"].reset_index(drop=True)
    base_b = strong.metrics.loc[strong.metrics["arm"] == "baseline"].reset_index(drop=True)
    pdt.assert_frame_equal(base_a, base_b, check_exact=True)


def test_result_carries_states_for_rendering():
    result = run_scenario(demo_tracking(), make_scenario())
    assert len(result.states) == int(round(1.0 / DT)) + 1
    assert len(result.baseline_states) == len(result.states)
    assert result.scenario is not None


def test_result_converts_back_to_a_tracking_frame():
    result = run_scenario(demo_tracking(), make_scenario())
    df = result.to_tracking(match_id="sim")
    assert set(df["team"]) == {"home", "away", "ball"}
    assert df["frame"].nunique() == len(result.states)


def test_raising_the_line_moves_the_defensive_block(tmp_path):
    """The away block is the defending one here, so a higher line means lower x."""
    del tmp_path
    tracking = demo_tracking()
    result = run_scenario(tracking, make_scenario(seed=3, defensive_line=6.0, n_rollouts=2))
    base_x = np.mean([s.pos[11:22, 0].mean() for s in result.baseline_states])
    scen_x = np.mean([s.pos[11:22, 0].mean() for s in result.states])
    assert scen_x < base_x


# --- rendering --------------------------------------------------------------


def test_render_rollout_gif_writes_a_small_file(tmp_path):
    result = run_scenario(demo_tracking(), make_scenario())
    out = render_rollout_gif(result.states, tmp_path / "rollout.gif", fps=10)
    assert out.exists()
    size = out.stat().st_size
    assert size > 0
    assert size < 400_000


def test_render_rejects_an_empty_rollout(tmp_path):
    with pytest.raises(ValueError, match="no states"):
        render_rollout_gif([], tmp_path / "empty.gif")


@pytest.mark.slow
def test_determinism_holds_at_a_realistic_rollout_count():
    tracking = demo_tracking()
    scenario = Scenario(
        name="full",
        initial_state=InitialState(),
        policy=PolicyConfig(defensive_line=2.0),
        horizon_s=5.0,
        n_rollouts=16,
        seed=17,
    )
    a = run_scenario(tracking, scenario)
    b = run_scenario(tracking, scenario)
    pdt.assert_frame_equal(a.metrics, b.metrics, check_exact=True)
    pdt.assert_frame_equal(a.deltas, b.deltas, check_exact=True)
