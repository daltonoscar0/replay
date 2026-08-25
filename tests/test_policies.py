"""Tests for the rule-based and behaviour-cloning policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from replay.schema import coerce_tracking_frame
from replay.sim.physics import MAX_SPEED, step_players
from replay.sim.policies import BCPolicy, RuleBasedPolicy, lane_clearance, point_segment_distance
from replay.sim.policies.rule_based import formation_template
from replay.sim.state import MatchState


def make_state(*, ball=(52.5, 34.0), possession=11, carrier="away") -> MatchState:
    pos = np.zeros((23, 2))
    pos[:11] = formation_template("home")
    pos[11:22] = formation_template("away")
    ball_xy = np.asarray(ball, dtype=float)
    pos[22] = ball_xy
    if possession >= 0:
        pos[possession] = ball_xy
    return MatchState(
        t=0.0,
        pos=pos,
        vel=np.zeros((23, 2)),
        possession=possession,
        ball_carrier_team=carrier,
    )


# --- geometry ---------------------------------------------------------------


def test_point_segment_distance_hand_computed():
    a = np.array([0.0, 0.0])
    b = np.array([10.0, 0.0])
    pts = np.array(
        [
            [5.0, 3.0],  # perpendicular from the middle
            [-4.0, 0.0],  # behind the start, clamps to a
            [14.0, 3.0],  # past the end, clamps to b -> sqrt(16 + 9)
            [0.0, 0.0],  # on the segment
            [10.0, -7.0],  # perpendicular at the end
        ]
    )
    got = point_segment_distance(pts, a, b)
    np.testing.assert_allclose(got, [3.0, 4.0, 5.0, 0.0, 7.0])


def test_point_segment_distance_degenerate_segment():
    a = np.array([3.0, 4.0])
    got = point_segment_distance(np.array([[0.0, 0.0]]), a, a)
    np.testing.assert_allclose(got, [5.0])


def test_lane_clearance_picks_the_nearest_blocker():
    origin = np.array([0.0, 0.0])
    targets = np.array([[10.0, 0.0], [0.0, 10.0]])
    opponents = np.array([[5.0, 3.0], [1.0, 5.0]])
    got = lane_clearance(origin, targets, opponents)
    # First lane: the (5,3) opponent is 3 m off it, the other is sqrt(1+25)=5.10.
    # Second lane: (1,5) is 1 m off it, (5,3) is 5 m off it.
    np.testing.assert_allclose(got, [3.0, 1.0])


def test_lane_clearance_with_no_opponents_is_finite():
    got = lane_clearance(np.zeros(2), np.array([[10.0, 0.0]]), np.zeros((0, 2)))
    assert np.isfinite(got).all()
    assert got[0] > 50.0


# --- defending --------------------------------------------------------------


def test_defender_inside_press_trigger_runs_at_the_carrier():
    state = make_state()
    carrier = state.pos[11]
    state.pos[3] = carrier + np.array([-6.5, 0.0])  # 6.5 m away, inside the 12 m trigger

    policy = RuleBasedPolicy(press_trigger_m=12.0)
    vel = policy.desired_velocity(state, "home", np.random.default_rng(0))

    to_carrier = carrier - state.pos[3]
    unit = to_carrier / np.linalg.norm(to_carrier)
    v = vel[3]
    assert np.linalg.norm(v) > 4.0
    assert float(v @ unit) / np.linalg.norm(v) > 0.99


def test_defenders_outside_press_trigger_hold_their_shape():
    state = make_state()
    policy = RuleBasedPolicy(press_trigger_m=12.0)
    rest = policy.resting_positions(state, "home", defending=True)
    state.pos[:11] = rest

    vel = policy.desired_velocity(state, "home", np.random.default_rng(0))
    dist = np.linalg.norm(state.pos[:11] - state.pos[11], axis=1)
    holding = dist > 12.0

    assert holding.sum() >= 6
    assert np.linalg.norm(vel[holding], axis=1).max() < 0.5


def test_speeds_never_exceed_the_physical_cap():
    state = make_state()
    policy = RuleBasedPolicy()
    for team in ("home", "away"):
        vel = policy.desired_velocity(state, team, np.random.default_rng(3))
        assert vel.shape == (11, 2)
        assert np.linalg.norm(vel, axis=1).max() <= MAX_SPEED + 1e-9


def test_policy_is_deterministic_given_the_rng():
    state = make_state()
    policy = RuleBasedPolicy()
    a = policy.desired_velocity(state, "home", np.random.default_rng(11))
    b = policy.desired_velocity(state, "home", np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)


# --- the defensive line override --------------------------------------------


def test_defensive_line_translates_the_resting_block():
    state = make_state()
    low = RuleBasedPolicy(defensive_line=0.0).resting_positions(state, "home")
    high = RuleBasedPolicy(defensive_line=1.5).resting_positions(state, "home")
    assert float(np.mean(high[:, 0] - low[:, 0])) == pytest.approx(1.5, abs=1e-9)


def test_defensive_line_is_relative_to_the_direction_of_attack():
    state = make_state(possession=0, carrier="home")
    low = RuleBasedPolicy(defensive_line=0.0).resting_positions(state, "away")
    high = RuleBasedPolicy(defensive_line=1.5).resting_positions(state, "away")
    # Away attacks towards -x, so "up the pitch" for the away block is -x.
    assert float(np.mean(high[:, 0] - low[:, 0])) == pytest.approx(-1.5, abs=1e-9)


def _settled_defence(defensive_line: float, ticks: int = 40) -> np.ndarray:
    """Let the home block converge against a stationary carrier."""
    state = make_state(ball=(40.0, 34.0))
    policy = RuleBasedPolicy(defensive_line=defensive_line)
    rng = np.random.default_rng(5)
    for _ in range(ticks):
        want = policy.desired_velocity(state, "home", rng)
        pos, vel = step_players(state.pos[:11], state.vel[:11], want)
        state.pos[:11] = pos
        state.vel[:11] = vel
    return state.pos[:11].copy()


def test_raising_the_line_pushes_the_settled_block_up_the_pitch():
    flat = _settled_defence(0.0)
    high = _settled_defence(1.5)
    # Measure the back four, which is what a coach means by "the line".
    delta = float(np.mean(high[1:5, 0]) - np.mean(flat[1:5, 0]))
    assert delta > 0.0
    assert 1.0 < delta < 2.0


def test_line_compactness_squeezes_the_block_towards_the_ball():
    state = make_state(ball=(30.0, 20.0))
    loose = RuleBasedPolicy(line_compactness=0.0).resting_positions(state, "home")
    tight = RuleBasedPolicy(line_compactness=2.0).resting_positions(state, "home")
    ball = state.ball
    assert np.linalg.norm(tight - ball, axis=1).mean() < np.linalg.norm(loose - ball, axis=1).mean()


# --- attacking --------------------------------------------------------------


def test_pass_lane_weight_changes_off_ball_movement():
    state = make_state(possession=6, carrier="home")
    low = RuleBasedPolicy(pass_lane_weight=0.0).desired_velocity(
        state, "home", np.random.default_rng(2)
    )
    high = RuleBasedPolicy(pass_lane_weight=4.0).desired_velocity(
        state, "home", np.random.default_rng(2)
    )
    assert not np.allclose(low, high)


def test_carrier_advances_towards_the_opposition_goal():
    state = make_state(possession=6, carrier="home")
    vel = RuleBasedPolicy().desired_velocity(state, "home", np.random.default_rng(0))
    assert vel[6, 0] > 0.0


# --- behaviour cloning ------------------------------------------------------


def synthetic_tracking(n_frames: int = 24, *, seed: int = 0) -> pd.DataFrame:
    """A short clip of players drifting on straight lines, ball included."""
    rng = np.random.default_rng(seed)
    home0 = formation_template("home")
    away0 = formation_template("away")
    vel_home = rng.normal(scale=1.0, size=(11, 2))
    vel_away = rng.normal(scale=1.0, size=(11, 2))
    ball0 = home0[6].copy()
    ball_v = np.array([2.0, 0.5])

    rows = []
    for f in range(n_frames):
        t = f * 0.1
        for team, base, vel in (("home", home0, vel_home), ("away", away0, vel_away)):
            xy = base + vel * t
            for j in range(11):
                rows.append(
                    {
                        "match_id": "syn",
                        "t": t,
                        "period": 1,
                        "frame": f,
                        "team": team,
                        "player_id": f"{team}_{j + 1}",
                        "jersey": j + 1,
                        "x": float(xy[j, 0]),
                        "y": float(xy[j, 1]),
                        "vx": float(vel[j, 0]),
                        "vy": float(vel[j, 1]),
                        "confidence": 1.0,
                    }
                )
        ball = ball0 + ball_v * t
        rows.append(
            {
                "match_id": "syn",
                "t": t,
                "period": 1,
                "frame": f,
                "team": "ball",
                "player_id": "ball",
                "jersey": pd.NA,
                "x": float(ball[0]),
                "y": float(ball[1]),
                "vx": float(ball_v[0]),
                "vy": float(ball_v[1]),
                "confidence": 1.0,
            }
        )
    return coerce_tracking_frame(pd.DataFrame(rows))


def test_bc_trains_and_the_loss_goes_down():
    policy = BCPolicy.train(synthetic_tracking(), hidden=8, epochs=12, seed=0)
    losses = policy.loss_history
    assert losses.shape == (12,)
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]


def test_bc_desired_velocity_shape_and_cap():
    policy = BCPolicy.train(synthetic_tracking(), hidden=8, epochs=4, seed=1)
    state = make_state()
    vel = policy.desired_velocity(state, "home", np.random.default_rng(0))
    assert vel.shape == (11, 2)
    assert np.isfinite(vel).all()
    assert np.linalg.norm(vel, axis=1).max() <= MAX_SPEED + 1e-9


def test_bc_save_load_round_trip_is_identical(tmp_path):
    policy = BCPolicy.train(synthetic_tracking(), hidden=8, epochs=4, seed=2)
    path = policy.save(tmp_path / "bc.npz")
    assert path.exists()

    reloaded = BCPolicy.load(path)
    state = make_state()
    a = policy.desired_velocity(state, "home", np.random.default_rng(0))
    b = reloaded.desired_velocity(state, "home", np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(policy.w1, reloaded.w1)
    np.testing.assert_array_equal(policy.feat_std, reloaded.feat_std)


def test_bc_training_is_seeded():
    frame = synthetic_tracking()
    a = BCPolicy.train(frame, hidden=8, epochs=4, seed=3)
    b = BCPolicy.train(frame, hidden=8, epochs=4, seed=3)
    c = BCPolicy.train(frame, hidden=8, epochs=4, seed=4)
    np.testing.assert_array_equal(a.w1, b.w1)
    assert not np.allclose(a.w1, c.w1)


def test_bc_refuses_a_frame_with_nothing_to_learn_from():
    empty = synthetic_tracking(n_frames=1)
    with pytest.raises(ValueError, match="consecutive frames"):
        BCPolicy.train(empty, hidden=4, epochs=2, seed=0)
