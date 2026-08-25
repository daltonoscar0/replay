import numpy as np
import pandas as pd
import pytest

from replay.schema import TRACKING_COLUMNS, coerce_tracking_frame
from replay.sim.state import (
    BALL_INDEX,
    N_PLAYERS_PER_TEAM,
    N_ROWS,
    MatchState,
    build_roster,
    empty_state,
    infer_possession,
    slot_jersey,
    slot_player_id,
    slot_team,
    state_from_tracking,
    states_to_tracking,
)

MATCH_ID = "test_match"


def _row(frame, team, jersey, x, y, vx=0.0, vy=0.0):
    player_id = "ball" if team == "ball" else f"{team}_{jersey}"
    return {
        "match_id": MATCH_ID,
        "t": frame * 0.1,
        "period": 1,
        "frame": frame,
        "team": team,
        "player_id": player_id,
        "jersey": pd.NA if team == "ball" else jersey,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "confidence": 1.0,
    }


def make_tracking(n_frames=3, drop=()):
    """Canonical tracking with jerseys 1..11 per team plus a ball."""
    rows = []
    for frame in range(n_frames):
        for team, base_y in (("home", 10.0), ("away", 40.0)):
            for jersey in range(1, N_PLAYERS_PER_TEAM + 1):
                if (frame, team, jersey) in drop:
                    continue
                rows.append(
                    _row(
                        frame,
                        team,
                        jersey,
                        x=5.0 * jersey + frame,
                        y=base_y + jersey,
                        vx=0.5 * jersey,
                        vy=-0.25 * jersey,
                    )
                )
        rows.append(_row(frame, "ball", None, x=52.5 + frame, y=34.0, vx=1.0, vy=2.0))
    return coerce_tracking_frame(pd.DataFrame(rows))


def test_slot_layout():
    assert slot_team(0) == "home"
    assert slot_team(10) == "home"
    assert slot_team(11) == "away"
    assert slot_team(21) == "away"
    assert slot_team(BALL_INDEX) == "ball"
    assert BALL_INDEX == 22
    assert N_ROWS == 23
    assert slot_jersey(0) == 1
    assert slot_jersey(11) == 1
    assert slot_jersey(21) == 11
    assert slot_player_id(6) == "home_7"
    assert slot_player_id(BALL_INDEX) == "ball"


def test_ball_row_is_last_and_addressed_by_index():
    df = make_tracking(n_frames=1)
    state = state_from_tracking(df, 0)
    assert state.pos.shape == (N_ROWS, 2)
    np.testing.assert_allclose(state.pos[BALL_INDEX], [52.5, 34.0])
    np.testing.assert_allclose(state.ball, state.pos[BALL_INDEX])
    assert state.home.shape == (11, 2)
    assert state.away.shape == (11, 2)
    np.testing.assert_allclose(state.home[0], [5.0, 11.0])
    np.testing.assert_allclose(state.away[0], [5.0, 41.0])


def test_players_are_ordered_by_jersey_regardless_of_row_order():
    df = make_tracking(n_frames=1)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    ordered = state_from_tracking(df, 0)
    from_shuffled = state_from_tracking(shuffled, 0)
    np.testing.assert_allclose(ordered.pos, from_shuffled.pos)
    np.testing.assert_allclose(ordered.vel, from_shuffled.vel)


def test_missing_player_leaves_nan_slot_without_shifting_others():
    # Home #4 is absent on frame 0 only.
    df = make_tracking(n_frames=2, drop=((0, "home", 4),))
    partial = state_from_tracking(df, 0)
    full = state_from_tracking(df, 1)

    assert np.all(np.isnan(partial.pos[3]))
    for slot in range(N_PLAYERS_PER_TEAM):
        if slot == 3:
            continue
        # x drifts by +1 per frame; identity of every other slot is unchanged.
        np.testing.assert_allclose(partial.pos[slot][0] + 1.0, full.pos[slot][0])
        np.testing.assert_allclose(partial.pos[slot][1], full.pos[slot][1])
    assert not np.any(np.isnan(partial.pos[N_PLAYERS_PER_TEAM:]))


def test_missing_ball_is_nan_and_possession_loose():
    df = make_tracking(n_frames=1)
    df = df[df["team"] != "ball"].reset_index(drop=True)
    state = state_from_tracking(df, 0)
    assert np.all(np.isnan(state.pos[BALL_INDEX]))
    assert state.possession == -1
    assert state.ball_carrier_team is None


def test_extra_players_are_dropped_not_overflowed(caplog):
    df = make_tracking(n_frames=1)
    extra = pd.DataFrame([_row(0, "home", 99, x=1.0, y=1.0)])
    df = coerce_tracking_frame(pd.concat([df, extra], ignore_index=True))
    state = state_from_tracking(df, 0)
    # Jersey 99 sorts last, so it is the one dropped and slot 0..10 are intact.
    np.testing.assert_allclose(state.home[0], [5.0, 11.0])
    assert not np.any(np.isnan(state.home))


def test_round_trip_preserves_positions_and_ids():
    df = make_tracking(n_frames=3)
    states = [state_from_tracking(df, f) for f in (0, 1, 2)]
    out = states_to_tracking(states, match_id=MATCH_ID)

    assert list(out.columns) == list(TRACKING_COLUMNS)
    assert len(out) == len(df)

    key = ["frame", "team", "player_id"]
    left = df.sort_values(key).reset_index(drop=True)
    right = out.sort_values(key).reset_index(drop=True)
    for col in ("x", "y", "vx", "vy"):
        np.testing.assert_allclose(left[col].to_numpy(), right[col].to_numpy())
    assert left["player_id"].tolist() == right["player_id"].tolist()
    assert left["team"].tolist() == right["team"].tolist()
    assert left["jersey"].tolist() == right["jersey"].tolist()
    assert left["frame"].tolist() == right["frame"].tolist()
    np.testing.assert_allclose(left["t"].to_numpy(), right["t"].to_numpy())


def test_round_trip_omits_missing_players():
    # Away #2 is absent on frame 0 but present on frame 1, so it holds a slot.
    df = make_tracking(n_frames=2, drop=((0, "away", 2),))
    out = states_to_tracking([state_from_tracking(df, 0)], match_id=MATCH_ID)
    assert "away_2" not in set(out["player_id"])
    assert len(out) == N_ROWS - 1
    assert not out[["x", "y"]].isna().to_numpy().any()


def test_build_roster_orders_by_jersey_over_the_whole_frame():
    df = make_tracking(n_frames=2, drop=((0, "home", 4),))
    roster = build_roster(df)
    assert roster["home"] == [f"home_{j}" for j in range(1, N_PLAYERS_PER_TEAM + 1)]
    assert roster["away"] == [f"away_{j}" for j in range(1, N_PLAYERS_PER_TEAM + 1)]
    assert "ball" not in roster


def test_explicit_roster_is_reused():
    df = make_tracking(n_frames=1)
    roster = build_roster(df)
    np.testing.assert_allclose(
        state_from_tracking(df, 0, roster=roster).pos,
        state_from_tracking(df, 0).pos,
    )


def test_player_outside_the_roster_is_skipped_not_placed():
    df = make_tracking(n_frames=1)
    roster = build_roster(df)
    # A roster listing a player who never appears, and omitting one who does.
    roster["home"] = ["home_ghost", *roster["home"][1:]]
    state = state_from_tracking(df, 0, roster=roster)
    assert np.all(np.isnan(state.home[0]))
    # home_2 keeps slot 1 rather than sliding into the empty slot 0.
    np.testing.assert_allclose(state.home[1], [10.0, 12.0])


def test_copy_does_not_alias():
    state = state_from_tracking(make_tracking(n_frames=1), 0)
    clone = state.copy()
    clone.pos[0, 0] = -999.0
    clone.vel[BALL_INDEX, 1] = -999.0
    clone.possession = 5
    clone.t = 99.0
    assert state.pos[0, 0] != -999.0
    assert state.vel[BALL_INDEX, 1] != -999.0
    assert state.possession != 5
    assert state.t != 99.0
    assert clone.pos is not state.pos
    assert clone.vel is not state.vel


def test_home_and_away_are_views_of_pos():
    state = empty_state()
    state.home[0] = (1.0, 2.0)
    state.away[10] = (3.0, 4.0)
    np.testing.assert_allclose(state.pos[0], [1.0, 2.0])
    np.testing.assert_allclose(state.pos[21], [3.0, 4.0])


def test_infer_possession_picks_nearest_player_within_radius():
    pos = np.full((N_ROWS, 2), 1000.0)
    pos[BALL_INDEX] = (50.0, 30.0)
    pos[4] = (50.5, 30.0)
    pos[15] = (52.0, 30.0)
    idx, team = infer_possession(pos)
    assert idx == 4
    assert team == "home"

    pos[4] = (60.0, 30.0)
    idx, team = infer_possession(pos)
    assert idx == 15
    assert team == "away"

    pos[15] = (80.0, 30.0)
    assert infer_possession(pos) == (-1, None)


def test_possession_from_tracking():
    df = make_tracking(n_frames=1)
    rows = df.to_dict("records")
    rows.append(_row(0, "away", 12, x=52.5, y=34.4))
    df2 = coerce_tracking_frame(pd.DataFrame(rows))
    # Away jersey 12 sorts last and is dropped, so nobody is on the ball.
    assert state_from_tracking(df2, 0).possession == -1


def test_unknown_frame_raises():
    df = make_tracking(n_frames=1)
    with pytest.raises(ValueError, match="frame 9"):
        state_from_tracking(df, 9)


def test_match_id_filter():
    df = make_tracking(n_frames=1)
    with pytest.raises(ValueError, match="other"):
        state_from_tracking(df, 0, match_id="other")
    state = state_from_tracking(df, 0, match_id=MATCH_ID)
    np.testing.assert_allclose(state.ball, [52.5, 34.0])


def test_bad_shapes_rejected():
    with pytest.raises(ValueError, match="pos must have shape"):
        MatchState(t=0.0, pos=np.zeros((22, 2)), vel=np.zeros((23, 2)))
    with pytest.raises(ValueError, match="vel must have shape"):
        MatchState(t=0.0, pos=np.zeros((23, 2)), vel=np.zeros((23, 3)))
    with pytest.raises(ValueError, match="possession"):
        MatchState(t=0.0, pos=np.zeros((23, 2)), vel=np.zeros((23, 2)), possession=99)


def test_states_to_tracking_empty():
    out = states_to_tracking([], match_id=MATCH_ID)
    assert list(out.columns) == list(TRACKING_COLUMNS)
    assert len(out) == 0
