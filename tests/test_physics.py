import numpy as np
import pytest

from replay.schema import PITCH_LENGTH, PITCH_WIDTH
from replay.sim.physics import (
    BALL_DRAG,
    DT,
    MAX_ACCEL,
    MAX_SPEED,
    clamp_to_pitch,
    clip_norm,
    step_ball,
    step_players,
)

TOL = 1e-9


def test_accel_cap_holds_by_magnitude():
    pos = np.zeros((1, 2))
    vel = np.zeros((1, 2))
    desired = np.array([[100.0, 0.0]])
    _, new_vel = step_players(pos, vel, desired)
    assert np.linalg.norm(new_vel[0]) <= MAX_ACCEL * DT + TOL
    np.testing.assert_allclose(np.linalg.norm(new_vel[0]), MAX_ACCEL * DT)


def test_accel_cap_on_45_degree_request():
    # A per-component clip would give each axis MAX_ACCEL and a norm of
    # MAX_ACCEL * sqrt(2). It must not.
    pos = np.zeros((1, 2))
    vel = np.zeros((1, 2))
    desired = np.array([[100.0, 100.0]])
    _, new_vel = step_players(pos, vel, desired)
    speed = float(np.linalg.norm(new_vel[0]))
    assert speed <= MAX_ACCEL * DT + TOL
    assert speed < MAX_ACCEL * DT * np.sqrt(2) - TOL
    # Direction is preserved: still exactly 45 degrees.
    np.testing.assert_allclose(new_vel[0, 0], new_vel[0, 1])


def test_speed_cap_on_45_degree_motion():
    pos = np.zeros((1, 2))
    diag = MAX_SPEED / np.sqrt(2.0)
    vel = np.array([[diag, diag]])
    desired = np.array([[1000.0, 1000.0]])
    new_pos, new_vel = step_players(pos, vel, desired)
    speed = float(np.linalg.norm(new_vel[0]))
    assert speed <= MAX_SPEED + TOL
    np.testing.assert_allclose(speed, MAX_SPEED)
    # Neither component may reach MAX_SPEED on a diagonal.
    assert np.all(np.abs(new_vel[0]) < MAX_SPEED)
    assert np.linalg.norm(new_pos[0] - pos[0]) <= MAX_SPEED * DT + TOL


def test_player_at_max_speed_asked_to_go_faster_does_not():
    pos = np.zeros((1, 2))
    vel = np.array([[MAX_SPEED, 0.0]])
    desired = np.array([[MAX_SPEED * 5, 0.0]])
    speeds = []
    for _ in range(20):
        pos, vel = step_players(pos, vel, desired)
        speeds.append(float(np.linalg.norm(vel[0])))
    assert max(speeds) <= MAX_SPEED + TOL
    np.testing.assert_allclose(speeds[-1], MAX_SPEED)


def test_speed_cap_holds_over_many_ticks_from_rest_in_every_direction():
    angles = np.linspace(0.0, 2 * np.pi, 17)
    for angle in angles:
        pos = np.zeros((1, 2))
        vel = np.zeros((1, 2))
        desired = np.array([[1000.0 * np.cos(angle), 1000.0 * np.sin(angle)]])
        for _ in range(100):
            pos, vel = step_players(pos, vel, desired)
            assert np.linalg.norm(vel[0]) <= MAX_SPEED + TOL
        np.testing.assert_allclose(np.linalg.norm(vel[0]), MAX_SPEED)


def test_deceleration_is_also_capped():
    pos = np.zeros((1, 2))
    vel = np.array([[MAX_SPEED, 0.0]])
    desired = np.zeros((1, 2))
    _, new_vel = step_players(pos, vel, desired)
    delta = np.linalg.norm(new_vel[0] - vel[0])
    assert delta <= MAX_ACCEL * DT + TOL
    np.testing.assert_allclose(delta, MAX_ACCEL * DT)


def test_desired_velocity_reached_when_within_the_cap():
    pos = np.zeros((1, 2))
    vel = np.zeros((1, 2))
    desired = np.array([[0.2, -0.1]])  # needs 2.24 m/s^2, under the cap
    new_pos, new_vel = step_players(pos, vel, desired)
    np.testing.assert_allclose(new_vel, desired)
    np.testing.assert_allclose(new_pos, desired * DT)


def test_step_players_handles_nan_rows_without_touching_others():
    pos = np.array([[0.0, 0.0], [np.nan, np.nan]])
    vel = np.zeros((2, 2))
    desired = np.array([[1.0, 0.0], [1.0, 0.0]])
    new_pos, _ = step_players(pos, vel, desired)
    # 1 m/s in one 0.1 s tick needs 10 m/s^2; the cap allows 6, so 0.6 m/s.
    np.testing.assert_allclose(new_pos[0], [MAX_ACCEL * DT * DT, 0.0])
    assert np.all(np.isnan(new_pos[1]))


def test_step_players_is_pure():
    pos = np.array([[1.0, 2.0], [3.0, 4.0]])
    vel = np.array([[0.5, 0.0], [0.0, -0.5]])
    desired = np.array([[9.0, 9.0], [-9.0, 9.0]])
    pos_before, vel_before, desired_before = pos.copy(), vel.copy(), desired.copy()
    new_pos, new_vel = step_players(pos, vel, desired)
    np.testing.assert_array_equal(pos, pos_before)
    np.testing.assert_array_equal(vel, vel_before)
    np.testing.assert_array_equal(desired, desired_before)
    assert new_pos is not pos
    assert new_vel is not vel


def test_step_players_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        step_players(np.zeros((2, 2)), np.zeros((3, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="dt must be positive"):
        step_players(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), dt=0.0)


def test_ball_decelerates_monotonically_and_comes_to_rest():
    pos = np.array([20.0, 30.0])
    vel = np.array([15.0, -5.0])
    speed = float(np.linalg.norm(vel))
    travelled = 0.0
    for _ in range(600):  # 60 s
        prev_pos, prev_speed = pos, speed
        pos, vel = step_ball(pos, vel)
        speed = float(np.linalg.norm(vel))
        assert speed < prev_speed
        step = float(np.linalg.norm(pos - prev_pos))
        travelled += step
        # Never reverses: still moving the same way it started.
        assert np.dot(vel, np.array([15.0, -5.0])) > 0
    assert speed < 1e-6
    # Total distance tends to v0 / drag.
    np.testing.assert_allclose(travelled, np.linalg.norm([15.0, -5.0]) / BALL_DRAG, rtol=1e-6)


def test_ball_velocity_matches_exact_exponential_decay():
    vel = np.array([[6.0, 0.0]])
    pos = np.zeros((1, 2))
    for i in range(1, 11):
        pos, vel = step_ball(pos, vel)
        expected = 6.0 * np.exp(-BALL_DRAG * DT * i)
        np.testing.assert_allclose(vel[0, 0], expected, rtol=1e-12)


def test_ball_step_is_dt_consistent():
    pos = np.array([[0.0, 0.0]])
    vel = np.array([[7.0, 3.0]])
    coarse = step_ball(pos, vel, dt=1.0)
    fine_pos, fine_vel = pos, vel
    for _ in range(10):
        fine_pos, fine_vel = step_ball(fine_pos, fine_vel, dt=0.1)
    np.testing.assert_allclose(coarse[0], fine_pos, rtol=1e-12)
    np.testing.assert_allclose(coarse[1], fine_vel, rtol=1e-12)


def test_ball_without_drag_is_straight_line():
    pos, vel = step_ball(np.array([1.0, 1.0]), np.array([2.0, 4.0]), dt=0.5, drag=0.0)
    np.testing.assert_allclose(pos, [2.0, 3.0])
    np.testing.assert_allclose(vel, [2.0, 4.0])


def test_step_ball_is_pure():
    pos = np.array([[10.0, 20.0]])
    vel = np.array([[5.0, -5.0]])
    pos_before, vel_before = pos.copy(), vel.copy()
    new_pos, new_vel = step_ball(pos, vel)
    np.testing.assert_array_equal(pos, pos_before)
    np.testing.assert_array_equal(vel, vel_before)
    assert new_pos is not pos
    assert new_vel is not vel


def test_step_ball_rejects_bad_parameters():
    with pytest.raises(ValueError, match="dt must be positive"):
        step_ball(np.zeros(2), np.zeros(2), dt=-1.0)
    with pytest.raises(ValueError, match="drag must be non-negative"):
        step_ball(np.zeros(2), np.zeros(2), drag=-0.1)


def test_clamp_to_pitch_all_four_edges():
    pos = np.array(
        [
            [-5.0, 34.0],  # past the left goal line
            [PITCH_LENGTH + 5.0, 34.0],  # past the right goal line
            [52.5, -3.0],  # past the bottom touchline
            [52.5, PITCH_WIDTH + 3.0],  # past the top touchline
            [52.5, 34.0],  # inside, untouched
        ]
    )
    out = clamp_to_pitch(pos)
    np.testing.assert_allclose(
        out,
        [
            [0.0, 34.0],
            [PITCH_LENGTH, 34.0],
            [52.5, 0.0],
            [52.5, PITCH_WIDTH],
            [52.5, 34.0],
        ],
    )


def test_clamp_to_pitch_corner_and_single_point():
    np.testing.assert_allclose(clamp_to_pitch(np.array([-1.0, -1.0])), [0.0, 0.0])
    np.testing.assert_allclose(
        clamp_to_pitch(np.array([999.0, 999.0])), [PITCH_LENGTH, PITCH_WIDTH]
    )


def test_clamp_to_pitch_preserves_nan_and_is_pure():
    pos = np.array([[-5.0, 100.0], [np.nan, np.nan]])
    before = pos.copy()
    out = clamp_to_pitch(pos)
    np.testing.assert_array_equal(pos[0], before[0])
    assert np.all(np.isnan(pos[1]))
    assert out is not pos
    np.testing.assert_allclose(out[0], [0.0, PITCH_WIDTH])
    assert np.all(np.isnan(out[1]))


def test_clip_norm_leaves_short_vectors_alone():
    v = np.array([[1.0, 1.0], [3.0, 4.0]])
    out = clip_norm(v, 5.0)
    np.testing.assert_allclose(out[0], [1.0, 1.0])
    np.testing.assert_allclose(out[1], [3.0, 4.0])
    np.testing.assert_allclose(clip_norm(np.array([[6.0, 8.0]]), 5.0), [[3.0, 4.0]])
    with pytest.raises(ValueError, match="non-negative"):
        clip_norm(v, -1.0)


def test_constants_are_plausible():
    assert MAX_SPEED == 8.0
    assert MAX_ACCEL == 6.0
    assert BALL_DRAG == 0.4
    assert DT == 0.1
