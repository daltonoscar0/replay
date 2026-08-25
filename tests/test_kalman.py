"""Tests for the constant-velocity Kalman/RTS smoother."""

from __future__ import annotations

import numpy as np
import pytest

from replay.track.kalman import smooth_track

TRUE_START = np.array([12.0, 40.0])
TRUE_VEL = np.array([3.0, -1.25])


def _truth(t: np.ndarray) -> np.ndarray:
    return TRUE_START[None, :] + TRUE_VEL[None, :] * t[:, None]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return float(np.sqrt(np.mean(np.sum((a[mask] - b[mask]) ** 2, axis=1))))


def test_smoothing_beats_raw_measurements():
    rng = np.random.default_rng(42)
    t = np.arange(80) * 0.1
    truth = _truth(t)
    noisy = truth + rng.normal(0.0, 0.3, size=truth.shape)

    smoothed, _ = smooth_track(t, noisy, process_var=0.5, measurement_var=0.09)

    assert smoothed.shape == truth.shape
    assert _rmse(smoothed, truth) < _rmse(noisy, truth)
    # A useful smoother should not merely be a hair better than the raw input.
    assert _rmse(smoothed, truth) < 0.5 * _rmse(noisy, truth)


def test_recovered_velocity_matches_truth():
    rng = np.random.default_rng(7)
    t = np.arange(120) * 0.1
    noisy = _truth(t) + rng.normal(0.0, 0.25, size=(120, 2))

    _, velocity = smooth_track(t, noisy, process_var=0.5, measurement_var=0.0625)

    assert velocity.shape == (120, 2)
    assert np.allclose(velocity.mean(axis=0), TRUE_VEL, atol=0.15)
    interior = velocity[10:-10]
    assert np.max(np.abs(interior - TRUE_VEL[None, :])) < 1.0


def test_nan_gap_is_filled_and_finite():
    rng = np.random.default_rng(3)
    t = np.arange(60) * 0.1
    truth = _truth(t)
    noisy = truth + rng.normal(0.0, 0.1, size=truth.shape)
    gap = slice(24, 34)
    noisy[gap] = np.nan

    smoothed, velocity = smooth_track(t, noisy, process_var=0.5, measurement_var=0.01)

    assert np.isfinite(smoothed).all()
    assert np.isfinite(velocity).all()
    # Across a gap in an otherwise straight track the fill should be close to
    # the straight-line answer, which is the entire reason for predicting
    # through NaNs instead of dropping the rows.
    assert np.max(np.abs(smoothed[gap] - truth[gap])) < 0.4
    assert np.max(np.abs(velocity[gap] - TRUE_VEL[None, :])) < 0.5


def test_leading_and_trailing_nans_are_filled():
    t = np.arange(30) * 0.2
    xy = _truth(t)
    xy[:3] = np.nan
    xy[-4:] = np.nan

    smoothed, velocity = smooth_track(t, xy, process_var=0.1, measurement_var=0.01)

    assert np.isfinite(smoothed).all()
    assert np.allclose(smoothed, _truth(t), atol=1e-3)
    assert np.allclose(velocity, TRUE_VEL[None, :], atol=1e-3)


def test_irregular_timestamps_match_regular_sampling():
    rng = np.random.default_rng(11)
    n = 100
    regular_t = np.arange(n) * 0.1
    irregular_t = np.concatenate([[0.0], np.cumsum(rng.uniform(0.04, 0.16, size=n - 1))])

    regular_xy = _truth(regular_t) + rng.normal(0.0, 0.2, size=(n, 2))
    irregular_xy = _truth(irregular_t) + rng.normal(0.0, 0.2, size=(n, 2))

    kw = {"process_var": 0.5, "measurement_var": 0.04}
    _, reg_vel = smooth_track(regular_t, regular_xy, **kw)
    irr_sm, irr_vel = smooth_track(irregular_t, irregular_xy, **kw)

    assert _rmse(irr_sm, _truth(irregular_t)) < _rmse(irregular_xy, _truth(irregular_t))
    assert np.allclose(irr_vel.mean(axis=0), TRUE_VEL, atol=0.2)
    # The two samplings describe the same motion, so the recovered velocities
    # must agree with each other, not just with the truth.
    assert np.allclose(irr_vel.mean(axis=0), reg_vel.mean(axis=0), atol=0.25)


def test_irregular_timestamps_are_not_treated_as_uniform():
    # A trajectory sampled with a long pause in the middle. If dt were assumed
    # constant the velocity would be badly wrong across the pause.
    t = np.array([0.0, 0.1, 0.2, 0.3, 2.3, 2.4, 2.5, 2.6])
    xy = _truth(t)

    smoothed, velocity = smooth_track(t, xy, process_var=0.1, measurement_var=0.01)

    assert np.allclose(smoothed, xy, atol=1e-3)
    assert np.allclose(velocity, TRUE_VEL[None, :], atol=1e-2)


def test_noise_free_input_is_essentially_unchanged():
    t = np.arange(50) * 0.1
    xy = _truth(t)

    smoothed, velocity = smooth_track(t, xy)

    assert np.allclose(smoothed, xy, atol=1e-6)
    assert np.allclose(velocity, TRUE_VEL[None, :], atol=1e-6)


def test_degenerate_inputs():
    empty_xy, empty_vel = smooth_track(np.zeros(0), np.zeros((0, 2)))
    assert empty_xy.shape == (0, 2)
    assert empty_vel.shape == (0, 2)

    one_xy, one_vel = smooth_track(np.array([0.0]), np.array([[5.0, 6.0]]))
    assert np.allclose(one_xy, [[5.0, 6.0]])
    assert np.allclose(one_vel, [[0.0, 0.0]])

    with pytest.raises(ValueError, match="finite"):
        smooth_track(np.arange(4.0), np.full((4, 2), np.nan))

    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        smooth_track(np.arange(4.0), np.zeros((4, 3)))

    with pytest.raises(ValueError, match="rows"):
        smooth_track(np.arange(5.0), np.zeros((4, 2)))


def test_repeated_timestamps_do_not_blow_up():
    t = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
    xy = _truth(t)

    smoothed, velocity = smooth_track(t, xy, process_var=0.1, measurement_var=0.01)

    assert np.isfinite(smoothed).all()
    assert np.isfinite(velocity).all()
