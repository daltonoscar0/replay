"""Tests for the image -> pitch homography.

Most of these build a real pinhole camera looking at the pitch, project the
named landmarks through it, and check the fit recovers the camera. That gives
exact ground truth, which is what makes sub-centimeter assertions meaningful.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from replay.schema import PITCH_LENGTH, PITCH_WIDTH
from replay.track.homography import (
    CENTER_CIRCLE_RADIUS,
    PENALTY_BOX_DEPTH,
    PENALTY_BOX_WIDTH,
    PENALTY_SPOT_DISTANCE,
    PITCH_KEYPOINTS,
    SIX_YARD_BOX_DEPTH,
    SIX_YARD_BOX_WIDTH,
    SOCCERNET_LINE_INTERSECTIONS,
    SOCCERNET_POINT_NAMES,
    Homography,
    estimate_from_lines,
    estimate_from_soccernet,
    pitch_marking_samples,
    soccernet_correspondences,
)

# --------------------------------------------------------------------------
# ground-truth camera
# --------------------------------------------------------------------------

FRAME_W = 1280
FRAME_H = 720


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward], axis=0)
    return rot, -rot @ eye


def pitch_to_image_matrix(
    eye=(52.5, -70.0, 40.0), target=(52.5, 34.0, 0.0), f=900.0, cx=FRAME_W / 2.0, cy=FRAME_H / 2.0
):
    """Homography taking pitch meters to image pixels for a pinhole camera.

    The pitch is the plane Z = 0, so the 3x4 projection collapses to the first
    two rotation columns plus the translation.
    """
    rot, t = look_at(eye, target)
    k = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return k @ np.column_stack((rot[:, 0], rot[:, 1], t))


def project(mat, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ mat.T
    return homo[:, :2] / homo[:, 2:3]


@pytest.fixture(scope="module")
def truth():
    mat = pitch_to_image_matrix()
    names = list(PITCH_KEYPOINTS)
    pitch = np.array([PITCH_KEYPOINTS[n] for n in names], dtype=np.float64)
    image = project(mat, pitch)
    h_true = np.linalg.inv(mat)
    return names, pitch, image, h_true / h_true[2, 2], mat


def render_pitch(mat, width=FRAME_W, height=FRAME_H, seed=3):
    """Draw the pitch markings as white lines on grass through ``mat``."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), (58, 132, 62), dtype=np.int16)
    img = np.clip(img + rng.normal(0.0, 4.0, img.shape), 0, 255).astype(np.uint8)

    def seg(p, q, thick=3):
        a, b = project(mat, [p, q])
        cv2.line(
            img,
            tuple(np.round(a).astype(int)),
            tuple(np.round(b).astype(int)),
            (250, 250, 250),
            thick,
            cv2.LINE_AA,
        )

    def poly(pts, thick=3):
        for i in range(len(pts) - 1):
            seg(pts[i], pts[i + 1], thick)

    lo_y, hi_y = 34.0 - PENALTY_BOX_WIDTH / 2, 34.0 + PENALTY_BOX_WIDTH / 2
    lo_s, hi_s = 34.0 - SIX_YARD_BOX_WIDTH / 2, 34.0 + SIX_YARD_BOX_WIDTH / 2
    poly([(0, 0), (PITCH_LENGTH, 0), (PITCH_LENGTH, PITCH_WIDTH), (0, PITCH_WIDTH), (0, 0)])
    seg((52.5, 0), (52.5, PITCH_WIDTH))
    for near, sign in ((0.0, 1.0), (PITCH_LENGTH, -1.0)):
        poly(
            [
                (near, lo_y),
                (near + sign * PENALTY_BOX_DEPTH, lo_y),
                (near + sign * PENALTY_BOX_DEPTH, hi_y),
                (near, hi_y),
            ]
        )
        poly(
            [
                (near, lo_s),
                (near + sign * SIX_YARD_BOX_DEPTH, lo_s),
                (near + sign * SIX_YARD_BOX_DEPTH, hi_s),
                (near, hi_s),
            ]
        )
    ang = np.linspace(0.0, 2 * np.pi, 80)
    poly(
        [
            (52.5 + CENTER_CIRCLE_RADIUS * np.cos(a), 34.0 + CENTER_CIRCLE_RADIUS * np.sin(a))
            for a in ang
        ]
    )
    return img


# --------------------------------------------------------------------------
# PITCH_KEYPOINTS
# --------------------------------------------------------------------------


def test_keypoints_use_real_dimensions():
    kp = PITCH_KEYPOINTS
    assert kp["corner_left_bottom"] == (0.0, 0.0)
    assert kp["corner_right_top"] == (105.0, 68.0)
    assert kp["center_spot"] == (52.5, 34.0)
    assert kp["penalty_spot_left"] == (11.0, 34.0)
    assert kp["penalty_spot_right"] == (94.0, 34.0)
    assert kp["halfway_bottom"] == (52.5, 0.0)

    # Penalty box: 16.5 m deep, 40.32 m wide, centred on the goal.
    assert kp["penalty_box_left_front_bottom"][0] == pytest.approx(16.5)
    assert kp["penalty_box_right_front_bottom"][0] == pytest.approx(88.5)
    width = kp["penalty_box_left_goal_top"][1] - kp["penalty_box_left_goal_bottom"][1]
    assert width == pytest.approx(40.32)

    # Six-yard box: 5.5 m deep, 18.32 m wide.
    assert kp["six_yard_left_front_top"][0] == pytest.approx(5.5)
    assert kp["six_yard_right_front_top"][0] == pytest.approx(99.5)
    width6 = kp["six_yard_left_goal_top"][1] - kp["six_yard_left_goal_bottom"][1]
    assert width6 == pytest.approx(18.32)

    assert kp["center_circle_top"][1] - kp["center_spot"][1] == pytest.approx(9.15)
    assert PENALTY_SPOT_DISTANCE == 11.0


def test_keypoints_are_on_the_pitch_and_distinct():
    pts = list(PITCH_KEYPOINTS.values())
    assert len(set(pts)) == len(pts)
    for name, (x, y) in PITCH_KEYPOINTS.items():
        assert 0.0 <= x <= PITCH_LENGTH, name
        assert 0.0 <= y <= PITCH_WIDTH, name
    # 4 corners + 2 halfway + centre spot + 2 circle + 2 spots + 8 + 8 box corners
    assert len(PITCH_KEYPOINTS) == 27


# --------------------------------------------------------------------------
# from_point_pairs: the DLT itself
# --------------------------------------------------------------------------


def test_recovers_known_homography_to_sub_centimeter(truth):
    _, pitch, image, h_true, _ = truth
    fit = Homography.from_point_pairs(image, pitch)

    recovered = fit.image_to_pitch(image)
    assert np.abs(recovered - pitch).max() < 1e-4  # well under a millimetre
    assert fit.reprojection_error < 1e-6
    assert np.abs(fit.H / fit.H[2, 2] - h_true).max() < 1e-9


def test_recovers_from_the_minimal_four_correspondences(truth):
    names, pitch, image, _, _ = truth
    corners = [
        "corner_left_bottom",
        "corner_left_top",
        "corner_right_bottom",
        "corner_right_top",
    ]
    idx = [names.index(n) for n in corners]
    fit = Homography.from_point_pairs(image[idx], pitch[idx])
    assert fit.reprojection_error < 1e-9

    # A point that took no part in the fit still lands where it should.
    held_out = names.index("penalty_spot_right")
    got = fit.image_to_pitch(image[held_out])
    assert np.allclose(got, pitch[held_out], atol=1e-6)


def test_normalization_survives_a_huge_coordinate_offset(truth):
    """A raw DLT on coordinates around 1e6 loses all precision; Hartley's does not."""
    _, pitch, image, _, _ = truth
    shifted = image + 1.0e6
    fit = Homography.from_point_pairs(shifted, pitch)
    assert fit.reprojection_error < 1e-3
    assert np.abs(fit.image_to_pitch(shifted) - pitch).max() < 1e-2


def test_fit_is_invariant_to_correspondence_order(truth):
    _, pitch, image, _, _ = truth
    order = np.arange(len(pitch))[::-1]
    a = Homography.from_point_pairs(image, pitch)
    b = Homography.from_point_pairs(image[order], pitch[order])
    assert np.abs(a.H / a.H[2, 2] - b.H / b.H[2, 2]).max() < 1e-9


def test_reprojection_error_is_nan_when_not_fitted(truth):
    _, _, _, h_true, _ = truth
    assert np.isnan(Homography(H=h_true).reprojection_error)


def test_reprojection_error_is_in_meters(truth):
    """A known 1 m offset on one landmark shows up as ~1/N meters of residual."""
    _, pitch, image, _, _ = truth
    n = len(pitch)
    perturbed = pitch.copy()
    perturbed[0] += np.array([1.0, 0.0])
    fit = Homography.from_point_pairs(image, perturbed)
    # The fit absorbs part of the offset, so the mean residual sits between the
    # trivial lower bound and the full displacement.
    assert 0.2 / n < fit.reprojection_error < 1.0


# --------------------------------------------------------------------------
# from_point_pairs: rejection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_raises_on_fewer_than_four_correspondences(truth, n):
    _, pitch, image, _, _ = truth
    with pytest.raises(ValueError, match="at least 4"):
        Homography.from_point_pairs(image[:n], pitch[:n])


def test_raises_on_length_mismatch(truth):
    _, pitch, image, _, _ = truth
    with pytest.raises(ValueError, match="same length"):
        Homography.from_point_pairs(image[:6], pitch[:5])


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((5, 3)),
        np.zeros(8),
        np.zeros((2, 2, 2)),
    ],
)
def test_raises_on_wrong_shape(truth, bad):
    _, pitch, _, _, _ = truth
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        Homography.from_point_pairs(bad, pitch[: len(bad)])


def test_raises_on_non_finite_input(truth):
    _, pitch, image, _, _ = truth
    bad = image.copy()
    bad[2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Homography.from_point_pairs(bad, pitch)


def test_raises_when_all_image_points_are_collinear(truth):
    _, pitch, _, _, _ = truth
    collinear = np.stack([np.arange(6.0) * 10.0, np.arange(6.0) * 10.0 + 5.0], axis=1)
    with pytest.raises(ValueError, match="degenerate"):
        Homography.from_point_pairs(collinear, pitch[:6])


def test_raises_when_three_of_four_points_are_collinear():
    image = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [50.0, 0.0]])
    pitch = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [0.0, 10.0]])
    with pytest.raises(ValueError, match="degenerate"):
        Homography.from_point_pairs(image, pitch)


def test_raises_when_all_points_coincide():
    image = np.tile([100.0, 200.0], (5, 1))
    pitch = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="degenerate"):
        Homography.from_point_pairs(image, pitch)


def test_raises_on_repeated_image_points_mapped_to_different_pitch_points(truth):
    _, pitch, image, _, _ = truth
    dup = image[:4].copy()
    dup[1] = dup[0]
    dup[2] = dup[0]
    with pytest.raises(ValueError, match="degenerate"):
        Homography.from_point_pairs(dup, pitch[:4])


def test_constructor_rejects_a_singular_matrix():
    with pytest.raises(ValueError, match="singular"):
        Homography(H=np.ones((3, 3)))


def test_constructor_rejects_wrong_shape_and_nan():
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        Homography(H=np.eye(4))
    bad = np.eye(3)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Homography(H=bad)


# --------------------------------------------------------------------------
# mapping both ways
# --------------------------------------------------------------------------


def test_image_to_pitch_and_pitch_to_image_are_inverses(truth):
    _, pitch, image, _, _ = truth
    fit = Homography.from_point_pairs(image, pitch)

    rng = np.random.default_rng(11)
    uv = np.stack(
        [rng.uniform(0.0, FRAME_W, 400), rng.uniform(FRAME_H * 0.55, FRAME_H, 400)], axis=1
    )
    assert np.allclose(fit.pitch_to_image(fit.image_to_pitch(uv)), uv, atol=1e-6)

    xy = np.stack([rng.uniform(0.0, PITCH_LENGTH, 400), rng.uniform(0.0, PITCH_WIDTH, 400)], axis=1)
    assert np.allclose(fit.image_to_pitch(fit.pitch_to_image(xy)), xy, atol=1e-9)


def test_accepts_a_single_point_and_returns_a_single_point(truth):
    _, pitch, image, _, _ = truth
    fit = Homography.from_point_pairs(image, pitch)
    one = fit.image_to_pitch(image[0])
    assert one.shape == (2,)
    assert np.allclose(one, pitch[0], atol=1e-6)
    assert fit.pitch_to_image(pitch[0]).shape == (2,)


def test_horizon_points_map_to_nan_not_infinity(truth):
    _, pitch, image, _, _ = truth
    fit = Homography.from_point_pairs(image, pitch)
    # Row 2 of H_inv dotted with the point is zero exactly on the horizon line.
    a, b, c = fit.H[2]
    horizon = np.array([[0.0, -c / b], [100.0, -(a * 100.0 + c) / b]])
    out = fit.image_to_pitch(horizon)
    assert np.isnan(out).all()


def test_empty_input_gives_empty_output(truth):
    _, pitch, image, _, _ = truth
    fit = Homography.from_point_pairs(image, pitch)
    assert fit.image_to_pitch(np.zeros((0, 2))).shape == (0, 2)


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def test_json_round_trip(tmp_path, truth):
    _, pitch, image, _, _ = truth
    fit = Homography.from_point_pairs(image, pitch)
    path = fit.to_json(tmp_path / "nested" / "H.json")
    assert path.exists()

    back = Homography.from_json(path)
    assert np.array_equal(back.H, fit.H)
    assert back.reprojection_error == pytest.approx(fit.reprojection_error)
    assert np.allclose(back.image_to_pitch(image), pitch, atol=1e-6)


def test_json_round_trip_preserves_unknown_error(tmp_path, truth):
    _, _, _, h_true, _ = truth
    path = Homography(H=h_true).to_json(tmp_path / "H.json")
    assert "NaN" not in path.read_text(encoding="utf-8")
    assert np.isnan(Homography.from_json(path).reprojection_error)


def test_from_json_rejects_a_file_without_H(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"units": "meters"}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain"):
        Homography.from_json(path)


# --------------------------------------------------------------------------
# noise behaviour
# --------------------------------------------------------------------------


def test_noise_degrades_error_monotonically_and_boundedly(truth):
    """Pixel noise must show up as proportionate meters, not as a blow-up."""
    _, pitch, image, _, _ = truth
    sigmas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    rng = np.random.default_rng(20250825)
    means = []
    worst = 0.0
    for sigma in sigmas:
        errs = []
        for _ in range(30):
            noisy = image + rng.normal(0.0, sigma, image.shape)
            fit = Homography.from_point_pairs(noisy, pitch)
            assert np.isfinite(fit.reprojection_error)
            errs.append(fit.reprojection_error)
            # The whole pitch stays somewhere near the whole pitch.
            mapped = fit.image_to_pitch(image)
            worst = max(worst, float(np.abs(mapped - pitch).max()))
        means.append(float(np.mean(errs)))

    assert means[0] < 1e-9
    assert all(a < b for a, b in zip(means[:-1], means[1:], strict=True))
    # Roughly linear in sigma: this camera is about 0.15 m per pixel at worst.
    for sigma, mean in zip(sigmas, means, strict=True):
        assert mean <= 0.4 * sigma + 0.02, (sigma, mean)
    assert worst < 10.0


def test_a_single_gross_outlier_does_not_destroy_the_fit(truth):
    """Least squares is not robust, but one bad point must stay a local problem."""
    _, pitch, image, _, _ = truth
    corrupted = image.copy()
    corrupted[5] += np.array([120.0, 90.0])
    fit = Homography.from_point_pairs(corrupted, pitch)
    mapped = fit.image_to_pitch(image)
    good = np.delete(np.arange(len(pitch)), 5)
    median_err = float(np.median(np.linalg.norm(mapped[good] - pitch[good], axis=1)))
    assert median_err < 5.0


# --------------------------------------------------------------------------
# estimate_from_lines
# --------------------------------------------------------------------------


def test_pitch_marking_samples_lie_on_the_pitch():
    samples = pitch_marking_samples()
    assert samples.ndim == 2 and samples.shape[1] == 2
    assert samples.shape[0] > 500
    assert samples[:, 0].min() >= -1e-9
    assert samples[:, 0].max() <= PITCH_LENGTH + 1e-9
    assert samples[:, 1].min() >= -1e-9
    assert samples[:, 1].max() <= PITCH_WIDTH + 1e-9


def test_estimate_from_lines_recovers_a_rendered_pitch():
    mat = pitch_to_image_matrix()
    frame = render_pitch(mat)
    fit = estimate_from_lines(frame)
    assert fit is not None
    assert fit.reprojection_error < 2.0

    for name in ("center_spot", "corner_left_bottom", "corner_right_top", "penalty_spot_left"):
        uv = project(mat, [PITCH_KEYPOINTS[name]])[0]
        got = fit.image_to_pitch(uv)
        assert np.linalg.norm(got - np.array(PITCH_KEYPOINTS[name])) < 2.0, name


def test_estimate_from_lines_is_deterministic():
    frame = render_pitch(pitch_to_image_matrix())
    a = estimate_from_lines(frame)
    b = estimate_from_lines(frame)
    assert a is not None and b is not None
    assert np.array_equal(a.H, b.H)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320, 3), (60, 140, 60), dtype=np.uint8),
        np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8),
    ],
    ids=["black", "bare-grass", "noise"],
)
def test_estimate_from_lines_returns_none_without_structure(frame):
    assert estimate_from_lines(frame) is None


def test_estimate_from_lines_returns_none_when_only_one_line_is_visible():
    """A lone halfway line cannot pin down eight degrees of freedom."""
    img = np.full((FRAME_H, FRAME_W, 3), (58, 132, 62), dtype=np.uint8)
    cv2.line(img, (300, 700), (700, 60), (250, 250, 250), 4)
    assert estimate_from_lines(img) is None


def test_estimate_from_lines_rejects_a_non_bgr_frame():
    with pytest.raises(ValueError, match="HxWx3"):
        estimate_from_lines(np.zeros((240, 320), dtype=np.uint8))


# --------------------------------------------------------------------------
# SoccerNet hook
# --------------------------------------------------------------------------

# Pitch geometry of each SoccerNet line, in our coordinates. This is the
# convention SOCCERNET_LINE_INTERSECTIONS is written against.
SOCCERNET_LINE_GEOMETRY = {
    "Side line top": ((0.0, 0.0), (105.0, 0.0)),
    "Side line bottom": ((0.0, 68.0), (105.0, 68.0)),
    "Side line left": ((0.0, 0.0), (0.0, 68.0)),
    "Side line right": ((105.0, 0.0), (105.0, 68.0)),
    "Middle line": ((52.5, 0.0), (52.5, 68.0)),
    "Big rect. left main": ((16.5, 13.84), (16.5, 54.16)),
    "Big rect. left top": ((0.0, 13.84), (16.5, 13.84)),
    "Big rect. left bottom": ((0.0, 54.16), (16.5, 54.16)),
    "Big rect. right main": ((88.5, 13.84), (88.5, 54.16)),
    "Big rect. right top": ((88.5, 13.84), (105.0, 13.84)),
    "Big rect. right bottom": ((88.5, 54.16), (105.0, 54.16)),
    "Small rect. left main": ((5.5, 24.84), (5.5, 43.16)),
    "Small rect. left top": ((0.0, 24.84), (5.5, 24.84)),
    "Small rect. left bottom": ((0.0, 43.16), (5.5, 43.16)),
    "Small rect. right main": ((99.5, 24.84), (99.5, 43.16)),
    "Small rect. right top": ((99.5, 24.84), (105.0, 24.84)),
    "Small rect. right bottom": ((99.5, 43.16), (105.0, 43.16)),
}


def test_soccernet_mapping_targets_are_real_keypoints():
    for pair, key in SOCCERNET_LINE_INTERSECTIONS.items():
        assert key in PITCH_KEYPOINTS, pair
        for name in pair:
            assert name in SOCCERNET_LINE_GEOMETRY, name
    for key in SOCCERNET_POINT_NAMES.values():
        assert key in PITCH_KEYPOINTS, key


def test_soccernet_line_pairs_intersect_at_the_named_keypoint():
    """The mapping is only correct if the two pitch lines actually meet there."""
    for (name_a, name_b), key in SOCCERNET_LINE_INTERSECTIONS.items():
        (ax, ay), (bx, by) = SOCCERNET_LINE_GEOMETRY[name_a]
        (cx, cy), (dx, dy) = SOCCERNET_LINE_GEOMETRY[name_b]
        la = np.cross([ax, ay, 1.0], [bx, by, 1.0])
        lb = np.cross([cx, cy, 1.0], [dx, dy, 1.0])
        p = np.cross(la, lb)
        assert abs(p[2]) > 1e-9, (name_a, name_b)
        assert np.allclose(p[:2] / p[2], PITCH_KEYPOINTS[key]), (name_a, name_b, key)


def test_soccernet_correspondences_recover_the_camera():
    mat = pitch_to_image_matrix()
    prediction = {
        name: [{"x": float(u), "y": float(v)} for u, v in project(mat, [ends[0], ends[1]])]
        for name, ends in SOCCERNET_LINE_GEOMETRY.items()
    }
    spot = project(mat, [PITCH_KEYPOINTS["penalty_spot_left"]])[0]
    prediction["Penalty mark left"] = {"x": float(spot[0]), "y": float(spot[1])}

    image_pts, pitch_pts, names = soccernet_correspondences(prediction)
    assert len(names) == len(SOCCERNET_LINE_INTERSECTIONS) + 1
    assert len(set(names)) == len(names)

    fit = Homography.from_point_pairs(image_pts, pitch_pts)
    assert fit.reprojection_error < 1e-6
    for key in ("center_spot", "corner_right_top"):
        uv = project(mat, [PITCH_KEYPOINTS[key]])[0]
        assert np.allclose(fit.image_to_pitch(uv), PITCH_KEYPOINTS[key], atol=1e-4)


def test_soccernet_correspondences_ignores_unusable_entries():
    image_pts, pitch_pts, names = soccernet_correspondences(
        {"Side line top": "nonsense", "Unknown thing": [[1, 2], [3, 4]]}
    )
    assert names == []
    assert image_pts.shape == (0, 2)
    assert pitch_pts.shape == (0, 2)


def test_estimate_from_soccernet_returns_none_and_says_why(caplog):
    frame = render_pitch(pitch_to_image_matrix())
    with caplog.at_level(logging.INFO, logger="replay.track.homography"):
        assert estimate_from_soccernet(frame) is None
    assert any("SoccerNet calibration unavailable" in r.message for r in caplog.records)


def test_estimate_from_soccernet_rejects_a_non_bgr_frame():
    with pytest.raises(ValueError, match="HxWx3"):
        estimate_from_soccernet(np.zeros((10, 10), dtype=np.uint8))
