"""Image-plane to pitch-plane homography.

The whole tracking pipeline is only as good as this mapping: a detection is a
box in pixels, and everything downstream wants meters on a 105x68 pitch. The
fit is a plain DLT on Hartley-normalized coordinates, which keeps the design
matrix well conditioned and makes the residual meaningful in meters.

Two estimators sit on top of it. ``estimate_from_lines`` is the offline path:
green mask, Hough segments, and a small search over which pitch lines the
detected image lines could be. ``estimate_from_soccernet`` is the hook for the
SoccerNet calibration model, which is far more reliable but needs weights that
are not shipped here.
"""

from __future__ import annotations

import functools
import itertools
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from replay.schema import PITCH_LENGTH, PITCH_WIDTH

logger = logging.getLogger(__name__)

# Real pitch markings, in meters, per IFAB Law 1 with the common
# 105 x 68 professional pitch.
PENALTY_BOX_DEPTH = 16.5
PENALTY_BOX_WIDTH = 40.32
SIX_YARD_BOX_DEPTH = 5.5
SIX_YARD_BOX_WIDTH = 18.32
PENALTY_SPOT_DISTANCE = 11.0
CENTER_CIRCLE_RADIUS = 9.15

_HALF_X = PITCH_LENGTH / 2.0
_MID_Y = PITCH_WIDTH / 2.0
_PBOX_HALF = PENALTY_BOX_WIDTH / 2.0
_SBOX_HALF = SIX_YARD_BOX_WIDTH / 2.0

#: Named landmarks in pitch meters. ``left`` / ``right`` are the two goal ends,
#: ``bottom`` / ``top`` are the low-y and high-y touchlines. Origin is the
#: bottom-left corner, matching ``schema``.
PITCH_KEYPOINTS: dict[str, tuple[float, float]] = {
    # Pitch corners.
    "corner_left_bottom": (0.0, 0.0),
    "corner_left_top": (0.0, PITCH_WIDTH),
    "corner_right_bottom": (PITCH_LENGTH, 0.0),
    "corner_right_top": (PITCH_LENGTH, PITCH_WIDTH),
    # Halfway line and center.
    "halfway_bottom": (_HALF_X, 0.0),
    "halfway_top": (_HALF_X, PITCH_WIDTH),
    "center_spot": (_HALF_X, _MID_Y),
    "center_circle_bottom": (_HALF_X, _MID_Y - CENTER_CIRCLE_RADIUS),
    "center_circle_top": (_HALF_X, _MID_Y + CENTER_CIRCLE_RADIUS),
    # Penalty spots.
    "penalty_spot_left": (PENALTY_SPOT_DISTANCE, _MID_Y),
    "penalty_spot_right": (PITCH_LENGTH - PENALTY_SPOT_DISTANCE, _MID_Y),
    # Penalty ("18-yard") boxes.
    "penalty_box_left_goal_bottom": (0.0, _MID_Y - _PBOX_HALF),
    "penalty_box_left_goal_top": (0.0, _MID_Y + _PBOX_HALF),
    "penalty_box_left_front_bottom": (PENALTY_BOX_DEPTH, _MID_Y - _PBOX_HALF),
    "penalty_box_left_front_top": (PENALTY_BOX_DEPTH, _MID_Y + _PBOX_HALF),
    "penalty_box_right_goal_bottom": (PITCH_LENGTH, _MID_Y - _PBOX_HALF),
    "penalty_box_right_goal_top": (PITCH_LENGTH, _MID_Y + _PBOX_HALF),
    "penalty_box_right_front_bottom": (PITCH_LENGTH - PENALTY_BOX_DEPTH, _MID_Y - _PBOX_HALF),
    "penalty_box_right_front_top": (PITCH_LENGTH - PENALTY_BOX_DEPTH, _MID_Y + _PBOX_HALF),
    # Six-yard boxes.
    "six_yard_left_goal_bottom": (0.0, _MID_Y - _SBOX_HALF),
    "six_yard_left_goal_top": (0.0, _MID_Y + _SBOX_HALF),
    "six_yard_left_front_bottom": (SIX_YARD_BOX_DEPTH, _MID_Y - _SBOX_HALF),
    "six_yard_left_front_top": (SIX_YARD_BOX_DEPTH, _MID_Y + _SBOX_HALF),
    "six_yard_right_goal_bottom": (PITCH_LENGTH, _MID_Y - _SBOX_HALF),
    "six_yard_right_goal_top": (PITCH_LENGTH, _MID_Y + _SBOX_HALF),
    "six_yard_right_front_bottom": (PITCH_LENGTH - SIX_YARD_BOX_DEPTH, _MID_Y - _SBOX_HALF),
    "six_yard_right_front_top": (PITCH_LENGTH - SIX_YARD_BOX_DEPTH, _MID_Y + _SBOX_HALF),
}

#: Straight pitch markings that run the length of the pitch (constant y).
PITCH_LINES_CONST_Y: tuple[float, ...] = (
    0.0,
    _MID_Y - _PBOX_HALF,
    _MID_Y - _SBOX_HALF,
    _MID_Y + _SBOX_HALF,
    _MID_Y + _PBOX_HALF,
    PITCH_WIDTH,
)

#: Straight pitch markings that run across the pitch (constant x).
PITCH_LINES_CONST_X: tuple[float, ...] = (
    0.0,
    SIX_YARD_BOX_DEPTH,
    PENALTY_BOX_DEPTH,
    _HALF_X,
    PITCH_LENGTH - PENALTY_BOX_DEPTH,
    PITCH_LENGTH - SIX_YARD_BOX_DEPTH,
    PITCH_LENGTH,
)

# Below this ratio between the second-smallest and largest singular value of
# the DLT design matrix the null space is effectively two-dimensional, i.e. the
# correspondences do not pin down a unique homography.
_DEGENERACY_TOL = 1e-8

# A homogeneous coordinate this close to zero is a point on the horizon; the
# projection is undefined there and we report NaN rather than a huge number.
_W_EPS = 1e-12

# How many low-residual line labellings get the (more expensive) pitch-support
# check in estimate_from_lines.
_MAX_VERIFIED_CANDIDATES = 60


def _as_points(pts: npt.ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    proj = homo @ h.T
    w = proj[:, 2]
    w = np.where(np.abs(w) < _W_EPS, np.nan, w)
    return proj[:, :2] / w[:, None]


def _hartley_normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate to the centroid and scale so the mean distance is sqrt(2).

    Without this the DLT design matrix mixes entries of order 1 with entries of
    order (pixels squared), and the SVD solution is dominated by rounding in
    the large column.
    """
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = float(np.sqrt((shifted**2).sum(axis=1)).mean())
    if mean_dist < 1e-12:
        raise ValueError("degenerate correspondences: all points coincide")
    scale = np.sqrt(2.0) / mean_dist
    t = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return shifted * scale, t


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve ``dst ~ H src`` in the least-squares sense on normalized points."""
    n = src.shape[0]
    a = np.zeros((2 * n, 9), dtype=np.float64)
    u, v = src[:, 0], src[:, 1]
    x, y = dst[:, 0], dst[:, 1]
    a[0::2, 0] = -u
    a[0::2, 1] = -v
    a[0::2, 2] = -1.0
    a[0::2, 6] = x * u
    a[0::2, 7] = x * v
    a[0::2, 8] = x
    a[1::2, 3] = -u
    a[1::2, 4] = -v
    a[1::2, 5] = -1.0
    a[1::2, 6] = y * u
    a[1::2, 7] = y * v
    a[1::2, 8] = y

    _, s, vt = np.linalg.svd(a)
    # With N >= 4 there are at least 8 singular values. Index 7 is the smallest
    # one that must stay away from zero for the null space to be 1-D.
    if s[0] <= 0.0 or s[7] / s[0] < _DEGENERACY_TOL:
        raise ValueError(
            "degenerate correspondences: the point configuration does not "
            "determine a unique homography (collinear or repeated points)"
        )
    return vt[-1].reshape(3, 3)


def _rescale(h: np.ndarray) -> np.ndarray:
    """Fix the overall scale so two homographies can be compared elementwise."""
    if abs(h[2, 2]) > 1e-12:
        return h / h[2, 2]
    norm = float(np.linalg.norm(h))
    if norm < 1e-12:
        raise ValueError("degenerate correspondences: homography collapsed to zero")
    return h / norm


@dataclass
class Homography:
    """Maps image pixels to pitch meters.

    ``H`` is the image -> pitch direction because that is what the tracker
    needs on every frame; the inverse is only used for drawing overlays.
    """

    H: np.ndarray
    error_m: float = field(default=float("nan"))

    def __post_init__(self) -> None:
        h = np.asarray(self.H, dtype=np.float64)
        if h.shape != (3, 3):
            raise ValueError(f"H must have shape (3, 3), got {h.shape}")
        if not np.isfinite(h).all():
            raise ValueError("H contains non-finite values")
        if abs(float(np.linalg.det(h))) < 1e-15:
            raise ValueError("H is singular and cannot be inverted")
        self.H = h

    @property
    def reprojection_error(self) -> float:
        """Mean Euclidean fit residual in meters, NaN if the fit is unknown."""
        return float(self.error_m)

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def image_to_pitch(self, pts: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        out = _apply(self.H, _as_points(arr, "pts"))
        return out[0] if single else out

    def pitch_to_image(self, pts: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        out = _apply(self.H_inv, _as_points(arr, "pts"))
        return out[0] if single else out

    @classmethod
    def from_point_pairs(
        cls,
        image_pts: npt.ArrayLike,
        pitch_pts: npt.ArrayLike,
    ) -> Homography:
        """Fit image -> pitch by DLT on Hartley-normalized coordinates.

        Raises ``ValueError`` for fewer than four correspondences, mismatched
        shapes, or a configuration that does not determine a homography.
        """
        img = _as_points(image_pts, "image_pts")
        pit = _as_points(pitch_pts, "pitch_pts")
        if img.shape[0] != pit.shape[0]:
            raise ValueError(
                f"image_pts and pitch_pts must have the same length, "
                f"got {img.shape[0]} and {pit.shape[0]}"
            )
        if img.shape[0] < 4:
            raise ValueError(f"need at least 4 correspondences, got {img.shape[0]}")

        img_n, t_img = _hartley_normalize(img)
        pit_n, t_pit = _hartley_normalize(pit)
        h_norm = _dlt(img_n, pit_n)
        h = _rescale(np.linalg.inv(t_pit) @ h_norm @ t_img)

        projected = _apply(h, img)
        if not np.isfinite(projected).all():
            raise ValueError("degenerate correspondences: a fitted point maps to the horizon")
        err = float(np.sqrt(((projected - pit) ** 2).sum(axis=1)).mean())
        return cls(H=h, error_m=err)

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        err = self.reprojection_error
        payload = {
            "H": self.H.tolist(),
            "direction": "image_to_pitch",
            "units": "meters",
            "pitch_length": PITCH_LENGTH,
            "pitch_width": PITCH_WIDTH,
            "reprojection_error_m": None if not np.isfinite(err) else err,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return out

    @classmethod
    def from_json(cls, path: str | Path) -> Homography:
        src = Path(path)
        payload: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
        if "H" not in payload:
            raise ValueError(f"{src} does not contain an 'H' entry")
        err = payload.get("reprojection_error_m")
        return cls(
            H=np.asarray(payload["H"], dtype=np.float64),
            error_m=float("nan") if err is None else float(err),
        )


# --------------------------------------------------------------------------
# Line-based estimation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ImageLine:
    """A merged image line: homogeneous coefficients plus how much ink backs it."""

    coeffs: np.ndarray  # (3,) with a^2 + b^2 == 1
    strength: float
    offset: float  # position along the cross axis, used for ordering


def _pitch_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Green pixels, cleaned up. White markings show up as holes in this mask."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _fit_line(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted total-least-squares line through ``points``, as [a, b, c]."""
    w = weights / weights.sum()
    centroid = (points * w[:, None]).sum(axis=0)
    centered = (points - centroid) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return np.array([normal[0], normal[1], -float(normal @ centroid)])


def _line_offset(coeffs: np.ndarray, width: int, height: int, horizontal: bool) -> float:
    """Where the line crosses the mid-axis of the image, for stable ordering."""
    a, b, c = coeffs
    if horizontal:
        if abs(b) < 1e-9:
            return float("nan")
        return float(-(a * (width / 2.0) + c) / b)
    if abs(a) < 1e-9:
        return float("nan")
    return float(-(b * (height / 2.0) + c) / a)


def _merge_segments(
    segments: list[tuple[np.ndarray, float]],
    width: int,
    height: int,
    horizontal: bool,
    offset_tol: float,
) -> list[_ImageLine]:
    """Collapse Hough segments that describe the same painted line.

    Each white marking gives two Canny edges a few pixels apart, and Hough
    breaks long lines into pieces, so raw segments badly over-count.
    """
    entries: list[tuple[float, np.ndarray, float]] = []
    for pts, length in segments:
        coeffs = _fit_line(pts, np.array([1.0, 1.0]))
        off = _line_offset(coeffs, width, height, horizontal)
        if not np.isfinite(off):
            continue
        entries.append((off, pts, length))
    if not entries:
        return []
    entries.sort(key=lambda e: e[0])

    merged: list[_ImageLine] = []
    cluster_pts: list[np.ndarray] = []
    cluster_w: list[float] = []
    cluster_ref = entries[0][0]

    def flush() -> None:
        if not cluster_pts:
            return
        pts = np.concatenate(cluster_pts, axis=0)
        w = np.repeat(np.array(cluster_w, dtype=np.float64), 2)
        coeffs = _fit_line(pts, w)
        off = _line_offset(coeffs, width, height, horizontal)
        if np.isfinite(off):
            merged.append(_ImageLine(coeffs=coeffs, strength=float(sum(cluster_w)), offset=off))

    for off, pts, length in entries:
        if abs(off - cluster_ref) > offset_tol and cluster_pts:
            flush()
            cluster_pts, cluster_w = [], []
        cluster_pts.append(pts)
        cluster_w.append(length)
        cluster_ref = off
    flush()
    return merged


def _marking_edges(frame_bgr: np.ndarray) -> np.ndarray:
    """Canny edges of the green mask: the outlines of the painted markings."""
    mask = _pitch_mask(frame_bgr)
    if int(mask.max()) == 0:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    return cv2.Canny(mask, 50, 150, apertureSize=3)


def _detect_line_families(
    frame_bgr: np.ndarray,
    edges: np.ndarray,
    *,
    hough_threshold: int,
    min_line_frac: float,
    max_line_gap: int,
) -> tuple[list[_ImageLine], list[_ImageLine]]:
    height, width = frame_bgr.shape[:2]
    if int(edges.max()) == 0:
        return [], []
    min_len = max(20, int(min_line_frac * min(width, height)))
    raw = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_len,
        maxLineGap=max_line_gap,
    )
    if raw is None:
        return [], []

    horiz: list[tuple[np.ndarray, float]] = []
    vert: list[tuple[np.ndarray, float]] = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        pts = np.array([[float(x1), float(y1)], [float(x2), float(y2)]])
        dx, dy = pts[1] - pts[0]
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            continue
        if abs(dx) >= abs(dy):
            horiz.append((pts, length))
        else:
            vert.append((pts, length))

    tol_h = max(8.0, 0.02 * height)
    tol_v = max(8.0, 0.02 * width)
    return (
        _merge_segments(horiz, width, height, True, tol_h),
        _merge_segments(vert, width, height, False, tol_v),
    )


def _intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    p = np.cross(a, b)
    if abs(p[2]) < 1e-9:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]])


def _ordered_subsets(values: tuple[float, ...], k: int) -> list[tuple[float, ...]]:
    """All k-subsets in both pitch orders; the camera may look from either side."""
    out: list[tuple[float, ...]] = []
    for combo in itertools.combinations(values, k):
        out.append(combo)
        if k > 1:
            out.append(tuple(reversed(combo)))
    return out


def _sample_segment(p0: tuple[float, float], p1: tuple[float, float], step: float) -> np.ndarray:
    a = np.array(p0, dtype=np.float64)
    b = np.array(p1, dtype=np.float64)
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


@functools.lru_cache(maxsize=4)
def pitch_marking_samples(step: float = 0.5) -> np.ndarray:
    """Every painted marking on the pitch, sampled every ``step`` meters.

    Used to verify a candidate homography: project these back into the image
    and see whether they land on marking pixels.
    """
    parts: list[np.ndarray] = []
    lo_y, hi_y = _MID_Y - _PBOX_HALF, _MID_Y + _PBOX_HALF
    lo_s, hi_s = _MID_Y - _SBOX_HALF, _MID_Y + _SBOX_HALF
    outline = [
        ((0.0, 0.0), (PITCH_LENGTH, 0.0)),
        ((PITCH_LENGTH, 0.0), (PITCH_LENGTH, PITCH_WIDTH)),
        ((PITCH_LENGTH, PITCH_WIDTH), (0.0, PITCH_WIDTH)),
        ((0.0, PITCH_WIDTH), (0.0, 0.0)),
        ((_HALF_X, 0.0), (_HALF_X, PITCH_WIDTH)),
    ]
    for near, far in ((0.0, PENALTY_BOX_DEPTH), (PITCH_LENGTH, PITCH_LENGTH - PENALTY_BOX_DEPTH)):
        outline += [
            ((near, lo_y), (far, lo_y)),
            ((far, lo_y), (far, hi_y)),
            ((far, hi_y), (near, hi_y)),
        ]
    for near, far in (
        (0.0, SIX_YARD_BOX_DEPTH),
        (PITCH_LENGTH, PITCH_LENGTH - SIX_YARD_BOX_DEPTH),
    ):
        outline += [
            ((near, lo_s), (far, lo_s)),
            ((far, lo_s), (far, hi_s)),
            ((far, hi_s), (near, hi_s)),
        ]
    parts.extend(_sample_segment(a, b, step) for a, b in outline)

    n_circle = max(16, int(2 * np.pi * CENTER_CIRCLE_RADIUS / step))
    ang = np.linspace(0.0, 2 * np.pi, n_circle, endpoint=False)
    parts.append(
        np.stack(
            [
                _HALF_X + CENTER_CIRCLE_RADIUS * np.cos(ang),
                _MID_Y + CENTER_CIRCLE_RADIUS * np.sin(ang),
            ],
            axis=1,
        )
    )
    out = np.concatenate(parts, axis=0)
    out.flags.writeable = False
    return out


def _support(h: np.ndarray, samples: np.ndarray, hit_mask: np.ndarray, min_samples: int) -> float:
    """Fraction of in-frame pitch markings that land on detected marking pixels.

    This is what separates a correct line labelling from a coincidental one:
    a wrong hypothesis still fits its own intersections exactly but sends the
    rest of the pitch onto bare grass.
    """
    height, width = hit_mask.shape
    uv = _apply(np.linalg.inv(h), samples)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if uv.size == 0:
        return 0.0
    u = uv[:, 0]
    v = uv[:, 1]
    inside = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    if int(inside.sum()) < min_samples:
        return 0.0
    hits = hit_mask[v[inside].astype(np.intp), u[inside].astype(np.intp)]
    return float((hits > 0).mean())


def _plausible(h: np.ndarray, width: int, height: int) -> bool:
    """Reject fits that send the middle of the frame nowhere near the pitch."""
    center = _apply(h, np.array([[width / 2.0, height / 2.0]]))
    if not np.isfinite(center).all():
        return False
    x, y = center[0]
    return -0.5 * PITCH_LENGTH <= x <= 1.5 * PITCH_LENGTH and (
        -0.5 * PITCH_WIDTH <= y <= 1.5 * PITCH_WIDTH
    )


def _matches_main_camera(h: np.ndarray, width: int, height: int) -> bool:
    """Does this fit put the pitch the way the main broadcast camera sees it?

    The markings are symmetric under a 180-degree rotation of the pitch, so
    line geometry alone can never choose between a homography and its rotated
    twin. Preferring the main-camera convention -- rightwards in the image is
    increasing x, downwards in the image is decreasing y -- at least makes the
    choice deterministic and matches the schema's left-to-right convention.
    """
    base = np.array(
        [
            [width / 2.0, height / 2.0],
            [width / 2.0 + 1.0, height / 2.0],
            [width / 2.0, height / 2.0 + 1.0],
        ]
    )
    mapped = _apply(h, base)
    if not np.isfinite(mapped).all():
        return False
    return bool(mapped[1, 0] > mapped[0, 0] and mapped[2, 1] < mapped[0, 1])


def estimate_from_lines(
    frame_bgr: np.ndarray,
    *,
    max_reprojection_error_m: float = 2.0,
    min_support: float = 0.6,
    max_lines_per_family: int = 3,
    hough_threshold: int = 60,
    min_line_frac: float = 0.12,
    max_line_gap: int = 12,
) -> Homography | None:
    """Fit a homography from the pitch markings visible in one frame.

    Green mask -> Canny -> probabilistic Hough -> two families of image lines
    (near-horizontal and near-vertical) -> a search over which pitch markings
    those families could be. Hough gives geometry but no labels, and several
    labellings fit their own intersections equally well, so surviving
    hypotheses are re-scored by projecting the whole pitch back into the frame
    and measuring how much of it lands on marking pixels.

    Returns ``None`` when the frame does not carry enough structure. A wrong
    homography is worse than no homography, so the bar is deliberately high:
    at least two lines in each family, at least six intersections, a fit
    residual under ``max_reprojection_error_m``, and pitch support above
    ``min_support``.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"frame_bgr must be an HxWx3 BGR image, got shape {frame_bgr.shape}")
    height, width = frame_bgr.shape[:2]

    edges = _marking_edges(frame_bgr)
    horiz, vert = _detect_line_families(
        frame_bgr,
        edges,
        hough_threshold=hough_threshold,
        min_line_frac=min_line_frac,
        max_line_gap=max_line_gap,
    )
    if len(horiz) < 2 or len(vert) < 2:
        logger.debug(
            "line-based calibration: only %d horizontal and %d vertical lines",
            len(horiz),
            len(vert),
        )
        return None

    def strongest(lines: list[_ImageLine]) -> list[_ImageLine]:
        keep = sorted(lines, key=lambda ln: ln.strength, reverse=True)[:max_lines_per_family]
        return sorted(keep, key=lambda ln: ln.offset)

    horiz = strongest(horiz)
    vert = strongest(vert)
    if len(horiz) * len(vert) < 6:
        # Four intersections fit any homography exactly, so the residual would
        # carry no information about whether the labelling is right.
        logger.debug("line-based calibration: only %d intersections", len(horiz) * len(vert))
        return None

    grid = [[_intersect(h.coeffs, v.coeffs) for v in vert] for h in horiz]
    if any(p is None for row in grid for p in row):
        logger.debug("line-based calibration: line families are not transversal")
        return None
    image_pts = np.array(grid, dtype=np.float64)

    candidates: list[Homography] = []
    for swap in (False, True):
        n_rows, n_cols = (len(horiz), len(vert)) if not swap else (len(vert), len(horiz))
        pts = image_pts if not swap else image_pts.transpose(1, 0, 2)
        flat_img = pts.reshape(-1, 2)
        for ys in _ordered_subsets(PITCH_LINES_CONST_Y, n_rows):
            for xs in _ordered_subsets(PITCH_LINES_CONST_X, n_cols):
                pitch = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
                try:
                    cand = Homography.from_point_pairs(flat_img, pitch)
                except ValueError:
                    continue
                if cand.reprojection_error > max_reprojection_error_m:
                    continue
                if _plausible(cand.H, width, height):
                    candidates.append(cand)

    if not candidates:
        logger.debug("line-based calibration: no labelling fit within the residual budget")
        return None

    candidates.sort(key=lambda c: c.reprojection_error)
    del candidates[_MAX_VERIFIED_CANDIDATES:]
    hit_mask = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    samples = pitch_marking_samples()
    min_samples = max(30, samples.shape[0] // 20)

    scored = [(_support(c.H, samples, hit_mask, min_samples), c) for c in candidates]
    best_support = max(s for s, _ in scored)
    if best_support < min_support:
        logger.debug(
            "line-based calibration: best pitch support %.2f is below %.2f",
            best_support,
            min_support,
        )
        return None

    tied = [c for s, c in scored if s >= best_support - 1e-9]
    for cand in tied:
        if _matches_main_camera(cand.H, width, height):
            return cand
    return tied[0]


# --------------------------------------------------------------------------
# SoccerNet calibration hook
# --------------------------------------------------------------------------

#: SoccerNet's calibration baseline predicts named pitch *lines*; a keypoint is
#: the intersection of two of them. Keys are the SoccerNet line names, values
#: are keys of ``PITCH_KEYPOINTS``. SoccerNet's pitch has its origin at the
#: center with -y towards the "top" touchline, so their "top" is our y = 0.
SOCCERNET_LINE_INTERSECTIONS: dict[tuple[str, str], str] = {
    ("Side line left", "Side line top"): "corner_left_bottom",
    ("Side line left", "Side line bottom"): "corner_left_top",
    ("Side line right", "Side line top"): "corner_right_bottom",
    ("Side line right", "Side line bottom"): "corner_right_top",
    ("Middle line", "Side line top"): "halfway_bottom",
    ("Middle line", "Side line bottom"): "halfway_top",
    ("Big rect. left main", "Big rect. left top"): "penalty_box_left_front_bottom",
    ("Big rect. left main", "Big rect. left bottom"): "penalty_box_left_front_top",
    ("Side line left", "Big rect. left top"): "penalty_box_left_goal_bottom",
    ("Side line left", "Big rect. left bottom"): "penalty_box_left_goal_top",
    ("Big rect. right main", "Big rect. right top"): "penalty_box_right_front_bottom",
    ("Big rect. right main", "Big rect. right bottom"): "penalty_box_right_front_top",
    ("Side line right", "Big rect. right top"): "penalty_box_right_goal_bottom",
    ("Side line right", "Big rect. right bottom"): "penalty_box_right_goal_top",
    ("Small rect. left main", "Small rect. left top"): "six_yard_left_front_bottom",
    ("Small rect. left main", "Small rect. left bottom"): "six_yard_left_front_top",
    ("Side line left", "Small rect. left top"): "six_yard_left_goal_bottom",
    ("Side line left", "Small rect. left bottom"): "six_yard_left_goal_top",
    ("Small rect. right main", "Small rect. right top"): "six_yard_right_front_bottom",
    ("Small rect. right main", "Small rect. right bottom"): "six_yard_right_front_top",
    ("Side line right", "Small rect. right top"): "six_yard_right_goal_bottom",
    ("Side line right", "Small rect. right bottom"): "six_yard_right_goal_top",
}

#: SoccerNet also emits a handful of already-named points directly.
SOCCERNET_POINT_NAMES: dict[str, str] = {
    "Center spot": "center_spot",
    "Penalty mark left": "penalty_spot_left",
    "Penalty mark right": "penalty_spot_right",
}

_SOCCERNET_WEIGHTS_ENV = "REPLAY_SOCCERNET_WEIGHTS"
_SOCCERNET_WEIGHT_FILES = ("SV_FT_TSWC_kp", "SV_FT_TSWC_lines")


def soccernet_correspondences(
    prediction: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Turn a SoccerNet calibration prediction into point correspondences.

    ``prediction`` is the baseline's per-frame output: a mapping from landmark
    name to something with image coordinates. Named points are taken directly;
    line names are paired up through ``SOCCERNET_LINE_INTERSECTIONS`` and their
    predicted extremities intersected.
    """
    image_pts: list[tuple[float, float]] = []
    pitch_pts: list[tuple[float, float]] = []
    names: list[str] = []

    def add(key: str, uv: tuple[float, float]) -> None:
        if key in names or key not in PITCH_KEYPOINTS:
            return
        names.append(key)
        image_pts.append(uv)
        pitch_pts.append(PITCH_KEYPOINTS[key])

    for sn_name, key in SOCCERNET_POINT_NAMES.items():
        entry = prediction.get(sn_name)
        uv = _soccernet_point(entry)
        if uv is not None:
            add(key, uv)

    lines = {name: _soccernet_line(prediction[name]) for name in prediction}
    for (name_a, name_b), key in SOCCERNET_LINE_INTERSECTIONS.items():
        la, lb = lines.get(name_a), lines.get(name_b)
        if la is None or lb is None:
            continue
        pt = _intersect(la, lb)
        if pt is not None:
            add(key, (float(pt[0]), float(pt[1])))

    img = np.array(image_pts, dtype=np.float64).reshape(-1, 2)
    pit = np.array(pitch_pts, dtype=np.float64).reshape(-1, 2)
    return img, pit, names


def _soccernet_point(entry: Any) -> tuple[float, float] | None:
    if isinstance(entry, dict) and "x" in entry and "y" in entry:
        return (float(entry["x"]), float(entry["y"]))
    if isinstance(entry, list | tuple) and len(entry) == 2:
        first = entry[0]
        if isinstance(first, int | float):
            return (float(entry[0]), float(entry[1]))
    return None


def _soccernet_line(entry: Any) -> np.ndarray | None:
    """The baseline stores a line as its two extremities in image coordinates."""
    if not isinstance(entry, list | tuple) or len(entry) != 2:
        return None
    a = _soccernet_point(entry[0])
    b = _soccernet_point(entry[1])
    if a is None or b is None or a == b:
        return None
    la = np.array([a[0], a[1], 1.0])
    lb = np.array([b[0], b[1], 1.0])
    return np.cross(la, lb)


def _soccernet_weights_dir(weights_dir: str | Path | None) -> Path:
    if weights_dir is not None:
        return Path(weights_dir)
    env = os.environ.get(_SOCCERNET_WEIGHTS_ENV)
    if env:
        return Path(env)
    return Path.home() / ".cache" / "replay" / "soccernet"


def estimate_from_soccernet(
    frame_bgr: np.ndarray,
    *,
    weights_dir: str | Path | None = None,
    device: str = "cpu",
) -> Homography | None:
    """Fit a homography with the SoccerNet camera-calibration baseline.

    The weights are gated behind the SoccerNet download agreement and are not
    vendored here, so on a clean machine this logs one line and returns
    ``None``. Nothing is downloaded.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"frame_bgr must be an HxWx3 BGR image, got shape {frame_bgr.shape}")

    try:
        from sn_calibration_baseline.detect_extremities import (  # type: ignore[import-not-found]
            SegmentationNetwork,
        )
    except ImportError:
        logger.info(
            "SoccerNet calibration unavailable: sn_calibration_baseline is not installed; "
            "falling back to line-based calibration"
        )
        return None

    root = _soccernet_weights_dir(weights_dir)
    missing = [f for f in _SOCCERNET_WEIGHT_FILES if not (root / f).exists()]
    if missing:
        logger.info(
            "SoccerNet calibration unavailable: weights %s not found under %s "
            "(set %s to the download directory); falling back to line-based calibration",
            ", ".join(missing),
            root,
            _SOCCERNET_WEIGHTS_ENV,
        )
        return None

    network = SegmentationNetwork(
        str(root / _SOCCERNET_WEIGHT_FILES[0]),
        str(root / _SOCCERNET_WEIGHT_FILES[1]),
        device=device,
    )
    prediction = network.analyse_image(frame_bgr)
    image_pts, pitch_pts, names = soccernet_correspondences(prediction)
    if len(names) < 4:
        logger.info(
            "SoccerNet calibration produced only %d usable landmarks; returning None", len(names)
        )
        return None
    try:
        return Homography.from_point_pairs(image_pts, pitch_pts)
    except ValueError as exc:
        logger.info("SoccerNet calibration landmarks did not yield a homography: %s", exc)
        return None
