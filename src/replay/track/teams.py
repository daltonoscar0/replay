"""Team assignment by jersey colour.

One colour summary per track, then two clusters over those summaries. The only
subtle part is hue: OpenCV packs it into 0..179 and it wraps, so a red shirt
gives pixel hues on both sides of the seam. Averaging those linearly lands on
cyan and splits one team across both clusters, which is why every hue statistic
here goes through the unit circle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from replay.schema import TeamLabel, TrackedBox

log = logging.getLogger(__name__)

# Torso window inside the box: skip the head and the legs, and drop a quarter of
# the width from each side so that background pitch leaking around the shoulders
# does not dominate the crop.
TORSO_TOP = 0.2
TORSO_BOTTOM = 0.6
TORSO_INSET = 0.25

# Pitch green, dropped from the crop before summarising. Same bounds as the
# colour detector uses.
GREEN_LOWER = np.array([35, 60, 40], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)
MIN_NON_GREEN_FRACTION = 0.1

# Feature weights. Hue carries the signal, so its two circle components are
# scaled up relative to brightness; saturation enters as the radius of the hue
# vector, which puts washed-out kits near the origin where they are genuinely
# ambiguous instead of pretending they have a hue.
HUE_WEIGHT = 2.0
VALUE_WEIGHT = 1.0

KMEANS_MAX_ITER = 100


@dataclass(slots=True)
class _TrackSummary:
    track_id: int
    hue: float
    saturation: float
    value: float
    mean_x: float

    @property
    def feature(self) -> np.ndarray:
        theta = self.hue * (np.pi / 90.0)
        chroma = self.saturation / 255.0
        return np.array(
            [
                HUE_WEIGHT * chroma * np.cos(theta),
                HUE_WEIGHT * chroma * np.sin(theta),
                VALUE_WEIGHT * self.value / 255.0,
            ],
            dtype=np.float64,
        )


def circular_median_hue(hues: np.ndarray) -> float:
    """Median of OpenCV hues (0..179) that respects the wrap at 180.

    The offsets are taken relative to the circular mean and then medianed, which
    is cheap and, unlike a direct circular median, does not need an O(n^2) scan.
    """
    h = np.asarray(hues, dtype=np.float64).ravel()
    if h.size == 0:
        return float("nan")
    ang = h * (np.pi / 90.0)
    mean_ang = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
    offsets = (ang - mean_ang + np.pi) % (2.0 * np.pi) - np.pi
    med = mean_ang + float(np.median(offsets))
    return float((med % (2.0 * np.pi)) * (90.0 / np.pi))


def _torso_crop(frame_bgr: np.ndarray, box: TrackedBox) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    bw = box.x2 - box.x1
    bh = box.y2 - box.y1
    if bw <= 0 or bh <= 0:
        return None
    x1 = int(round(box.x1 + TORSO_INSET * bw))
    x2 = int(round(box.x2 - TORSO_INSET * bw))
    y1 = int(round(box.y1 + TORSO_TOP * bh))
    y2 = int(round(box.y1 + TORSO_BOTTOM * bh))
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def _sample_frames(frames: Sequence[int], n_samples: int) -> list[int]:
    """Evenly spaced sample over the track's lifetime, without an RNG."""
    if n_samples <= 0 or not frames:
        return []
    if len(frames) <= n_samples:
        return list(frames)
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, n_samples)).astype(int))
    return [frames[int(i)] for i in idx]


def _summarise(pixels_hsv: np.ndarray) -> tuple[float, float, float]:
    """Circular-median hue plus median saturation and value over torso pixels."""
    hue = circular_median_hue(pixels_hsv[:, 0])
    sat = float(np.median(pixels_hsv[:, 1]))
    val = float(np.median(pixels_hsv[:, 2]))
    return hue, sat, val


def _two_means(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with a deterministic init.

    The two mutually most distant points seed the centroids, so repeated calls
    on the same input give byte-identical labels with no seed to thread through.
    """
    n = feats.shape[0]
    diffs = feats[:, None, :] - feats[None, :, :]
    pair_dist = np.linalg.norm(diffs, axis=2)
    flat = int(np.argmax(pair_dist))
    i, j = divmod(flat, n)
    centroids = feats[[i, j]].astype(np.float64).copy()

    labels = np.zeros(n, dtype=np.int64)
    dist = np.zeros((n, 2), dtype=np.float64)
    for _ in range(KMEANS_MAX_ITER):
        dist = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        updated = centroids.copy()
        for k in (0, 1):
            member = labels == k
            if member.any():
                updated[k] = feats[member].mean(axis=0)
        if np.allclose(updated, centroids):
            break
        centroids = updated
    dist = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dist, axis=1)
    return labels, dist


def _margin_confidence(dist_row: np.ndarray) -> float:
    """Normalized margin between the near and far centroid, in [0, 1]."""
    near, far = float(dist_row.min()), float(dist_row.max())
    total = near + far
    if total <= 0.0:
        return 0.0
    return float(np.clip((far - near) / total, 0.0, 1.0))


def assign_teams(
    boxes_by_frame: Mapping[int, list[TrackedBox]],
    frame_reader: Callable[[int], np.ndarray | None],
    *,
    n_samples: int = 20,
) -> dict[int, tuple[TeamLabel, float]]:
    """Label every track ``home``, ``away`` or ``ball`` with a confidence.

    Which cluster is called home is arbitrary, so it is pinned by a stable rule:
    the cluster whose tracks have the lower mean image x is ``home``, ties broken
    by the lower track id. Tracks with no readable crop still get a label, with
    confidence 0.0, because callers need a row for every track.
    """
    by_track: dict[int, list[TrackedBox]] = {}
    is_ball: dict[int, bool] = {}
    for frame_idx in sorted(boxes_by_frame):
        for box in boxes_by_frame[frame_idx]:
            by_track.setdefault(box.track_id, []).append(box)
            is_ball[box.track_id] = is_ball.get(box.track_id, False) or box.cls == "ball"

    out: dict[int, tuple[TeamLabel, float]] = {}
    player_ids = []
    for track_id in sorted(by_track):
        if is_ball[track_id]:
            out[track_id] = ("ball", 1.0)
        else:
            player_ids.append(track_id)

    frame_cache: dict[int, np.ndarray | None] = {}

    def read(frame_idx: int) -> np.ndarray | None:
        if frame_idx not in frame_cache:
            frame_cache[frame_idx] = frame_reader(frame_idx)
        return frame_cache[frame_idx]

    summaries: list[_TrackSummary] = []
    unusable: list[int] = []
    for track_id in player_ids:
        boxes = sorted(by_track[track_id], key=lambda b: b.frame)
        wanted = set(_sample_frames([b.frame for b in boxes], n_samples))
        chunks: list[np.ndarray] = []
        xs: list[float] = []
        for box in boxes:
            xs.append(box.center[0])
            if box.frame not in wanted:
                continue
            frame = read(box.frame)
            if frame is None or frame.size == 0:
                continue
            crop = _torso_crop(frame, box)
            if crop is None or crop.size == 0:
                continue
            hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2HSV)
            flat = hsv.reshape(-1, 3)
            green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER).reshape(-1) > 0
            keep = ~green
            # Only drop grass if something is left; an all-green kit would
            # otherwise erase its own crop.
            if keep.mean() >= MIN_NON_GREEN_FRACTION:
                flat = flat[keep]
            chunks.append(flat.astype(np.float64))

        if not chunks:
            log.debug("track %d has no usable torso crop", track_id)
            unusable.append(track_id)
            continue
        hue, sat, val = _summarise(np.concatenate(chunks, axis=0))
        summaries.append(
            _TrackSummary(
                track_id=track_id,
                hue=hue,
                saturation=sat,
                value=val,
                mean_x=float(np.mean(xs)) if xs else 0.0,
            )
        )

    if len(summaries) >= 2:
        feats = np.stack([s.feature for s in summaries], axis=0)
        labels, dist = _two_means(feats)

        home_cluster = 0
        if (labels == 0).any() and (labels == 1).any():
            mean_x = [
                float(np.mean([s.mean_x for s, k in zip(summaries, labels, strict=True) if k == c]))
                for c in (0, 1)
            ]
            if mean_x[1] < mean_x[0]:
                home_cluster = 1
            elif mean_x[1] == mean_x[0]:
                first = [
                    min(s.track_id for s, k in zip(summaries, labels, strict=True) if k == c)
                    for c in (0, 1)
                ]
                home_cluster = 0 if first[0] <= first[1] else 1

        for summary, label, row in zip(summaries, labels, dist, strict=True):
            team: TeamLabel = "home" if int(label) == home_cluster else "away"
            out[summary.track_id] = (team, _margin_confidence(row))
    else:
        for summary in summaries:
            out[summary.track_id] = ("home", 0.0)

    for track_id in unusable:
        out[track_id] = ("home", 0.0)

    return dict(sorted(out.items()))
