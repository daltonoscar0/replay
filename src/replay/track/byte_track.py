"""ByteTrack multi-object tracking over per-frame detections.

The idea ByteTrack adds to plain IoU tracking is that low-confidence detections
are not garbage: an occluded or motion-blurred player usually still produces a
weak box roughly where the tracker predicted one. Throwing those away is what
causes a track to die mid-occlusion and come back with a new id. So association
runs twice per frame -- once with the confident detections, then again over
whatever tracks are still unmatched using the weak ones -- and only detections
that fail both stages are allowed to start a new track.

Motion is modelled per track with a constant-velocity Kalman filter in box
space ``(cx, cy, a, h)`` with ``a = w / h``, plus the four velocities. Working
in aspect/height rather than width/height keeps the aspect nearly constant for
an upright person, so the filter spends its uncertainty budget on the things
that actually move.

Emission policy: a track emits a ``TrackedBox`` only on frames where it was
matched to a detection *and* it has accumulated at least ``min_hits``
detections. There is no warm-up exemption for brand-new tracks -- a single
spurious detection should never reach the output -- so the first ``min_hits - 1``
frames of a genuinely new object are silently swallowed. Tracks that are alive
but unmatched (coasting through an occlusion) emit nothing; the caller sees a
gap and then the original id again, which is what the downstream smoother in
``replay.track.kalman`` is built to fill.
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from replay.schema import Detection, ObjectClass, TrackedBox

__all__ = ["ByteTrack"]

# Kalman noise is scaled by box height: a player near the camera is bigger and
# moves more pixels per frame, so a fixed pixel variance would be wrong at both
# ends of the pitch. Same weights as the reference implementation.
_STD_WEIGHT_POSITION = 1.0 / 20.0
_STD_WEIGHT_VELOCITY = 1.0 / 160.0

# Aspect ratio is nearly constant for a standing person and meaningless for a
# ball, so it gets its own small fixed noise instead of a height-scaled one.
_STD_ASPECT_PROCESS = 1.0e-2
_STD_ASPECT_VELOCITY = 1.0e-5
_STD_ASPECT_MEASURE = 1.0e-1

# IoU distance gate for the second (low-score) association. Weak detections are
# only trusted when they land almost exactly on the prediction, otherwise they
# would drag tracks onto background clutter.
_SECOND_MATCH_THRESH = 0.5

# Stand-in for an infinite assignment cost. linear_sum_assignment rejects inf,
# and every match is filtered against its gate afterwards anyway.
_BLOCKED = 1.0e6

_TrackState = Literal["tentative", "confirmed"]


def _to_xyah(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Box corners to the filter's measurement space ``(cx, cy, aspect, h)``."""
    width = max(x2 - x1, 1.0e-6)
    height = max(y2 - y1, 1.0e-6)
    return np.array(
        [x1 + width / 2.0, y1 + height / 2.0, width / height, height],
        dtype=np.float64,
    )


def _to_xyxy(mean: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, aspect, height = (float(v) for v in mean[:4])
    height = max(height, 1.0e-6)
    width = max(aspect * height, 1.0e-6)
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def iou_matrix(tracks_xyxy: np.ndarray, dets_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(T, 4)`` and ``(D, 4)`` boxes, shape ``(T, D)``."""
    if tracks_xyxy.size == 0 or dets_xyxy.size == 0:
        return np.zeros((tracks_xyxy.shape[0], dets_xyxy.shape[0]), dtype=np.float64)
    tl_x = np.maximum(tracks_xyxy[:, None, 0], dets_xyxy[None, :, 0])
    tl_y = np.maximum(tracks_xyxy[:, None, 1], dets_xyxy[None, :, 1])
    br_x = np.minimum(tracks_xyxy[:, None, 2], dets_xyxy[None, :, 2])
    br_y = np.minimum(tracks_xyxy[:, None, 3], dets_xyxy[None, :, 3])
    inter = np.clip(br_x - tl_x, 0.0, None) * np.clip(br_y - tl_y, 0.0, None)
    area_t = np.clip(tracks_xyxy[:, 2] - tracks_xyxy[:, 0], 0.0, None) * np.clip(
        tracks_xyxy[:, 3] - tracks_xyxy[:, 1], 0.0, None
    )
    area_d = np.clip(dets_xyxy[:, 2] - dets_xyxy[:, 0], 0.0, None) * np.clip(
        dets_xyxy[:, 3] - dets_xyxy[:, 1], 0.0, None
    )
    union = area_t[:, None] + area_d[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0.0, inter / union, 0.0)
    return np.asarray(out, dtype=np.float64)


class _Track:
    """One tracked object: its Kalman state, identity and liveness counters."""

    __slots__ = (
        "track_id",
        "cls",
        "mean",
        "covariance",
        "hits",
        "age",
        "time_since_update",
        "state",
        "score",
        "start_frame",
        "last_frame",
    )

    def __init__(self, track_id: int, detection: Detection, min_hits: int) -> None:
        self.track_id = track_id
        self.cls: ObjectClass = detection.cls
        measurement = _to_xyah(detection.x1, detection.y1, detection.x2, detection.y2)
        height = measurement[3]
        self.mean = np.concatenate([measurement, np.zeros(4, dtype=np.float64)])
        std = np.array(
            [
                2.0 * _STD_WEIGHT_POSITION * height,
                2.0 * _STD_WEIGHT_POSITION * height,
                _STD_ASPECT_PROCESS,
                2.0 * _STD_WEIGHT_POSITION * height,
                10.0 * _STD_WEIGHT_VELOCITY * height,
                10.0 * _STD_WEIGHT_VELOCITY * height,
                _STD_ASPECT_VELOCITY,
                10.0 * _STD_WEIGHT_VELOCITY * height,
            ],
            dtype=np.float64,
        )
        self.covariance = np.diag(np.square(std))
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.score = float(detection.score)
        self.start_frame = int(detection.frame)
        self.last_frame = int(detection.frame)
        self.state: _TrackState = "confirmed" if self.hits >= min_hits else "tentative"

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return _to_xyxy(self.mean)

    def predict(self) -> None:
        height = max(float(self.mean[3]), 1.0e-6)
        std_pos = np.array(
            [
                _STD_WEIGHT_POSITION * height,
                _STD_WEIGHT_POSITION * height,
                _STD_ASPECT_PROCESS,
                _STD_WEIGHT_POSITION * height,
            ],
            dtype=np.float64,
        )
        std_vel = np.array(
            [
                _STD_WEIGHT_VELOCITY * height,
                _STD_WEIGHT_VELOCITY * height,
                _STD_ASPECT_VELOCITY,
                _STD_WEIGHT_VELOCITY * height,
            ],
            dtype=np.float64,
        )
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        trans = _transition_matrix()
        self.mean = trans @ self.mean
        self.covariance = trans @ self.covariance @ trans.T + motion_cov
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, min_hits: int) -> None:
        measurement = _to_xyah(detection.x1, detection.y1, detection.x2, detection.y2)
        height = max(float(self.mean[3]), 1.0e-6)
        std = np.array(
            [
                _STD_WEIGHT_POSITION * height,
                _STD_WEIGHT_POSITION * height,
                _STD_ASPECT_MEASURE,
                _STD_WEIGHT_POSITION * height,
            ],
            dtype=np.float64,
        )
        meas_cov = np.diag(np.square(std))
        obs = _observation_matrix()
        innovation = measurement - obs @ self.mean
        innovation_cov = obs @ self.covariance @ obs.T + meas_cov
        gain = np.linalg.solve(innovation_cov.T, (self.covariance @ obs.T).T).T
        self.mean = self.mean + gain @ innovation
        eye_minus = np.eye(8, dtype=np.float64) - gain @ obs
        self.covariance = eye_minus @ self.covariance @ eye_minus.T + gain @ meas_cov @ gain.T

        self.hits += 1
        self.time_since_update = 0
        self.score = float(detection.score)
        self.last_frame = int(detection.frame)
        if self.hits >= min_hits:
            self.state = "confirmed"

    def to_box(self, frame_idx: int) -> TrackedBox:
        x1, y1, x2, y2 = self.xyxy
        return TrackedBox(
            frame=int(frame_idx),
            track_id=self.track_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=self.score,
            cls=self.cls,
        )


def _transition_matrix() -> np.ndarray:
    trans = np.eye(8, dtype=np.float64)
    trans[0, 4] = 1.0
    trans[1, 5] = 1.0
    trans[2, 6] = 1.0
    trans[3, 7] = 1.0
    return trans


def _observation_matrix() -> np.ndarray:
    obs = np.zeros((4, 8), dtype=np.float64)
    obs[0, 0] = 1.0
    obs[1, 1] = 1.0
    obs[2, 2] = 1.0
    obs[3, 3] = 1.0
    return obs


def _associate(
    tracks: list[_Track],
    detections: list[Detection],
    max_distance: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment on IoU distance, gated by ``max_distance``.

    Returns ``(matches, unmatched_track_indices, unmatched_detection_indices)``
    where indices are into the argument lists. Pairs whose classes differ are
    made unassignable, which is what stops a ball detection from being absorbed
    into a player track when the ball is dribbled at a player's feet.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    track_boxes = np.array([t.xyxy for t in tracks], dtype=np.float64)
    det_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float64)
    cost = 1.0 - iou_matrix(track_boxes, det_boxes)

    track_cls = np.array([t.cls for t in tracks], dtype=object)
    det_cls = np.array([d.cls for d in detections], dtype=object)
    cost[track_cls[:, None] != det_cls[None, :]] = _BLOCKED

    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for row, col in zip(rows, cols, strict=True):
        if cost[row, col] > max_distance:
            continue
        matches.append((int(row), int(col)))
        matched_tracks.add(int(row))
        matched_dets.add(int(col))
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


class ByteTrack:
    """Two-stage IoU tracker with per-track constant-velocity motion.

    ``match_thresh`` is a maximum on IoU *distance* (``1 - IoU``), matching the
    reference implementation: the default 0.8 accepts a first-stage match at
    IoU >= 0.2. Detections scoring below ``low_thresh`` are ignored entirely,
    detections in ``[low_thresh, high_thresh)`` may only extend an existing
    confirmed track, and only detections at or above ``high_thresh`` may start
    one.
    """

    def __init__(
        self,
        *,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        if not 0.0 <= low_thresh <= high_thresh:
            raise ValueError("low_thresh must be in [0, high_thresh]")
        if max_age < 1:
            raise ValueError("max_age must be at least 1")
        if min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._tracks: list[_Track] = []
        # itertools.count so ids are monotonic from 1 and an id is never handed
        # out twice, even after the track that owned it dies.
        self._ids = itertools.count(1)
        self._frame_idx = -1

    @property
    def tracks(self) -> list[_Track]:
        """Live tracks, in creation order. Exposed for inspection and tests."""
        return list(self._tracks)

    def update(self, detections: list[Detection], frame_idx: int) -> list[TrackedBox]:
        """Advance the tracker by one frame and return this frame's boxes."""
        self._frame_idx = int(frame_idx)

        high: list[Detection] = []
        low: list[Detection] = []
        for det in detections:
            if det.score >= self.high_thresh:
                high.append(det)
            elif det.score >= self.low_thresh:
                low.append(det)

        for track in self._tracks:
            track.predict()

        confirmed = [t for t in self._tracks if t.state == "confirmed"]
        tentative = [t for t in self._tracks if t.state == "tentative"]

        # Stage 1: confident detections against confirmed tracks, including the
        # ones currently coasting through an occlusion.
        matches, unmatched_conf, unmatched_high = _associate(confirmed, high, self.match_thresh)
        for track_i, det_j in matches:
            confirmed[track_i].update(high[det_j], self.min_hits)

        # Stage 2: the whole point of ByteTrack. Weak detections get a shot at
        # the tracks stage 1 could not explain.
        pool = [confirmed[i] for i in unmatched_conf]
        matches, unmatched_pool, _ = _associate(pool, low, _SECOND_MATCH_THRESH)
        for track_i, det_j in matches:
            pool[track_i].update(low[det_j], self.min_hits)
        still_unmatched = [pool[i] for i in unmatched_pool]

        # Stage 3a: leftover confident detections against not-yet-confirmed
        # tracks, so a new object can accumulate the hits it needs.
        leftover = [high[j] for j in unmatched_high]
        matches, unmatched_tentative, unmatched_leftover = _associate(
            tentative, leftover, self.match_thresh
        )
        for track_i, det_j in matches:
            tentative[track_i].update(leftover[det_j], self.min_hits)

        # Stage 3b: anything still unexplained and confident starts a track.
        for det_j in unmatched_leftover:
            self._tracks.append(_Track(next(self._ids), leftover[det_j], self.min_hits))

        # A tentative track that misses a frame was probably a false positive,
        # so it dies immediately; only confirmed tracks earn the max_age grace.
        doomed = {id(t) for t in [tentative[i] for i in unmatched_tentative]}
        for track in still_unmatched:
            if track.time_since_update > self.max_age:
                doomed.add(id(track))
        self._tracks = [t for t in self._tracks if id(t) not in doomed]

        out = [
            t.to_box(self._frame_idx)
            for t in self._tracks
            if t.time_since_update == 0 and t.hits >= self.min_hits
        ]
        out.sort(key=lambda b: b.track_id)
        return out
