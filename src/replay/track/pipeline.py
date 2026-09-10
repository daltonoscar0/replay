"""End-to-end broadcast video -> pitch-coordinate tracking.

The stages are deliberately separable: each one is a pure-ish function over the
previous stage's output, so a caller can swap the detector, supply a known
homography, or re-run team assignment without redoing the expensive parts.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from replay.schema import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    Detection,
    TrackedBox,
    coerce_tracking_frame,
    empty_tracking_frame,
)
from replay.track.byte_track import ByteTrack
from replay.track.detect import build_detector, resolve_device
from replay.track.homography import Homography, estimate_from_lines, estimate_from_soccernet
from replay.track.kalman import smooth_track
from replay.track.teams import assign_teams

log = logging.getLogger(__name__)


@dataclass
class TrackConfig:
    """Everything that changes what `replay track` does, in one object."""

    detector: str = "yolo"
    device: str = "auto"
    out_fps: float = 10.0
    conf: float = 0.25
    weights: str = "yolov8n.pt"
    frame_stride: int = 1
    max_frames: int | None = None
    homography_path: Path | None = None
    match_id: str = "match"
    smooth: bool = True
    detector_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackResult:
    tracks: pd.DataFrame
    boxes_by_frame: dict[int, list[TrackedBox]]
    homography: Homography | None
    video_fps: float
    n_frames_read: int
    notes: list[str] = field(default_factory=list)


class VideoReader:
    """Random-access-ish frame reader with a one-frame cache.

    Team assignment wants to revisit specific frames after tracking has already
    run. Seeking with `CAP_PROP_POS_FRAMES` is unreliable on some codecs, so
    frames sampled for team assignment are cached during the forward pass
    instead of being sought for afterwards.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"video not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise OSError(f"could not open video: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cache: dict[int, np.ndarray] = {}

    def frames(
        self, *, stride: int = 1, limit: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield `(frame_index, image)` for every `stride`-th frame."""
        idx = 0
        emitted = 0
        while True:
            ok, img = self.cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, img
                emitted += 1
                if limit is not None and emitted >= limit:
                    break
            idx += 1

    def cache(self, idx: int, img: np.ndarray) -> None:
        self._cache[idx] = img

    def cached(self, idx: int) -> np.ndarray | None:
        return self._cache.get(idx)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def resolve_homography(
    reader: VideoReader,
    cfg: TrackConfig,
    probe_frames: list[np.ndarray],
    notes: list[str],
) -> Homography | None:
    """Pick a homography: explicit file, then SoccerNet model, then line fitting.

    Order matters. A calibration file the user supplied is always right; the
    learned model beats line fitting when it is installed; line fitting is the
    fallback that works with nothing but the frame. Any of them may fail, and
    failing loudly beats projecting with a wrong matrix.
    """
    if cfg.homography_path is not None:
        h = Homography.from_json(cfg.homography_path)
        notes.append(f"homography loaded from {cfg.homography_path}")
        return h

    for img in probe_frames:
        h = estimate_from_soccernet(img)
        if h is not None:
            notes.append("homography from SoccerNet calibration model")
            return h

    best: Homography | None = None
    for img in probe_frames:
        h = estimate_from_lines(img)
        if h is None:
            continue
        if best is None or h.reprojection_error < best.reprojection_error:
            best = h
    if best is not None:
        notes.append(f"homography from line fitting, reproj {best.reprojection_error:.2f} m")
        return best

    notes.append(
        "no homography could be estimated; output stays in image pixels and "
        "pitch coordinates are not written"
    )
    return None


def detect_and_track(
    reader: VideoReader,
    cfg: TrackConfig,
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[int, list[TrackedBox]], list[np.ndarray], int]:
    """Run detection and ByteTrack over the video in one forward pass."""
    detector = build_detector(
        cfg.detector,
        **(
            {"device": resolve_device(cfg.device), "conf": cfg.conf, "weights": cfg.weights}
            if cfg.detector == "yolo"
            else {}
        ),
        **cfg.detector_kwargs,
    )
    tracker = ByteTrack()
    boxes_by_frame: dict[int, list[TrackedBox]] = {}
    ball_dets: dict[int, list[Detection]] = {}
    probe_frames: list[np.ndarray] = []
    n_read = 0

    for idx, img in reader.frames(stride=cfg.frame_stride, limit=cfg.max_frames):
        dets: list[Detection] = detector.detect(img, idx)
        # Players stream straight into the tracker. Ball candidates are held
        # back, because deciding which of them are painted on takes the whole
        # clip (see _track_ball).
        balls = [d for d in dets if d.cls == "ball"]
        if balls:
            ball_dets[idx] = balls
        boxes_by_frame[idx] = tracker.update([d for d in dets if d.cls != "ball"], idx)
        # Cache a sparse set of frames for team assignment and homography
        # probing so neither stage has to seek backwards later.
        if n_read % 25 == 0:
            reader.cache(idx, img)
            if len(probe_frames) < 8:
                probe_frames.append(img)
        n_read += 1
        if progress is not None:
            progress(n_read)

    for idx, boxes in _track_ball(ball_dets).items():
        boxes_by_frame.setdefault(idx, []).extend(boxes)

    return boxes_by_frame, probe_frames, n_read


#: Ball candidates are binned into cells this many pixels across, and a cell
#: that is occupied in more than this fraction of frames is taken to be paint
#: rather than a ball.
BALL_STATIC_CELL_PX = 6
BALL_STATIC_FRAME_FRACTION = 0.2

#: Ball track ids are offset past the player tracker's so the two independent
#: trackers cannot hand out the same id.
BALL_TRACK_ID_OFFSET = 1_000_000


def _track_ball(ball_dets: dict[int, list[Detection]]) -> dict[int, list[TrackedBox]]:
    """Track the ball on its own, after discarding the painted impostors.

    The ball detector is tuned for recall and returns a couple of dozen
    candidates a frame: the penalty spots, the centre mark and the junctions
    where lines meet are all small, round, white and surrounded by grass. No
    single frame can tell them apart from the ball. A sequence of frames can.
    Paint stays exactly where it is, so it lights up the same handful of image
    cells all clip long, while the ball is somewhere new every frame. Measuring
    how often each cell is occupied separates them cleanly, and it costs
    nothing that a per-frame appearance test could have bought.

    Filtering has to finish before the first frame can be tracked, so the
    candidates are buffered rather than streamed — a few tens of thousands of
    small boxes for a clip, and the reason this is worth keeping an eye on for a
    full match.

    This reads image position, so it assumes the camera is not panning far. A
    pan smears each mark across many cells and weakens it; the pitch-space
    movement gate in `_resolve_ball` is what catches them after that.
    """
    if not ball_dets:
        return {}

    n_frames = len(ball_dets)
    occupancy: Counter[tuple[int, int]] = Counter()
    for dets in ball_dets.values():
        # One vote per cell per frame, so a cluster of candidates on one mark
        # cannot out-vote its own frame count.
        seen = set()
        for d in dets:
            cu, cv = d.center
            seen.add((round(cu / BALL_STATIC_CELL_PX), round(cv / BALL_STATIC_CELL_PX)))
        occupancy.update(seen)

    limit = BALL_STATIC_FRAME_FRACTION * n_frames
    static = {cell for cell, count in occupancy.items() if count > limit}

    tracker = ByteTrack()
    out: dict[int, list[TrackedBox]] = {}
    kept = dropped = 0
    for idx in sorted(ball_dets):
        live = []
        for d in ball_dets[idx]:
            cu, cv = d.center
            cell = (round(cu / BALL_STATIC_CELL_PX), round(cv / BALL_STATIC_CELL_PX))
            if cell in static:
                dropped += 1
            else:
                live.append(d)
                kept += 1
        boxes = tracker.update(live, idx)
        if boxes:
            out[idx] = [
                replace(b, track_id=b.track_id + BALL_TRACK_ID_OFFSET) for b in boxes
            ]
    log.info(
        "ball: %d candidates kept, %d dropped at %d stationary image cells",
        kept,
        dropped,
        len(static),
    )
    return out


def _resample_to_fps(
    t: np.ndarray, xy: np.ndarray, target_t: np.ndarray, *, max_gap_s: float = np.inf
) -> np.ndarray:
    """Linear resample of a (N,2) trajectory onto `target_t`, NaN outside range.

    Interpolation is refused across a gap wider than ``max_gap_s``. Straight-line
    filling is only defensible over a gap short enough that the object could not
    have gone anywhere interesting, and the ball breaks that badly: it is tracked
    in fragments, travels at up to 20 m/s, and a chord drawn between two
    fragments a second apart can put it twenty metres from where it really was.
    Emitting nothing is the honest output for a stretch that was not observed.
    """
    out = np.full((target_t.size, 2), np.nan, dtype=np.float64)
    finite = np.isfinite(xy).all(axis=1)
    if finite.sum() < 2:
        return out
    tf, xyf = t[finite], xy[finite]
    inside = (target_t >= tf[0]) & (target_t <= tf[-1])
    for j in range(2):
        out[inside, j] = np.interp(target_t[inside], tf, xyf[:, j])

    if np.isfinite(max_gap_s):
        # Width of the observed interval each target time falls into.
        hi = np.clip(np.searchsorted(tf, target_t, side="left"), 1, tf.size - 1)
        too_wide = (tf[hi] - tf[hi - 1]) > max_gap_s
        out[inside & too_wide] = np.nan
    return out


#: How far a ball candidate must travel over its whole life, in meters, before
#: it is believed. Penalty spots, the centre spot and goal-frame corners are
#: white, round, and sit in open grass, so they pass every appearance test the
#: detector can apply, but they do not move. On the demo clip the gap is stark:
#: marking tracks cover at most 0.7 m while real ball fragments cover 2.5 m or
#: more, so anything in between is a safe place to draw the line.
BALL_MIN_PATH_M = 1.5

#: Longest unobserved stretch that may be bridged by interpolation, per class.
#: A player runs at up to 8 m/s and turns slowly, so half a second of straight
#: line costs a couple of metres at worst. A ball leaves a boot at 20 m/s and
#: changes direction instantly, so the same half second is meaningless.
PLAYER_MAX_GAP_S = 1.0
BALL_MAX_GAP_S = 0.3

Observation = tuple[int, float, float, float, str]


def _resolve_ball(by_track: dict[int, list[Observation]], homography: Homography | None) -> None:
    """Reduce every ball-class track to one moving ball, in place.

    Two problems at once. Each spurious white blob opens its own track, and all
    of them would be written under the same ``player_id``, colliding in a single
    frame. And the obvious tiebreak — keep the longest-lived — is exactly wrong:
    a painted marking never moves, so it tracks cleanly for the entire clip,
    while the real ball is fast and keeps fragmenting. Longest-lived reliably
    picks the penalty spot.

    So candidates are judged on movement instead, measured in pitch meters
    rather than pixels: a marking is stationary on the pitch whether or not the
    camera pans, and a pixel threshold would only hold for a locked-off shot.
    Whatever survives is fragments of the one real ball, so they are merged into
    a single track rather than a winner being chosen among them. Where two
    fragments claim the same frame the longer one wins, on the grounds that it
    had more evidence behind it.

    The cost is a ball that genuinely never moves — a long dead-ball period —
    which this discards. Mistaking a penalty spot for the ball is the worse
    failure, because it is silent and puts the ball metres from every player.
    """
    ball_tracks = {tid: obs for tid, obs in by_track.items() if obs and obs[0][4] == "ball"}
    if not ball_tracks:
        return
    for tid in ball_tracks:
        del by_track[tid]
    if homography is None:
        return

    moving: dict[int, float] = {}
    for tid, obs in ball_tracks.items():
        pts = homography.image_to_pitch(np.array([[o[1], o[2]] for o in sorted(obs)]))
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        path = float(np.nansum(steps)) if steps.size else 0.0
        if path >= BALL_MIN_PATH_M:
            moving[tid] = path

    if not moving:
        log.warning(
            "no ball track moved at least %.1f m; %d stationary candidate(s) discarded "
            "as pitch markings, so the output has no ball",
            BALL_MIN_PATH_M,
            len(ball_tracks),
        )
        return

    # Longest last, so its observations overwrite shorter fragments on any frame
    # the two disagree about.
    order = sorted(moving, key=lambda tid: len(ball_tracks[tid]))
    merged: dict[int, Observation] = {}
    for tid in order:
        for o in ball_tracks[tid]:
            merged[o[0]] = o

    keep = order[-1]
    by_track[keep] = [merged[f] for f in sorted(merged)]
    log.info(
        "merged %d moving ball fragment(s) into %d observations and discarded %d "
        "stationary candidate(s)",
        len(moving),
        len(merged),
        len(ball_tracks) - len(moving),
    )


def to_tracking_frame(
    boxes_by_frame: dict[int, list[TrackedBox]],
    homography: Homography | None,
    teams: dict[int, tuple[str, float]],
    *,
    video_fps: float,
    out_fps: float,
    match_id: str,
    smooth: bool = True,
) -> pd.DataFrame:
    """Project tracked boxes to the pitch, smooth, resample, and lay out rows.

    Resampling happens after smoothing rather than before: the Kalman smoother
    wants every measurement it can get, and decimating first would throw away
    information the smoother would have used.
    """
    if homography is None:
        return empty_tracking_frame()

    by_track: dict[int, list[tuple[int, float, float, float, str]]] = {}
    for frame_idx, boxes in boxes_by_frame.items():
        for b in boxes:
            u, v = b.foot
            by_track.setdefault(b.track_id, []).append((frame_idx, u, v, b.score, b.cls))

    if not by_track:
        return empty_tracking_frame()

    _resolve_ball(by_track, homography)

    all_frames = sorted(boxes_by_frame)
    t_start = all_frames[0] / video_fps
    t_end = all_frames[-1] / video_fps
    step = 1.0 / out_fps
    target_t = np.arange(t_start, t_end + 1e-9, step)

    rows: list[dict[str, Any]] = []
    for track_id, obs in sorted(by_track.items()):
        obs.sort()
        frames = np.array([o[0] for o in obs], dtype=np.float64)
        t = frames / video_fps
        img_pts = np.array([[o[1], o[2]] for o in obs], dtype=np.float64)
        conf = np.array([o[3] for o in obs], dtype=np.float64)
        cls = obs[0][4]

        pitch = homography.image_to_pitch(img_pts)
        # Anything projecting well outside the pitch is a bad detection or a bad
        # projection; drop the point rather than let it drag the smoother.
        off = (
            (pitch[:, 0] < -5)
            | (pitch[:, 0] > PITCH_LENGTH + 5)
            | (pitch[:, 1] < -5)
            | (pitch[:, 1] > PITCH_WIDTH + 5)
        )
        pitch[off] = np.nan

        if smooth and np.isfinite(pitch).all(axis=1).sum() >= 3:
            sm, vel = smooth_track(t, pitch)
        else:
            sm, vel = pitch, np.full_like(pitch, np.nan)

        # A player drifts predictably and can be bridged across a short dropout;
        # the ball cannot, so it gets a much tighter budget.
        max_gap = BALL_MAX_GAP_S if cls == "ball" else PLAYER_MAX_GAP_S
        rs = _resample_to_fps(t, sm, target_t, max_gap_s=max_gap)
        rv = _resample_to_fps(t, vel, target_t, max_gap_s=max_gap)
        rc = np.interp(target_t, t, conf, left=np.nan, right=np.nan)

        team_label, team_conf = teams.get(track_id, ("unknown", 0.0))
        if cls == "ball":
            team_label, team_conf = "ball", 1.0

        for k, tt in enumerate(target_t):
            if not np.isfinite(rs[k]).all():
                continue
            rows.append(
                {
                    "match_id": match_id,
                    "t": float(tt),
                    "period": 1,
                    "frame": int(round(tt * out_fps)),
                    "team": team_label,
                    "player_id": f"{team_label}_{track_id}" if cls != "ball" else "ball",
                    "jersey": pd.NA,
                    "x": float(rs[k, 0]),
                    "y": float(rs[k, 1]),
                    "vx": float(rv[k, 0]) if np.isfinite(rv[k, 0]) else np.nan,
                    "vy": float(rv[k, 1]) if np.isfinite(rv[k, 1]) else np.nan,
                    # Detection confidence and team confidence are different
                    # things; the column carries their product so a caller
                    # filtering on it drops both kinds of doubt at once.
                    "confidence": float(np.nan_to_num(rc[k], nan=0.0) * max(team_conf, 1e-3)),
                }
            )

    if not rows:
        return empty_tracking_frame()
    df = pd.DataFrame(rows).sort_values(["frame", "team", "player_id"], kind="stable")
    return coerce_tracking_frame(df)


def run_track(video: str | Path, cfg: TrackConfig) -> TrackResult:
    """The whole `replay track` pipeline."""
    notes: list[str] = []
    with VideoReader(video) as reader:
        boxes_by_frame, probe_frames, n_read = detect_and_track(reader, cfg)
        homography = resolve_homography(reader, cfg, probe_frames, notes)
        teams = assign_teams(boxes_by_frame, reader.cached)
        tracks = to_tracking_frame(
            boxes_by_frame,
            homography,
            teams,
            video_fps=reader.fps,
            out_fps=cfg.out_fps,
            match_id=cfg.match_id,
            smooth=cfg.smooth,
        )
        return TrackResult(
            tracks=tracks,
            boxes_by_frame=boxes_by_frame,
            homography=homography,
            video_fps=reader.fps,
            n_frames_read=n_read,
            notes=notes,
        )
