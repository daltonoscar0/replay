"""Draw tracking results back onto the source video.

Two input shapes are supported and they carry different information.

``TrackedBox`` input is image-space: boxes are drawn where the tracker put
them, no calibration needed. A canonical tracking dataframe is pitch-space:
the boxes are long gone, so the positions have to be projected back through
the homography and marked on the ground plane. A ground ellipse under the
player is what broadcast overlays use, and it is also the honest shape here,
since the only thing the pitch coordinate pins down is where the player's
feet touch the grass.

Encoding goes through ffmpeg rather than ``cv2.VideoWriter`` because the
codecs OpenCV ships with vary by wheel and platform, and half of them produce
files that will not play in a browser. libx264 + yuv420p plays everywhere.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from replay.schema import PITCH_LENGTH, PITCH_WIDTH, TrackedBox, coerce_tracking_frame
from replay.track.homography import Homography

logger = logging.getLogger(__name__)

#: Prefix of the scratch directory the PNG frames are staged in. Exposed so
#: tests can assert nothing is left behind.
TEMP_DIR_PREFIX = "replay-overlay-"

#: BGR, because that is what OpenCV draws in.
TEAM_COLORS: dict[str, tuple[int, int, int]] = {
    "home": (70, 70, 235),
    "away": (235, 160, 60),
    "ball": (40, 220, 250),
    "referee": (205, 70, 205),
}

#: Anything unlabelled, unrecognised, or below the confidence floor.
UNKNOWN_COLOR: tuple[int, int, int] = (170, 170, 170)

#: Team assignment below this is drawn as unknown rather than as a wrong team.
LOW_CONFIDENCE_TEAM_THRESHOLD = 0.35

#: Two tracking samples further apart than this are treated as separate
#: appearances of the object; interpolating across the gap would draw a marker
#: gliding through positions the object was never in.
MAX_INTERPOLATION_GAP_S = 1.0

#: Used when the container reports no usable frame rate.
FALLBACK_FPS = 25.0

PLAYER_MARKER_RADIUS_M = 0.55
BALL_MARKER_RADIUS_M = 0.28

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg binary is not on PATH."""


@dataclass(slots=True)
class _Marker:
    """One object to draw on one output frame."""

    label: str
    color: tuple[int, int, int]
    cls: str
    box: tuple[float, float, float, float] | None = None
    pitch: tuple[float, float] | None = None

    @property
    def depth_key(self) -> float:
        """Sort key so nearer objects are drawn last and overlap correctly."""
        if self.box is not None:
            return self.box[3]
        if self.pitch is not None:
            return -self.pitch[1]
        return 0.0


@dataclass(slots=True)
class _Track:
    """One object's samples on a shared seconds axis.

    ``values`` is whatever the input shape carries: ``(x, y)`` in meters for a
    tracking dataframe, ``(x1, y1, x2, y2)`` in pixels for tracked boxes.
    """

    key: str
    label: str
    team: str | None
    cls: str
    t: np.ndarray
    values: np.ndarray
    confidence: np.ndarray
    hold_s: float

    def sample(self, t: float) -> tuple[np.ndarray, float] | None:
        """Values at time ``t``, or None if the object is not present then.

        Between two tracking samples the values are interpolated linearly.
        Tracking commonly runs at 10 Hz against 25 fps video, so most output
        frames land between samples; holding the previous value instead would
        make every marker visibly stutter twice per sample.
        """
        ts = self.t
        if t < ts[0] - self.hold_s or t > ts[-1] + self.hold_s:
            return None
        idx = int(np.searchsorted(ts, t, side="left"))
        if idx <= 0:
            return self.values[0], float(self.confidence[0])
        if idx >= ts.size:
            return self.values[-1], float(self.confidence[-1])
        t0 = float(ts[idx - 1])
        t1 = float(ts[idx])
        if t1 - t0 > MAX_INTERPOLATION_GAP_S:
            # The object disappeared and came back. Hold each end of the gap
            # for the usual sample duration, then draw nothing at all rather
            # than sliding a marker across positions it never occupied.
            if t - t0 <= self.hold_s:
                return self.values[idx - 1], float(self.confidence[idx - 1])
            if t1 - t <= self.hold_s:
                return self.values[idx], float(self.confidence[idx])
            return None
        w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        values = self.values[idx - 1] * (1.0 - w) + self.values[idx] * w
        conf = float(self.confidence[idx - 1] * (1.0 - w) + self.confidence[idx] * w)
        return values, conf


def render_overlay(
    video_path: str | Path,
    tracks: pd.DataFrame | Mapping[int, list[TrackedBox]],
    out_path: str | Path,
    *,
    homography: Homography | None = None,
    fps: float | None = None,
    draw_minimap: bool = True,
) -> Path:
    """Render ``tracks`` over ``video_path`` and encode the result to ``out_path``.

    ``fps`` overrides the output frame rate; the default is the source video's.
    Tracking positions are resampled onto the output timeline, so a 10 Hz
    tracking file over 25 fps video yields a smooth 25 fps overlay.
    """
    ffmpeg = _require_ffmpeg()
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"source video not found: {src}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {src}")
    try:
        src_fps = _read_fps(cap, src)
        out_fps = float(fps) if fps else src_fps
        if out_fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        if isinstance(tracks, pd.DataFrame):
            built = _tracks_from_dataframe(tracks, src_fps)
            pitch_space = True
        else:
            built = _tracks_from_boxes(tracks, src_fps)
            pitch_space = False

        if pitch_space and homography is None:
            logger.warning(
                "no homography supplied with a tracking dataframe: pitch positions cannot "
                "be projected into the image, drawing the minimap only"
            )
        minimap = _minimap_enabled(draw_minimap, pitch_space=pitch_space, homography=homography)

        n_frames = _write_frames(
            cap,
            built,
            src_fps=src_fps,
            out_fps=out_fps,
            pitch_space=pitch_space,
            homography=homography,
            minimap=minimap,
            ffmpeg=ffmpeg,
            out=out,
        )
    finally:
        cap.release()

    logger.info("wrote %s (%d frames @ %.3f fps)", out, n_frames, out_fps if fps else src_fps)
    return out


def draw_frame(
    frame_bgr: np.ndarray,
    objects: Sequence[TrackedBox] | Mapping[str, tuple[float, float]],
    *,
    homography: Homography | None = None,
    draw_minimap: bool = False,
) -> np.ndarray:
    """Draw one frame and return it, without touching the input.

    The single-frame entry point behind ``render_overlay``; useful for
    inspecting what an overlay will look like without encoding a video.
    ``objects`` is either tracked boxes or a mapping of label to pitch
    coordinates in meters.
    """
    if isinstance(objects, Mapping):
        markers = [
            _Marker(
                label=str(key),
                color=UNKNOWN_COLOR,
                cls="player",
                pitch=(float(p[0]), float(p[1])),
            )
            for key, p in objects.items()
        ]
    else:
        markers = [_marker_from_box(box, homography) for box in objects]
    canvas = frame_bgr.copy()
    minimap = _minimap_enabled(
        draw_minimap,
        pitch_space=isinstance(objects, Mapping),
        homography=homography,
    )
    _render_markers(canvas, markers, homography=homography, minimap=minimap)
    return canvas


# --------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------


def _require_ffmpeg() -> str:
    """Locate ffmpeg up front so a long render cannot fail at the last step."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FFmpegNotFoundError(
            "the 'ffmpeg' binary was not found on PATH. replay encodes overlay video with "
            "ffmpeg (libx264 / yuv420p) so the result plays in any player; install it "
            "(macOS: 'brew install ffmpeg', Debian/Ubuntu: 'sudo apt install ffmpeg') and "
            "make sure 'ffmpeg' is on PATH, then re-run."
        )
    return exe


def _encode(ffmpeg: str, frame_dir: Path, out: Path, fps: float) -> None:
    """Encode the staged PNGs. Scaling to even dimensions keeps yuv420p legal."""
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:.6f}",
        "-start_number",
        "0",
        "-i",
        str(frame_dir / "%06d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-r",
        f"{fps:.6f}",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise RuntimeError(
            f"ffmpeg failed with exit code {proc.returncode} while encoding {out}:\n"
            + "\n".join(tail)
        )


def _write_frames(
    cap: cv2.VideoCapture,
    tracks: Sequence[_Track],
    *,
    src_fps: float,
    out_fps: float,
    pitch_space: bool,
    homography: Homography | None,
    minimap: bool,
    ffmpeg: str,
    out: Path,
) -> int:
    """Stage every output frame as a PNG, then encode.

    The scratch directory is a context manager so it is removed even when
    drawing or encoding raises part way through.
    """
    # When the container reports a frame count the output length is fixed up
    # front, so an fps override resamples to exactly the same duration. When
    # it does not, fall back to rendering until the decoder runs dry.
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_out = max(1, int(round(n_src / src_fps * out_fps))) if n_src > 0 else None

    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as tmp:
        frame_dir = Path(tmp)
        n_written = 0
        src_idx = -1
        current: np.ndarray | None = None
        exhausted = False
        i = 0
        while not exhausted and (n_out is None or i < n_out):
            t = i / out_fps
            target = int(round(t * src_fps))
            if n_src > 0:
                target = min(target, n_src - 1)
            while src_idx < target:
                ok, img = cap.read()
                if not ok:
                    exhausted = True
                    break
                src_idx += 1
                current = img
            if exhausted or current is None:
                break
            canvas = current.copy()
            markers = _markers_at(tracks, t, pitch_space=pitch_space, homography=homography)
            _render_markers(canvas, markers, homography=homography, minimap=minimap)
            if not cv2.imwrite(str(frame_dir / f"{i:06d}.png"), canvas):
                raise RuntimeError(f"failed to stage overlay frame {i} in {frame_dir}")
            n_written += 1
            i += 1

        if n_written == 0:
            raise ValueError("source video contained no decodable frames")
        _encode(ffmpeg, frame_dir, out, out_fps)
    return n_written


def _read_fps(cap: cv2.VideoCapture, src: Path) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        logger.warning("%s reports no frame rate; assuming %.1f fps", src, FALLBACK_FPS)
        return FALLBACK_FPS
    return fps


# --------------------------------------------------------------------------
# Track construction
# --------------------------------------------------------------------------


def _tracks_from_dataframe(df: pd.DataFrame, src_fps: float) -> list[_Track]:
    """Split a canonical tracking frame into per-object time series."""
    tidy = coerce_tracking_frame(df)
    if tidy.empty:
        logger.warning("tracking frame is empty: the overlay will be the source video")
        return []

    t = tidy["t"].to_numpy(dtype=np.float64)
    if not np.isfinite(t).all():
        logger.warning("tracking frame has non-finite 't'; deriving times from frame numbers")
        t = tidy["frame"].to_numpy(dtype=np.float64) / src_fps
    tidy = tidy.assign(_t=t)

    keys = tidy["player_id"].astype("object").where(tidy["player_id"].notna(), None)
    fallback = tidy["team"].astype("object").fillna("obj") + "_" + tidy.index.astype(str)
    resolved = [k if k is not None else f for k, f in zip(keys, fallback, strict=True)]
    tidy = tidy.assign(_key=resolved)

    out: list[_Track] = []
    for key, grp in tidy.groupby("_key", sort=True):
        grp = grp.sort_values("_t")
        team = _first_str(grp["team"])
        jersey = _first_jersey(grp["jersey"])
        cls = "ball" if team == "ball" else "player"
        xy = grp[["x", "y"]].to_numpy(dtype=np.float64)
        conf = grp["confidence"].to_numpy(dtype=np.float64)
        conf = np.where(np.isfinite(conf), conf, 0.0)
        keep = np.isfinite(xy).all(axis=1)
        if not keep.any():
            continue
        ts = grp["_t"].to_numpy(dtype=np.float64)[keep]
        out.append(
            _Track(
                key=str(key),
                label=_label_for(str(key), jersey, cls),
                team=team,
                cls=cls,
                t=ts,
                values=xy[keep],
                confidence=conf[keep],
                hold_s=_hold_for(ts, src_fps),
            )
        )
    return out


def _tracks_from_boxes(
    boxes_by_frame: Mapping[int, list[TrackedBox]],
    src_fps: float,
) -> list[_Track]:
    """Split frame-keyed tracked boxes into per-track time series.

    Frame keys are source-video frame indices, so they map onto the shared
    seconds axis through the source frame rate.
    """
    per_track: dict[int, list[tuple[float, TrackedBox]]] = {}
    for frame, boxes in boxes_by_frame.items():
        for box in boxes:
            per_track.setdefault(box.track_id, []).append((float(frame) / src_fps, box))
    if not per_track:
        logger.warning("no tracked boxes supplied: the overlay will be the source video")

    out: list[_Track] = []
    for track_id, samples in sorted(per_track.items()):
        samples.sort(key=lambda s: s[0])
        ts = np.array([s[0] for s in samples], dtype=np.float64)
        xyxy = np.array([s[1].xyxy for s in samples], dtype=np.float64)
        conf = np.array([float(s[1].team_confidence) for s in samples], dtype=np.float64)
        last = samples[-1][1]
        out.append(
            _Track(
                key=str(track_id),
                label=_label_for(f"#{track_id}", None, last.cls),
                team=last.team if last.cls != "ball" else "ball",
                cls=last.cls,
                t=ts,
                values=xyxy,
                confidence=conf,
                hold_s=_hold_for(ts, src_fps),
            )
        )
    return out


def _hold_for(ts: np.ndarray, src_fps: float) -> float:
    """How long past its last sample an object stays on screen.

    One sampling interval: enough to bridge the tail of a 10 Hz sample at
    25 fps, short enough that a track that ended does not linger.
    """
    if ts.size < 2:
        return max(1.0 / src_fps, 0.04)
    return float(np.median(np.diff(ts)))


def _first_str(col: pd.Series) -> str | None:
    values = col.dropna()
    return None if values.empty else str(values.iloc[0])


def _first_jersey(col: pd.Series) -> int | None:
    values = col.dropna()
    return None if values.empty else int(values.iloc[0])


def _label_for(ident: str, jersey: int | None, cls: str) -> str:
    if cls == "ball":
        return "ball"
    if jersey is None:
        return ident
    return f"{ident} #{jersey}"


def _markers_at(
    tracks: Sequence[_Track],
    t: float,
    *,
    pitch_space: bool,
    homography: Homography | None,
) -> list[_Marker]:
    markers: list[_Marker] = []
    for track in tracks:
        sampled = track.sample(t)
        if sampled is None:
            continue
        values, conf = sampled
        color = _color_for(track.team, conf, track.cls)
        if pitch_space:
            markers.append(
                _Marker(
                    label=track.label,
                    color=color,
                    cls=track.cls,
                    pitch=(float(values[0]), float(values[1])),
                )
            )
        else:
            box = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
            pitch = _box_to_pitch(box, homography)
            markers.append(
                _Marker(label=track.label, color=color, cls=track.cls, box=box, pitch=pitch)
            )
    markers.sort(key=lambda m: m.depth_key)
    return markers


def _box_to_pitch(
    box: tuple[float, float, float, float],
    homography: Homography | None,
) -> tuple[float, float] | None:
    if homography is None:
        return None
    foot = np.array([[(box[0] + box[2]) / 2.0, box[3]]], dtype=np.float64)
    pitch = homography.image_to_pitch(foot)[0]
    if not np.isfinite(pitch).all():
        return None
    return (float(pitch[0]), float(pitch[1]))


def _color_for(team: str | None, confidence: float, cls: str) -> tuple[int, int, int]:
    """Team colour, or the unknown colour when the label is not trustworthy."""
    if cls == "ball":
        return TEAM_COLORS["ball"]
    if team is None or team not in TEAM_COLORS:
        return UNKNOWN_COLOR
    if team in ("home", "away") and confidence < LOW_CONFIDENCE_TEAM_THRESHOLD:
        return UNKNOWN_COLOR
    return TEAM_COLORS[team]


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _minimap_enabled(
    requested: bool,
    *,
    pitch_space: bool,
    homography: Homography | None,
) -> bool:
    if not requested:
        return False
    if pitch_space or homography is not None:
        return True
    logger.info(
        "minimap skipped: it needs pitch coordinates, so pass a tracking dataframe or a "
        "homography to project the boxes with"
    )
    return False


def _render_markers(
    canvas: np.ndarray,
    markers: Sequence[_Marker],
    *,
    homography: Homography | None,
    minimap: bool,
) -> None:
    for marker in markers:
        if marker.box is not None:
            _draw_box(canvas, marker)
        elif marker.pitch is not None and homography is not None:
            _draw_ground_marker(canvas, marker, homography)
    if minimap:
        _draw_minimap(canvas, markers)


def _stroke(canvas: np.ndarray) -> int:
    """Line weight that stays visible at 240p and does not smother a 4K frame."""
    return max(2, int(round(canvas.shape[0] / 320.0)))


def _draw_box(canvas: np.ndarray, marker: _Marker) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in marker.box)  # type: ignore[misc]
    thickness = _stroke(canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), marker.color, thickness, cv2.LINE_AA)
    _put_label(canvas, marker.label, (x1, y1), marker.color, center=False)


def _draw_ground_marker(canvas: np.ndarray, marker: _Marker, homography: Homography) -> None:
    """A circle on the ground plane, which projects to an ellipse in image space."""
    assert marker.pitch is not None
    radius = BALL_MARKER_RADIUS_M if marker.cls == "ball" else PLAYER_MARKER_RADIUS_M
    pts = _ground_circle(homography, marker.pitch, radius)
    if pts is None:
        return
    poly = pts.astype(np.int32)
    thickness = _stroke(canvas)
    cv2.polylines(canvas, [poly], True, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.polylines(canvas, [poly], True, marker.color, thickness, cv2.LINE_AA)
    if marker.cls == "ball":
        cv2.fillPoly(canvas, [poly], marker.color, cv2.LINE_AA)
    top = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].min()))))
    _put_label(canvas, marker.label, top, marker.color, center=True)


def _ground_circle(
    homography: Homography,
    pitch_xy: tuple[float, float],
    radius_m: float,
    n: int = 28,
) -> np.ndarray | None:
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.stack(
        [pitch_xy[0] + radius_m * np.cos(ang), pitch_xy[1] + radius_m * np.sin(ang)],
        axis=1,
    )
    pts = homography.pitch_to_image(circle)
    if not np.isfinite(pts).all() or np.abs(pts).max() > 1e5:
        return None
    return pts


def _put_label(
    canvas: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    *,
    center: bool,
) -> None:
    """Text just above ``anchor``, on a filled plate so it reads over grass."""
    h = canvas.shape[0]
    scale = max(0.32, min(0.7, h / 900.0))
    thickness = 1 if h < 720 else 2
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    x = anchor[0] - tw // 2 if center else anchor[0]
    y = anchor[1] - 4
    x = int(np.clip(x, 1, max(1, canvas.shape[1] - tw - 2)))
    y = int(np.clip(y, th + baseline + 2, canvas.shape[0] - 2))
    cv2.rectangle(
        canvas,
        (x - 2, y - th - baseline),
        (x + tw + 2, y + baseline // 2),
        color,
        -1,
    )
    cv2.putText(canvas, text, (x, y), _FONT, scale, (16, 16, 16), thickness, cv2.LINE_AA)


def _minimap_rect(canvas: np.ndarray) -> tuple[int, int, int, int]:
    """Bottom-right inset, pitch aspect preserved, clamped to the frame."""
    h, w = canvas.shape[:2]
    margin = max(4, int(round(0.02 * w)))
    mw = min(max(int(round(0.26 * w)), 90), max(20, w - 2 * margin))
    mh = int(round(mw * PITCH_WIDTH / PITCH_LENGTH))
    max_h = max(16, h - 2 * margin)
    if mh > max_h:
        mh = max_h
        mw = int(round(mh * PITCH_LENGTH / PITCH_WIDTH))
    x0 = w - margin - mw
    y0 = h - margin - mh
    return x0, y0, mw, mh


def _draw_minimap(canvas: np.ndarray, markers: Sequence[_Marker]) -> None:
    x0, y0, mw, mh = _minimap_rect(canvas)
    if mw < 20 or mh < 12:
        return
    roi = canvas[y0 : y0 + mh, x0 : x0 + mw]
    plate = np.full(roi.shape, (32, 74, 38), dtype=np.uint8)
    cv2.addWeighted(plate, 0.72, roi, 0.28, 0.0, dst=roi)

    line = (225, 225, 225)
    cv2.rectangle(canvas, (x0, y0), (x0 + mw - 1, y0 + mh - 1), line, 1, cv2.LINE_AA)
    mid = x0 + mw // 2
    cv2.line(canvas, (mid, y0), (mid, y0 + mh - 1), line, 1, cv2.LINE_AA)
    cv2.circle(
        canvas, (mid, y0 + mh // 2), max(2, int(mw * 9.15 / PITCH_LENGTH)), line, 1, cv2.LINE_AA
    )
    box_w = int(round(mw * 16.5 / PITCH_LENGTH))
    box_h = int(round(mh * 40.32 / PITCH_WIDTH))
    box_y0 = y0 + (mh - box_h) // 2
    cv2.rectangle(canvas, (x0, box_y0), (x0 + box_w, box_y0 + box_h), line, 1, cv2.LINE_AA)
    cv2.rectangle(
        canvas, (x0 + mw - 1 - box_w, box_y0), (x0 + mw - 1, box_y0 + box_h), line, 1, cv2.LINE_AA
    )

    dot = max(2, mw // 55)
    for marker in markers:
        if marker.pitch is None:
            continue
        px, py = marker.pitch
        if not (-5.0 <= px <= PITCH_LENGTH + 5.0 and -5.0 <= py <= PITCH_WIDTH + 5.0):
            continue
        u = x0 + int(round(np.clip(px / PITCH_LENGTH, 0.0, 1.0) * (mw - 1)))
        # Pitch y grows upward from the bottom-left origin; image v grows down.
        v = y0 + int(round((1.0 - np.clip(py / PITCH_WIDTH, 0.0, 1.0)) * (mh - 1)))
        r = dot if marker.cls != "ball" else max(1, dot - 1)
        cv2.circle(canvas, (u, v), r + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, (u, v), r, marker.color, -1, cv2.LINE_AA)


def _marker_from_box(box: TrackedBox, homography: Homography | None = None) -> _Marker:
    team = box.team if box.cls != "ball" else "ball"
    xyxy = (box.x1, box.y1, box.x2, box.y2)
    return _Marker(
        label=_label_for(f"#{box.track_id}", None, box.cls),
        color=_color_for(team, box.team_confidence, box.cls),
        cls=box.cls,
        box=xyxy,
        pitch=_box_to_pitch(xyxy, homography),
    )
