"""Tests for replay.overlay."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from replay import overlay as ov
from replay.schema import TrackedBox, coerce_tracking_frame
from replay.track.homography import Homography

FRAME_W = 320
FRAME_H = 240
PITCH_PX_PER_M = 16.0

pytestmark = pytest.mark.skipif(
    ov.shutil.which("ffmpeg") is None or ov.shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are required to build and inspect test clips",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _make_video(path: Path, *, n_frames: int, fps: float, size=(FRAME_W, FRAME_H)) -> Path:
    """A tiny green clip with a moving white square so frames are distinguishable."""
    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_frames):
            img = np.full((h, w, 3), (60, 130, 70), dtype=np.uint8)
            x = 8 + (i * 5) % max(1, w - 24)
            img[4:20, x : x + 16] = (250, 250, 250)
            cv2.imwrite(str(Path(tmp) / f"{i:06d}.png"), img)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", f"{fps}", "-start_number", "0",
                "-i", str(Path(tmp) / "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    return path


def _probe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,pix_fmt,nb_read_frames,width,height",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)["streams"][0]


def _homography() -> Homography:
    """Image rectangle onto a 20 x 15 m patch of pitch around the centre circle."""
    image_pts = [(0.0, 0.0), (FRAME_W, 0.0), (FRAME_W, FRAME_H), (0.0, FRAME_H)]
    pitch_pts = [
        (40.0, 55.0),
        (40.0 + FRAME_W / PITCH_PX_PER_M, 55.0),
        (40.0 + FRAME_W / PITCH_PX_PER_M, 55.0 - FRAME_H / PITCH_PX_PER_M),
        (40.0, 55.0 - FRAME_H / PITCH_PX_PER_M),
    ]
    return Homography.from_point_pairs(image_pts, pitch_pts)


def _tracking_df(*, n_samples: int, hz: float, video_fps: float) -> pd.DataFrame:
    """Two players and a ball crossing the visible patch at a constant speed."""
    rows = []
    for i in range(n_samples):
        t = i / hz
        frame = int(round(t * video_fps))
        for team, pid, jersey, x0, y0 in (
            ("home", "home_7", 7, 44.0, 48.0),
            ("away", "away_4", 4, 52.0, 44.0),
            ("ball", "ball", None, 48.0, 46.0),
        ):
            rows.append(
                {
                    "match_id": "clip",
                    "t": t,
                    "period": 1,
                    "frame": frame,
                    "team": team,
                    "player_id": pid,
                    "jersey": pd.NA if jersey is None else jersey,
                    "x": x0 + 4.0 * t,
                    "y": y0 + 1.0 * t,
                    "vx": 4.0,
                    "vy": 1.0,
                    "confidence": 0.9,
                }
            )
    return coerce_tracking_frame(pd.DataFrame(rows))


def _boxes(n_frames: int) -> dict[int, list[TrackedBox]]:
    out: dict[int, list[TrackedBox]] = {}
    for f in range(n_frames):
        out[f] = [
            TrackedBox(f, 1, 40.0 + 3 * f, 90.0, 70.0 + 3 * f, 160.0, 0.9, "player", "home", 0.9),
            TrackedBox(f, 2, 180.0, 100.0, 210.0, 170.0, 0.8, "player", "away", 0.8),
            TrackedBox(f, 3, 150.0, 150.0, 158.0, 158.0, 0.7, "ball", None, 0.0),
        ]
    return out


def _overlay_temp_dirs() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(ov.TEMP_DIR_PREFIX + "*"))


# --------------------------------------------------------------------------
# end-to-end encoding
# --------------------------------------------------------------------------


def test_dataframe_overlay_encodes_playable_video(tmp_path):
    src = _make_video(tmp_path / "src.mp4", n_frames=10, fps=10.0)
    df = _tracking_df(n_samples=10, hz=10.0, video_fps=10.0)

    out = ov.render_overlay(src, df, tmp_path / "out.mp4", homography=_homography())

    assert out.exists()
    assert out.stat().st_size > 0
    info = _probe(out)
    assert info["codec_name"] == "h264"
    assert info["pix_fmt"] == "yuv420p"
    assert int(info["nb_read_frames"]) == 10
    assert (int(info["width"]), int(info["height"])) == (FRAME_W, FRAME_H)


def test_tracked_box_overlay_encodes_playable_video(tmp_path):
    src = _make_video(tmp_path / "src.mp4", n_frames=8, fps=8.0)

    out = ov.render_overlay(src, _boxes(8), tmp_path / "boxes.mp4", draw_minimap=False)

    info = _probe(out)
    assert info["codec_name"] == "h264"
    assert info["pix_fmt"] == "yuv420p"
    assert int(info["nb_read_frames"]) == 8


def test_fps_override_resamples_to_the_same_duration(tmp_path):
    src = _make_video(tmp_path / "src.mp4", n_frames=10, fps=10.0)
    df = _tracking_df(n_samples=10, hz=10.0, video_fps=10.0)

    out = ov.render_overlay(src, df, tmp_path / "fast.mp4", homography=_homography(), fps=20.0)

    # One second of source at 20 fps is twenty output frames.
    assert int(_probe(out)["nb_read_frames"]) == 20


def test_tracking_at_10hz_fills_every_25fps_frame(tmp_path):
    src = _make_video(tmp_path / "src.mp4", n_frames=25, fps=25.0)
    df = _tracking_df(n_samples=10, hz=10.0, video_fps=25.0)

    out = ov.render_overlay(src, df, tmp_path / "smooth.mp4", homography=_homography())

    assert int(_probe(out)["nb_read_frames"]) == 25


# --------------------------------------------------------------------------
# interpolation
# --------------------------------------------------------------------------


def test_interpolation_between_tracking_samples_is_monotone():
    df = _tracking_df(n_samples=10, hz=10.0, video_fps=25.0)
    tracks = {tr.key: tr for tr in ov._tracks_from_dataframe(df, 25.0)}
    home = tracks["home_7"]

    xs = []
    for i in range(25):
        sampled = home.sample(i / 25.0)
        assert sampled is not None
        xs.append(float(sampled[0][0]))

    assert len(xs) == 25
    assert all(b >= a for a, b in zip(xs[:-1], xs[1:], strict=True))
    # Strictly increasing everywhere the samples actually cover (t <= 0.9);
    # past the last sample the position is held, not extrapolated.
    covered = xs[:23]
    assert all(b > a for a, b in zip(covered[:-1], covered[1:], strict=True))
    assert xs[-1] == pytest.approx(float(home.values[-1][0]))
    # A sample landing between two 10 Hz observations lands strictly between
    # their positions rather than being held at the earlier one.
    mid = home.sample(0.05)
    assert mid is not None
    assert home.values[0][0] < mid[0][0] < home.values[1][0]


def test_track_is_dropped_outside_its_own_span():
    df = _tracking_df(n_samples=5, hz=10.0, video_fps=25.0)
    home = next(tr for tr in ov._tracks_from_dataframe(df, 25.0) if tr.key == "home_7")

    assert home.sample(-1.0) is None
    assert home.sample(10.0) is None
    assert home.sample(0.42) is not None  # just past the last sample, still held


def test_long_gaps_are_not_bridged():
    track = ov._Track(
        key="k",
        label="k",
        team="home",
        cls="player",
        t=np.array([0.0, 5.0]),
        values=np.array([[0.0, 0.0], [50.0, 50.0]]),
        confidence=np.array([1.0, 1.0]),
        hold_s=0.1,
    )
    assert track.sample(2.5) is None
    assert track.sample(0.05) is not None


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------


def test_tracked_boxes_change_pixels_inside_the_box(tmp_path):
    frame = np.full((FRAME_H, FRAME_W, 3), (60, 130, 70), dtype=np.uint8)
    box = TrackedBox(0, 5, 60.0, 90.0, 110.0, 170.0, 0.9, "player", "home", 0.9)

    drawn = ov.draw_frame(frame, [box])

    assert drawn is not frame
    assert np.array_equal(frame, np.full((FRAME_H, FRAME_W, 3), (60, 130, 70), dtype=np.uint8))
    edge = drawn[90:171, 58:63]
    assert not np.array_equal(edge, frame[90:171, 58:63])
    far = drawn[200:240, 250:320]
    assert np.array_equal(far, frame[200:240, 250:320])


def test_pitch_markers_are_drawn_at_the_projected_position():
    h = _homography()
    frame = np.full((FRAME_H, FRAME_W, 3), (60, 130, 70), dtype=np.uint8)
    pitch_xy = (48.0, 47.0)

    drawn = ov.draw_frame(frame, {"home_7": pitch_xy}, homography=h, draw_minimap=False)

    u, v = h.pitch_to_image(np.array(pitch_xy))
    u, v = int(round(float(u))), int(round(float(v)))
    patch = drawn[v - 12 : v + 13, u - 14 : u + 15]
    assert not np.array_equal(patch, frame[v - 12 : v + 13, u - 14 : u + 15])
    assert np.array_equal(drawn[FRAME_H - 20 :, :40], frame[FRAME_H - 20 :, :40])


def test_teams_get_distinct_colours_and_low_confidence_does_not():
    assert ov._color_for("home", 0.9, "player") == ov.TEAM_COLORS["home"]
    assert ov._color_for("away", 0.9, "player") == ov.TEAM_COLORS["away"]
    assert ov._color_for("referee", 0.9, "player") == ov.TEAM_COLORS["referee"]
    assert ov._color_for(None, 0.0, "ball") == ov.TEAM_COLORS["ball"]
    assert ov._color_for(None, 0.9, "player") == ov.UNKNOWN_COLOR
    assert ov._color_for("home", 0.05, "player") == ov.UNKNOWN_COLOR
    assert len({tuple(c) for c in ov.TEAM_COLORS.values()} | {ov.UNKNOWN_COLOR}) == 5


def test_labels_carry_track_id_and_jersey():
    df = _tracking_df(n_samples=3, hz=10.0, video_fps=10.0)
    labels = {tr.key: tr.label for tr in ov._tracks_from_dataframe(df, 10.0)}
    assert labels["home_7"] == "home_7 #7"
    assert labels["ball"] == "ball"

    box_labels = {tr.key: tr.label for tr in ov._tracks_from_boxes(_boxes(2), 10.0)}
    assert box_labels["1"] == "#1"
    assert box_labels["3"] == "ball"


def test_minimap_draws_into_the_corner_only():
    h = _homography()
    frame = np.full((FRAME_H, FRAME_W, 3), (60, 130, 70), dtype=np.uint8)

    drawn = ov.draw_frame(frame, {"home_7": (48.0, 47.0)}, homography=h, draw_minimap=True)

    x0, y0, mw, mh = ov._minimap_rect(frame)
    assert not np.array_equal(drawn[y0 : y0 + mh, x0 : x0 + mw], frame[y0 : y0 + mh, x0 : x0 + mw])
    # The inset stays in its corner: the opposite corner is untouched.
    assert np.array_equal(drawn[:60, :60], frame[:60, :60])


def test_minimap_skipped_without_pitch_information(caplog):
    frame = np.full((FRAME_H, FRAME_W, 3), (60, 130, 70), dtype=np.uint8)
    box = TrackedBox(0, 1, 10.0, 10.0, 40.0, 60.0, 0.9, "player", "home", 0.9)

    with caplog.at_level(logging.INFO, logger="replay.overlay"):
        drawn = ov.draw_frame(frame, [box], homography=None, draw_minimap=True)

    assert "minimap skipped" in caplog.text
    x0, y0, mw, mh = ov._minimap_rect(frame)
    assert np.array_equal(drawn[y0 : y0 + mh, x0 : x0 + mw], frame[y0 : y0 + mh, x0 : x0 + mw])


def test_dataframe_without_homography_logs_and_still_renders(tmp_path, caplog):
    src = _make_video(tmp_path / "src.mp4", n_frames=6, fps=6.0)
    df = _tracking_df(n_samples=6, hz=6.0, video_fps=6.0)

    with caplog.at_level(logging.WARNING, logger="replay.overlay"):
        out = ov.render_overlay(src, df, tmp_path / "nohom.mp4")

    assert "no homography" in caplog.text
    assert int(_probe(out)["nb_read_frames"]) == 6


# --------------------------------------------------------------------------
# failure modes and cleanup
# --------------------------------------------------------------------------


def test_missing_ffmpeg_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ov.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError) as excinfo:
        ov.render_overlay(tmp_path / "missing.mp4", _boxes(2), tmp_path / "out.mp4")

    message = str(excinfo.value)
    assert "ffmpeg" in message
    assert "PATH" in message
    assert isinstance(excinfo.value, ov.FFmpegNotFoundError)


def test_missing_source_video_raises_before_encoding(tmp_path):
    with pytest.raises(FileNotFoundError, match="source video"):
        ov.render_overlay(tmp_path / "nope.mp4", _boxes(2), tmp_path / "out.mp4")


def test_encode_failure_reports_stderr_and_cleans_up(tmp_path, monkeypatch):
    src = _make_video(tmp_path / "src.mp4", n_frames=4, fps=4.0)
    before = _overlay_temp_dirs()

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="x264 exploded")

    monkeypatch.setattr(ov.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="x264 exploded"):
        ov.render_overlay(src, _boxes(4), tmp_path / "out.mp4", draw_minimap=False)

    assert _overlay_temp_dirs() == before


def test_temp_frames_are_removed_on_success_and_on_draw_failure(tmp_path, monkeypatch):
    src = _make_video(tmp_path / "src.mp4", n_frames=4, fps=4.0)
    before = _overlay_temp_dirs()

    ov.render_overlay(src, _boxes(4), tmp_path / "ok.mp4", draw_minimap=False)
    assert _overlay_temp_dirs() == before

    def boom(*_args, **_kwargs):
        raise ValueError("drawing blew up")

    monkeypatch.setattr(ov, "_render_markers", boom)
    with pytest.raises(ValueError, match="drawing blew up"):
        ov.render_overlay(src, _boxes(4), tmp_path / "bad.mp4", draw_minimap=False)
    assert _overlay_temp_dirs() == before


def test_empty_tracking_frame_still_produces_video(tmp_path, caplog):
    src = _make_video(tmp_path / "src.mp4", n_frames=4, fps=4.0)
    empty = _tracking_df(n_samples=1, hz=10.0, video_fps=4.0).iloc[0:0]

    with caplog.at_level(logging.WARNING, logger="replay.overlay"):
        out = ov.render_overlay(src, empty, tmp_path / "empty.mp4", homography=_homography())

    assert "empty" in caplog.text
    assert int(_probe(out)["nb_read_frames"]) == 4
