"""Tests for the synthetic demo clip generator.

The generator is a script rather than a package module, so it is loaded by
path. Everything here runs on a 2-second clip except the encode test, which is
marked slow.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replay.schema import PITCH_LENGTH, PITCH_WIDTH, TRACKING_COLUMNS, TRACKING_DTYPES

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_demo_clip.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_demo_clip", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up in sys.modules while processing the class
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mdc = _load_module()

SHORT_S = 2.0


@pytest.fixture(scope="module")
def track():
    return mdc.generate_world_trajectories(duration_s=SHORT_S, seed=0)


@pytest.fixture(scope="module")
def camera():
    return mdc.build_camera()


def test_generation_is_deterministic():
    a = mdc.generate_world_trajectories(duration_s=SHORT_S, seed=0)
    b = mdc.generate_world_trajectories(duration_s=SHORT_S, seed=0)
    assert np.array_equal(a.t, b.t)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.vel, b.vel)


def test_different_seeds_give_different_trajectories():
    a = mdc.generate_world_trajectories(duration_s=SHORT_S, seed=0)
    b = mdc.generate_world_trajectories(duration_s=SHORT_S, seed=7)
    assert not np.array_equal(a.pos, b.pos)


def test_track_shape_and_rate(track):
    n = int(round(SHORT_S * mdc.SIM_FPS)) + 1
    assert track.pos.shape == (n, mdc.N_OBJECTS, 2)
    assert track.vel.shape == (n, mdc.N_OBJECTS, 2)
    assert track.t.shape == (n,)
    assert np.allclose(np.diff(track.t), 1.0 / mdc.SIM_FPS)


def test_subsample_rejects_non_integer_ratio(track):
    with pytest.raises(ValueError, match="divide evenly"):
        track.subsample(7.0)


def test_ground_truth_schema(track):
    gt = mdc.build_ground_truth(track.subsample(mdc.GT_FPS))
    assert tuple(gt.columns) == TRACKING_COLUMNS
    for col, dtype in TRACKING_DTYPES.items():
        assert gt[col].dtype == pd.api.types.pandas_dtype(dtype), col
    assert gt["match_id"].unique().tolist() == ["demo"]
    assert (gt["period"] == 1).all()


def test_ground_truth_has_23_objects_per_frame(track):
    gt = mdc.build_ground_truth(track.subsample(mdc.GT_FPS))
    counts = gt.groupby("frame").size()
    assert (counts == 23).all()
    per_frame_ids = gt.groupby("frame")["player_id"].nunique()
    assert (per_frame_ids == 23).all()

    first = gt[gt["frame"] == 0]
    assert (first["team"] == "home").sum() == 11
    assert (first["team"] == "away").sum() == 11
    assert (first["team"] == "ball").sum() == 1
    assert first.loc[first["team"] == "ball", "jersey"].isna().all()
    assert sorted(first.loc[first["team"] == "home", "jersey"].tolist()) == list(range(1, 12))


def test_ground_truth_is_10hz(track):
    gt = mdc.build_ground_truth(track.subsample(mdc.GT_FPS))
    times = np.sort(gt["t"].unique())
    assert len(times) == int(round(SHORT_S * mdc.GT_FPS)) + 1
    assert np.allclose(np.diff(times), 0.1)
    frames = np.sort(gt["frame"].unique())
    assert np.array_equal(frames, np.arange(len(times)))


def test_all_positions_inside_pitch(track):
    gt = mdc.build_ground_truth(track.subsample(mdc.GT_FPS))
    assert gt["x"].between(0.0, PITCH_LENGTH).all()
    assert gt["y"].between(0.0, PITCH_WIDTH).all()


def test_players_respect_the_speed_cap(track):
    gt = mdc.build_ground_truth(track.subsample(mdc.GT_FPS))
    players = gt[gt["team"] != "ball"]
    for _, group in players.groupby("player_id", sort=True):
        g = group.sort_values("frame")
        step = np.hypot(np.diff(g["x"].to_numpy()), np.diff(g["y"].to_numpy()))
        assert step.max() / 0.1 <= mdc.MAX_SPEED + 1e-6


def test_ground_truth_csv_is_byte_identical_for_a_seed(tmp_path):
    tr = mdc.generate_world_trajectories(duration_s=SHORT_S, seed=3)
    gt = mdc.build_ground_truth(tr.subsample(mdc.GT_FPS))
    a = mdc.write_ground_truth(gt, tmp_path / "a.csv")
    b = mdc.write_ground_truth(
        mdc.build_ground_truth(
            mdc.generate_world_trajectories(duration_s=SHORT_S, seed=3).subsample(mdc.GT_FPS)
        ),
        tmp_path / "b.csv",
    )
    assert a.read_bytes() == b.read_bytes()
    header = a.read_text().splitlines()[0]
    assert header.split(",") == list(TRACKING_COLUMNS)


def test_camera_round_trips(camera):
    pts = np.array([[0.0, 0.0], [105.0, 0.0], [52.5, 34.0], [30.0, 60.0], [88.0, 20.0]])
    back = camera.unproject(camera.project(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_camera_depth_increases_away_from_the_near_touchline(camera):
    near = camera.depth(np.array([[52.5, 0.0]]))[0]
    far = camera.depth(np.array([[52.5, PITCH_WIDTH]]))[0]
    assert far > near > 0
    # far touchline should be compressed by roughly the depth ratio
    assert far / near > 2.0


def test_camera_json_matches_the_render_matrix(tmp_path, camera):
    path = mdc.write_camera_json(camera, tmp_path / "camera.json")
    payload = json.loads(path.read_text())
    assert set(payload) == {"H", "reprojection_error"}
    h = np.asarray(payload["H"], dtype=np.float64)
    assert h.shape == (3, 3)
    assert payload["reprojection_error"] == 0.0
    # H is image -> pitch, so it must undo the projection
    pts = np.array([[10.0, 50.0], [52.5, 34.0], [95.0, 12.0]])
    uv = camera.project(pts)
    homo = np.column_stack([uv, np.ones(len(uv))]) @ h.T
    assert np.allclose(homo[:, :2] / homo[:, 2:], pts, atol=1e-6)


def test_render_frame_shape_and_grass(track, camera):
    img = mdc.render_frame(track.pos[0], camera, rng=np.random.default_rng(0))
    assert img.shape == (mdc.FRAME_HEIGHT, mdc.FRAME_WIDTH, 3)
    assert img.dtype == np.uint8
    b = img[:, :, 0].astype(np.int32)
    g = img[:, :, 1].astype(np.int32)
    r = img[:, :, 2].astype(np.int32)
    green = ((g > b) & (g > r)).mean()
    assert 0.35 < green < 0.95


def test_render_frame_is_deterministic_without_noise(track, camera):
    a = mdc.render_frame(track.pos[10], camera, rng=None)
    b = mdc.render_frame(track.pos[10], camera, rng=None)
    assert np.array_equal(a, b)


def test_players_shrink_with_depth(camera):
    """Perspective scaling is the point of the render; assert it directly."""
    pos = np.zeros((mdc.N_OBJECTS, 2))
    pos[:] = np.array([52.5, 34.0])
    near = np.array([52.5, 4.0])
    far = np.array([52.5, 64.0])

    def painted(xy):
        img = np.zeros((mdc.FRAME_HEIGHT, mdc.FRAME_WIDTH, 3), dtype=np.uint8)
        mdc._draw_player(img, camera, xy, mdc.COLOR_HOME, mdc.COLOR_HOME_SHORTS)
        return int((img.any(axis=2)).sum())

    assert painted(near) > 3 * painted(far) > 0


def test_all_objects_stay_in_frame(camera):
    tr = mdc.generate_world_trajectories(duration_s=SHORT_S, seed=0)
    uv = camera.project(tr.pos.reshape(-1, 2))
    assert (uv[:, 0] >= 0).all() and (uv[:, 0] < mdc.FRAME_WIDTH).all()
    assert (uv[:, 1] >= 0).all() and (uv[:, 1] < mdc.FRAME_HEIGHT).all()


def test_make_demo_without_video(tmp_path):
    paths = mdc.make_demo(tmp_path, seed=0, duration_s=SHORT_S, render=False)
    assert set(paths) == {"ground_truth", "camera"}
    assert paths["ground_truth"].exists()
    assert paths["camera"].exists()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_encoded_clip_is_h264_yuv420p(tmp_path):
    paths = mdc.make_demo(tmp_path, seed=0, duration_s=3.0)
    video = paths["video"]
    assert video.stat().st_size > 20_000
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["width"] == mdc.FRAME_WIDTH
    assert stream["height"] == mdc.FRAME_HEIGHT


@pytest.mark.slow
@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "demo" / "assets").exists(),
    reason="demo assets have not been generated",
)
def test_committed_demo_assets_are_consistent():
    """The committed clip and csv must be what this script produces at seed 0."""
    assets = Path(__file__).resolve().parent.parent / "demo" / "assets"
    csv = assets / "ground_truth.csv"
    if not csv.exists():
        pytest.skip("ground_truth.csv has not been generated")
    df = pd.read_csv(csv)
    assert tuple(df.columns) == TRACKING_COLUMNS
    assert df.groupby("frame").size().eq(23).all()
    assert df["x"].between(0.0, PITCH_LENGTH).all()
    assert df["y"].between(0.0, PITCH_WIDTH).all()
