"""Tests for the colour detector, device resolution and detector construction."""

from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from replay.schema import Detection
from replay.track.detect import ColorDetector, YoloDetector, build_detector, resolve_device

PITCH_BGR = (40, 130, 40)
BLUE = (200, 60, 60)
RED = (50, 50, 200)
WHITE = (255, 255, 255)

PLAYER_CENTERS = [(60, 80), (140, 120), (220, 90), (270, 170)]
PLAYER_COLORS = [BLUE, BLUE, RED, RED]
BALL_CENTER = (180, 200)


def make_frame(
    *,
    height: int = 240,
    width: int = 320,
    with_ball: bool = True,
    specks: tuple[tuple[int, int], ...] = (),
    ring: tuple[int, int] | None = None,
):
    img = np.full((height, width, 3), PITCH_BGR, dtype=np.uint8)
    for (cx, cy), colour in zip(PLAYER_CENTERS, PLAYER_COLORS, strict=True):
        img[cy - 14 : cy + 15, cx - 7 : cx + 8] = colour
    if with_ball:
        cv2.circle(img, BALL_CENTER, 5, WHITE, -1)
    for sx, sy in specks:
        img[sy : sy + 2, sx : sx + 2] = WHITE
    if ring is not None:
        cv2.circle(img, ring, 15, RED, 3)
    return img


def contains(det: Detection, point: tuple[int, int]) -> bool:
    x, y = point
    return det.x1 <= x <= det.x2 and det.y1 <= y <= det.y2


def test_color_detector_finds_players_and_one_ball():
    det = ColorDetector()
    dets = det.detect(make_frame(), 7)

    players = [d for d in dets if d.cls == "player"]
    balls = [d for d in dets if d.cls == "ball"]
    assert len(players) == len(PLAYER_CENTERS)
    assert len(balls) == 1
    assert all(d.frame == 7 for d in dets)

    for center in PLAYER_CENTERS:
        assert any(contains(p, center) for p in players), center
    assert contains(balls[0], BALL_CENTER)


def test_boxes_are_tight_around_the_blobs():
    dets = ColorDetector().detect(make_frame(), 0)
    players = sorted((d for d in dets if d.cls == "player"), key=lambda d: d.center[0])
    for det, center in zip(players, sorted(PLAYER_CENTERS), strict=True):
        w, h = det.wh
        assert w == pytest.approx(15, abs=1)
        assert h == pytest.approx(29, abs=1)
        assert det.center == pytest.approx(center, abs=1.0)


def test_area_filter_rejects_noise_specks():
    specks = ((10, 10), (300, 220), (150, 30))
    dets = ColorDetector().detect(make_frame(specks=specks), 0)
    assert len(dets) == len(PLAYER_CENTERS) + 1
    for sx, sy in specks:
        assert not any(contains(d, (sx, sy)) for d in dets)


def test_min_area_above_every_blob_yields_nothing():
    assert ColorDetector(min_area=5000).detect(make_frame(), 0) == []


def test_ball_max_area_reclassifies_the_ball_as_a_player():
    dets = ColorDetector(ball_max_area=10).detect(make_frame(), 0)
    assert [d for d in dets if d.cls == "ball"] == []
    assert len(dets) == len(PLAYER_CENTERS) + 1


def test_score_tracks_blob_quality():
    dets = ColorDetector().detect(make_frame(with_ball=False, ring=(180, 190)), 0)
    solid = [d for d in dets if contains(d, PLAYER_CENTERS[0])]
    hollow = [d for d in dets if contains(d, (180, 190))]
    assert len(solid) == 1
    assert len(hollow) == 1
    assert all(0.0 < d.score <= 1.0 for d in dets)
    assert solid[0].score == pytest.approx(1.0, abs=1e-6)
    assert hollow[0].score < 0.6


def test_objects_outside_the_pitch_outline_are_ignored():
    img = make_frame()
    img[:60] = (70, 70, 70)  # stand, above the grass
    img[20:40, 100:160] = (30, 220, 240)  # a banner up in the stand
    dets = ColorDetector().detect(img, 0)
    assert not any(contains(d, (130, 30)) for d in dets)
    assert len([d for d in dets if d.cls == "player"]) == len(PLAYER_CENTERS)


def test_empty_and_all_green_frames_give_no_detections():
    det = ColorDetector()
    assert det.detect(np.full((60, 80, 3), PITCH_BGR, dtype=np.uint8), 0) == []
    assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8), 0) == []


def test_non_bgr_frame_raises():
    with pytest.raises(ValueError, match="BGR"):
        ColorDetector().detect(np.zeros((10, 10), dtype=np.uint8), 0)


def test_resolve_device_returns_a_valid_backend():
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}
    assert resolve_device() in {"cuda", "mps", "cpu"}
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("CUDA") == "cuda"


def test_build_detector_dispatch():
    det = build_detector("color", min_area=44)
    assert isinstance(det, ColorDetector)
    assert det.min_area == 44


def test_build_detector_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown detector"):
        build_detector("rcnn")  # type: ignore[arg-type]


def test_module_import_does_not_pull_in_ultralytics():
    code = "import sys; import replay.track.detect; assert 'ultralytics' not in sys.modules"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("REPLAY_NETWORK_TESTS"),
    reason="downloads YOLO weights; set REPLAY_NETWORK_TESTS=1 to run",
)
def test_yolo_detector_on_a_synthetic_frame():
    det = YoloDetector(weights="yolov8n.pt", device="cpu", conf=0.1)
    dets = det.detect(make_frame(), 3)
    assert all(d.cls in {"player", "ball"} for d in dets)
    assert all(d.frame == 3 for d in dets)
