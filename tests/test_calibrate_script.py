"""Tests for scripts/calibrate.py, the manual camera calibration fallback."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from replay.track.homography import PITCH_KEYPOINTS, Homography

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate.py"


def load_script():
    spec = importlib.util.spec_from_file_location("calibrate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


calib = load_script()

TRUE = Homography.from_json(
    Path(__file__).resolve().parents[1] / "demo" / "assets" / "camera.json"
)


def clicks(names: list[str]) -> dict[str, list[float]]:
    """Where those landmarks land in the image under the known camera."""
    return {
        n: TRUE.pitch_to_image(np.array([PITCH_KEYPOINTS[n]], dtype=np.float64))[0].tolist()
        for n in names
    }


SPREAD = [
    "halfway_top",
    "halfway_bottom",
    "penalty_box_left_front_top",
    "penalty_box_right_front_bottom",
]


def test_four_landmarks_recover_the_camera_they_came_from():
    picked = {k: tuple(v) for k, v in clicks(SPREAD).items()}
    h = calib.solve(picked)
    pitch = np.array(list(PITCH_KEYPOINTS.values()), dtype=np.float64)
    back = h.image_to_pitch(TRUE.pitch_to_image(pitch))
    assert np.nanmax(np.linalg.norm(back - pitch, axis=1)) < 1e-6


def test_more_landmarks_still_recover_it():
    names = SPREAD + ["center_circle_top", "center_circle_bottom", "penalty_spot_left"]
    h = calib.solve({k: tuple(v) for k, v in clicks(names).items()})
    assert h.reprojection_error < 1e-6


def test_fewer_than_four_landmarks_is_refused():
    with pytest.raises(SystemExit, match="at least 4"):
        calib.solve({k: tuple(v) for k, v in clicks(SPREAD[:3]).items()})


def test_unknown_landmark_names_are_named_back_to_the_user(tmp_path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps({"halfway_middle_ish": [10, 20], "corner_left_top": [1, 2]}))
    with pytest.raises(SystemExit, match="halfway_middle_ish"):
        calib.main(
            [
                str(Path(__file__).resolve().parents[1] / "demo" / "assets" / "demo_clip.mp4"),
                "--pairs",
                str(pairs),
                "--out",
                str(tmp_path / "camera.json"),
            ]
        )


def test_end_to_end_writes_a_camera_and_a_check_image(tmp_path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps(clicks(SPREAD)))
    out = tmp_path / "camera.json"
    rc = calib.main(
        [
            str(Path(__file__).resolve().parents[1] / "demo" / "assets" / "demo_clip.mp4"),
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--frame",
            "30",
        ]
    )
    assert rc == 0
    assert out.exists()
    check = out.with_name("camera_check.png")
    assert check.exists(), "a check image must always be written"
    img = cv2.imread(str(check))
    assert img is not None and img.size > 0

    reloaded = Homography.from_json(out)
    pitch = np.array(list(PITCH_KEYPOINTS.values()), dtype=np.float64)
    back = reloaded.image_to_pitch(TRUE.pitch_to_image(pitch))
    assert np.nanmax(np.linalg.norm(back - pitch, axis=1)) < 1e-6


def test_check_image_actually_draws_the_model(tmp_path):
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    out = calib.draw_check(frame, TRUE, tmp_path / "check.png")
    drawn = cv2.imread(str(out))
    # The model is drawn in yellow; on a black frame that is the only thing there.
    assert (drawn.sum(axis=2) > 0).any(), "nothing was drawn onto the frame"
