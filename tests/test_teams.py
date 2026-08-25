"""Tests for jersey-colour team assignment."""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest

from replay.schema import TrackedBox
from replay.track.teams import assign_teams, circular_median_hue

PITCH_BGR = (40, 130, 40)
FRAME_H, FRAME_W = 240, 480
N_FRAMES = 8


def bgr_from_hsv(h: int, s: int = 230, v: int = 200) -> tuple[int, int, int]:
    px = np.array([[[h, s, v]]], dtype=np.uint8)
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def build_clip(specs, n_frames: int = N_FRAMES):
    """specs: sequence of (track_id, x, colour_bgr, cls). Boxes are static."""
    frames: dict[int, np.ndarray] = {}
    boxes_by_frame: dict[int, list[TrackedBox]] = {}
    for f in range(n_frames):
        img = np.full((FRAME_H, FRAME_W, 3), PITCH_BGR, dtype=np.uint8)
        boxes: list[TrackedBox] = []
        for track_id, x, colour, cls in specs:
            x1, y1, x2, y2 = x, 80, x + 24, 140
            img[y1:y2, x1:x2] = colour
            boxes.append(
                TrackedBox(
                    frame=f,
                    track_id=track_id,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=0.9,
                    cls=cls,
                )
            )
        frames[f] = img
        boxes_by_frame[f] = boxes
    return frames, boxes_by_frame


def reader(frames):
    def _read(frame_idx: int):
        return frames.get(frame_idx)

    return _read


def two_team_specs():
    blue = bgr_from_hsv(110)
    yellow = bgr_from_hsv(25)
    specs = [(i + 1, 20 + 40 * i, blue, "player") for i in range(3)]
    specs += [(i + 4, 300 + 40 * i, yellow, "player") for i in range(3)]
    return specs


def test_two_colours_split_cleanly_with_high_confidence():
    specs = two_team_specs()
    frames, boxes = build_clip(specs)
    result = assign_teams(boxes, reader(frames))

    assert set(result) == {1, 2, 3, 4, 5, 6}
    left = {result[i][0] for i in (1, 2, 3)}
    right = {result[i][0] for i in (4, 5, 6)}
    assert len(left) == 1
    assert len(right) == 1
    assert left != right
    assert all(conf > 0.9 for _, conf in result.values())


def test_lower_mean_x_cluster_is_home():
    frames, boxes = build_clip(two_team_specs())
    result = assign_teams(boxes, reader(frames))
    assert result[1][0] == "home"
    assert result[6][0] == "away"


def test_red_team_is_not_split_across_the_hue_wrap():
    """The regression test for a linear hue mean: red straddles the 0/180 seam."""
    reds = [177, 179, 1, 3]
    blue = bgr_from_hsv(110)
    specs = [(i + 1, 20 + 40 * i, bgr_from_hsv(h), "player") for i, h in enumerate(reds)]
    specs += [(i + 5, 260 + 40 * i, blue, "player") for i in range(4)]
    frames, boxes = build_clip(specs)
    result = assign_teams(boxes, reader(frames))

    red_labels = {result[i][0] for i in (1, 2, 3, 4)}
    blue_labels = {result[i][0] for i in (5, 6, 7, 8)}
    assert len(red_labels) == 1, f"red team split across clusters: {result}"
    assert len(blue_labels) == 1
    assert red_labels != blue_labels


def test_circular_median_hue_handles_the_wrap():
    hues = np.array([178, 179, 0, 1, 2], dtype=np.float64)
    got = circular_median_hue(hues)
    assert min(got, 180.0 - got) < 2.0

    # No wrap involved: it must still behave like a plain median.
    assert circular_median_hue(np.array([50.0, 52.0, 54.0])) == pytest.approx(52.0, abs=0.5)

    # Outliers must not drag the summary the way a mean would.
    plain = np.concatenate([np.full(20, 100.0), np.full(2, 20.0)])
    assert circular_median_hue(plain) == pytest.approx(100.0, abs=1.0)

    assert np.isnan(circular_median_hue(np.array([])))


def test_greyish_tracks_get_low_confidence():
    blue = bgr_from_hsv(110)
    orange = bgr_from_hsv(20)
    grey = (200, 200, 200)
    specs = [(i + 1, 10 + 30 * i, blue, "player") for i in range(5)]
    specs += [(i + 6, 300 + 30 * i, orange, "player") for i in range(5)]
    specs += [(11, 190, grey, "player"), (12, 220, grey, "player")]
    frames, boxes = build_clip(specs)
    result = assign_teams(boxes, reader(frames))

    team_conf = [result[i][1] for i in range(1, 11)]
    grey_conf = [result[i][1] for i in (11, 12)]
    assert min(team_conf) > 0.6
    assert max(grey_conf) < 0.3
    assert all(0.0 <= c <= 1.0 for _, c in result.values())
    # Ambiguous tracks are still labelled; the confidence is the caller's cue.
    assert all(result[i][0] in {"home", "away"} for i in (11, 12))


def test_ball_tracks_short_circuit():
    specs = two_team_specs()
    specs.append((99, 200, (255, 255, 255), "ball"))
    frames, boxes = build_clip(specs)
    result = assign_teams(boxes, reader(frames))

    assert result[99] == ("ball", 1.0)
    assert {result[i][0] for i in range(1, 7)} == {"home", "away"}


def test_deterministic_across_calls_and_box_order():
    specs = two_team_specs()
    frames, boxes = build_clip(specs)
    first = assign_teams(boxes, reader(frames))
    second = assign_teams(boxes, reader(frames))
    assert first == second

    rng = random.Random(0)
    shuffled = {f: rng.sample(v, len(v)) for f, v in boxes.items()}
    assert assign_teams(shuffled, reader(frames)) == first


def test_missing_frames_are_tolerated():
    specs = two_team_specs()
    frames, boxes = build_clip(specs)
    full = assign_teams(boxes, reader(frames))

    partial = {f: img for f, img in frames.items() if f % 2 == 0}
    assert assign_teams(boxes, reader(partial)) == full


def test_all_frames_unreadable_still_labels_every_track():
    specs = two_team_specs()
    _, boxes = build_clip(specs)
    result = assign_teams(boxes, lambda _idx: None)
    assert set(result) == {1, 2, 3, 4, 5, 6}
    assert all(value == ("home", 0.0) for value in result.values())


def test_single_track_gets_a_label_without_clustering():
    specs = [(1, 100, bgr_from_hsv(110), "player")]
    frames, boxes = build_clip(specs)
    result = assign_teams(boxes, reader(frames))
    assert result == {1: ("home", 0.0)}


def test_empty_input():
    assert assign_teams({}, lambda _idx: None) == {}


def test_n_samples_limits_frame_reads():
    specs = two_team_specs()
    frames, boxes = build_clip(specs, n_frames=30)
    seen: set[int] = set()

    def counting_reader(frame_idx: int):
        seen.add(frame_idx)
        return frames.get(frame_idx)

    assign_teams(boxes, counting_reader, n_samples=4)
    assert len(seen) <= 4
