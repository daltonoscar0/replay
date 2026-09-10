"""Tests for the parts of the pipeline that decide what reaches the output.

The ball is what these are really about. Every one of them is a regression test
for a bug that produced confident, plausible, wrong tracking rather than an
error: a penalty spot tracked as the ball for a whole clip, and a ball
interpolated across a gap it was never observed in.
"""

from __future__ import annotations

import numpy as np
import pytest

from replay.schema import Detection, TrackedBox
from replay.track.homography import Homography
from replay.track.pipeline import (
    BALL_MIN_PATH_M,
    BALL_STATIC_CELL_PX,
    _resample_to_fps,
    _resolve_ball,
    _track_ball,
    to_tracking_frame,
)

# Identity homography: pitch meters equal image pixels, so a track's path length
# in meters is whatever the test wrote in pixels.
IDENTITY = Homography(np.eye(3))


def obs(frames, xs, ys, *, cls="ball", score=0.9):
    return [(f, float(x), float(y), score, cls) for f, x, y in zip(frames, xs, ys, strict=True)]


def still(frames, x=50.0, y=30.0, **kw):
    return obs(frames, [x] * len(frames), [y] * len(frames), **kw)


def moving(frames, x0=10.0, step=1.0, **kw):
    xs = [x0 + step * i for i in range(len(frames))]
    return obs(frames, xs, [30.0] * len(frames), **kw)


def test_stationary_ball_candidate_is_discarded_however_long_it_lives():
    # The original bug. A penalty spot never moves, so it tracks cleanly for the
    # whole clip while the real ball fragments, and any rule that prefers the
    # longest-lived track picks the paint every time.
    by_track = {1: still(range(500)), 2: moving(range(20))}
    _resolve_ball(by_track, IDENTITY)
    assert 1 not in by_track, "the 500-frame stationary candidate must not survive"
    assert len(by_track) == 1


def test_moving_fragments_are_merged_rather_than_one_being_chosen():
    by_track = {1: moving(range(0, 10)), 2: moving(range(20, 30), x0=40.0)}
    _resolve_ball(by_track, IDENTITY)
    assert len(by_track) == 1
    (merged,) = by_track.values()
    assert [o[0] for o in merged] == list(range(0, 10)) + list(range(20, 30))


def test_overlapping_fragments_resolve_in_favour_of_the_longer_one():
    short = obs([5], [99.0], [99.0])
    long_ = moving(range(0, 12))
    by_track = {1: short, 2: long_}
    _resolve_ball(by_track, IDENTITY)
    (merged,) = by_track.values()
    at_five = next(o for o in merged if o[0] == 5)
    assert at_five[1] != 99.0


def test_a_track_below_the_path_threshold_counts_as_stationary():
    crawl = obs(range(10), np.linspace(0.0, BALL_MIN_PATH_M * 0.5, 10), [30.0] * 10)
    by_track = {1: crawl}
    _resolve_ball(by_track, IDENTITY)
    assert by_track == {}


def test_players_are_never_touched_by_ball_resolution():
    by_track = {1: still(range(50), cls="player"), 2: moving(range(20))}
    _resolve_ball(by_track, IDENTITY)
    assert 1 in by_track, "a stationary player is a player standing still, not paint"


def test_without_a_homography_no_ball_is_emitted():
    # Movement is judged in meters. With no way to reach meters, guessing from
    # pixels would reintroduce exactly the bug this gate exists to stop.
    by_track = {1: moving(range(20))}
    _resolve_ball(by_track, None)
    assert by_track == {}


def test_recurring_image_cells_are_rejected_before_tracking():
    # A mark sits in one cell all clip; the ball is somewhere new each frame.
    n = 100
    mark = (300.0, 200.0)
    ball_dets = {}
    for f in range(n):
        ball_dets[f] = [
            Detection(frame=f, x1=mark[0], y1=mark[1], x2=mark[0] + 6, y2=mark[1] + 6,
                      score=0.9, cls="ball"),
            Detection(frame=f, x1=10.0 + 4 * f, y1=50.0, x2=16.0 + 4 * f, y2=56.0,
                      score=0.9, cls="ball"),
        ]
    tracked = _track_ball(ball_dets)
    seen = [b.center for boxes in tracked.values() for b in boxes]
    assert seen, "the moving candidate should still be tracked"
    for cu, cv in seen:
        in_mark_cell = (
            round(cu / BALL_STATIC_CELL_PX) == round((mark[0] + 3) / BALL_STATIC_CELL_PX)
            and round(cv / BALL_STATIC_CELL_PX) == round((mark[1] + 3) / BALL_STATIC_CELL_PX)
        )
        assert not in_mark_cell


def test_ball_track_ids_cannot_collide_with_player_track_ids():
    ball_dets = {
        f: [Detection(frame=f, x1=10.0 + 4 * f, y1=50.0, x2=16.0 + 4 * f, y2=56.0,
                      score=0.9, cls="ball")]
        for f in range(30)
    }
    ids = {b.track_id for boxes in _track_ball(ball_dets).values() for b in boxes}
    assert ids and min(ids) > 1000, "ball ids must be offset clear of the player tracker's"


def test_empty_input_is_handled():
    assert _track_ball({}) == {}


class TestResampleGap:
    t = np.array([0.0, 0.1, 0.2, 2.0, 2.1])
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [50.0, 0.0], [51.0, 0.0]])

    def test_wide_gaps_are_not_bridged(self):
        target = np.arange(0.0, 2.11, 0.1)
        out = _resample_to_fps(self.t, self.xy, target, max_gap_s=0.3)
        # Inside the observed runs the trajectory is present...
        assert np.isfinite(out[0]).all()
        assert np.isfinite(out[-1]).all()
        # ...and across the 1.8 s hole it is not invented.
        mid = (target > 0.25) & (target < 1.95)
        assert not np.isfinite(out[mid]).any()

    def test_unbounded_gap_keeps_the_old_straight_line_behaviour(self):
        target = np.arange(0.0, 2.11, 0.1)
        out = _resample_to_fps(self.t, self.xy, target)
        assert np.isfinite(out).all()

    def test_narrow_gaps_are_still_bridged(self):
        t = np.array([0.0, 0.1, 0.3])
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        out = _resample_to_fps(t, xy, np.array([0.2]), max_gap_s=0.3)
        assert out[0, 0] == pytest.approx(2.0)


def _box(frame, track_id, x, cls):
    return TrackedBox(
        frame=frame, track_id=track_id, x1=x, y1=30.0, x2=x + 4.0, y2=34.0,
        score=0.9, cls=cls,
    )


def test_pipeline_bridges_a_player_across_a_gap_but_not_the_ball():
    # A gap of 0.6 s at 25 fps: longer than a ball may be interpolated over,
    # shorter than a player may. The two classes must be treated differently,
    # and this is the wiring rather than the arithmetic.
    seen = list(range(0, 6)) + list(range(21, 27))
    boxes: dict[int, list[TrackedBox]] = {f: [] for f in range(0, 27)}
    for f in seen:
        boxes[f].append(_box(f, 1, 10.0 + f, "player"))
        boxes[f].append(_box(f, 2, 60.0 + f, "ball"))

    df = to_tracking_frame(
        boxes, IDENTITY, {}, video_fps=25.0, out_fps=10.0, match_id="t", smooth=False
    )
    gap_t = 0.4  # inside the hole, which runs from 0.2 s to 0.84 s
    in_gap = df[np.isclose(df["t"], gap_t)]
    kinds = set(in_gap["team"].astype(str))
    assert "ball" not in kinds, "the ball must not be interpolated across 0.6 s"
    assert kinds, "the player should still be bridged across the same gap"
