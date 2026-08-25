"""Tests for the two-stage ByteTrack association."""

from __future__ import annotations

import numpy as np

from replay.schema import Detection, TrackedBox
from replay.track.byte_track import ByteTrack, iou_matrix

PLAYER_W = 24.0
PLAYER_H = 48.0


def make_det(
    frame: int,
    cx: float,
    cy: float,
    *,
    score: float = 0.9,
    cls: str = "player",
    w: float = PLAYER_W,
    h: float = PLAYER_H,
) -> Detection:
    return Detection(
        frame=frame,
        x1=cx - w / 2.0,
        y1=cy - h / 2.0,
        x2=cx + w / 2.0,
        y2=cy + h / 2.0,
        score=score,
        cls=cls,  # type: ignore[arg-type]
    )


def nearest_id(boxes: list[TrackedBox], cx: float, cy: float) -> int | None:
    """Track id of the emitted box whose center sits closest to (cx, cy)."""
    if not boxes:
        return None
    best = min(boxes, key=lambda b: (b.center[0] - cx) ** 2 + (b.center[1] - cy) ** 2)
    return best.track_id


def test_iou_matrix_basics():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    got = iou_matrix(a, b)
    assert got.shape == (1, 3)
    assert got[0, 0] == 1.0
    assert abs(got[0, 1] - 50.0 / 150.0) < 1e-12
    assert got[0, 2] == 0.0
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_two_objects_crossing_keep_their_ids():
    """Two players walk through each other; ids must not swap at the crossing."""
    rng = np.random.default_rng(0)
    tracker = ByteTrack(min_hits=3, max_age=30)

    a_ids: list[int] = []
    b_ids: list[int] = []
    for frame in range(24):
        a_cx, a_cy = 20.0 + 8.0 * frame, 100.0
        b_cx, b_cy = 180.0 - 8.0 * frame, 112.0
        jitter = rng.uniform(-1.0, 1.0, size=4)
        dets = [
            make_det(frame, a_cx + jitter[0], a_cy + jitter[1]),
            make_det(frame, b_cx + jitter[2], b_cy + jitter[3]),
        ]
        out = tracker.update(dets, frame)
        if frame < 3:
            continue
        assert len(out) == 2, f"frame {frame} emitted {len(out)} boxes"
        a_ids.append(nearest_id(out, a_cx, a_cy))
        b_ids.append(nearest_id(out, b_cx, b_cy))

    # The two objects are at the same cx around frame 10, so if the tracker
    # were going to swap it would happen there.
    assert len(set(a_ids)) == 1, f"left-to-right track changed id: {a_ids}"
    assert len(set(b_ids)) == 1, f"right-to-left track changed id: {b_ids}"
    assert a_ids[0] != b_ids[0]


def test_short_occlusion_reacquires_original_id():
    tracker = ByteTrack(min_hits=3, max_age=10)
    gap = range(12, 17)

    ids_before: list[int] = []
    ids_after: list[int] = []
    for frame in range(30):
        cx, cy = 30.0 + 6.0 * frame, 150.0
        dets = [] if frame in gap else [make_det(frame, cx, cy)]
        out = tracker.update(dets, frame)
        if frame in gap:
            assert out == [], "a track with no detection must not emit a box"
        elif frame < 12:
            if out:
                ids_before.append(out[0].track_id)
        elif out:
            ids_after.append(out[0].track_id)

    assert ids_before, "track never confirmed before the occlusion"
    assert ids_after, "track never came back after the occlusion"
    assert set(ids_before) == set(ids_after) == {ids_before[0]}


def test_occlusion_longer_than_max_age_gets_a_new_id():
    tracker = ByteTrack(min_hits=3, max_age=5)

    ids_before: list[int] = []
    ids_after: list[int] = []
    for frame in range(40):
        cx, cy = 30.0 + 6.0 * frame, 150.0
        present = frame < 10 or frame >= 25
        out = tracker.update([make_det(frame, cx, cy)] if present else [], frame)
        if not out:
            continue
        if frame < 10:
            ids_before.append(out[0].track_id)
        else:
            ids_after.append(out[0].track_id)

    assert ids_before and ids_after
    assert set(ids_before).isdisjoint(ids_after)
    assert min(ids_after) > max(ids_before), "ids must keep increasing"


def test_low_score_detections_keep_a_track_alive():
    """Stage 2: a stretch where every detection is below high_thresh."""
    tracker = ByteTrack(high_thresh=0.5, low_thresh=0.1, min_hits=3, max_age=30)
    weak = range(5, 20)

    seen: list[int] = []
    weak_frames_with_output = 0
    for frame in range(28):
        cx, cy = 40.0 + 4.0 * frame, 200.0
        score = 0.2 if frame in weak else 0.9
        out = tracker.update([make_det(frame, cx, cy, score=score)], frame)
        if out:
            seen.append(out[0].track_id)
            if frame in weak:
                weak_frames_with_output += 1

    assert len(set(seen)) == 1, f"track was lost or re-ided: {sorted(set(seen))}"
    assert weak_frames_with_output == len(weak), (
        "every low-score frame should have been recovered by the second association"
    )


def test_low_score_detections_alone_never_start_a_track():
    tracker = ByteTrack(high_thresh=0.5, low_thresh=0.1, min_hits=3, max_age=30)
    for frame in range(25):
        out = tracker.update([make_det(frame, 40.0 + 4.0 * frame, 200.0, score=0.2)], frame)
        assert out == []
    assert tracker.tracks == []


def test_detections_below_low_thresh_are_ignored():
    tracker = ByteTrack(high_thresh=0.5, low_thresh=0.1, min_hits=1, max_age=2)
    for frame in range(5):
        assert tracker.update([make_det(frame, 50.0, 50.0, score=0.9)], frame)
    for frame in range(5, 12):
        out = tracker.update([make_det(frame, 50.0, 50.0, score=0.05)], frame)
        assert out == []
    assert tracker.tracks == []


def test_ids_are_never_reused():
    """Track an object, let it die, then put a new one in the same place."""
    tracker = ByteTrack(min_hits=2, max_age=3)
    first_ids: set[int] = set()
    second_ids: set[int] = set()

    for frame in range(12):
        out = tracker.update([make_det(frame, 300.0, 300.0)], frame)
        first_ids.update(b.track_id for b in out)
    for frame in range(12, 25):
        assert tracker.update([], frame) == []
    for frame in range(25, 37):
        out = tracker.update([make_det(frame, 300.0, 300.0)], frame)
        second_ids.update(b.track_id for b in out)

    assert first_ids and second_ids
    assert first_ids.isdisjoint(second_ids)


def test_no_id_reappears_after_dying_in_a_busy_scene():
    tracker = ByteTrack(min_hits=2, max_age=4)
    last_seen: dict[int, int] = {}
    first_seen: dict[int, int] = {}

    rng = np.random.default_rng(5)
    for frame in range(80):
        dets: list[Detection] = []
        # A persistent object plus two that come and go on disjoint schedules.
        dets.append(make_det(frame, 40.0 + 3.0 * frame, 60.0))
        if frame % 20 < 9:
            dets.append(make_det(frame, 400.0 - 3.0 * frame, 200.0))
        if 30 <= frame < 44 or 60 <= frame < 72:
            dets.append(make_det(frame, 150.0 + rng.uniform(-1.0, 1.0), 320.0))
        out = tracker.update(dets, frame)
        for box in out:
            if box.track_id in last_seen:
                gap = frame - last_seen[box.track_id] - 1
                assert gap <= tracker.max_age, (
                    f"id {box.track_id} reappeared after {gap} missing frames"
                )
            else:
                first_seen[box.track_id] = frame
            last_seen[box.track_id] = frame

    assert len(first_seen) >= 4, "scenario should have created several tracks"
    order = [tid for tid, _ in sorted(first_seen.items(), key=lambda kv: (kv[1], kv[0]))]
    assert order == sorted(order), f"ids were not handed out monotonically: {order}"


def test_ball_is_never_absorbed_into_a_player_track():
    """The ball box overlaps the player enough to match on IoU alone."""
    tracker = ByteTrack(min_hits=3, max_age=30)
    ball_dets_only = range(10, 15)

    def player_center(frame: int) -> tuple[float, float]:
        return (112.0 + 1.0 * frame, 144.0)

    def ball_center(frame: int) -> tuple[float, float]:
        return (112.0 + 1.0 * frame, 160.0)

    # Sanity check the geometry: without the class gate stage 1 would happily
    # pair the ball detection with the player track.
    px, py = player_center(10)
    bx, by = ball_center(10)
    overlap = iou_matrix(
        np.array([[px - 12.0, py - 24.0, px + 12.0, py + 24.0]]),
        np.array([[bx - 10.0, by - 10.0, bx + 10.0, by + 10.0]]),
    )[0, 0]
    assert overlap > 1.0 - 0.8, "test geometry no longer exercises the class gate"

    player_ids: list[int] = []
    ball_ids: list[int] = []
    for frame in range(22):
        dets: list[Detection] = []
        if frame not in ball_dets_only:
            dets.append(make_det(frame, *player_center(frame)))
        if frame >= 10:
            dets.append(make_det(frame, *ball_center(frame), cls="ball", w=20.0, h=20.0))
        out = tracker.update(dets, frame)

        by_cls = {b.cls for b in out}
        if frame in ball_dets_only:
            assert "player" not in by_cls, "player track absorbed a ball detection"
        for box in out:
            if box.cls == "player":
                player_ids.append(box.track_id)
            else:
                ball_ids.append(box.track_id)

    assert len(set(player_ids)) == 1, f"player id changed: {sorted(set(player_ids))}"
    assert len(set(ball_ids)) == 1, f"ball id changed: {sorted(set(ball_ids))}"
    assert set(player_ids).isdisjoint(ball_ids)
    # The player must have survived the stretch where only the ball was visible
    # and resumed under its original id.
    assert player_ids[0] == player_ids[-1]


def test_ball_and_player_tracked_in_the_same_pass():
    tracker = ByteTrack(min_hits=2, max_age=10)
    out: list[TrackedBox] = []
    for frame in range(6):
        out = tracker.update(
            [
                make_det(frame, 100.0 + 2.0 * frame, 150.0),
                make_det(frame, 260.0 - 4.0 * frame, 155.0, cls="ball", w=12.0, h=12.0),
            ],
            frame,
        )
    assert sorted(b.cls for b in out) == ["ball", "player"]
    assert len({b.track_id for b in out}) == 2


def test_ids_start_at_one_and_output_is_sorted():
    tracker = ByteTrack(min_hits=1, max_age=5)
    out = tracker.update(
        [make_det(0, 500.0, 300.0), make_det(0, 100.0, 300.0), make_det(0, 300.0, 300.0)],
        0,
    )
    ids = [b.track_id for b in out]
    assert ids == sorted(ids)
    assert set(ids) == {1, 2, 3}
    assert all(b.frame == 0 for b in out)


def test_min_hits_suppresses_output_until_confirmed():
    tracker = ByteTrack(min_hits=4, max_age=5)
    emitted = []
    for frame in range(8):
        emitted.append(len(tracker.update([make_det(frame, 200.0 + frame, 200.0)], frame)))
    assert emitted[:3] == [0, 0, 0]
    assert all(n == 1 for n in emitted[3:])


def test_empty_update_is_safe():
    tracker = ByteTrack()
    assert tracker.update([], 0) == []
    assert tracker.tracks == []
