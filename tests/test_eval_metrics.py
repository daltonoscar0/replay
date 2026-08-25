"""CLEAR-MOT correctness tests, built from hand-reasoned toy sequences."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from replay.eval.metrics import AccuracyReport, evaluate
from replay.schema import coerce_tracking_frame, empty_tracking_frame


def build(records, *, match_id="toy"):
    """Canonical tracking frame from ``(frame, object_id, x, y)`` tuples."""
    rows = []
    for frame, obj_id, x, y in records:
        rows.append(
            {
                "match_id": match_id,
                "t": frame / 10.0,
                "period": 1,
                "frame": frame,
                "team": "home",
                "player_id": obj_id,
                "jersey": None,
                "x": float(x),
                "y": float(y),
                "vx": 0.0,
                "vy": 0.0,
                "confidence": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    df["jersey"] = pd.array(df["jersey"], dtype="Int64")
    return coerce_tracking_frame(df)


# Two static objects 10 m apart, far enough that a 2 m gate can never confuse
# them: any id switch in these fixtures is a real one.
A_XY = (10.0, 10.0)
B_XY = (20.0, 10.0)
N_FRAMES = 10


def two_object_gt():
    records = []
    for f in range(N_FRAMES):
        records.append((f, "gtA", *A_XY))
        records.append((f, "gtB", *B_XY))
    return build(records)


def test_perfect_prediction():
    gt = two_object_gt()
    pred = build(
        [(f, "p1", *A_XY) for f in range(N_FRAMES)] + [(f, "p2", *B_XY) for f in range(N_FRAMES)]
    )
    r = evaluate(gt, pred, dataset="perfect")

    assert isinstance(r, AccuracyReport)
    assert r.n_frames == N_FRAMES
    assert r.n_gt == 2 * N_FRAMES
    assert r.n_pred == 2 * N_FRAMES
    assert r.misses == 0
    assert r.false_positives == 0
    assert r.id_switches == 0
    assert r.mota == pytest.approx(1.0)
    assert r.motp == pytest.approx(0.0)
    assert r.rmse_m == pytest.approx(0.0)
    assert r.median_error_m == pytest.approx(0.0)
    assert r.idf1 == pytest.approx(1.0)


def test_deliberate_id_swap_counts_exactly_two_switches():
    """Halfway through, the two prediction ids trade places.

    Frames 0-4 have p1 on A and p2 on B; frames 5-9 have p1 on B and p2 on A.
    At frame 5 neither previous pair survives the gate (they are 10 m apart), so
    the assignment re-pairs both objects and each one changes its last-known
    prediction id: exactly two switches, and none afterwards because the new
    pairing then persists.
    """
    gt = two_object_gt()
    pred_records = []
    for f in range(N_FRAMES):
        if f < 5:
            pred_records.append((f, "p1", *A_XY))
            pred_records.append((f, "p2", *B_XY))
        else:
            pred_records.append((f, "p1", *B_XY))
            pred_records.append((f, "p2", *A_XY))
    r = evaluate(gt, build(pred_records), dataset="swap")

    assert r.id_switches == 2
    assert r.misses == 0
    assert r.false_positives == 0
    assert r.mota == pytest.approx(1.0 - 2 / 20)
    assert r.motp == pytest.approx(0.0)
    # Each gt id is explained by either prediction id for exactly half the
    # sequence, so the best global identity assignment recovers 10 of 20
    # detections: IDF1 = 2*10 / (2*10 + 10 + 10) = 0.5.
    assert r.idf1 == pytest.approx(0.5)


def test_close_pass_does_not_invent_switches():
    """Match preservation, the detail that makes or breaks the switch count.

    Two objects sit 1 m apart. On frame 2 the predictions wobble so that the
    *crossed* pairing is strictly cheaper in raw distance. A tracker that simply
    ran Hungarian on that frame would swap both objects and then swap them back,
    inventing four id switches. CLEAR-MOT keeps a pair that was matched last
    frame and is still inside the gate, so the correct pairing survives.
    """
    gt = build(
        [(f, "gtA", 10.0, 10.0) for f in range(5)] + [(f, "gtB", 11.0, 10.0) for f in range(5)]
    )
    pred_records = []
    for f in range(5):
        if f == 2:
            pred_records.append((f, "p1", 10.9, 10.0))
            pred_records.append((f, "p2", 10.2, 10.0))
        else:
            pred_records.append((f, "p1", 10.0, 10.0))
            pred_records.append((f, "p2", 11.0, 10.0))

    # The counterfactual: on frame 2 the distance-optimal assignment is crossed.
    frame2 = np.array([[0.9, 0.2], [0.1, 0.8]])
    _, cols = linear_sum_assignment(frame2)
    assert list(cols) == [1, 0]

    r = evaluate(gt, build(pred_records), dataset="close-pass")
    assert r.id_switches == 0
    assert r.misses == 0
    assert r.false_positives == 0
    assert r.mota == pytest.approx(1.0)
    assert r.motp == pytest.approx((0.9 + 0.8) / 10)


def test_gap_and_return_with_same_id_is_not_a_switch():
    """B vanishes for three frames and comes back on the same prediction id."""
    gt = two_object_gt()
    pred_records = [(f, "p1", *A_XY) for f in range(N_FRAMES)]
    pred_records += [(f, "p2", *B_XY) for f in range(N_FRAMES) if f not in (4, 5, 6)]
    r = evaluate(gt, build(pred_records), dataset="gap")

    assert r.id_switches == 0
    assert r.misses == 3
    assert r.false_positives == 0
    assert r.mota == pytest.approx(1.0 - 3 / 20)


def test_dropped_detections_count_as_misses():
    gt = build(
        [(f, "gtA", *A_XY) for f in range(N_FRAMES)]
        + [(f, "gtB", *B_XY) for f in range(N_FRAMES)]
        + [(f, "gtC", 30.0, 10.0) for f in range(N_FRAMES)]
    )
    pred = build(
        [(f, "p1", *A_XY) for f in range(N_FRAMES)] + [(f, "p2", *B_XY) for f in range(N_FRAMES)]
    )
    r = evaluate(gt, pred, dataset="drops")

    assert r.n_gt == 3 * N_FRAMES
    assert r.misses == N_FRAMES
    assert r.false_positives == 0
    assert r.id_switches == 0
    assert r.mota == pytest.approx(1.0 - 10 / 30)
    # gtC is never explained by any prediction: IDTP = 20, IDFN = 10, IDFP = 0.
    assert r.idf1 == pytest.approx(2 * 20 / (2 * 20 + 0 + 10))


def test_spurious_detections_count_as_false_positives_and_mota_goes_negative():
    gt = two_object_gt()
    extras = [(f, f"ghost{k}", 50.0 + 10.0 * k, 50.0) for f in range(N_FRAMES) for k in range(3)]
    pred = build(
        [(f, "p1", *A_XY) for f in range(N_FRAMES)]
        + [(f, "p2", *B_XY) for f in range(N_FRAMES)]
        + extras
    )
    r = evaluate(gt, pred, dataset="ghosts")

    assert r.false_positives == 3 * N_FRAMES
    assert r.misses == 0
    assert r.id_switches == 0
    # 30 false positives against 20 ground-truth objects. MOTA is unclamped.
    assert r.mota == pytest.approx(1.0 - 30 / 20)
    assert r.mota < 0


def test_prediction_outside_gate_is_miss_plus_false_positive():
    gt = build([(f, "gtA", 10.0, 10.0) for f in range(5)])
    pred = build([(f, "p1", 13.0, 10.0) for f in range(5)])
    r = evaluate(gt, pred, dataset="ungated", match_dist_m=2.0)

    assert r.misses == 5
    assert r.false_positives == 5
    assert r.id_switches == 0
    assert r.mota == pytest.approx(1.0 - 10 / 5)
    assert math.isnan(r.motp)
    assert math.isnan(r.rmse_m)
    assert r.idf1 == pytest.approx(0.0)

    # Widen the gate past 3 m and the same pair becomes a clean match.
    wide = evaluate(gt, pred, dataset="ungated", match_dist_m=4.0)
    assert wide.misses == 0
    assert wide.false_positives == 0
    assert wide.motp == pytest.approx(3.0)
    assert wide.rmse_m == pytest.approx(3.0)
    assert wide.idf1 == pytest.approx(1.0)


def test_empty_prediction_is_all_misses():
    gt = two_object_gt()
    r = evaluate(gt, empty_tracking_frame(), dataset="empty-pred")

    assert r.n_pred == 0
    assert r.misses == 2 * N_FRAMES
    assert r.false_positives == 0
    assert r.id_switches == 0
    assert r.mota == pytest.approx(0.0)
    assert math.isnan(r.motp)
    assert math.isnan(r.median_error_m)
    assert r.idf1 == pytest.approx(0.0)


def test_empty_ground_truth_raises():
    pred = build([(0, "p1", *A_XY)])
    with pytest.raises(ValueError, match="nothing to evaluate against"):
        evaluate(empty_tracking_frame(), pred, dataset="empty-gt")


def test_rmse_is_over_matched_pairs_only():
    gt = build([(0, "gtA", 10.0, 10.0), (1, "gtA", 10.0, 10.0), (2, "gtA", 10.0, 10.0)])
    # Errors of 0, 1 and 2 m -> RMSE sqrt(5/3), MOTP 1.0, median 1.0.
    pred = build([(0, "p1", 10.0, 10.0), (1, "p1", 11.0, 10.0), (2, "p1", 12.0, 10.0)])
    r = evaluate(gt, pred, dataset="errors", match_dist_m=2.5)

    assert r.motp == pytest.approx(1.0)
    assert r.median_error_m == pytest.approx(1.0)
    assert r.rmse_m == pytest.approx(math.sqrt(5.0 / 3.0))


def test_rows_without_position_are_dropped_and_noted():
    gt = two_object_gt()
    pred_records = [(f, "p1", *A_XY) for f in range(N_FRAMES)]
    pred_records += [(f, "p2", *B_XY) for f in range(N_FRAMES)]
    pred = build(pred_records)
    pred.loc[0, "x"] = np.nan

    r = evaluate(gt, pred, dataset="nan-rows")
    assert r.n_pred == 2 * N_FRAMES - 1
    assert r.misses == 1
    assert "dropped" in r.notes


def test_missing_columns_raise():
    gt = two_object_gt()
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate(gt.drop(columns=["x"]), gt, dataset="broken")


def test_bad_gate_raises():
    gt = two_object_gt()
    with pytest.raises(ValueError, match="match_dist_m must be positive"):
        evaluate(gt, gt, dataset="bad-gate", match_dist_m=0.0)


def test_to_row_exposes_every_number():
    gt = two_object_gt()
    row = evaluate(gt, gt, dataset="rows").to_row()
    assert set(row) == {
        "dataset",
        "n_frames",
        "n_gt",
        "n_pred",
        "mota",
        "motp_m",
        "idf1",
        "id_switches",
        "false_positives",
        "misses",
        "rmse_m",
        "median_error_m",
        "notes",
    }
    assert row["dataset"] == "rows"
    assert row["mota"] == pytest.approx(1.0)


def test_row_order_does_not_change_the_result():
    gt = two_object_gt()
    pred = build(
        [(f, "p1", *A_XY) for f in range(N_FRAMES)] + [(f, "p2", *B_XY) for f in range(N_FRAMES)]
    )
    shuffled = pred.iloc[::-1].reset_index(drop=True)
    assert evaluate(gt, pred, dataset="d").to_row() == evaluate(gt, shuffled, dataset="d").to_row()
