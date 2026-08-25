"""CLEAR-MOT and IDF1 accuracy metrics over canonical tracking frames.

Ground truth and predictions are both canonical tracking frames (see
``replay.schema``). Object identity is the ``player_id`` column, position is
``(x, y)`` in pitch meters, and association is per-frame Hungarian assignment on
Euclidean distance with a hard gate.

Matching ignores ``team`` and any notion of object class: a prediction is free
to match any ground-truth object within the gate. Two meters is small enough on
a 105x68 pitch that a ball rarely steals a player's match, and keeping the
association class-blind means a tracker is not rewarded for guessing a label it
never had to commit to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

__all__ = ["AccuracyReport", "evaluate"]

#: Finite stand-in for "not allowed" in the assignment cost matrix. Infinity
#: makes ``linear_sum_assignment`` raise on infeasible problems, so gated-out
#: pairs get a cost that is merely enormous and are filtered out afterwards.
_FORBIDDEN = 1e9

_REQUIRED_COLUMNS = ("frame", "player_id", "x", "y")


@dataclass(slots=True)
class AccuracyReport:
    """Accuracy of one prediction set against one ground-truth set.

    ``motp`` is the mean Euclidean error of matched pairs in meters, so lower is
    better. Some papers report MOTP as a normalized similarity in ``[0, 1]``
    where higher is better; this is not that. ``mota`` is unclamped and goes
    negative when false positives outnumber ground-truth objects.
    """

    dataset: str
    n_frames: int
    n_gt: int
    n_pred: int
    mota: float
    motp: float
    idf1: float
    id_switches: int
    false_positives: int
    misses: int
    rmse_m: float
    median_error_m: float
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        """Flat mapping for tabulation. The only supported way to read numbers out."""
        return {
            "dataset": self.dataset,
            "n_frames": self.n_frames,
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "mota": self.mota,
            "motp_m": self.motp,
            "idf1": self.idf1,
            "id_switches": self.id_switches,
            "false_positives": self.false_positives,
            "misses": self.misses,
            "rmse_m": self.rmse_m,
            "median_error_m": self.median_error_m,
            "notes": self.notes,
        }


def _frames_by_index(
    df: pd.DataFrame, *, role: str
) -> tuple[dict[int, tuple[list[str], np.ndarray]], int, list[str]]:
    """Bucket a tracking frame into ``{frame: (ids, xy)}``.

    Returns the buckets, the number of usable object-frames, and any notes about
    rows that had to be discarded. Rows without an identity or without a finite
    position cannot participate in an assignment, and silently treating them as
    misses would hide a broken upstream stage, so they are counted and reported.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{role} frame is missing required columns: {missing}")

    sub = df.loc[:, list(_REQUIRED_COLUMNS)].copy()
    sub["x"] = pd.to_numeric(sub["x"], errors="coerce").astype(np.float64)
    sub["y"] = pd.to_numeric(sub["y"], errors="coerce").astype(np.float64)
    usable = sub["player_id"].notna() & np.isfinite(sub["x"]) & np.isfinite(sub["y"])

    notes: list[str] = []
    n_dropped = int((~usable).sum())
    if n_dropped:
        notes.append(f"{n_dropped} {role} row(s) dropped for missing id or position")
    sub = sub.loc[usable]

    sub["player_id"] = sub["player_id"].astype(str)
    sub["frame"] = sub["frame"].astype(np.int64)

    duplicated = sub.duplicated(subset=["frame", "player_id"], keep="first")
    n_dupes = int(duplicated.sum())
    if n_dupes:
        notes.append(f"{n_dupes} duplicate {role} (frame, player_id) row(s) ignored")
        sub = sub.loc[~duplicated]

    # Sort so the assignment sees a stable row order regardless of input order.
    sub = sub.sort_values(["frame", "player_id"], kind="stable")

    buckets: dict[int, tuple[list[str], np.ndarray]] = {}
    for frame, group in sub.groupby("frame", sort=True):
        ids = group["player_id"].tolist()
        xy = group.loc[:, ["x", "y"]].to_numpy(dtype=np.float64)
        buckets[int(frame)] = (ids, xy)
    return buckets, int(len(sub)), notes


def _pairwise_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between two ``(N, 2)`` / ``(M, 2)`` arrays."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _idf1(
    potential: dict[tuple[str, str], int], n_gt: int, n_pred: int
) -> tuple[float, int, int, int]:
    """Global identity matching, then IDF1.

    IDF1 is not a per-frame quantity: it asks which single prediction id best
    explains each ground-truth id over the whole sequence. The cost matrix holds
    the number of frames in which a (gt, pred) pair sits within the gate, and the
    assignment maximizes that total. Forced pairings on a rectangular matrix are
    harmless because an unmatched identity contributes zero and every pairing
    contributes at least zero.
    """
    if not potential or n_gt == 0:
        return 0.0, 0, n_pred, n_gt

    gt_ids = sorted({g for g, _ in potential})
    pred_ids = sorted({p for _, p in potential})
    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    gt_pos = {g: i for i, g in enumerate(gt_ids)}
    pred_pos = {p: j for j, p in enumerate(pred_ids)}
    for (g, p), n in potential.items():
        counts[gt_pos[g], pred_pos[p]] = float(n)

    rows, cols = linear_sum_assignment(counts, maximize=True)
    idtp = int(round(float(counts[rows, cols].sum())))
    idfn = n_gt - idtp
    idfp = n_pred - idtp
    denom = 2 * idtp + idfp + idfn
    idf1 = (2.0 * idtp / denom) if denom > 0 else 0.0
    return idf1, idtp, idfp, idfn


def evaluate(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    dataset: str,
    match_dist_m: float = 2.0,
) -> AccuracyReport:
    """Score ``pred`` against ``gt`` with CLEAR-MOT plus IDF1.

    ``match_dist_m`` is the association gate in meters: a prediction further than
    this from a ground-truth object can never match it, and so becomes a false
    positive while the ground-truth object becomes a miss.
    """
    if match_dist_m <= 0:
        raise ValueError(f"match_dist_m must be positive, got {match_dist_m}")

    gt_frames, n_gt, gt_notes = _frames_by_index(gt, role="gt")
    pred_frames, n_pred, pred_notes = _frames_by_index(pred, role="pred")

    if n_gt == 0:
        raise ValueError(
            "ground truth contains no usable rows; there is nothing to evaluate against. "
            "Check that the gt frame has finite x/y and a non-null player_id."
        )

    frames = sorted(set(gt_frames) | set(pred_frames))

    misses = 0
    false_positives = 0
    id_switches = 0
    distances: list[float] = []
    potential: dict[tuple[str, str], int] = {}

    # gt id -> pred id matched in the immediately preceding frame.
    prev_matches: dict[str, str] = {}
    # gt id -> the last pred id it was ever matched to, however long ago.
    last_pred_for_gt: dict[str, str] = {}

    empty_xy = np.zeros((0, 2), dtype=np.float64)

    for frame in frames:
        g_ids, g_xy = gt_frames.get(frame, ([], empty_xy))
        p_ids, p_xy = pred_frames.get(frame, ([], empty_xy))

        if g_ids and p_ids:
            dist = _pairwise_distance(g_xy, p_xy)
        else:
            dist = np.zeros((len(g_ids), len(p_ids)), dtype=np.float64)

        # Accumulate the identity co-occurrence counts IDF1 needs. This is every
        # pair inside the gate, not just the pairs the frame ends up assigning.
        for i in range(len(g_ids)):
            for j in range(len(p_ids)):
                if dist[i, j] <= match_dist_m:
                    key = (g_ids[i], p_ids[j])
                    potential[key] = potential.get(key, 0) + 1

        gt_pos = {g: i for i, g in enumerate(g_ids)}
        pred_pos = {p: j for j, p in enumerate(p_ids)}
        matches: list[tuple[str, str, float]] = []
        taken_gt: set[str] = set()
        taken_pred: set[str] = set()

        # CLEAR-MOT preserves existing correspondences before it optimizes.
        # A pair that was matched last frame and is still inside the gate keeps
        # its pairing even when some other assignment would be cheaper. Without
        # this, two objects passing close to each other get re-optimized into the
        # crossed pairing and the sequence collects two phantom id switches every
        # time players cross, which is most of a football match.
        for g_id, p_id in prev_matches.items():
            i = gt_pos.get(g_id)
            j = pred_pos.get(p_id)
            if i is None or j is None:
                continue
            d = float(dist[i, j])
            if d <= match_dist_m:
                matches.append((g_id, p_id, d))
                taken_gt.add(g_id)
                taken_pred.add(p_id)

        rem_g = [i for i, g in enumerate(g_ids) if g not in taken_gt]
        rem_p = [j for j, p in enumerate(p_ids) if p not in taken_pred]
        if rem_g and rem_p:
            sub = dist[np.ix_(rem_g, rem_p)]
            cost = np.where(sub <= match_dist_m, sub, _FORBIDDEN)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                d = float(sub[r, c])
                if d <= match_dist_m:
                    matches.append((g_ids[rem_g[r]], p_ids[rem_p[c]], d))

        misses += len(g_ids) - len(matches)
        false_positives += len(p_ids) - len(matches)

        new_prev: dict[str, str] = {}
        for g_id, p_id, d in matches:
            distances.append(d)
            # An id switch is a change against the last id this gt object was
            # ever matched to. Comparing against the previous frame alone would
            # invent a switch every time an object is occluded and comes back
            # with the same, correct, prediction id.
            previous = last_pred_for_gt.get(g_id)
            if previous is not None and previous != p_id:
                id_switches += 1
            last_pred_for_gt[g_id] = p_id
            new_prev[g_id] = p_id
        prev_matches = new_prev

    err = np.asarray(distances, dtype=np.float64)
    if err.size:
        motp = float(err.mean())
        rmse = float(np.sqrt(np.mean(err**2)))
        median = float(np.median(err))
    else:
        motp = float("nan")
        rmse = float("nan")
        median = float("nan")

    mota = 1.0 - (misses + false_positives + id_switches) / n_gt
    idf1, _idtp, _idfp, _idfn = _idf1(potential, n_gt=n_gt, n_pred=n_pred)

    return AccuracyReport(
        dataset=dataset,
        n_frames=len(frames),
        n_gt=n_gt,
        n_pred=n_pred,
        mota=float(mota),
        motp=motp,
        idf1=float(idf1),
        id_switches=int(id_switches),
        false_positives=int(false_positives),
        misses=int(misses),
        rmse_m=rmse,
        median_error_m=median,
        notes="; ".join(gt_notes + pred_notes),
    )
