"""Accuracy evaluation: CLEAR-MOT metrics, SoccerNet ground truth, markdown reports."""

from replay.eval.metrics import AccuracyReport, evaluate
from replay.eval.report import render_table, update_readme
from replay.eval.soccernet import SoccerNetUnavailable, available, load_soccernet_gt

__all__ = [
    "AccuracyReport",
    "SoccerNetUnavailable",
    "available",
    "evaluate",
    "load_soccernet_gt",
    "render_table",
    "update_readme",
]
