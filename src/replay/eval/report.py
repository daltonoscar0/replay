"""Render accuracy reports to markdown and splice them into the README.

Every number in the output is read off an :class:`~replay.eval.metrics.AccuracyReport`
by attribute. There is no formatting path that accepts a bare float, a dict or a
string, so the table cannot contain a figure that ``evaluate`` did not produce.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from replay.eval.metrics import AccuracyReport

__all__ = ["render_table", "update_readme", "MarkerError"]

_NA = "n/a"


class MarkerError(ValueError):
    """The README does not contain a usable pair of accuracy markers."""


def _fmt_float(value: Any, digits: int) -> str:
    number = float(value)
    if number != number:  # NaN
        return _NA
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    return f"{int(value):d}"


def _fmt_text(value: Any) -> str:
    """Escape a cell so it cannot break out of the table row."""
    text = "" if value is None else str(value)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|").strip()


def _score(digits: int) -> Callable[[Any], str]:
    def formatter(value: Any) -> str:
        return _fmt_float(value, digits)

    return formatter


#: ``(header, AccuracyReport attribute, formatter)``. The attribute names are the
#: only channel through which values reach the table.
_COLUMNS: tuple[tuple[str, str, Callable[[Any], str]], ...] = (
    ("Dataset", "dataset", _fmt_text),
    ("Frames", "n_frames", _fmt_int),
    ("MOTA", "mota", _score(3)),
    ("MOTP (m)", "motp", _score(3)),
    ("IDF1", "idf1", _score(3)),
    ("ID switches", "id_switches", _fmt_int),
    ("FP", "false_positives", _fmt_int),
    ("Misses", "misses", _fmt_int),
    ("RMSE (m)", "rmse_m", _score(3)),
    ("Notes", "notes", _fmt_text),
)

_EMPTY_TABLE = "_No accuracy runs recorded._"


def render_table(reports: Sequence[AccuracyReport]) -> str:
    """A markdown table with one row per report, newest-first order preserved."""
    if not isinstance(reports, Sequence) or isinstance(reports, str | bytes):
        raise TypeError("reports must be a sequence of AccuracyReport")
    for report in reports:
        if not isinstance(report, AccuracyReport):
            raise TypeError(
                f"render_table only formats AccuracyReport instances, got {type(report).__name__}"
            )
    if not reports:
        return _EMPTY_TABLE

    header = "| " + " | ".join(name for name, _, _ in _COLUMNS) + " |"
    rule = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [header, rule]
    for report in reports:
        cells = [fmt(getattr(report, attr)) for _, attr, fmt in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_readme(
    reports: Sequence[AccuracyReport],
    readme_path: str | Path,
    *,
    marker: str = "ACCURACY",
) -> Path:
    """Replace the marked block in ``readme_path`` with the rendered table.

    Everything outside ``<!-- BEGIN {marker} -->`` / ``<!-- END {marker} -->`` is
    written back unchanged, byte for byte. Missing markers are an error rather
    than a cue to guess a location: appending to a README is a destructive edit
    to a file the author owns.
    """
    path = Path(readme_path)
    begin = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"

    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"README not found at {path}") from exc
    text = raw.decode("utf-8")

    if text.count(begin) != 1 or text.count(end) != 1:
        raise MarkerError(
            f"{path} must contain exactly one '{begin}' and one '{end}' "
            f"(found {text.count(begin)} and {text.count(end)}). "
            "Add the marker pair where the accuracy table belongs."
        )

    start = text.index(begin) + len(begin)
    stop = text.index(end)
    if stop < start:
        raise MarkerError(f"{path} has '{end}' before '{begin}'; the markers are out of order.")

    table = render_table(reports)
    replacement = f"\n\n{table}\n\n"
    updated = text[:start] + replacement + text[stop:]
    path.write_bytes(updated.encode("utf-8"))
    return path
