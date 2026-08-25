"""Markdown rendering and README splicing."""

from __future__ import annotations

import math

import pytest

from replay.eval.metrics import AccuracyReport
from replay.eval.report import MarkerError, render_table, update_readme

BEGIN = "<!-- BEGIN ACCURACY -->"
END = "<!-- END ACCURACY -->"

PREFIX = "# replay\n\nSome prose the author wrote.\n\n## Accuracy\n\n" + BEGIN
SUFFIX = END + "\n\n## Next section\n\nMore prose.\n"


def make_report(**overrides):
    fields = {
        "dataset": "synthetic",
        "n_frames": 120,
        "n_gt": 2640,
        "n_pred": 2600,
        "mota": 0.87654,
        "motp": 0.4321,
        "idf1": 0.79999,
        "id_switches": 14,
        "false_positives": 31,
        "misses": 275,
        "rmse_m": 0.6789,
        "median_error_m": 0.5,
        "notes": "color detector",
    }
    fields.update(overrides)
    return AccuracyReport(**fields)


def test_render_table_has_one_row_per_report():
    table = render_table([make_report(dataset="a"), make_report(dataset="b")])
    lines = table.splitlines()

    assert len(lines) == 4  # header, rule, two rows
    assert lines[0].startswith("| Dataset | Frames | MOTA | MOTP (m) | IDF1 |")
    assert "ID switches" in lines[0]
    assert "RMSE (m)" in lines[0]
    assert "Notes" in lines[0]
    assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}
    assert lines[2].startswith("| a |")
    assert lines[3].startswith("| b |")


def test_render_table_formats_numbers_from_the_report():
    table = render_table([make_report()])
    row = table.splitlines()[2]
    cells = [c.strip() for c in row.strip("|").split("|")]

    assert cells == [
        "synthetic",
        "120",
        "0.877",
        "0.432",
        "0.800",
        "14",
        "31",
        "275",
        "0.679",
        "color detector",
    ]


def test_render_table_renders_nan_as_not_available():
    table = render_table([make_report(motp=math.nan, rmse_m=math.nan)])
    row = table.splitlines()[2]
    assert row.count("n/a") == 2


def test_render_table_escapes_pipes_in_text_cells():
    table = render_table([make_report(dataset="a|b", notes="one | two\nthree")])
    row = table.splitlines()[2]
    assert len(row.splitlines()) == 1
    assert "a\\|b" in row
    assert "one \\| two three" in row


def test_render_table_of_no_reports_is_a_placeholder_not_an_empty_table():
    assert "|" not in render_table([])


def test_render_table_rejects_anything_that_is_not_a_report():
    with pytest.raises(TypeError):
        render_table([{"dataset": "fake", "mota": 1.0}])
    with pytest.raises(TypeError):
        render_table("synthetic")


def test_update_readme_preserves_surrounding_bytes(tmp_path):
    readme = tmp_path / "README.md"
    original = PREFIX + "\n\nstale table that must go\n\n" + SUFFIX
    readme.write_bytes(original.encode("utf-8"))

    out = update_readme([make_report()], readme)
    assert out == readme

    text = readme.read_bytes().decode("utf-8")
    assert text.startswith(PREFIX)
    assert text.endswith(SUFFIX)
    assert "stale table that must go" not in text

    body = text[len(PREFIX) : len(text) - len(SUFFIX)]
    assert body.strip().splitlines()[0].startswith("| Dataset |")
    assert "synthetic" in body


def test_update_readme_is_idempotent(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_bytes((PREFIX + "\n" + SUFFIX).encode("utf-8"))
    reports = [make_report()]

    update_readme(reports, readme)
    once = readme.read_bytes()
    update_readme(reports, readme)
    assert readme.read_bytes() == once


def test_update_readme_writes_nothing_but_report_numbers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_bytes((PREFIX + "\n" + SUFFIX).encode("utf-8"))
    update_readme([make_report(mota=0.5, id_switches=7)], readme)

    body = readme.read_bytes().decode("utf-8")
    body = body[body.index(BEGIN) + len(BEGIN) : body.index(END)]
    assert "0.500" in body
    assert "| 7 |" in body
    assert "0.877" not in body


def test_update_readme_raises_when_markers_are_missing(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_bytes(b"# replay\n\nNo markers here.\n")
    with pytest.raises(MarkerError, match="exactly one"):
        update_readme([make_report()], readme)
    assert readme.read_bytes() == b"# replay\n\nNo markers here.\n"


def test_update_readme_raises_when_only_one_marker_is_present(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_bytes((PREFIX + "\nno end marker\n").encode("utf-8"))
    with pytest.raises(MarkerError):
        update_readme([make_report()], readme)


def test_update_readme_raises_on_duplicate_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_bytes((PREFIX + "\n" + SUFFIX + PREFIX + "\n" + SUFFIX).encode("utf-8"))
    with pytest.raises(MarkerError):
        update_readme([make_report()], readme)


def test_update_readme_raises_when_markers_are_out_of_order(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_bytes((END + "\nbody\n" + BEGIN + "\n").encode("utf-8"))
    with pytest.raises(MarkerError, match="out of order"):
        update_readme([make_report()], readme)


def test_update_readme_honours_a_custom_marker(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_bytes(b"top\n<!-- BEGIN SIMACC -->\n<!-- END SIMACC -->\nbottom\n")
    update_readme([make_report()], readme, marker="SIMACC")

    text = readme.read_bytes().decode("utf-8")
    assert text.startswith("top\n<!-- BEGIN SIMACC -->")
    assert text.endswith("<!-- END SIMACC -->\nbottom\n")
    assert "| Dataset |" in text


def test_update_readme_on_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        update_readme([make_report()], tmp_path / "nope.md")
