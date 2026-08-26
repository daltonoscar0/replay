from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from replay.io import (
    _adapter,
    backend_name,
    from_canonical_csv,
    from_metrica,
    metrica,
    read_tracking,
    to_canonical_csv,
    to_metrica,
)
from replay.schema import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    TRACKING_COLUMNS,
    coerce_tracking_frame,
)

ID_COLUMNS = ["match_id", "period", "frame", "team", "player_id", "jersey"]
FLOAT_COLUMNS = ["t", "x", "y", "vx", "vy", "confidence"]

# Column order to_metrica produces: home jerseys ascending, away jerseys
# ascending, ball last.
OBJECTS = [
    ("home", "home_1", 1),
    ("home", "home_7", 7),
    ("away", "away_2", 2),
    ("away", "away_9", 9),
    ("ball", "ball", None),
]


def synthetic_frame(match_id: str = "m1", n_frames: int = 4) -> pd.DataFrame:
    """Dense (frame x object) canonical frame with a couple of NaN gaps."""
    rows = []
    for i in range(n_frames):
        for j, (team, player_id, jersey) in enumerate(OBJECTS):
            x = 3.5 + 7.25 * j + 1.125 * i
            y = 2.75 + 5.5 * j + 0.875 * i
            # home_7 is off-frame at i == 1, the ball at i == 2.
            if (j == 1 and i == 1) or (j == 4 and i == 2):
                x = np.nan
                y = np.nan
            rows.append(
                {
                    "match_id": match_id,
                    "t": i / 10.0,
                    "period": 1,
                    "frame": 100 + i,
                    "team": team,
                    "player_id": player_id,
                    "jersey": jersey,
                    "x": x,
                    "y": y,
                    "vx": np.nan,
                    "vy": np.nan,
                    "confidence": np.nan,
                }
            )
    return coerce_tracking_frame(pd.DataFrame(rows))


def assert_frames_match(got: pd.DataFrame, want: pd.DataFrame) -> None:
    assert list(got.columns) == list(TRACKING_COLUMNS)
    assert len(got) == len(want)
    pd.testing.assert_frame_equal(got[ID_COLUMNS], want[ID_COLUMNS])
    for col in FLOAT_COLUMNS:
        np.testing.assert_allclose(
            got[col].to_numpy(dtype=float),
            want[col].to_numpy(dtype=float),
            atol=1e-9,
            rtol=0,
            equal_nan=True,
            err_msg=f"column {col} drifted",
        )


def test_metrica_round_trip_is_exact(tmp_path):
    want = synthetic_frame()
    path = to_metrica(want, tmp_path / "wide.csv")
    got = from_metrica(path, match_id="m1")
    assert_frames_match(got, want)


def test_metrica_header_and_column_order(tmp_path):
    path = to_metrica(synthetic_frame(), tmp_path / "wide.csv")
    header = path.read_text().splitlines()[0].split(",")
    assert header == [
        "Period",
        "Frame",
        "Time [s]",
        "Home_1_X",
        "Home_1_Y",
        "Home_7_X",
        "Home_7_Y",
        "Away_2_X",
        "Away_2_Y",
        "Away_9_X",
        "Away_9_Y",
        "Ball_X",
        "Ball_Y",
    ]


def test_positions_are_normalized_to_unit_square(tmp_path):
    want = synthetic_frame()
    path = to_metrica(want, tmp_path / "wide.csv")
    raw = pd.read_csv(path)
    x_cols = [c for c in raw.columns if c.endswith("_X")]
    y_cols = [c for c in raw.columns if c.endswith("_Y")]
    assert x_cols and y_cols
    for col in x_cols + y_cols:
        vals = raw[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        assert finite.size > 0
        assert finite.min() >= 0.0
        assert finite.max() <= 1.0
    first = raw.iloc[0]
    assert first["Home_1_X"] == pytest.approx(3.5 / PITCH_LENGTH)
    assert first["Home_1_Y"] == pytest.approx(2.75 / PITCH_WIDTH)


def test_missing_positions_are_empty_cells(tmp_path):
    path = to_metrica(synthetic_frame(), tmp_path / "wide.csv")
    lines = path.read_text().splitlines()
    # Second data row is frame 101, where home_7 is off-frame.
    cells = lines[2].split(",")
    header = lines[0].split(",")
    assert cells[header.index("Home_7_X")] == ""
    assert cells[header.index("Home_7_Y")] == ""
    assert cells[header.index("Home_1_X")] != ""

    back = from_metrica(path)
    gap = back[(back["frame"] == 101) & (back["player_id"] == "home_7")]
    assert len(gap) == 1
    assert np.isnan(gap["x"].to_numpy(dtype=float)[0])


def test_from_metrica_stamps_match_id(tmp_path):
    path = to_metrica(synthetic_frame(), tmp_path / "wide.csv")
    assert set(from_metrica(path)["match_id"]) == {"unknown"}
    assert set(from_metrica(path, match_id="abc")["match_id"]) == {"abc"}


def test_from_metrica_leaves_velocity_and_confidence_unknown(tmp_path):
    path = to_metrica(synthetic_frame(), tmp_path / "wide.csv")
    back = from_metrica(path)
    for col in ("vx", "vy", "confidence"):
        assert back[col].isna().all()


def test_time_falls_back_to_fps_when_t_is_missing(tmp_path):
    df = synthetic_frame()
    df["t"] = np.nan
    path = to_metrica(df, tmp_path / "wide.csv", fps=25.0)
    raw = pd.read_csv(path)
    np.testing.assert_allclose(raw["Time [s]"].to_numpy(), np.arange(100, 104) / 25.0)


def test_multiple_periods_keep_their_own_frame_numbering(tmp_path):
    first = synthetic_frame(n_frames=2)
    second = first.copy()
    second["period"] = 2
    want = coerce_tracking_frame(pd.concat([first, second], ignore_index=True))
    path = to_metrica(want, tmp_path / "wide.csv")
    raw = pd.read_csv(path)
    assert list(raw["Period"]) == [1, 1, 2, 2]
    assert list(raw["Frame"]) == [100, 101, 100, 101]
    assert_frames_match(from_metrica(path, match_id="m1"), want)


def test_unnumbered_players_use_their_player_id_as_label(tmp_path):
    df = synthetic_frame(n_frames=1)
    df.loc[df["player_id"] == "home_7", "jersey"] = pd.NA
    df.loc[df["player_id"] == "home_7", "player_id"] = "home_gk"
    want = coerce_tracking_frame(df)
    path = to_metrica(want, tmp_path / "wide.csv")
    assert "Home_gk_X" in pd.read_csv(path).columns
    back = from_metrica(path, match_id="m1")
    assert "home_gk" in set(back["player_id"])
    assert back.loc[back["player_id"] == "home_gk", "jersey"].isna().all()


def test_duplicate_object_in_one_frame_is_rejected(tmp_path):
    df = synthetic_frame(n_frames=1)
    dup = coerce_tracking_frame(pd.concat([df, df.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="more than one row"):
        to_metrica(dup, tmp_path / "wide.csv")


def test_missing_jersey_and_player_id_is_rejected(tmp_path):
    df = synthetic_frame(n_frames=1)
    df.loc[0, "jersey"] = pd.NA
    df.loc[0, "player_id"] = pd.NA
    with pytest.raises(ValueError, match="neither a jersey nor a player_id"):
        to_metrica(coerce_tracking_frame(df), tmp_path / "wide.csv")


def test_non_positive_fps_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="fps must be positive"):
        to_metrica(synthetic_frame(n_frames=1), tmp_path / "wide.csv", fps=0.0)


def test_from_metrica_rejects_a_non_metrica_file(tmp_path):
    path = tmp_path / "nope.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not a Metrica wide file"):
        from_metrica(path)


def test_from_metrica_rejects_an_unpaired_column(tmp_path):
    path = tmp_path / "half.csv"
    path.write_text("Period,Frame,Time [s],Home_1_X\n1,0,0.0,0.5\n")
    with pytest.raises(ValueError, match="Home_1_X but no Home_1_Y"):
        from_metrica(path)


def test_canonical_csv_round_trip(tmp_path):
    want = synthetic_frame()
    path = to_canonical_csv(want, tmp_path / "nested" / "long.csv")
    assert path.exists()
    assert list(pd.read_csv(path).columns) == list(TRACKING_COLUMNS)
    assert_frames_match(from_canonical_csv(path), want)


def test_canonical_csv_round_trip_keeps_velocity(tmp_path):
    want = synthetic_frame()
    rng = np.random.default_rng(0)
    want["vx"] = rng.normal(size=len(want))
    want["vy"] = rng.normal(size=len(want))
    want["confidence"] = rng.uniform(size=len(want))
    want = coerce_tracking_frame(want)
    path = to_canonical_csv(want, tmp_path / "long.csv")
    assert_frames_match(from_canonical_csv(path), want)


def test_from_canonical_csv_rejects_a_short_file(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("frame,x,y\n0,1.0,2.0\n")
    with pytest.raises(ValueError, match="not a canonical tracking CSV"):
        from_canonical_csv(path)


def test_backend_is_vendored_without_sportio():
    assert "sportio" not in sys.modules
    assert backend_name() == "vendored"


def _fake_sportio(calls: list[str]) -> types.ModuleType:
    module = types.ModuleType("sportio")

    def fake_to_metrica(df, path, *, fps=10.0):
        calls.append("to_metrica")
        return metrica.to_metrica(df, path, fps=fps)

    def fake_from_metrica(path, *, match_id="unknown"):
        calls.append("from_metrica")
        return metrica.from_metrica(path, match_id=match_id)

    module.to_metrica = fake_to_metrica
    module.from_metrica = fake_from_metrica
    return module


def test_sportio_backend_is_preferred_when_present(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "sportio", _fake_sportio(calls))

    assert backend_name() == "sportio"
    assert _adapter.backend_name() == "sportio"

    want = synthetic_frame()
    path = to_metrica(want, tmp_path / "wide.csv")
    got = from_metrica(path, match_id="m1")
    assert calls == ["to_metrica", "from_metrica"]
    assert_frames_match(got, want)


def test_backend_flips_back_when_sportio_goes_away(tmp_path, monkeypatch):
    calls: list[str] = []
    with monkeypatch.context() as ctx:
        ctx.setitem(sys.modules, "sportio", _fake_sportio(calls))
        assert backend_name() == "sportio"
    assert backend_name() == "vendored"
    to_metrica(synthetic_frame(n_frames=1), tmp_path / "wide.csv")
    assert calls == []


def test_sportio_without_to_metrica_is_ignored(monkeypatch):
    module = types.ModuleType("sportio")
    module.to_metrica = "not callable"
    monkeypatch.setitem(sys.modules, "sportio", module)
    assert backend_name() == "vendored"


def test_sportio_reader_result_is_coerced(tmp_path, monkeypatch):
    """A backend that returns extra columns or loose dtypes still yields canonical."""
    want = synthetic_frame()
    path = metrica.to_metrica(want, tmp_path / "wide.csv")

    module = types.ModuleType("sportio")

    def fake_to_metrica(df, path, *, fps=10.0):
        return metrica.to_metrica(df, path, fps=fps)

    def fake_from_metrica(path, *, match_id="unknown"):
        out = metrica.from_metrica(path, match_id=match_id)
        out["scratch"] = 1
        return out[["scratch", *TRACKING_COLUMNS]]

    module.to_metrica = fake_to_metrica
    module.from_metrica = fake_from_metrica
    monkeypatch.setitem(sys.modules, "sportio", module)

    got = from_metrica(path, match_id="m1")
    assert list(got.columns) == list(TRACKING_COLUMNS)
    assert_frames_match(got, want)


# --- format sniffing --------------------------------------------------------


def _sample_frame() -> pd.DataFrame:
    rows = []
    for f in range(4):
        for team, jersey in (("home", 7), ("away", 9)):
            rows.append(
                {
                    "match_id": "m", "t": f / 10.0, "period": 1, "frame": f,
                    "team": team, "player_id": f"{team}_{jersey}", "jersey": jersey,
                    "x": 10.0 + f, "y": 20.0 + f, "vx": np.nan, "vy": np.nan,
                    "confidence": np.nan,
                }
            )
        rows.append(
            {
                "match_id": "m", "t": f / 10.0, "period": 1, "frame": f,
                "team": "ball", "player_id": "ball", "jersey": pd.NA,
                "x": 50.0, "y": 34.0, "vx": np.nan, "vy": np.nan, "confidence": np.nan,
            }
        )
    return coerce_tracking_frame(pd.DataFrame(rows))


def test_read_tracking_detects_metrica_whatever_the_file_is_called(tmp_path):
    path = tmp_path / "ground_truth_labels.csv"  # no format hint in the name
    to_metrica(_sample_frame(), path)
    out = read_tracking(path)
    assert list(out.columns) == list(TRACKING_COLUMNS)
    assert set(out["team"].dropna().astype(str)) == {"home", "away", "ball"}


def test_read_tracking_detects_the_canonical_layout_whatever_the_file_is_called(tmp_path):
    path = tmp_path / "my_hand_labels.csv"  # the case that used to fail
    to_canonical_csv(_sample_frame(), path)
    out = read_tracking(path)
    pdt.assert_frame_equal(out, _sample_frame())


def test_read_tracking_rejects_a_csv_that_is_neither(tmp_path):
    path = tmp_path / "something_else.csv"
    path.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="neither a Metrica wide file nor a canonical"):
        read_tracking(path)
