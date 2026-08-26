"""Metrica Sports wide-format tracking IO, plus the canonical long-format CSV.

The wide format is one row per (period, frame): a ``Period, Frame, Time [s]``
header block, then an ``X, Y`` column pair per tracked object in stable jersey
order, then ``Ball_X, Ball_Y``. Positions are normalized to ``[0, 1]`` by the
pitch dimensions, matching the Metrica sample-data release.

Deviation from the released Metrica data: that release splits home and away
into two files (``*_Home_Team.csv`` and ``*_Away_Team.csv``), each with a
three-line header carrying the team name, the jersey numbers, and then the
column labels. We write a single file with one header line and
``Home_<jersey>_X`` / ``Away_<jersey>_X`` column prefixes. One file keeps the
round trip lossless -- two files cannot express which frames the two teams
share without an extra convention -- and one header line means the file opens
correctly in any CSV reader. Anything that needs the released layout has to
split the columns on the ``Home_`` / ``Away_`` prefix.

The wide format has no place to put velocity or per-detection confidence, so
``from_metrica`` returns those columns as NaN rather than inventing them.
``player_id`` is likewise not stored; it is reconstructed as
``"<team>_<label>"`` (and ``"ball"`` for the ball), which is the convention
``to_metrica`` assumes when it builds the column names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from replay.schema import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    TRACKING_COLUMNS,
    coerce_tracking_frame,
)

__all__ = [
    "from_canonical_csv",
    "from_metrica",
    "to_canonical_csv",
    "to_metrica",
]

_META_COLUMNS: tuple[str, str, str] = ("Period", "Frame", "Time [s]")
_BALL_PREFIX = "Ball"

# Objects are written home, away, anything else (e.g. referees), ball last.
# The ball goes last because that is where the Metrica release puts it and
# because it is the column readers most often want to find by position.
_TEAM_RANK: dict[str, int] = {"home": 0, "away": 1, "ball": 3}


@dataclass(frozen=True, slots=True)
class _Object:
    """One tracked object and the column prefix it owns in the wide file."""

    key: tuple[str, str | None, int | None]
    team: str
    player_id: str
    jersey: int | None
    label: str

    @property
    def prefix(self) -> str:
        if self.team == "ball":
            return _BALL_PREFIX
        return f"{self.team.capitalize()}_{self.label}"

    @property
    def sort_key(self) -> tuple[int, int, int, str]:
        # Jerseyed players sort numerically ahead of unnumbered ones, so the
        # column order is stable across runs and across matches.
        return (
            _TEAM_RANK.get(self.team, 2),
            0 if self.jersey is not None else 1,
            self.jersey if self.jersey is not None else 0,
            self.label,
        )


def _scalar(value: object) -> object | None:
    return None if pd.isna(value) else value


def _object_label(team: str, player_id: str | None, jersey: int | None) -> str:
    """Column label for an object: its jersey, else its bare player id.

    The team prefix is stripped from the player id when present so that the
    reconstructed id on read matches the one that went in.
    """
    if team == "ball":
        return "ball"
    if jersey is not None:
        return str(jersey)
    if player_id is None:
        raise ValueError(f"object on team {team!r} has neither a jersey nor a player_id")
    stripped = player_id.removeprefix(f"{team}_")
    return stripped or player_id


def _row_key(team: object, player_id: object, jersey: object) -> tuple[str, str | None, int | None]:
    team_v = _scalar(team)
    if team_v is None:
        raise ValueError("tracking frame has rows with a missing team; cannot lay out columns")
    pid = _scalar(player_id)
    jer = _scalar(jersey)
    return (str(team_v), None if pid is None else str(pid), None if jer is None else int(jer))


def _objects(df: pd.DataFrame) -> list[_Object]:
    """Distinct objects in the frame, in stable column order."""
    seen: dict[tuple[str, str | None, int | None], _Object] = {}
    for team, pid, jersey in zip(df["team"], df["player_id"], df["jersey"], strict=True):
        key = _row_key(team, pid, jersey)
        if key in seen:
            continue
        label = _object_label(key[0], key[1], key[2])
        seen[key] = _Object(
            key=key,
            team=key[0],
            player_id=key[1] if key[1] is not None else f"{key[0]}_{label}",
            jersey=key[2],
            label=label,
        )
    objects = sorted(seen.values(), key=lambda o: o.sort_key)
    prefixes: dict[str, _Object] = {}
    for obj in objects:
        clash = prefixes.get(obj.prefix)
        if clash is not None:
            raise ValueError(
                f"objects {clash.player_id!r} and {obj.player_id!r} both map to column "
                f"prefix {obj.prefix!r}; give them distinct jerseys or player ids"
            )
        prefixes[obj.prefix] = obj
    return objects


def to_metrica(df: pd.DataFrame, path: str | Path, *, fps: float = 10.0) -> Path:
    """Write a canonical tracking frame as a wide Metrica CSV.

    ``fps`` only supplies the ``Time [s]`` column for frames whose ``t`` is
    missing; where ``t`` is present it is written through unchanged, because
    real tracking clocks drift off an exact frame multiple.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    frame = coerce_tracking_frame(df)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    objects = _objects(frame)
    prefix_by_key = {o.key: o.prefix for o in objects}
    work = frame.copy()
    work["_col"] = [
        prefix_by_key[_row_key(t, p, j)]
        for t, p, j in zip(work["team"], work["player_id"], work["jersey"], strict=True)
    ]

    try:
        wide_x = work.pivot(index=["period", "frame"], columns="_col", values="x").sort_index()
        wide_y = work.pivot(index=["period", "frame"], columns="_col", values="y").sort_index()
    except ValueError as exc:
        raise ValueError(
            "tracking frame has more than one row for the same object in the same "
            "(period, frame); Metrica wide format cannot represent that"
        ) from exc

    times = work.groupby(["period", "frame"], sort=True)["t"].first().reindex(wide_x.index)
    periods = wide_x.index.get_level_values("period").to_numpy(dtype=np.int64)
    frames = wide_x.index.get_level_values("frame").to_numpy(dtype=np.int64)
    t_values = times.to_numpy(dtype=np.float64)
    t_values = np.where(np.isnan(t_values), frames / float(fps), t_values)

    data: dict[str, np.ndarray] = {
        "Period": periods,
        "Frame": frames,
        "Time [s]": t_values,
    }
    for obj in objects:
        data[f"{obj.prefix}_X"] = wide_x[obj.prefix].to_numpy(dtype=np.float64) / PITCH_LENGTH
        data[f"{obj.prefix}_Y"] = wide_y[obj.prefix].to_numpy(dtype=np.float64) / PITCH_WIDTH

    pd.DataFrame(data).to_csv(out_path, index=False)
    return out_path


def _parse_prefix(prefix: str) -> tuple[str, str, int | None, str]:
    """``"Home_7"`` -> ``("home", "7", 7, "home_7")``."""
    if prefix == _BALL_PREFIX:
        return ("ball", "ball", None, "ball")
    team_part, sep, label = prefix.partition("_")
    if not sep or not label:
        raise ValueError(f"column prefix {prefix!r} is not '<Team>_<label>' or {_BALL_PREFIX!r}")
    team = team_part.lower()
    jersey = int(label) if label.isdigit() else None
    return (team, label, jersey, f"{team}_{label}")


def from_metrica(path: str | Path, *, match_id: str = "unknown") -> pd.DataFrame:
    """Read a wide Metrica CSV back into a canonical tracking frame.

    Every object column produces a row for every frame in the file. Empty
    cells become NaN positions rather than dropped rows, so the frame keeps a
    dense grid and the round trip through :func:`to_metrica` is exact.
    """
    src = Path(path)
    raw = pd.read_csv(src)
    missing = [c for c in _META_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{src} is not a Metrica wide file; missing columns {missing}")

    prefixes = [c[:-2] for c in raw.columns if c.endswith("_X") and c not in _META_COLUMNS]
    for prefix in prefixes:
        if f"{prefix}_Y" not in raw.columns:
            raise ValueError(f"{src} has {prefix}_X but no {prefix}_Y")
    parsed = [_parse_prefix(p) for p in prefixes]

    n_rows = len(raw)
    n_obj = len(parsed)
    if n_obj == 0:
        xs = np.empty((n_rows, 0), dtype=np.float64)
        ys = xs
    else:
        xs = raw[[f"{p}_X" for p in prefixes]].to_numpy(dtype=np.float64) * PITCH_LENGTH
        ys = raw[[f"{p}_Y" for p in prefixes]].to_numpy(dtype=np.float64) * PITCH_WIDTH

    teams = np.array([p[0] for p in parsed], dtype=object)
    player_ids = np.array([p[3] for p in parsed], dtype=object)
    jerseys = [p[2] for p in parsed]
    nan = np.full(n_rows * n_obj, np.nan, dtype=np.float64)

    long = pd.DataFrame(
        {
            "match_id": np.repeat(np.array([match_id], dtype=object), n_rows * n_obj),
            "t": np.repeat(raw["Time [s]"].to_numpy(dtype=np.float64), n_obj),
            "period": np.repeat(raw["Period"].to_numpy(dtype=np.int64), n_obj),
            "frame": np.repeat(raw["Frame"].to_numpy(dtype=np.int64), n_obj),
            "team": np.tile(teams, n_rows),
            "player_id": np.tile(player_ids, n_rows),
            "jersey": pd.array(jerseys * n_rows, dtype="Int64"),
            "x": xs.reshape(-1),
            "y": ys.reshape(-1),
            "vx": nan,
            "vy": nan.copy(),
            "confidence": nan.copy(),
        }
    )
    return coerce_tracking_frame(long)


def to_canonical_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the long-format canonical frame: one row per object per frame."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coerce_tracking_frame(df).to_csv(out_path, index=False)
    return out_path


def from_canonical_csv(path: str | Path) -> pd.DataFrame:
    """Read a long-format canonical CSV written by :func:`to_canonical_csv`."""
    src = Path(path)
    raw = pd.read_csv(
        src,
        dtype={
            "match_id": "string",
            "team": "string",
            "player_id": "string",
            "jersey": "Int64",
        },
    )
    missing = [c for c in TRACKING_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{src} is not a canonical tracking CSV; missing columns {missing}")
    return coerce_tracking_frame(raw)


def read_tracking(path: str | Path, *, match_id: str = "unknown") -> pd.DataFrame:
    """Load a tracking CSV in either supported layout, deciding from the file.

    Both writers here produce a `.csv` and the extension says nothing about
    which. Sniffing the header is the only honest way to tell them apart: a
    filename convention puts the burden on the caller to name their ground-truth
    file a particular way, and silently misreads it when they do not.
    """
    header = pd.read_csv(path, nrows=0).columns
    if {"Period", "Frame", "Time [s]"}.issubset(set(header)):
        return from_metrica(path, match_id=match_id)
    missing = [c for c in TRACKING_COLUMNS if c not in header]
    if missing:
        raise ValueError(
            f"{path} is neither a Metrica wide file nor a canonical tracking CSV; "
            f"as canonical it is missing {missing}"
        )
    return from_canonical_csv(path)
