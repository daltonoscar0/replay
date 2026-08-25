"""The simulator's view of a moment in a match.

Everything in the sandbox is 23 rows of ``(x, y)``: eleven home players, eleven
away players, then the ball. Fixing the row layout means a policy can index a
teammate by slot without carrying a lookup table, and it means the physics step
is one array operation instead of a loop over objects.

The slot layout is identity: slot ``i`` is the same player for the whole
rollout. That is why a frame with a missing player leaves the slot as NaN
instead of packing the remaining players down; shifting them would silently
hand one player's trajectory to another.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from replay.schema import coerce_tracking_frame

log = logging.getLogger(__name__)

N_PLAYERS_PER_TEAM = 11
BALL_INDEX = 22
N_ROWS = 23

HOME_SLICE = slice(0, N_PLAYERS_PER_TEAM)
AWAY_SLICE = slice(N_PLAYERS_PER_TEAM, 2 * N_PLAYERS_PER_TEAM)

#: A player this close to the ball is treated as carrying it. Loose enough to
#: survive tracking noise, tight enough that two opponents rarely both qualify.
POSSESSION_RADIUS_M = 2.0

TeamSide = Literal["home", "away"]


def slot_team(index: int) -> Literal["home", "away", "ball"]:
    """Which team owns a row index."""
    if index == BALL_INDEX:
        return "ball"
    if index < 0 or index >= N_ROWS:
        raise IndexError(f"row index {index} outside 0..{N_ROWS - 1}")
    return "home" if index < N_PLAYERS_PER_TEAM else "away"


def slot_jersey(index: int) -> int:
    """Synthetic jersey number for a row index, 1..11 within each team."""
    if index == BALL_INDEX:
        raise IndexError("the ball row has no jersey")
    return index % N_PLAYERS_PER_TEAM + 1


def slot_player_id(index: int) -> str:
    """Synthetic player id for a row index, e.g. ``home_7``.

    ``MatchState`` carries no identity beyond the slot, so anything written
    back out has to reconstruct ids from the slot. Round-tripping a real
    tracking frame therefore only preserves ids when the source already uses
    this ``<team>_<jersey>`` convention with jerseys 1..11.
    """
    if index == BALL_INDEX:
        return "ball"
    return f"{slot_team(index)}_{slot_jersey(index)}"


@dataclass
class MatchState:
    """Positions, velocities and possession at one instant.

    ``pos`` and ``vel`` are ``(23, 2)`` float64 in pitch meters and m/s. Rows
    0..10 are home, 11..21 are away, row 22 (``BALL_INDEX``) is the ball. A row
    of NaN means that object is unknown at this instant.
    """

    t: float
    pos: np.ndarray
    vel: np.ndarray
    possession: int = -1
    ball_carrier_team: Literal["home", "away"] | None = None

    def __post_init__(self) -> None:
        self.t = float(self.t)
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.vel = np.asarray(self.vel, dtype=np.float64)
        for name, arr in (("pos", self.pos), ("vel", self.vel)):
            if arr.shape != (N_ROWS, 2):
                raise ValueError(f"{name} must have shape ({N_ROWS}, 2), got {arr.shape}")
        self.possession = int(self.possession)
        if self.possession < -1 or self.possession >= N_ROWS:
            raise ValueError(f"possession must be -1 or a row index, got {self.possession}")

    def copy(self) -> MatchState:
        """A state that shares nothing with this one."""
        return MatchState(
            t=self.t,
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            possession=self.possession,
            ball_carrier_team=self.ball_carrier_team,
        )

    @property
    def home(self) -> np.ndarray:
        return self.pos[HOME_SLICE]

    @property
    def away(self) -> np.ndarray:
        return self.pos[AWAY_SLICE]

    @property
    def ball(self) -> np.ndarray:
        return self.pos[BALL_INDEX]

    @property
    def home_vel(self) -> np.ndarray:
        return self.vel[HOME_SLICE]

    @property
    def away_vel(self) -> np.ndarray:
        return self.vel[AWAY_SLICE]

    @property
    def ball_vel(self) -> np.ndarray:
        return self.vel[BALL_INDEX]

    def team_pos(self, team: TeamSide) -> np.ndarray:
        return self.home if team == "home" else self.away

    def team_vel(self, team: TeamSide) -> np.ndarray:
        return self.home_vel if team == "home" else self.away_vel


def empty_state(t: float = 0.0) -> MatchState:
    """A state with every row unknown."""
    return MatchState(t=t, pos=np.full((N_ROWS, 2), np.nan), vel=np.zeros((N_ROWS, 2)))


def infer_possession(
    pos: np.ndarray, *, radius: float = POSSESSION_RADIUS_M
) -> tuple[int, Literal["home", "away"] | None]:
    """Nearest player to the ball, if one is within ``radius``.

    Tracking data does not label possession, so the simulator infers it. NaN
    rows never win, and a ball with an unknown position leaves it loose.
    """
    ball = pos[BALL_INDEX]
    if not np.all(np.isfinite(ball)):
        return -1, None
    players = pos[: 2 * N_PLAYERS_PER_TEAM]
    dist = np.linalg.norm(players - ball, axis=1)
    if not np.any(np.isfinite(dist)):
        return -1, None
    idx = int(np.nanargmin(dist))
    if not np.isfinite(dist[idx]) or dist[idx] > radius:
        return -1, None
    return idx, "home" if idx < N_PLAYERS_PER_TEAM else "away"


def _player_key(player_id: object, jersey: object) -> str | None:
    """Stable identity for one tracking row, or None if it has neither id."""
    if pd.notna(player_id) and str(player_id) != "":
        return str(player_id)
    if pd.notna(jersey):
        return f"#{int(jersey)}"
    return None


def build_roster(df: pd.DataFrame, *, match_id: str | None = None) -> dict[str, list[str]]:
    """Slot order for each team, derived from the whole tracking frame.

    Slots have to be assigned from the roster rather than from whoever happens
    to be visible on a given frame. Ranking within the frame would slide every
    player above a missing one down by a slot, so a rollout would swap
    identities the moment tracking dropped somebody.

    Returns ``{"home": [key, ...], "away": [key, ...]}`` where the key is the
    ``player_id``, or ``#<jersey>`` when the id is missing. Ordered by jersey,
    then by key. A team with more than eleven distinct players over the frame
    (substitutions, or a mislabelled referee) keeps the eleven lowest jerseys
    and logs a warning.
    """
    work = df if match_id is None else df[df["match_id"] == match_id]
    roster: dict[str, list[str]] = {}
    for team in ("home", "away"):
        side = work.loc[work["team"] == team, ["player_id", "jersey"]]
        entries: dict[str, float] = {}
        seen: Counter[str] = Counter()
        for player_id, jersey in zip(side["player_id"], side["jersey"], strict=True):
            key = _player_key(player_id, jersey)
            if key is None:
                continue
            entries.setdefault(key, float(jersey) if pd.notna(jersey) else float("inf"))
            seen[key] += 1
        # Jersey first, because a file that carries numbers means them. Tracker
        # output carries none, and then every entry ties at infinity: falling
        # back to the id string would rank by an arbitrary label and let a
        # two-frame fragment take a slot from a player tracked all match. Row
        # count is the tiebreak that actually says which identities are real.
        ordered = sorted(entries, key=lambda k: (entries[k], -seen[k], k))
        if len(ordered) > N_PLAYERS_PER_TEAM:
            unnumbered = all(not np.isfinite(v) for v in entries.values())
            log.warning(
                "%s has %d distinct players; keeping the %d %s (%s dropped)",
                team,
                len(ordered),
                N_PLAYERS_PER_TEAM,
                "most-tracked" if unnumbered else "lowest jerseys",
                ", ".join(ordered[N_PLAYERS_PER_TEAM:]),
            )
            ordered = ordered[:N_PLAYERS_PER_TEAM]
        roster[team] = ordered
    return roster


def _first_finite(values: pd.Series, default: float = 0.0) -> float:
    for v in values:
        f = float(v) if pd.notna(v) else np.nan
        if np.isfinite(f):
            return f
    return default


def state_from_tracking(
    df: pd.DataFrame,
    frame: int,
    *,
    match_id: str | None = None,
    roster: Mapping[str, Sequence[str]] | None = None,
) -> MatchState:
    """Pull one frame of a canonical tracking frame into a ``MatchState``.

    Players are ordered by jersey within each team, from the roster of the
    whole tracking frame (see ``build_roster``), so the same player lands in
    the same slot on every frame. A team with fewer than eleven players on this
    frame leaves the absent slots as NaN; the players that are present keep
    their slots rather than sliding up into the gaps, which would hand one
    player's trajectory to another. A player not in the roster is skipped with
    a logged warning.

    Pass ``roster`` to reuse one built earlier: it is derived from every row of
    ``df``, so building it per frame is wasted work on a long match.
    """
    work = df
    if match_id is not None:
        work = work[work["match_id"] == match_id]
    rows = work[work["frame"] == frame]
    if rows.empty:
        where = "" if match_id is None else f" for match {match_id!r}"
        raise ValueError(f"frame {frame} not present in tracking frame{where}")

    if roster is None:
        roster = build_roster(work)

    pos = np.full((N_ROWS, 2), np.nan, dtype=np.float64)
    vel = np.full((N_ROWS, 2), np.nan, dtype=np.float64)

    for team, offset in (("home", 0), ("away", N_PLAYERS_PER_TEAM)):
        slots = {key: i for i, key in enumerate(roster.get(team, ()))}
        for row in rows[rows["team"] == team].itertuples(index=False):
            key = _player_key(row.player_id, row.jersey)
            slot = slots.get(key) if key is not None else None
            if slot is None:
                log.warning("frame %s: %s player %s is not in the roster", frame, team, key)
                continue
            pos[offset + slot] = (row.x, row.y)
            vel[offset + slot] = (row.vx, row.vy)

    ball_rows = rows[rows["team"] == "ball"]
    if len(ball_rows) > 1:
        log.warning("frame %s has %d ball rows; keeping the first", frame, len(ball_rows))
    if not ball_rows.empty:
        ball = ball_rows.iloc[0]
        pos[BALL_INDEX] = (ball["x"], ball["y"])
        vel[BALL_INDEX] = (ball["vx"], ball["vy"])

    vel = np.nan_to_num(vel, nan=0.0)
    possession, carrier_team = infer_possession(pos)
    return MatchState(
        t=_first_finite(rows["t"]),
        pos=pos,
        vel=vel,
        possession=possession,
        ball_carrier_team=carrier_team,
    )


def states_to_tracking(states: Sequence[MatchState], *, match_id: str) -> pd.DataFrame:
    """Emit a canonical tracking frame from a rollout.

    Frame numbers are the index into ``states``; identities come from the slot
    via ``slot_player_id`` and ``slot_jersey``, because a ``MatchState`` does
    not carry the original ids. NaN slots are omitted rather than written as
    NaN rows, which keeps the output the same shape as tracking data produced
    by the detector for a frame where a player was not seen.
    """
    records: list[dict[str, object]] = []
    for frame, state in enumerate(states):
        for idx in range(N_ROWS):
            x, y = state.pos[idx]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            vx, vy = state.vel[idx]
            is_ball = idx == BALL_INDEX
            records.append(
                {
                    "match_id": match_id,
                    "t": state.t,
                    "period": 1,
                    "frame": frame,
                    "team": "ball" if is_ball else slot_team(idx),
                    "player_id": slot_player_id(idx),
                    "jersey": pd.NA if is_ball else slot_jersey(idx),
                    "x": float(x),
                    "y": float(y),
                    "vx": float(vx),
                    "vy": float(vy),
                    "confidence": 1.0,
                }
            )
    frame_df = pd.DataFrame.from_records(records, columns=list(_RECORD_COLUMNS))
    return coerce_tracking_frame(frame_df)


_RECORD_COLUMNS: tuple[str, ...] = (
    "match_id",
    "t",
    "period",
    "frame",
    "team",
    "player_id",
    "jersey",
    "x",
    "y",
    "vx",
    "vy",
    "confidence",
)
