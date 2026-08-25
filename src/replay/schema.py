"""Canonical data contracts shared by every module in replay.

Everything that crosses a module boundary is defined here. Detection and
tracking stages exchange the small dataclasses; anything that reaches disk or
another tool in the suite goes through the tracking dataframe schema, which
matches sportio's canonical ``tracking`` frame exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# Pitch dimensions in meters. Origin is the bottom-left corner as seen by the
# home team attacking left-to-right, so x runs 0..105 and y runs 0..68.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

ObjectClass = Literal["player", "ball"]
TeamLabel = Literal["home", "away", "ball", "referee"]

#: Column order of the canonical tracking frame. Matches sportio's ``tracking``
#: schema; downstream tools (pressing) rely on this exact ordering.
TRACKING_COLUMNS: tuple[str, ...] = (
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

#: dtypes for the canonical tracking frame. ``jersey`` and ``player_id`` are
#: nullable because a track can exist before it is assigned an identity.
TRACKING_DTYPES: dict[str, str] = {
    "match_id": "string",
    "t": "float64",
    "period": "int64",
    "frame": "int64",
    "team": "string",
    "player_id": "string",
    "jersey": "Int64",
    "x": "float64",
    "y": "float64",
    "vx": "float64",
    "vy": "float64",
    "confidence": "float64",
}


@dataclass(slots=True)
class Detection:
    """One detected object in one frame, in image pixel coordinates."""

    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: ObjectClass = "player"

    @property
    def xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-center of the box: where the object meets the ground plane.

        This is the point the homography maps to pitch coordinates. Using the
        box center instead would project players onto the pitch roughly half a
        body-height further from the camera.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def wh(self) -> tuple[float, float]:
        return (self.x2 - self.x1, self.y2 - self.y1)


@dataclass(slots=True)
class TrackedBox:
    """A detection that has been assigned a persistent track id."""

    frame: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: ObjectClass = "player"
    team: TeamLabel | None = None
    team_confidence: float = 0.0

    @property
    def xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(slots=True)
class FrameDetections:
    """All detections for a single video frame, plus the frame's timestamp."""

    frame: int
    t: float
    detections: list[Detection] = field(default_factory=list)


def empty_tracking_frame() -> pd.DataFrame:
    """An empty dataframe with the canonical tracking columns and dtypes."""
    df = pd.DataFrame({c: pd.Series(dtype=TRACKING_DTYPES[c]) for c in TRACKING_COLUMNS})
    return df[list(TRACKING_COLUMNS)]


def coerce_tracking_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with canonical columns, order and dtypes.

    Raises ``ValueError`` if a required column is missing. Extra columns are
    dropped so that writers never leak internal state into the output file.
    """
    missing = [c for c in TRACKING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"tracking frame is missing required columns: {missing}")
    out = df.loc[:, list(TRACKING_COLUMNS)].copy()
    for col, dtype in TRACKING_DTYPES.items():
        out[col] = out[col].astype(dtype)
    return out.reset_index(drop=True)
