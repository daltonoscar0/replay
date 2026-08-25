"""Read SoccerNet tracking / Game State Reconstruction ground truth.

Two on-disk layouts are supported, both as documented by the SoccerNet
challenge releases.

MOT-style (SoccerNet-Tracking)::

    <sequence>/
        gameinfo.ini      # [Sequence] plus trackletID_<n>=<role> team <side>; <jersey>
        seqinfo.ini       # [Sequence] name, imDir, frameRate, seqLength, imWidth, ...
        gt/gt.txt         # frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1
        img1/000001.jpg

Game State Reconstruction (SoccerNet-GSR)::

    <sequence>/
        Labels-GameState.json   # COCO-like: images / annotations / categories,
                                # annotations carry bbox_image and bbox_pitch
        img1/000001.jpg

Only the GSR labels contain pitch coordinates. MOT-style ground truth is image
boxes and nothing else, so converting it to the canonical tracking frame needs a
homography supplied by the caller.

The data itself is behind a signed NDA and a password-protected download. This
module never attempts to fetch or unlock anything; when the data is not on disk
it raises :class:`SoccerNetUnavailable` naming exactly what the user has to do.
"""

from __future__ import annotations

import configparser
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from replay.schema import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    TRACKING_COLUMNS,
    coerce_tracking_frame,
    empty_tracking_frame,
)

__all__ = [
    "SoccerNetUnavailable",
    "SequenceInfo",
    "available",
    "find_sequences",
    "load_soccernet_gt",
]

log = logging.getLogger(__name__)

_GT_RELPATH = ("gt", "gt.txt")
_GSR_LABEL_NAMES = ("Labels-GameState.json", "labels-gamestate.json")

_OBTAIN_HELP = (
    "SoccerNet data is distributed under an NDA and the archives are "
    "password-protected. Download it yourself from https://www.soccer-net.org "
    "(accept the NDA to receive the password), extract the sequences, and point "
    "this loader at the directory that contains them. Nothing here will fetch or "
    "unlock the data for you."
)

#: SoccerNet role words mapped onto the canonical ``team`` vocabulary. Roles that
#: are not one of the two sides get their own label so they can be filtered out
#: rather than silently counted as a player of some team.
_SIDE_TO_TEAM = {"left": "home", "right": "away"}


class SoccerNetUnavailable(RuntimeError):
    """Raised when SoccerNet ground truth is absent, empty, or still packed."""


@dataclass(slots=True)
class SequenceInfo:
    """One SoccerNet sequence directory and which annotation style it carries."""

    path: Path
    name: str
    kind: str  # "mot" or "gsr"
    frame_rate: float = 25.0
    seq_length: int = 0

    @property
    def gt_path(self) -> Path:
        if self.kind == "mot":
            return self.path.joinpath(*_GT_RELPATH)
        return _gsr_label_path(self.path) or self.path / _GSR_LABEL_NAMES[0]


def _gsr_label_path(seq_dir: Path) -> Path | None:
    for name in _GSR_LABEL_NAMES:
        candidate = seq_dir / name
        if candidate.is_file():
            return candidate
    # The release has been shipped with other capitalisations; fall back to a
    # case-insensitive scan of the directory before giving up.
    for candidate in sorted(seq_dir.glob("*.json")):
        if candidate.name.lower() in {n.lower() for n in _GSR_LABEL_NAMES}:
            return candidate
    return None


def _read_ini(path: Path) -> dict[str, str]:
    """Parse a SoccerNet ``.ini`` into a flat lowercase-keyed mapping.

    ``gameinfo.ini`` values contain ``;`` which configparser treats as an inline
    comment by default, so inline comments are disabled explicitly.
    """
    parser = configparser.ConfigParser(inline_comment_prefixes=(), strict=False)
    parser.optionxform = str  # type: ignore[method-assign]
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error as exc:
        raise ValueError(f"could not parse {path}: {exc}") from exc
    out: dict[str, str] = {}
    for section in parser.sections():
        for key, value in parser.items(section):
            out[key.strip().lower()] = value.strip()
    return out


def _classify(seq_dir: Path) -> SequenceInfo | None:
    """Return a :class:`SequenceInfo` if ``seq_dir`` looks like a sequence."""
    if not seq_dir.is_dir():
        return None
    kind: str | None = None
    if seq_dir.joinpath(*_GT_RELPATH).is_file():
        kind = "mot"
    elif _gsr_label_path(seq_dir) is not None:
        kind = "gsr"
    if kind is None:
        return None

    meta: dict[str, str] = {}
    for ini_name in ("seqinfo.ini", "gameinfo.ini"):
        ini = seq_dir / ini_name
        if ini.is_file():
            meta.update(_read_ini(ini))

    name = meta.get("name") or seq_dir.name
    try:
        frame_rate = float(meta.get("framerate", 25.0))
    except ValueError:
        frame_rate = 25.0
    try:
        seq_length = int(meta.get("seqlength", 0))
    except ValueError:
        seq_length = 0
    return SequenceInfo(
        path=seq_dir,
        name=name,
        kind=kind,
        frame_rate=frame_rate if frame_rate > 0 else 25.0,
        seq_length=seq_length,
    )


def find_sequences(path: str | Path) -> list[SequenceInfo]:
    """All SoccerNet sequences at or below ``path``, sorted by name.

    ``path`` may be a single sequence directory, a split directory holding many
    of them, or a dataset root holding split directories.
    """
    root = Path(path)
    if not root.exists():
        return []

    direct = _classify(root)
    if direct is not None:
        return [direct]

    found: dict[Path, SequenceInfo] = {}
    for child in sorted(root.iterdir()):
        info = _classify(child)
        if info is not None:
            found[child] = info
            continue
        if child.is_dir():
            for grandchild in sorted(child.iterdir()):
                info = _classify(grandchild)
                if info is not None:
                    found[grandchild] = info
    return [found[k] for k in sorted(found)]


def available(path: str | Path) -> bool:
    """True if at least one readable SoccerNet sequence sits at ``path``."""
    try:
        return bool(find_sequences(path))
    except OSError as exc:  # unreadable mount, permission denied
        log.warning("cannot inspect SoccerNet path %s: %s", path, exc)
        return False


def _unavailable(root: Path) -> SoccerNetUnavailable:
    """Build the most specific unavailability message the filesystem supports."""
    if not root.exists():
        reason = f"nothing exists at {root}"
    elif root.is_file():
        reason = f"{root} is a file, not a sequence directory"
    else:
        archives = [p.name for p in sorted(root.glob("*.zip"))]
        if archives:
            reason = (
                f"{root} contains only archives ({', '.join(archives[:3])}"
                f"{'...' if len(archives) > 3 else ''}) and no extracted sequences; "
                "they are password-protected and must be extracted by hand"
            )
        else:
            reason = (
                f"{root} exists but holds no sequence with either gt/gt.txt or "
                "Labels-GameState.json"
            )
    return SoccerNetUnavailable(f"SoccerNet ground truth unavailable: {reason}. {_OBTAIN_HELP}")


def _parse_tracklets(seq_dir: Path) -> dict[int, dict[str, Any]]:
    """Map tracklet id -> role/team/jersey from ``gameinfo.ini``.

    Values look like ``player team left; 12``, ``goalkeeper team right``,
    ``referee``, ``ball`` or ``other``. Anything unrecognised is left with a
    ``None`` team rather than being forced into home or away.
    """
    ini = seq_dir / "gameinfo.ini"
    if not ini.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for key, value in _read_ini(ini).items():
        m = re.fullmatch(r"trackletid_(\d+)", key)
        if m is None:
            continue
        tid = int(m.group(1))
        text = value.strip()
        jersey: int | None = None
        if ";" in text:
            text, _, jersey_text = text.partition(";")
            jersey_digits = re.search(r"\d+", jersey_text)
            if jersey_digits is not None:
                jersey = int(jersey_digits.group(0))
        text = text.strip().lower()
        role = text.split()[0] if text else "other"
        team: str | None = None
        side = re.search(r"team\s+(left|right)", text)
        if side is not None:
            team = _SIDE_TO_TEAM[side.group(1)]
        elif role == "ball":
            team = "ball"
        elif role == "referee":
            team = "referee"
        out[tid] = {"role": role, "team": team, "jersey": jersey}
    return out


def _load_mot(
    info: SequenceInfo,
    *,
    match_id: str,
    pitch_from_image: Callable[[np.ndarray], np.ndarray] | None,
) -> pd.DataFrame:
    gt_file = info.gt_path
    rows = np.loadtxt(gt_file, delimiter=",", ndmin=2, usecols=range(6))
    if rows.size == 0:
        return empty_tracking_frame()

    frames = rows[:, 0].astype(np.int64)
    track_ids = rows[:, 1].astype(np.int64)
    left = rows[:, 2].astype(np.float64)
    top = rows[:, 3].astype(np.float64)
    width = rows[:, 4].astype(np.float64)
    height = rows[:, 5].astype(np.float64)

    # The ground-plane contact point of a box is its bottom-center; the box
    # center would project half a body height too far from the camera.
    foot = np.stack([left + width / 2.0, top + height], axis=1)

    if pitch_from_image is None:
        raise ValueError(
            f"{gt_file} is MOT-style SoccerNet-Tracking ground truth, which stores image "
            "boxes and no pitch coordinates. Pass pitch_from_image=<homography>.image_to_pitch "
            "to project it, or use a SoccerNet-GSR sequence whose annotations carry bbox_pitch."
        )
    pitch = np.asarray(pitch_from_image(foot), dtype=np.float64)
    if pitch.shape != foot.shape:
        raise ValueError(f"pitch_from_image returned shape {pitch.shape}, expected {foot.shape}")

    tracklets = _parse_tracklets(info.path)
    teams = [tracklets.get(int(t), {}).get("team") for t in track_ids]
    jerseys = [tracklets.get(int(t), {}).get("jersey") for t in track_ids]

    df = pd.DataFrame(
        {
            "match_id": match_id,
            "t": (frames - 1) / info.frame_rate,
            "period": 1,
            "frame": frames,
            "team": pd.array(teams, dtype="string"),
            "player_id": pd.array([str(int(t)) for t in track_ids], dtype="string"),
            "jersey": pd.array(jerseys, dtype="Int64"),
            "x": pitch[:, 0],
            "y": pitch[:, 1],
            "vx": np.nan,
            "vy": np.nan,
            "confidence": 1.0,
        }
    )
    return coerce_tracking_frame(df)


def _gsr_team(attributes: dict[str, Any], supercategory: str, category: str) -> str | None:
    role = str(attributes.get("role") or category or supercategory or "").lower()
    team = attributes.get("team")
    if team is not None:
        side = str(team).strip().lower()
        if side in _SIDE_TO_TEAM:
            return _SIDE_TO_TEAM[side]
    if "ball" in role:
        return "ball"
    if "referee" in role:
        return "referee"
    return None


def _gsr_jersey(attributes: dict[str, Any]) -> int | None:
    raw = attributes.get("jersey")
    if raw is None:
        return None
    digits = re.search(r"\d+", str(raw))
    return int(digits.group(0)) if digits is not None else None


def _load_gsr(info: SequenceInfo, *, match_id: str) -> pd.DataFrame:
    label_file = info.gt_path
    with label_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    images = payload.get("images") or []
    annotations = payload.get("annotations") or []
    categories = {int(c["id"]): str(c.get("name", "")) for c in payload.get("categories") or []}

    # image_id -> frame number. GSR image ids are strings that encode the video
    # and frame; prefer the explicit file name digits, which are the frame index.
    frame_of_image: dict[str, int] = {}
    for idx, image in enumerate(images, start=1):
        image_id = str(image.get("image_id", image.get("id", idx)))
        digits = re.search(r"(\d+)", str(image.get("file_name", "")))
        frame_of_image[image_id] = int(digits.group(1)) if digits is not None else idx

    records: list[dict[str, Any]] = []
    skipped_no_pitch = 0
    for ann in annotations:
        pitch_box = ann.get("bbox_pitch")
        if not pitch_box:
            skipped_no_pitch += 1
            continue
        x_raw = pitch_box.get("x_bottom_middle")
        y_raw = pitch_box.get("y_bottom_middle")
        if x_raw is None or y_raw is None:
            skipped_no_pitch += 1
            continue

        image_id = str(ann.get("image_id"))
        frame = frame_of_image.get(image_id)
        if frame is None:
            digits = re.search(r"(\d+)$", image_id)
            frame = int(digits.group(1)) if digits is not None else 0

        track_id = ann.get("track_id", ann.get("id"))
        attributes = ann.get("attributes") or {}
        supercategory = str(ann.get("supercategory", ""))
        category = categories.get(int(ann["category_id"]), "") if "category_id" in ann else ""

        # GSR pitch coordinates are meters from the center circle, +y toward the
        # top of the broadcast image. Canonical pitch coordinates put the origin
        # at the bottom-left corner with +y up, hence the shift and the flip.
        x_m = float(x_raw) + PITCH_LENGTH / 2.0
        y_m = PITCH_WIDTH / 2.0 - float(y_raw)

        records.append(
            {
                "match_id": match_id,
                "t": (frame - 1) / info.frame_rate,
                "period": 1,
                "frame": int(frame),
                "team": _gsr_team(attributes, supercategory, category),
                "player_id": str(track_id),
                "jersey": _gsr_jersey(attributes),
                "x": x_m,
                "y": y_m,
                "vx": np.nan,
                "vy": np.nan,
                "confidence": 1.0,
            }
        )

    if skipped_no_pitch:
        log.warning(
            "%s: %d annotation(s) had no bbox_pitch and were skipped",
            label_file,
            skipped_no_pitch,
        )
    if not records:
        return empty_tracking_frame()

    df = pd.DataFrame.from_records(records, columns=list(TRACKING_COLUMNS))
    df["team"] = df["team"].astype("string")
    df["jersey"] = pd.array(df["jersey"], dtype="Int64")
    return coerce_tracking_frame(df.sort_values(["frame", "player_id"], kind="stable"))


def load_soccernet_gt(
    path: str | Path,
    *,
    match_id: str | None = None,
    pitch_from_image: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Load SoccerNet ground truth at ``path`` as a canonical tracking frame.

    ``path`` may be one sequence directory or a directory of them; several
    sequences are concatenated with ``match_id`` set per sequence unless
    ``match_id`` is given explicitly.

    ``pitch_from_image`` maps an ``(N, 2)`` array of image points to pitch
    meters; ``Homography.image_to_pitch`` satisfies it. It is only consulted for
    MOT-style sequences, which have no pitch annotation of their own.

    Raises :class:`SoccerNetUnavailable` when there is no data to read.
    """
    root = Path(path)
    sequences = find_sequences(root)
    if not sequences:
        raise _unavailable(root)

    parts: list[pd.DataFrame] = []
    for info in sequences:
        seq_match_id = match_id if match_id is not None else info.name
        if info.kind == "mot":
            part = _load_mot(info, match_id=seq_match_id, pitch_from_image=pitch_from_image)
        else:
            part = _load_gsr(info, match_id=seq_match_id)
        parts.append(part)

    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        raise SoccerNetUnavailable(
            f"SoccerNet ground truth unavailable: found {len(sequences)} sequence(s) under "
            f"{root} but every annotation file was empty. {_OBTAIN_HELP}"
        )
    return coerce_tracking_frame(pd.concat(non_empty, ignore_index=True))
