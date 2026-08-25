"""Chooses between a real ``sportio`` install and the vendored Metrica IO.

``sportio`` is the shared IO library for this suite. When it is importable and
exposes a callable ``to_metrica``, we delegate to it so that replay and the
other tools cannot drift apart on the on-disk format. When it is absent --
which is the normal case for a standalone replay checkout -- the vendored
implementation in :mod:`replay.io.metrica` runs instead.

The lookup is redone on every call rather than cached at import time, because
whether ``sportio`` is importable can change inside a process (an install into
the running env, or a test substituting a stub) and a cached answer would then
be silently wrong.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import ModuleType
from typing import Literal

import pandas as pd

from replay.io.metrica import from_canonical_csv, to_canonical_csv
from replay.io.metrica import from_metrica as _vendored_from_metrica
from replay.io.metrica import to_metrica as _vendored_to_metrica
from replay.schema import coerce_tracking_frame

__all__ = [
    "backend_name",
    "from_canonical_csv",
    "from_metrica",
    "to_canonical_csv",
    "to_metrica",
]

log = logging.getLogger(__name__)

#: A module only counts as a usable backend if it provides these. ``to_metrica``
#: is the marker the contract names; a module carrying only ``from_metrica``
#: would leave the writer and reader on different backends.
_REQUIRED_ATTRS: tuple[str, ...] = ("to_metrica",)


def _sportio() -> ModuleType | None:
    try:
        module = importlib.import_module("sportio")
    except ImportError:
        return None
    for attr in _REQUIRED_ATTRS:
        if not callable(getattr(module, attr, None)):
            log.debug("sportio is installed but has no callable %s; using vendored IO", attr)
            return None
    return module


def backend_name() -> Literal["sportio", "vendored"]:
    """Which implementation :func:`to_metrica` will actually run."""
    return "sportio" if _sportio() is not None else "vendored"


def to_metrica(df: pd.DataFrame, path: str | Path, *, fps: float = 10.0) -> Path:
    module = _sportio()
    if module is not None:
        return Path(module.to_metrica(df, path, fps=fps))
    return _vendored_to_metrica(df, path, fps=fps)


def from_metrica(path: str | Path, *, match_id: str = "unknown") -> pd.DataFrame:
    module = _sportio()
    reader = getattr(module, "from_metrica", None) if module is not None else None
    if callable(reader):
        # Coerce even the sportio result: callers downstream index by
        # TRACKING_COLUMNS and must not have to care which backend ran.
        return coerce_tracking_frame(reader(path, match_id=match_id))
    return _vendored_from_metrica(path, match_id=match_id)
