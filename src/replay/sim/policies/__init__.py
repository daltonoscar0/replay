"""Policies that turn a :class:`MatchState` into desired player velocities."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from replay.sim.policies.bc import BCPolicy
from replay.sim.policies.rule_based import (
    RuleBasedPolicy,
    lane_clearance,
    point_segment_distance,
)
from replay.sim.state import MatchState


class Policy(Protocol):
    """Anything the rollout driver can ask for a team's next velocities."""

    def desired_velocity(
        self, state: MatchState, team: Literal["home", "away"], rng: np.random.Generator
    ) -> np.ndarray:
        """(11, 2) desired velocities in m/s, in row order 0..10 for that team."""
        ...


__all__ = [
    "BCPolicy",
    "Policy",
    "RuleBasedPolicy",
    "lane_clearance",
    "point_segment_distance",
]
