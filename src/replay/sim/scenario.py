"""Scenario files: the tactical question, written down.

A scenario says where to start, which policy to run, and how many rollouts to
average over. It is a YAML file because the person editing it is a coach or an
analyst, not a programmer, which also sets the bar for error messages: a typo
must come back as one line naming the field, never as a stack trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

#: Policy knobs a scenario is allowed to override, in the order they are
#: documented in ``scenarios/README.md``.
OVERRIDE_FIELDS: tuple[str, ...] = (
    "defensive_line",
    "press_trigger_m",
    "line_compactness",
    "pass_lane_weight",
)

AUTO_SOURCE = "auto"


class ScenarioError(ValueError):
    """A scenario file could not be read, parsed or validated."""


class InitialState(BaseModel):
    """Where the rollout starts.

    Either an explicit ``source`` tracking CSV plus a ``frame`` number, or
    ``source: auto``, which means "pick a sensible frame from the tracking
    passed on the command line". ``auto`` is what the shipped scenarios use, so
    they are not pinned to one analyst's local file path.
    """

    model_config = ConfigDict(extra="forbid")

    source: str = AUTO_SOURCE
    frame: int | None = Field(default=None, ge=0)

    @property
    def is_auto(self) -> bool:
        return self.source == AUTO_SOURCE

    @model_validator(mode="after")
    def _check_pairing(self) -> InitialState:
        if self.is_auto:
            if self.frame is not None:
                raise ValueError("frame must be omitted when source is 'auto'")
        elif self.frame is None:
            raise ValueError("frame is required when source is a tracking file")
        return self


class PolicyConfig(BaseModel):
    """Which policy to run and the overrides applied to it.

    The override fields are flat so the YAML reads like a set of instructions,
    but a nested ``overrides:`` mapping is accepted too and folded up before
    validation; both spellings turn up in hand-written files.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["rule_based", "bc"] = "rule_based"
    #: Weights file for ``kind: bc``. Ignored by the rule-based policy.
    weights: str | None = None
    #: Meters to push the defending block up (positive) or drop it (negative).
    defensive_line: float = 0.0
    #: Distance to the ball at which the nearest defender commits to a press.
    press_trigger_m: float = Field(default=12.0, gt=0.0)
    #: How hard the block squeezes towards its own centroid. 0 disables it.
    line_compactness: float = Field(default=1.0, ge=0.0)
    #: Weight on blocking passing lanes versus closing the carrier down.
    pass_lane_weight: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _flatten_overrides(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "overrides" not in data:
            return data
        nested = data["overrides"]
        if not isinstance(nested, dict):
            raise ValueError("overrides must be a mapping of policy knobs")
        merged = {k: v for k, v in data.items() if k != "overrides"}
        # Explicit top-level values win over the nested block.
        for key, value in nested.items():
            merged.setdefault(key, value)
        return merged

    @model_validator(mode="after")
    def _check_weights(self) -> PolicyConfig:
        if self.kind == "bc" and not self.weights:
            raise ValueError("weights is required when kind is 'bc'")
        return self

    @property
    def overrides(self) -> dict[str, float]:
        """The tuning knobs as constructor kwargs for the policy."""
        return {name: float(getattr(self, name)) for name in OVERRIDE_FIELDS}


class Scenario(BaseModel):
    """A complete counterfactual: start here, play this way, this many times."""

    model_config = ConfigDict(extra="forbid")

    name: str = "unnamed"
    description: str = ""
    initial_state: InitialState = Field(default_factory=InitialState)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    horizon_s: float = Field(default=10.0, gt=0.0)
    n_rollouts: int = Field(default=32, ge=1)
    seed: int = Field(default=0, ge=0)


def _format_location(loc: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        elif parts:
            parts.append(f".{item}")
        else:
            parts.append(str(item))
    return "".join(parts) or "<root>"


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    """One readable line naming every offending field."""
    problems = []
    for err in exc.errors():
        field = _format_location(err.get("loc", ()))
        message = err.get("msg", "invalid value")
        if err.get("type") == "extra_forbidden":
            message = "unknown field"
        problems.append(f"{field}: {message}")
    return f"{path.name}: " + "; ".join(problems)


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate a scenario YAML file.

    Every failure mode comes back as a ``ScenarioError`` whose message names
    the file and, where relevant, the field.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"cannot read scenario file {p}: {exc.strerror or exc}") from None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        detail = str(exc).replace("\n", " ")
        raise ScenarioError(f"{p.name} is not valid YAML: {detail}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ScenarioError(
            f"{p.name}: top level must be a mapping of fields, got {type(raw).__name__}"
        )
    try:
        return Scenario.model_validate(raw)
    except ValidationError as exc:
        raise ScenarioError(_format_validation_error(p, exc)) from None


def dump_scenario(scenario: Scenario, path: str | Path) -> Path:
    """Write a scenario back out as YAML. Comments are not preserved."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(scenario.model_dump(), sort_keys=False), encoding="utf-8")
    return p
