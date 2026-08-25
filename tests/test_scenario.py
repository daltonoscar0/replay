from pathlib import Path

import pytest

from replay.sim.scenario import (
    InitialState,
    PolicyConfig,
    Scenario,
    ScenarioError,
    load_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED = REPO_ROOT / "scenarios" / "press_higher.yaml"

MINIMAL = """
name: minimal
initial_state:
  source: auto
policy:
  kind: rule_based
"""


def write(tmp_path, text, name="s.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_press_higher_loads():
    scenario = load_scenario(SHIPPED)
    assert scenario.name == "press_higher"
    assert scenario.initial_state.is_auto
    assert scenario.initial_state.frame is None
    assert scenario.policy.kind == "rule_based"
    assert scenario.policy.defensive_line == pytest.approx(1.5)
    assert scenario.horizon_s > 0
    assert scenario.n_rollouts >= 1


def test_shipped_scenarios_all_load():
    files = sorted((REPO_ROOT / "scenarios").glob("*.yaml"))
    assert files, "no scenario files shipped"
    for path in files:
        assert load_scenario(path).name


def test_defaults_apply_when_fields_omitted(tmp_path):
    scenario = load_scenario(write(tmp_path, MINIMAL))
    assert scenario.horizon_s == 10.0
    assert scenario.n_rollouts == 32
    assert scenario.seed == 0
    assert scenario.description == ""
    assert scenario.policy.defensive_line == 0.0
    assert scenario.policy.press_trigger_m == 12.0
    assert scenario.policy.line_compactness == 1.0
    assert scenario.policy.pass_lane_weight == 1.0


def test_empty_file_is_all_defaults(tmp_path):
    scenario = load_scenario(write(tmp_path, "# nothing here\n"))
    assert scenario == Scenario()
    assert scenario.initial_state.is_auto


def test_overrides_property_is_policy_kwargs(tmp_path):
    scenario = load_scenario(write(tmp_path, MINIMAL))
    assert scenario.policy.overrides == {
        "defensive_line": 0.0,
        "press_trigger_m": 12.0,
        "line_compactness": 1.0,
        "pass_lane_weight": 1.0,
    }


def test_nested_overrides_block_is_accepted(tmp_path):
    text = """
policy:
  kind: rule_based
  overrides:
    defensive_line: -2.5
    press_trigger_m: 20.0
"""
    scenario = load_scenario(write(tmp_path, text))
    assert scenario.policy.defensive_line == -2.5
    assert scenario.policy.press_trigger_m == 20.0


def test_defensive_line_may_be_negative(tmp_path):
    scenario = load_scenario(write(tmp_path, "policy:\n  defensive_line: -4.0\n"))
    assert scenario.policy.defensive_line == -4.0


def test_bad_field_type_names_the_field(tmp_path):
    path = write(tmp_path, "policy:\n  defensive_line: high\n")
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "policy.defensive_line" in message
    assert "\n" not in message
    assert "Traceback" not in message


def test_bad_top_level_type_names_the_field(tmp_path):
    path = write(tmp_path, "horizon_s: soon\n")
    with pytest.raises(ScenarioError, match="horizon_s"):
        load_scenario(path)


def test_unknown_field_is_rejected(tmp_path):
    path = write(tmp_path, "presing_trigger_m: 9.0\n")
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "presing_trigger_m" in str(excinfo.value)
    assert "unknown field" in str(excinfo.value)


def test_unknown_nested_field_is_rejected(tmp_path):
    path = write(tmp_path, "policy:\n  kind: rule_based\n  defence_line: 2\n")
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "policy.defence_line" in str(excinfo.value)


def test_unknown_policy_kind_is_rejected(tmp_path):
    path = write(tmp_path, "policy:\n  kind: vibes\n")
    with pytest.raises(ScenarioError, match="policy.kind"):
        load_scenario(path)


def test_bc_policy_requires_weights(tmp_path):
    with pytest.raises(ScenarioError, match="weights"):
        load_scenario(write(tmp_path, "policy:\n  kind: bc\n"))
    scenario = load_scenario(write(tmp_path, "policy:\n  kind: bc\n  weights: bc.npz\n"))
    assert scenario.policy.weights == "bc.npz"


def test_explicit_source_requires_frame(tmp_path):
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(write(tmp_path, "initial_state:\n  source: data/derby.csv\n"))
    assert "initial_state" in str(excinfo.value)
    assert "frame is required" in str(excinfo.value)

    scenario = load_scenario(
        write(tmp_path, "initial_state:\n  source: data/derby.csv\n  frame: 4120\n")
    )
    assert scenario.initial_state.source == "data/derby.csv"
    assert scenario.initial_state.frame == 4120
    assert not scenario.initial_state.is_auto


def test_auto_source_rejects_a_frame(tmp_path):
    with pytest.raises(ScenarioError, match="frame must be omitted"):
        load_scenario(write(tmp_path, "initial_state:\n  source: auto\n  frame: 10\n"))


def test_out_of_range_values_are_rejected(tmp_path):
    with pytest.raises(ScenarioError, match="horizon_s"):
        load_scenario(write(tmp_path, "horizon_s: 0\n"))
    with pytest.raises(ScenarioError, match="n_rollouts"):
        load_scenario(write(tmp_path, "n_rollouts: 0\n"))
    with pytest.raises(ScenarioError, match="seed"):
        load_scenario(write(tmp_path, "seed: -1\n"))
    with pytest.raises(ScenarioError, match="policy.press_trigger_m"):
        load_scenario(write(tmp_path, "policy:\n  press_trigger_m: 0\n"))


def test_malformed_yaml_is_a_scenario_error(tmp_path):
    path = write(tmp_path, "policy: [unclosed\n")
    with pytest.raises(ScenarioError, match="not valid YAML"):
        load_scenario(path)


def test_non_mapping_yaml_is_a_scenario_error(tmp_path):
    with pytest.raises(ScenarioError, match="must be a mapping"):
        load_scenario(write(tmp_path, "- press\n- higher\n"))


def test_missing_file_is_a_scenario_error(tmp_path):
    with pytest.raises(ScenarioError, match="cannot read scenario file"):
        load_scenario(tmp_path / "nope.yaml")


def test_models_constructible_directly():
    scenario = Scenario(
        name="deep_block",
        initial_state=InitialState(source="x.csv", frame=3),
        policy=PolicyConfig(kind="rule_based", defensive_line=-6.0),
        horizon_s=5.0,
        n_rollouts=4,
        seed=11,
    )
    assert scenario.policy.overrides["defensive_line"] == -6.0
    assert scenario.initial_state.frame == 3
