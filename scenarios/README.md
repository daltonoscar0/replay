# Scenarios

A scenario file is a tactical question written down: start from this moment,
play this way, and see what changes. Each file is YAML, validated on load by
`replay.sim.scenario.load_scenario`. A bad field comes back as one line naming
the field, no stack trace.

```bash
replay sim --tracking data/derby.csv --scenario scenarios/press_higher.yaml
```

Every field is optional; the defaults below are what you get if you leave one
out. Unknown fields are rejected rather than ignored, so a typo fails loudly
instead of silently doing nothing.

## Top level

| Field | Type | Default | Units | Meaning |
|---|---|---|---|---|
| `name` | string | `unnamed` |, | Short label used in reports and output filenames. |
| `description` | string | `""` |, | Free text. What question is this asking? |
| `initial_state` | mapping | `{source: auto}` |, | Where the rollout starts. See below. |
| `policy` | mapping | rule-based baseline |, | Which policy runs, and its overrides. See below. |
| `horizon_s` | float > 0 | `10.0` | seconds | Simulated duration of one rollout. |
| `n_rollouts` | int >= 1 | `32` | rollouts | How many rollouts to average over. |
| `seed` | int >= 0 | `0` |, | Root seed. The same seed gives bit-identical metrics. |

## `initial_state`

| Field | Type | Default | Units | Meaning |
|---|---|---|---|---|
| `source` | string | `auto` |, | Path to a canonical tracking CSV, or the literal `auto`. |
| `frame` | int >= 0 | none | frame index | Frame to start from. Required with a file, forbidden with `auto`. |

`auto` means "use the tracking file passed on the command line and pick a
sensible frame from it". Prefer it for scenarios you intend to share, since a
hard-coded path only works on the machine it was written on. Pin `source` and
`frame` when you want a specific moment, a particular turnover, say, and the
same start on every run.

## `policy`

| Field | Type | Default | Units | Meaning |
|---|---|---|---|---|
| `kind` | `rule_based` \| `bc` | `rule_based` |, | Hand-written policy, or a behaviour-cloned one. |
| `weights` | string | none |, | Path to the trained policy. Required when `kind: bc`. |
| `defensive_line` | float | `0.0` | meters | Shift of the defending block's resting depth. Positive is higher up the pitch, negative is deeper. |
| `press_trigger_m` | float > 0 | `12.0` | meters | Distance to the ball at which the nearest defender commits to a press. Lower is more passive. |
| `line_compactness` | float >= 0 | `1.0` | multiplier | How hard the block squeezes towards its own centroid. `0` turns it off, `>1` is narrower. |
| `pass_lane_weight` | float >= 0 | `1.0` | multiplier | Weight on occupying passing lanes versus closing the carrier down. |

The four tuning knobs may also be written as a nested block, which validates
identically:

```yaml
policy:
  kind: rule_based
  overrides:
    defensive_line: 1.5
```

## Shipped scenarios

- `press_higher.yaml`, pushes the defending line up 1.5 m and triggers the
  press 2 m earlier, holding everything else at the baseline.

## Writing your own

Copy `press_higher.yaml`, change `name` and one knob. Changing one knob at a
time is the point: the report compares the scenario against a baseline rollout
from the same start with default settings, so a single-field diff is a clean
attribution and a five-field diff is not.
