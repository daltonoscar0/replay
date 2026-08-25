# Internal interfaces

The contract between modules. Every public name below is what other modules
import; changing a signature here means changing every caller, so don't.
Shared dataclasses (`Detection`, `TrackedBox`, `FrameDetections`) and the
canonical tracking frame live in `replay/schema.py`.

Conventions used throughout:

- Image coordinates are pixels, `(u, v)`, origin top-left, `+v` downward.
- Pitch coordinates are meters, `(x, y)`, origin bottom-left, `x in [0, 105]`,
  `y in [0, 68]`, home attacking left-to-right. Constants: `PITCH_LENGTH`,
  `PITCH_WIDTH`.
- Arrays of points are `(N, 2)` float64 unless stated otherwise.
- Every module is fully type-hinted and passes `ruff check` with the repo config.

## replay.io.metrica

```python
def to_metrica(df: pd.DataFrame, path: str | Path, *, fps: float = 10.0) -> Path
def from_metrica(path: str | Path, *, match_id: str = "unknown") -> pd.DataFrame
def to_canonical_csv(df: pd.DataFrame, path: str | Path) -> Path
def from_canonical_csv(path: str | Path) -> pd.DataFrame
```

`to_metrica` writes Metrica Sports wide format: a header block of
`Period, Frame, Time [s]`, then an `X, Y` column pair per player in stable
jersey order, then `Ball_X, Ball_Y`. Positions are normalized to `[0, 1]` by
pitch dimensions, matching the Metrica sample-data release. Home and away are
written to one file with `Home_<jersey>_X` / `Away_<jersey>_X` prefixes; the
released Metrica data splits these across two files, and that deviation is
documented in the module docstring and the README.

`from_metrica` inverts it and returns a canonical tracking frame
(`schema.TRACKING_COLUMNS`, dtypes per `schema.TRACKING_DTYPES`). Round-trip
must be exact to float tolerance.

The canonical CSV writers are the long-format passthrough: one row per
object per frame, exactly `TRACKING_COLUMNS`.

## replay.io (package `__init__`)

Re-exports `to_metrica`, `from_metrica`, and `backend_name()` from
`replay.io._adapter`. The adapter prefers a real `sportio` install and falls
back to the vendored implementation:

```python
def backend_name() -> Literal["sportio", "vendored"]
```

## replay.track.detect

```python
class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]: ...

class ColorDetector:      # jersey/ball blobs against a green pitch mask
    def __init__(self, *, min_area: int = 30, ball_max_area: int = 120) -> None
class YoloDetector:       # ultralytics, lazy-imported
    def __init__(self, *, weights: str = "yolov8n.pt", device: str = "auto",
                 conf: float = 0.25) -> None

def build_detector(name: Literal["color", "yolo"], **kwargs: Any) -> Detector
def resolve_device(device: str = "auto") -> str   # -> "cuda" | "mps" | "cpu"
```

`ColorDetector` is the offline path used by `replay demo` on the synthetic
clip: no weights, no downloads, CPU only. `YoloDetector` is the real-footage
path and must not be imported at module import time.

## replay.track.byte_track

```python
class ByteTrack:
    def __init__(self, *, high_thresh: float = 0.5, low_thresh: float = 0.1,
                 match_thresh: float = 0.8, max_age: int = 30,
                 min_hits: int = 3) -> None
    def update(self, detections: list[Detection], frame_idx: int) -> list[TrackedBox]
```

Two-stage association exactly as in the ByteTrack paper: high-score detections
matched to confirmed tracks by IoU first, then low-score detections matched to
whatever is left over, then unmatched high-score detections spawn new tracks.
Track ids are stable and monotonically increasing from 1.

## replay.track.kalman

```python
def smooth_track(t: np.ndarray, xy: np.ndarray, *, process_var: float = 1.0,
                 measurement_var: float = 0.25) -> tuple[np.ndarray, np.ndarray]
```

Constant-velocity Kalman filter plus RTS backward smoother over pitch-meter
positions. Returns `(xy_smoothed, velocity)`, both `(N, 2)`. `t` is seconds and
may be irregularly spaced. Gaps (NaN rows in `xy`) are predicted through, not
dropped.

## replay.track.homography

```python
@dataclass
class Homography:
    H: np.ndarray                                   # (3, 3), image -> pitch
    def image_to_pitch(self, pts: np.ndarray) -> np.ndarray
    def pitch_to_image(self, pts: np.ndarray) -> np.ndarray
    @classmethod
    def from_point_pairs(cls, image_pts, pitch_pts) -> "Homography"
    def to_json(self, path) -> Path
    @classmethod
    def from_json(cls, path) -> "Homography"
    @property
    def reprojection_error(self) -> float            # meters, NaN if unknown

PITCH_KEYPOINTS: dict[str, tuple[float, float]]      # named landmarks in meters
def estimate_from_lines(frame_bgr: np.ndarray) -> Homography | None
def estimate_from_soccernet(frame_bgr: np.ndarray) -> Homography | None
```

`from_point_pairs` solves the DLT with normalized coordinates (Hartley
normalization) and stores the mean reprojection error in meters.
`estimate_from_lines` masks the pitch by green hue, finds lines with a
probabilistic Hough transform, and fits them to `PITCH_KEYPOINTS`; it returns
`None` rather than a bad guess when it cannot find enough structure.
`estimate_from_soccernet` is the hook for the SoccerNet calibration model and
returns `None` with a logged note when the model is unavailable.

## replay.track.teams

```python
def assign_teams(
    boxes_by_frame: Mapping[int, list[TrackedBox]],
    frame_reader: Callable[[int], np.ndarray | None],
    *, n_samples: int = 20,
) -> dict[int, tuple[TeamLabel, float]]
```

Samples up to `n_samples` torso crops per track (upper-middle 40% of the box,
horizontally inset 25% each side to drop background), converts to HSV, takes a
robust hue/saturation summary per track, then 2-means over tracks. Returns
`{track_id: (team_label, confidence)}` where confidence is the normalized
margin between the two cluster distances, in `[0, 1]`. Tracks whose class is
`"ball"` map to `("ball", 1.0)`. Low-confidence tracks still get a label; the
confidence column is how a caller decides to distrust it.

## replay.overlay

```python
def render_overlay(video_path, tracks: pd.DataFrame | Mapping[int, list[TrackedBox]],
                   out_path, *, homography: Homography | None = None,
                   fps: float | None = None, draw_minimap: bool = True) -> Path
```

Draws boxes coloured by team with track id and jersey, and optionally a pitch
minimap inset showing pitch-coordinate positions. Encodes with ffmpeg
(`libx264`, `yuv420p`) so the result plays anywhere; raises a clear error if
`ffmpeg` is not on PATH.

## replay.eval.metrics

```python
@dataclass
class AccuracyReport:
    dataset: str; n_frames: int; n_gt: int; n_pred: int
    mota: float; motp: float; idf1: float
    id_switches: int; false_positives: int; misses: int
    rmse_m: float; median_error_m: float
    notes: str = ""
    def to_row(self) -> dict[str, Any]

def evaluate(gt: pd.DataFrame, pred: pd.DataFrame, *, dataset: str,
             match_dist_m: float = 2.0) -> AccuracyReport
```

Both frames are canonical tracking frames. Matching is per-frame Hungarian
assignment on Euclidean pitch distance with a `match_dist_m` gate; MOTA, MOTP,
IDF1 and ID switches follow the CLEAR-MOT and IDF1 definitions. RMSE is over
matched pairs only and is reported in meters.

## replay.eval.report

```python
def render_table(reports: Sequence[AccuracyReport]) -> str        # markdown
def update_readme(reports, readme_path, *, marker: str = "ACCURACY") -> Path
```

`update_readme` replaces whatever sits between
`<!-- BEGIN ACCURACY -->` and `<!-- END ACCURACY -->`. It never writes numbers
that were not produced by `evaluate`.

## replay.sim.state

```python
N_PLAYERS_PER_TEAM = 11
BALL_INDEX = 22            # rows 0..10 home, 11..21 away, 22 ball

@dataclass
class MatchState:
    t: float
    pos: np.ndarray        # (23, 2) meters
    vel: np.ndarray        # (23, 2) m/s
    possession: int        # row index of the player in possession, -1 if loose
    ball_carrier_team: Literal["home", "away", None]
    def copy(self) -> "MatchState"
    @property
    def home(self) -> np.ndarray          # (11, 2) view of positions
    @property
    def away(self) -> np.ndarray
    @property
    def ball(self) -> np.ndarray          # (2,)

def state_from_tracking(df: pd.DataFrame, frame: int, *, match_id: str | None = None) -> MatchState
def states_to_tracking(states: Sequence[MatchState], *, match_id: str) -> pd.DataFrame
```

## replay.sim.physics

```python
MAX_SPEED = 8.0            # m/s
MAX_ACCEL = 6.0            # m/s^2
BALL_DRAG = 0.4            # 1/s, first-order
DT = 0.1

def step_players(pos, vel, desired_vel, *, dt=DT, max_speed=MAX_SPEED,
                 max_accel=MAX_ACCEL) -> tuple[np.ndarray, np.ndarray]
def step_ball(pos, vel, *, dt=DT, drag=BALL_DRAG) -> tuple[np.ndarray, np.ndarray]
def clamp_to_pitch(pos: np.ndarray) -> np.ndarray
```

All pure functions on arrays: no globals, no RNG. Acceleration is clipped by
magnitude, then speed is clipped by magnitude, so caps hold on every axis
combination and not just per-component.

## replay.sim.policies

```python
class Policy(Protocol):
    def desired_velocity(self, state: MatchState, team: Literal["home", "away"],
                         rng: np.random.Generator) -> np.ndarray:   # (11, 2)
        ...

class RuleBasedPolicy:
    def __init__(self, *, press_trigger_m: float = 12.0, defensive_line: float = 0.0,
                 line_compactness: float = 1.0, pass_lane_weight: float = 1.0) -> None

class BCPolicy:            # numpy MLP, no torch
    @classmethod
    def train(cls, tracking: pd.DataFrame, *, hidden: int = 64, epochs: int = 40,
              seed: int = 0) -> "BCPolicy"
    def save(self, path) -> Path
    @classmethod
    def load(cls, path) -> "BCPolicy"
```

`defensive_line` is the scenario override in meters, added to the resting depth
of the defending block. Policies are deterministic given the `rng` passed in.

## replay.sim.scenario

```python
class Scenario(pydantic.BaseModel):
    initial_state: InitialState        # source file/frame, or "auto"
    policy: PolicyConfig               # kind + overrides such as defensive_line
    horizon_s: float = 10.0
    n_rollouts: int = 32
    seed: int = 0

def load_scenario(path: str | Path) -> Scenario
```

Validation errors must name the offending YAML field and be readable without
a stack trace.

## replay.sim.calibrate / render / run

```python
def calibration_report(sim: pd.DataFrame, source: pd.DataFrame) -> CalibrationReport
def render_rollout_gif(states: Sequence[MatchState], out_path, *, fps: int = 10) -> Path
def run_scenario(tracking: pd.DataFrame, scenario: Scenario) -> SimResult
```

`SimResult` carries the per-rollout metrics frame (`metrics`), the
baseline-vs-scenario delta table (`deltas`), and rollout 0 of each arm
(`states`, `baseline_states`) for rendering. Identical `seed` must
produce a bit-identical metrics table.
