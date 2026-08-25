# TODO

Everything here is a real gap, not a wish list. Items marked **manual** cannot
be completed by a run of this repo — they need data, credentials or hardware
that is not available offline.

## Manual steps

- **SoccerNet full evaluation** — *manual*. The tracking/GSR archives are
  distributed under an NDA and are password-protected. Nothing in this repo
  will fetch or unlock them. To fill the `soccernet-test` row of the accuracy
  table: accept the NDA at <https://www.soccer-net.org>, download and extract
  the test sequences to `data/raw/soccernet/`, then run
  `replay eval --dataset soccernet-test --pred <tracks.csv>` and
  `make eval-table`. Until then the row reads "not run", which is the honest
  state — it is not omitted.
  - `replay/eval/soccernet.py` is written against the documented MOT-style
    (`gameinfo.ini` + `gt/gt.txt`) and GSR JSON layouts but has only ever been
    run against fixtures matching those documents. Two assumptions need
    checking against the real release: that GSR `bbox_pitch`
    `x_bottom_middle`/`y_bottom_middle` are **meters from the pitch centre**
    (converted as `x + 52.5`, `34 - y`), and that MOT-style sequences carry no
    pitch coordinates and therefore need a homography passed in via
    `pitch_from_image`.
- **SoccerNet camera-calibration model** — *manual*. `estimate_from_soccernet`
  detects that `sn_calibration_baseline` and its weights are absent, logs one
  line and returns `None`; only that branch has ever executed. The live branch
  (`SegmentationNetwork`, `SV_FT_TSWC_kp` / `SV_FT_TSWC_lines` under
  `$REPLAY_SOCCERNET_WEIGHTS`) is written from the package docs and unverified.
- **YOLO detector on real footage** — *manual/network*. `YoloDetector` is
  implemented but never executed here: constructing it downloads `yolov8n.pt`.
  Its one test is `@pytest.mark.network` and skipped unless
  `REPLAY_NETWORK_TESTS=1`. Nothing in the tracker has been run against real
  broadcast video.

## Known gaps

- **`sportio` is not installed**, so `replay.io._adapter` runs the vendored
  Metrica writer. The sportio branch is exercised only against a stub injected
  into `sys.modules`, assuming `sportio.to_metrica(df, path, fps=...)` and
  `sportio.from_metrica(path, match_id=...)`. When the real package lands,
  check those signatures and delete nothing else — the swap is one file.
- **The Metrica wide format loses columns.** It cannot carry `vx`, `vy`,
  `confidence` or `match_id`, so `from_metrica` returns them as NaN / caller
  supplied. Use `to_canonical_csv` / `from_canonical_csv` when a round trip has
  to preserve everything. Velocities can be recomputed with
  `replay.track.kalman.smooth_track`.
- **The sandbox fails its own possession calibration.** `replay demo` prints
  `possession_loss_per_min ... FAIL`: the simulator churns possession several
  times more often than the source footage does. The tolerance
  (`POSSESSION_LOSS_TOL_PER_MIN = 4.0`) was deliberately *not* widened to make
  it pass. The turnover model in `replay/sim/run.py` (`TURNOVER_RATE_HZ`,
  `TACKLE_SCALE_M`, `PASS_TRIGGER_M`) is what needs work. Treat the delta table
  as directional until this passes.
- **`SimResult.deltas` and `calibrate` count possession differently** — the
  driver knows who holds the ball, `calibrate` re-infers it from proximity
  because canonical tracking has no possession column. They disagree by roughly
  2x on the same rollout. Reconcile before quoting either as *the* number.
- **Ball detection is the weak link in the colour path.** The ball, the pitch
  markings and the home team's shorts are all the same white, so the ball is
  separated by shape and by sitting in open grass. On the demo clip that finds
  it in roughly half of frames and still admits some marking fragments; the
  tracker's `min_hits` and the single-ball rule in `track/pipeline.py` clean up
  the rest. Most of the accuracy table's false positives are this.
- **`ColorDetector` is tuned for the synthetic clip.** Two players whose blobs
  touch merge into one detection. Its HSV bounds and shape filters are module
  constants, not parameters. It is the offline demo path, not a real detector.
- **`estimate_from_lines` is conservative and slow** — 0.5-1.5 s per 720p frame,
  and it returns `None` on many real frames rather than guess (it needs two
  merged lines in each image family and six intersections). It also cannot
  resolve the pitch's 180-degree rotational symmetry from markings alone and
  breaks the tie by assuming the main-camera convention.
- **`states_to_tracking` cannot preserve identities.** `MatchState` carries slot
  indices, so it writes `home_<n>` / `away_<n>` with jersey `n+1`. Round-tripping
  real tracking through the sim renames everyone.
- **No substitution handling.** Eleven slots per side is the contract; a team
  with more distinct ids over a file keeps eleven and logs which were dropped.
- **`assign_teams` home/away naming is arbitrary** — the cluster with the lower
  mean image x is called "home". It is stable per input but carries no relation
  to the actual home side.
- **The demo clip's camera is static**, which makes homography estimation on it
  easier than on real footage. Its ball is drawn ~3x life size and its players
  ~30% too tall so they survive at the far touchline. The accuracy table says
  the demo row is not representative; this is why.
- `demo/assets/ground_truth.csv` `frame` is the **10 Hz sample index**, not the
  video frame index (10 Hz does not land on integer 25 fps frames). `t` is the
  join key.

## Nice to have

- Per-class evaluation. `evaluate` association is class-blind, so a ball
  prediction can match a player within the 2 m gate. SoccerNet scores the ball
  separately.
- Re-ID head trained on jersey crops, replacing HSV clustering (the spec's
  preferred route when the data is available).
- Pipe raw frames into ffmpeg for overlay rendering instead of staging PNGs;
  the current path is disk-heavy for a full match.
