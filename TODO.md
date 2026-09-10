# TODO

Everything here is a real gap, not a wish list. Items marked **manual** cannot
be completed by a run of this repo, they need data, credentials or hardware
that is not available offline.

## Manual steps

- **SoccerNet full evaluation**, *manual*. The tracking/GSR archives are
  distributed under an NDA and are password-protected. Nothing in this repo
  will fetch or unlock them. To fill the `soccernet-test` row of the accuracy
  table: accept the NDA at <https://www.soccer-net.org>, download and extract
  the test sequences to `data/raw/soccernet/`, then run
  `replay eval --dataset soccernet-test --pred <tracks.csv>` and
  `make eval-table`. Until then the row reads "not run", which is the honest
  state, it is not omitted.
  - `replay/eval/soccernet.py` is written against the documented MOT-style
    (`gameinfo.ini` + `gt/gt.txt`) and GSR JSON layouts but has only ever been
    run against fixtures matching those documents. Two assumptions need
    checking against the real release: that GSR `bbox_pitch`
    `x_bottom_middle`/`y_bottom_middle` are **meters from the pitch centre**
    (converted as `x + 52.5`, `34 - y`), and that MOT-style sequences carry no
    pitch coordinates and therefore need a homography passed in via
    `pitch_from_image`.
- **SoccerNet camera-calibration model**, *manual*. `estimate_from_soccernet`
  detects that `sn_calibration_baseline` and its weights are absent, logs one
  line and returns `None`; only that branch has ever executed. The live branch
  (`SegmentationNetwork`, `SV_FT_TSWC_kp` / `SV_FT_TSWC_lines` under
  `$REPLAY_SOCCERNET_WEIGHTS`) is written from the package docs and unverified.
- **YOLO detector on real footage**, *manual*. The path now runs end to end:
  weights download, the device resolves to `mps` on Apple Silicon, and
  `replay track --detector yolo` completes without error. What has *not*
  happened is a run against real broadcast video, so its detection quality is
  still unmeasured. On the synthetic demo clip it correctly finds nothing -
  YOLO looks for people and the clip draws rectangles, which exercises the
  plumbing but says nothing about accuracy.
- **Automatic calibration is unreliable, and manual calibration is the way in.**
  `estimate_from_lines` returns `None` even on the synthetic clip, so assume it
  will on real footage too. `scripts/calibrate.py` covers this: click four or
  more landmarks and it solves for the camera, writing a check image alongside.
  Recovering the demo clip's known camera from four landmarks is exact to
  1e-6 m, but that is a self-consistency check, not evidence about hand-clicked
  points on real video.

- **`pressing` compatibility check**, *blocked*. The build sequence calls for
  verifying that `replay track` output feeds the suite's `pressing` tool, but
  `pressing` does not exist yet. What stands in for it: the writer emits the
  canonical `tracking` schema from `00_SPORTIO.md` (`replay/schema.py`,
  `TRACKING_COLUMNS`), which is the interface `pressing` is specified to
  consume, and `to_metrica`/`from_metrica` round-trip is tested. Run the actual
  check once `pressing` lands.

## Known gaps

- **`sportio` is not installed**, so `replay.io._adapter` runs the vendored
  Metrica writer. The sportio branch is exercised only against a stub injected
  into `sys.modules`, assuming `sportio.to_metrica(df, path, fps=...)` and
  `sportio.from_metrica(path, match_id=...)`. When the real package lands,
  check those signatures and delete nothing else, the swap is one file.
- **The Metrica wide format loses columns.** It cannot carry `vx`, `vy`,
  `confidence` or `match_id`, so `from_metrica` returns them as NaN / caller
  supplied. Use `to_canonical_csv` / `from_canonical_csv` when a round trip has
  to preserve everything. Velocities can be recomputed with
  `replay.track.kalman.smooth_track`.
- **Ball coverage is low, by choice.** The colour detector finds the ball in
  about one frame in seven, and after the stationary-candidate and movement
  filters roughly 9% of output frames carry one. What does get written is
  accurate, median 0.86 m from truth, worst case 1.8 m on the demo clip.
  The permissive alternative was built and measured: segmenting the ball off the
  raw white mask by thickness lifts detection to three frames in four, but
  perspective makes a near-camera marking thicker than a far-touchline ball, so
  no single threshold separates them and the leftovers are moving stubs of line
  that survive every downstream filter. End to end that put the ball ~20 m from
  the truth. Getting both needs a depth-aware threshold, scale the core radius
  by the homography's local metres-per-pixel, which is the real fix and is not
  done.
- **9% ball coverage is too sparse to infer possession from**, so calibration
  reports the possession row as `n/a` when run against tracker output rather
  than scoring it. Against exact ground truth the row is measured and passes.
- **Which eleven tracks become the roster changes the sim.** The tracker emits
  ~18 home identities over 30 s for 11 players, and `build_roster` keeps eleven.
  Reading the same tracking through the canonical CSV and through Metrica picks
  different elevens, jersey numbers survive one and not the other, and the two
  give different enough initial states that `control_share_home` calibrates
  inside tolerance from one and outside it from the other. Consolidating
  fragmented identities before roster selection is the fix.
- **`ColorDetector` is tuned for the synthetic clip.** Two players whose blobs
  touch merge into one detection. Its HSV bounds and shape filters are module
  constants, not parameters. It is the offline demo path, not a real detector.
- **`estimate_from_lines` is conservative and slow**, 0.5-1.5 s per 720p frame,
  and it returns `None` on many real frames rather than guess (it needs two
  merged lines in each image family and six intersections). It also cannot
  resolve the pitch's 180-degree rotational symmetry from markings alone and
  breaks the tie by assuming the main-camera convention.
- **`states_to_tracking` cannot preserve identities.** `MatchState` carries slot
  indices, so it writes `home_<n>` / `away_<n>` with jersey `n+1`. Round-tripping
  real tracking through the sim renames everyone.
- **No substitution handling.** Eleven slots per side is the contract; a team
  with more distinct ids over a file keeps eleven and logs which were dropped.
- **`assign_teams` home/away naming is arbitrary**, the cluster with the lower
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
