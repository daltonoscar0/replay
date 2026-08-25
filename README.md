# replay

Broadcast video in, pitch-coordinate tracking out — plus a sandbox for asking
what would have happened if the defensive line had been higher.

Two packages in one repo:

- **`replay.track`** — MP4 broadcast video → 10 Hz tracking CSV in Metrica
  schema, plus an overlay video.
- **`replay.sim`** — tracking CSV + scenario YAML → simulated tactical
  counterfactuals.

## Accuracy

Every number below is produced by `make eval-table`, which runs a real
evaluation and rewrites this section. Nothing here is typed by hand. Rows that
could not be run say so rather than being omitted.

<!-- BEGIN ACCURACY -->

| Dataset | Frames | MOTA | MOTP (m) | IDF1 | ID switches | FP | Misses | RMSE (m) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo (synthetic clip) | 301 | 0.928 | 0.331 | 0.883 | 16 | 97 | 384 | 0.398 | synthetic clip, colour detector, camera calibration supplied; not representative of broadcast footage |
| soccernet-test | 0 | n/a | n/a | n/a | 0 | 0 | 0 | n/a | not run: SoccerNet ground truth unavailable: nothing exists at data/raw/soccernet. SoccerNet data is distributed under an NDA and the archives are password-protected. Download it yourself from https://www.soccer-net.org (accept the NDA to receive the password), extract the sequences, and point this loader at the directory that contains them. Nothing here will fetch or unlock the data for you. |

<!-- END ACCURACY -->

## Install

```sh
uv sync                 # core: colour detector, tracking, sim, eval, overlay
uv sync --extra yolo    # adds ultralytics + torch for real broadcast footage
```

Python 3.11 or 3.12. `ffmpeg` must be on your PATH — overlay rendering shells
out to it, and the test suite skips the overlay module without it.

## Quickstart

```sh
replay demo                                          # offline, CPU, no downloads, ~2 min
replay track match.mp4 --out tracks.csv --fps 10     # needs the yolo extra
replay overlay match.mp4 tracks.csv --out overlay.mp4 --homography camera.json
replay eval --dataset all
replay sim tracks.csv --scenario scenarios/press_higher.yaml
```

`replay demo` runs the whole pipeline on the bundled 30 s clip and writes
`tracks.csv`, `overlay.mp4` and `rollout.gif` to `out/demo/`, then prints the
accuracy table and the calibration verdict.

`make help` lists the rest. The ones that matter: `make demo-assets`
regenerates the bundled clip and its ground truth, `make eval-table` reruns the
evaluation and rewrites the accuracy section above, `make test` runs everything.

## How it works

### track

1. **Detect** — `ColorDetector` (green-pitch segmentation, no weights, no
   network) or `YoloDetector` (ultralytics, person + sports-ball classes,
   device resolved at runtime to cuda/mps/cpu).
2. **Track** — ByteTrack: high-score detections associated to confirmed tracks
   by IoU, then low-score detections to whatever is left, which is what
   recovers a player through a brief occlusion.
3. **Calibrate** — a supplied `camera.json`, else the SoccerNet calibration
   model if installed, else line-based estimation from the markings. Each falls
   through to the next, and returning nothing beats projecting through a wrong
   matrix.
4. **Smooth** — constant-velocity Kalman filter plus an RTS backward smoother
   over pitch meters, which also fills the gaps.
5. **Assign teams** — torso crops clustered in HSV, using a *circular* hue mean
   so a red kit straddling the 0/180 wrap does not split across both clusters.
6. **Write** — Metrica-schema CSV, or long canonical CSV with `--canonical`.

### sim

22 agents and a ball on 105x68 at 0.1 s ticks, with magnitude-clipped
acceleration and speed caps and a first-order drag model for the ball.
Policies are rule-based (press trigger, formation block, passing lanes) or
behaviour-cloned (a small numpy MLP fitted to a tracking file). A scenario YAML
sets the starting state, the policy overrides, the horizon and the seed.

Each scenario runs twice — once as a baseline, once with the overrides — using
the same seeds pairwise, so the delta between arms is not swamped by rollout
variance. Same seed gives a bit-identical metrics table.

## Limits

Read this before trusting a number out of this tool.

- **The demo row is not a benchmark.** The bundled clip is synthetic: scripted
  trajectories rendered through a fixed, known camera, with a ball drawn about
  three times life size so it survives at the far touchline. The demo is
  supplied with that camera rather than estimating it. Those figures are a
  best case and are in the table only because they are reproducible on a fresh
  clone. The SoccerNet row is the one to judge the tracker by, and it has not
  been run — see `TODO.md`.
- **Nothing here has been run on real broadcast footage.** The YOLO path is
  implemented but never executed: constructing it downloads weights, so its
  only test is network-marked and skipped by default.
- **The sandbox currently fails its own possession calibration.** `replay demo`
  prints it. The simulator turns the ball over several times more often than
  the source footage does; the tolerance was not widened to hide it. Read the
  scenario delta table as directional until that is fixed.
- **The ball is the weak link in the colour detector.** Ball, pitch markings
  and the home team's white shorts are all the same white, so the ball is
  separated by roundness and by sitting in open grass. That finds it in about
  half the frames and still lets some marking fragments through — most of the
  false positives in the table above are this.
- **The Metrica writer is vendored** under `replay/io/`, and is used until the
  suite's `sportio` package is installed, at which point the adapter switches
  automatically. Two deviations from the released Metrica layout: home and away
  go in one file rather than two, and the format cannot carry `vx`, `vy`,
  `confidence` or `match_id` — use `to_canonical_csv` when a round trip must
  preserve those.

`TODO.md` has the full list, including what needs a manual step and why.

## Layout

```
src/replay/
  schema.py        canonical tracking frame + shared dataclasses
  io/              Metrica reader/writer, sportio adapter
  track/           detect, byte_track, kalman, homography, teams, pipeline
  sim/             state, physics, policies, scenario, calibrate, render, run
  eval/            CLEAR-MOT metrics, SoccerNet loader, README table writer
  overlay.py       boxes/markers + pitch minimap, encoded with ffmpeg
docs/INTERFACES.md the contract between those modules
scripts/           synthetic demo clip generator
scenarios/         example scenario YAML
```

## License

MIT. See `LICENSE`.
