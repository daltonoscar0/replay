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
_Not yet generated. Run `make eval-table`._
<!-- END ACCURACY -->

## Install

```sh
uv sync                 # core: tracking from the colour detector, sim, eval
uv sync --extra yolo    # adds ultralytics + torch for real broadcast footage
```

`ffmpeg` must be on your PATH for overlay rendering.

## Quickstart

```sh
replay demo                                          # offline, CPU, no downloads
replay track match.mp4 --out tracks.csv --fps 10
replay overlay match.mp4 tracks.csv --out overlay.mp4
replay eval --dataset demo
replay sim tracks.csv --scenario scenarios/press_higher.yaml
```

## Limits

Read this before trusting a number out of this tool.

- The bundled demo clip is **synthetic** — scripted trajectories rendered
  through a known camera. Its accuracy figures are close to a best case and are
  not representative of real broadcast footage. They are in the table because
  they are reproducible on a fresh clone; the SoccerNet row is the one to judge
  the tracker by.
- The Metrica writer is vendored under `replay/io/`. When the suite's `sportio`
  package is installed it is used instead, automatically. See `TODO.md`.

## License

See `LICENSE`.
