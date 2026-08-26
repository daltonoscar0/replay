"""Command line entry points.

Imports of the heavy stages are deliberately deferred into the command bodies:
`replay --help` should not pay for opencv, matplotlib and torch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Broadcast-to-tracking converter and tactical sandbox.",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    _setup_logging(verbose)


@app.command()
def track(
    video: Annotated[Path, typer.Argument(help="Broadcast video (mp4).")],
    out: Annotated[Path, typer.Option("--out", help="Output tracking CSV.")] = Path("tracks.csv"),
    fps: Annotated[float, typer.Option(help="Output sample rate, Hz.")] = 10.0,
    device: Annotated[str, typer.Option(help="auto | cpu | cuda | mps.")] = "auto",
    detector: Annotated[str, typer.Option(help="yolo | color.")] = "yolo",
    weights: Annotated[str, typer.Option(help="YOLO weights.")] = "yolov8n.pt",
    conf: Annotated[float, typer.Option(help="Detection confidence floor.")] = 0.25,
    stride: Annotated[int, typer.Option(help="Process every Nth frame.")] = 1,
    max_frames: Annotated[int | None, typer.Option(help="Stop after N frames.")] = None,
    homography: Annotated[
        Path | None, typer.Option(help="Calibration JSON; skips estimation.")
    ] = None,
    match_id: Annotated[str, typer.Option(help="match_id written into the CSV.")] = "match",
    canonical: Annotated[
        bool, typer.Option(help="Write long canonical format instead of Metrica wide.")
    ] = False,
) -> None:
    """Convert broadcast video into pitch-coordinate tracking."""
    from replay.io import backend_name, to_canonical_csv, to_metrica
    from replay.track.pipeline import TrackConfig, run_track

    cfg = TrackConfig(
        detector=detector,
        device=device,
        out_fps=fps,
        conf=conf,
        weights=weights,
        frame_stride=stride,
        max_frames=max_frames,
        homography_path=homography,
        match_id=match_id,
    )
    result = run_track(video, cfg)
    for note in result.notes:
        typer.echo(f"note: {note}")

    if result.tracks.empty:
        typer.secho(
            "no tracking rows produced — see the notes above (most often no "
            "homography could be estimated)",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        to_canonical_csv(result.tracks, out)
    else:
        to_metrica(result.tracks, out, fps=fps)
    n_objects = result.tracks["player_id"].nunique()
    typer.echo(
        f"wrote {len(result.tracks)} rows / {n_objects} objects to {out} "
        f"(writer: {backend_name()})"
    )


@app.command()
def overlay(
    video: Annotated[Path, typer.Argument(help="Source video.")],
    tracks: Annotated[Path, typer.Argument(help="Tracking CSV from `replay track`.")],
    out: Annotated[Path, typer.Option("--out", help="Output video.")] = Path("overlay.mp4"),
    homography: Annotated[Path | None, typer.Option(help="Calibration JSON.")] = None,
    minimap: Annotated[bool, typer.Option(help="Draw the pitch minimap inset.")] = True,
) -> None:
    """Render tracked positions back onto the source video."""
    from replay.io import from_metrica
    from replay.overlay import render_overlay
    from replay.track.homography import Homography

    df = from_metrica(tracks)
    h = Homography.from_json(homography) if homography else None
    path = render_overlay(video, df, out, homography=h, draw_minimap=minimap)
    typer.echo(f"wrote {path}")


@app.command("eval")
def eval_cmd(
    dataset: Annotated[
        str, typer.Option("--dataset", help="soccernet-test | demo | <path to gt csv>")
    ] = "demo",
    pred: Annotated[Path | None, typer.Option(help="Predicted tracking CSV.")] = None,
    gt: Annotated[Path | None, typer.Option(help="Ground-truth tracking CSV.")] = None,
    write_readme: Annotated[
        bool, typer.Option(help="Splice the table into the README accuracy section.")
    ] = False,
    readme: Annotated[Path, typer.Option(help="README to update.")] = Path("README.md"),
) -> None:
    """Evaluate tracking accuracy and print the table."""
    from replay.eval import evaluate, render_table, update_readme

    reports = _collect_reports(dataset, pred, gt, evaluate)
    table = render_table(reports)
    typer.echo(table)
    if write_readme:
        update_readme(reports, readme)
        typer.echo(f"updated accuracy section in {readme}")


def _collect_reports(dataset: str, pred: Path | None, gt: Path | None, evaluate):  # noqa: ANN001, ANN202
    """Resolve `--dataset` into a list of AccuracyReport.

    `demo` regenerates predictions from the bundled clip so the published table
    is never stale relative to the code that produced it.
    """
    from replay.eval.metrics import AccuracyReport
    from replay.eval.soccernet import SoccerNetUnavailable, load_soccernet_gt
    from replay.io import from_canonical_csv, from_metrica

    def _read(p: Path):  # noqa: ANN202
        return from_canonical_csv(p) if "canonical" in p.name else from_metrica(p)

    if pred is not None and gt is not None:
        return [evaluate(_read(gt), _read(pred), dataset=dataset)]

    if dataset == "demo":
        return [_demo_report(evaluate)]

    if dataset == "all":
        # Every dataset the table is meant to cover, including the ones that
        # could not be run. A row saying "not run" is the honest output; a table
        # that quietly contains only what happened to work reads as a clean
        # sweep.
        return [_demo_report(evaluate), *_collect_reports("soccernet-test", pred, gt, evaluate)]

    if dataset.startswith("soccernet"):
        try:
            gt_df = load_soccernet_gt(Path("data/raw/soccernet"))
        except SoccerNetUnavailable as exc:
            return [
                AccuracyReport(
                    dataset="soccernet-test",
                    n_frames=0,
                    n_gt=0,
                    n_pred=0,
                    mota=float("nan"),
                    motp=float("nan"),
                    idf1=float("nan"),
                    id_switches=0,
                    false_positives=0,
                    misses=0,
                    rmse_m=float("nan"),
                    median_error_m=float("nan"),
                    notes=f"not run: {exc}",
                )
            ]
        if pred is None:
            raise typer.BadParameter("--pred is required to evaluate against SoccerNet")
        return [evaluate(gt_df, _read(pred), dataset="soccernet-test")]

    p = Path(dataset)
    if p.exists() and pred is not None:
        return [evaluate(_read(p), _read(pred), dataset=p.stem)]
    raise typer.BadParameter(f"do not know how to evaluate dataset {dataset!r}")


def _demo_report(evaluate):  # noqa: ANN001, ANN202
    """Track the bundled synthetic clip and score it against exact ground truth."""
    from replay.io import from_canonical_csv
    from replay.track.homography import Homography
    from replay.track.pipeline import TrackConfig, run_track

    assets = Path("demo/assets")
    clip = assets / "demo_clip.mp4"
    gt_path = assets / "ground_truth.csv"
    cam = assets / "camera.json"
    if not clip.exists():
        raise typer.BadParameter(
            f"{clip} is missing; run `python scripts/make_demo_clip.py --out demo/assets`"
        )
    cfg = TrackConfig(
        detector="color",
        out_fps=10.0,
        homography_path=cam if cam.exists() else None,
        match_id="demo",
    )
    result = run_track(clip, cfg)
    gt_df = from_canonical_csv(gt_path)
    rep = evaluate(gt_df, result.tracks, dataset="demo (synthetic clip)")
    rep.notes = (
        "synthetic clip, colour detector, camera calibration supplied; "
        "not representative of broadcast footage"
    )
    _ = Homography  # keep the import meaningful if calibration is added later
    return rep


@app.command()
def sim(
    tracks: Annotated[Path, typer.Argument(help="Tracking CSV.")],
    scenario: Annotated[Path, typer.Option("--scenario", help="Scenario YAML.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Where to write outputs.")] = Path(
        "out"
    ),
    gif: Annotated[bool, typer.Option(help="Render a GIF of rollout 0.")] = True,
) -> None:
    """Run a tactical counterfactual against a tracking file."""
    from replay.io import from_metrica
    from replay.sim.calibrate import calibration_report
    from replay.sim.render import render_rollout_gif
    from replay.sim.run import run_scenario
    from replay.sim.scenario import load_scenario

    df = from_metrica(tracks)
    scn = load_scenario(scenario)
    result = run_scenario(df, scn)

    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(result.deltas.to_string(index=False))
    result.deltas.to_csv(out_dir / "deltas.csv", index=False)

    # calibrate compares simulated *tracking* against the source file, not the
    # per-rollout metrics table. The possession rate comes from every rollout
    # rather than the one that gets rendered.
    cal = calibration_report(result.to_tracking(arm="baseline"), df, **_loss_rate(result))
    typer.echo("")
    typer.echo(cal.format())

    if gif:
        path = render_rollout_gif(result.states, out_dir / "rollout.gif")
        typer.echo(f"wrote {path}")


def _loss_rate(result: Any) -> dict[str, Any]:
    """Possession-loss rate over every rollout, for the calibration report.

    A single ten-second rollout resolves the rate only to the nearest six per
    minute, which is coarse enough to fail a sim that is fine. The driver
    already ran dozens; use them.
    """
    metrics = getattr(result, "metrics", None)
    if metrics is None or "possession_losses" not in getattr(metrics, "columns", []):
        return {}
    # The baseline arm, not the scenario arm: calibration asks whether the
    # simulator reproduces the footage, and the scenario is the arm deliberately
    # altered so that it does not.
    scen = metrics.loc[metrics["arm"] == "baseline"]
    if scen.empty:
        return {}
    horizon_s = float(getattr(getattr(result, "scenario", None), "horizon_s", 0.0) or 0.0)
    if horizon_s <= 0.0:
        return {}
    return {
        "sim_loss_rate_per_min": float(scen["possession_losses"].mean()) * 60.0 / horizon_s,
        "n_rollouts": int(len(scen)),
    }



@app.command()
def demo(
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("out/demo"),
) -> None:
    """Run the whole pipeline offline on the bundled 30s clip."""
    from replay.io import from_canonical_csv, to_metrica
    from replay.overlay import render_overlay
    from replay.sim.render import render_rollout_gif
    from replay.sim.run import run_scenario
    from replay.sim.scenario import load_scenario
    from replay.track.homography import Homography
    from replay.track.pipeline import TrackConfig, run_track

    assets = Path("demo/assets")
    clip = assets / "demo_clip.mp4"
    cam = assets / "camera.json"
    if not clip.exists():
        typer.secho(
            f"{clip} is missing. Regenerate it with:\n"
            f"  python scripts/make_demo_clip.py --out demo/assets",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("[1/4] tracking the bundled clip (colour detector, CPU, no downloads)")
    cfg = TrackConfig(
        detector="color",
        out_fps=10.0,
        homography_path=cam if cam.exists() else None,
        match_id="demo",
    )
    result = run_track(clip, cfg)
    tracks_csv = out_dir / "tracks.csv"
    to_metrica(result.tracks, tracks_csv)
    typer.echo(f"      {len(result.tracks)} rows -> {tracks_csv}")

    typer.echo("[2/4] rendering the overlay")
    h = Homography.from_json(cam) if cam.exists() else result.homography
    render_overlay(clip, result.tracks, out_dir / "overlay.mp4", homography=h)
    typer.echo(f"      -> {out_dir / 'overlay.mp4'}")

    typer.echo("[3/4] scoring against ground truth")
    from replay.eval import evaluate, render_table

    gt_df = from_canonical_csv(assets / "ground_truth.csv")
    rep = evaluate(gt_df, result.tracks, dataset="demo (synthetic clip)")
    typer.echo(render_table([rep]))

    typer.echo("[4/4] simulating a higher defensive line")
    scn = load_scenario(Path("scenarios/press_higher.yaml"))
    source_tracks = result.tracks if not result.tracks.empty else gt_df
    sim_result = run_scenario(source_tracks, scn)
    typer.echo(sim_result.deltas.to_string(index=False))
    gif_path = render_rollout_gif(sim_result.states, out_dir / "rollout.gif")
    typer.echo(f"      -> {gif_path}")

    # The calibration verdict belongs in the demo, not just in `replay sim`.
    # This is the one command anyone runs on a fresh clone, and a sandbox whose
    # possession churn is out of tolerance should say so there rather than let
    # the delta table imply the numbers can be taken at face value.
    from replay.sim.calibrate import calibration_report

    typer.echo("")
    typer.echo(
        calibration_report(
            sim_result.to_tracking(arm="baseline"), source_tracks, **_loss_rate(sim_result)
        ).format()
    )

    typer.echo(f"\ndone. outputs in {out_dir}")


if __name__ == "__main__":
    app()
