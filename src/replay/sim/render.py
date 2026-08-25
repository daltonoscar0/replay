"""Top-down GIF of a rollout.

Deliberately small output: a scenario report embeds several of these, so each
one is a few hundred pixels wide with a short palette rather than something you
would put on a big screen. Matplotlib is pinned to Agg here because this runs in
CI and from a CLI, never inside a window manager.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from replay.schema import PITCH_LENGTH, PITCH_WIDTH
from replay.sim.state import N_PLAYERS_PER_TEAM, MatchState

matplotlib.use("Agg")

PITCH_COLOR = "#33683f"
LINE_COLOR = "#e8f0e8"
HOME_COLOR = "#2f6fd0"
AWAY_COLOR = "#d1483b"
BALL_COLOR = "#ffffff"

#: Output is 6.0 x 4.1 inches at 64 dpi, i.e. 384 x 262 px.
FIG_W_IN = 6.0
FIG_H_IN = 4.1
FIG_DPI = 64
#: GIFs are palette images; 32 colours is plenty for six flat fills plus the
#: antialiasing between them, and keeps the file a fraction of the 256-colour size.
PALETTE_SIZE = 32

# Standard markings in meters.
PENALTY_DEPTH = 16.5
PENALTY_WIDTH = 40.32
SIX_YARD_DEPTH = 5.5
SIX_YARD_WIDTH = 18.32
CENTER_CIRCLE_R = 9.15
PENALTY_SPOT = 11.0
GOAL_WIDTH = 7.32
GOAL_DEPTH = 1.8


def draw_pitch(ax: plt.Axes) -> None:
    """Draw the markings for a 105 x 68 pitch onto ``ax``."""
    ax.set_facecolor(PITCH_COLOR)
    ax.set_xlim(-3.0, PITCH_LENGTH + 3.0)
    ax.set_ylim(-3.0, PITCH_WIDTH + 5.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    kw = {"edgecolor": LINE_COLOR, "facecolor": "none", "linewidth": 1.0}
    ax.add_patch(Rectangle((0, 0), PITCH_LENGTH, PITCH_WIDTH, **kw))
    ax.plot(
        [PITCH_LENGTH / 2, PITCH_LENGTH / 2],
        [0, PITCH_WIDTH],
        color=LINE_COLOR,
        linewidth=1.0,
    )
    ax.add_patch(
        Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), CENTER_CIRCLE_R, **kw)
    )
    ax.plot(
        [PITCH_LENGTH / 2], [PITCH_WIDTH / 2], marker=".", color=LINE_COLOR, markersize=2
    )

    for side in (0.0, 1.0):
        sign = 1.0 if side == 0.0 else -1.0
        x0 = 0.0 if side == 0.0 else PITCH_LENGTH
        ax.add_patch(
            Rectangle(
                (x0 if sign > 0 else x0 - PENALTY_DEPTH, (PITCH_WIDTH - PENALTY_WIDTH) / 2),
                PENALTY_DEPTH,
                PENALTY_WIDTH,
                **kw,
            )
        )
        ax.add_patch(
            Rectangle(
                (
                    x0 if sign > 0 else x0 - SIX_YARD_DEPTH,
                    (PITCH_WIDTH - SIX_YARD_WIDTH) / 2,
                ),
                SIX_YARD_DEPTH,
                SIX_YARD_WIDTH,
                **kw,
            )
        )
        ax.add_patch(
            Rectangle(
                (x0 - GOAL_DEPTH if sign > 0 else x0, (PITCH_WIDTH - GOAL_WIDTH) / 2),
                GOAL_DEPTH,
                GOAL_WIDTH,
                **kw,
            )
        )
        spot_x = PENALTY_SPOT if sign > 0 else PITCH_LENGTH - PENALTY_SPOT
        ax.plot([spot_x], [PITCH_WIDTH / 2], marker=".", color=LINE_COLOR, markersize=2)


def _team_xy(state: MatchState, team: str) -> np.ndarray:
    lo = 0 if team == "home" else N_PLAYERS_PER_TEAM
    return np.asarray(state.pos[lo : lo + N_PLAYERS_PER_TEAM], dtype=np.float64)


def render_rollout_gif(
    states: Sequence[MatchState], out_path: str | Path, *, fps: int = 10
) -> Path:
    """Write ``states`` to an animated GIF and return the path.

    One GIF frame per state, so the playback rate matches the simulator's tick
    rate when ``fps`` is 1/DT.
    """
    if len(states) == 0:
        raise ValueError("cannot render a rollout with no states")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI, facecolor=PITCH_COLOR)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    try:
        draw_pitch(ax)
        home = ax.scatter([], [], s=36, c=HOME_COLOR, edgecolors="none", zorder=3)
        away = ax.scatter([], [], s=36, c=AWAY_COLOR, edgecolors="none", zorder=3)
        ball = ax.scatter(
            [], [], s=14, c=BALL_COLOR, edgecolors="#111111", linewidths=0.5, zorder=4
        )
        label = ax.text(
            1.0,
            PITCH_WIDTH + 1.6,
            "",
            color=LINE_COLOR,
            fontsize=7,
            family="monospace",
            va="bottom",
        )

        frames: list[np.ndarray] = []
        for i, state in enumerate(states):
            home.set_offsets(_team_xy(state, "home"))
            away.set_offsets(_team_xy(state, "away"))
            ball.set_offsets(np.asarray(state.ball, dtype=np.float64).reshape(1, 2))
            label.set_text(f"frame {i:4d}   t {float(state.t):6.1f}s")
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(rgba[..., :3].copy())
    finally:
        plt.close(fig)

    _write_gif(frames, out, fps=fps)
    return out


def _write_gif(frames: list[np.ndarray], out: Path, *, fps: int) -> None:
    """imageio's GIF plugins disagree about kwargs across versions; degrade."""
    duration_ms = 1000.0 / float(max(fps, 1))
    attempts: list[dict[str, object]] = [
        {"duration": duration_ms, "loop": 0, "palettesize": PALETTE_SIZE},
        {"duration": duration_ms, "loop": 0},
        {},
    ]
    last: Exception | None = None
    for kwargs in attempts:
        try:
            imageio.mimwrite(out, frames, format="GIF", **kwargs)
            return
        except TypeError as exc:  # unsupported kwarg for this plugin
            last = exc
    raise RuntimeError(f"could not write GIF to {out}") from last
