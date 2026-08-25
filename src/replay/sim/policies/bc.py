"""Behaviour cloning with a one-hidden-layer MLP written directly in numpy.

Torch is an optional extra in this project, so the network here is a few dozen
lines of explicit forward and backward passes instead. That is enough for the
job it is actually asked to do: reproduce the average velocity field of a short
clip well enough that a rollout looks like the source rather than like noise.

What this will not do is learn football. A single hidden layer fitted to a few
hundred player-ticks from one possession has no way to represent phases of
play, off-ball runs it never saw, or anything about the opponent's intentions.
Treat it as a smooth interpolator over the states in the training clip, and
expect it to drift once a rollout leaves that neighbourhood. The rule-based
policy is the one to use when you need behaviour you can defend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from replay.schema import PITCH_LENGTH, PITCH_WIDTH
from replay.sim.physics import MAX_SPEED
from replay.sim.policies.rule_based import cap_speed, team_slice
from replay.sim.state import N_PLAYERS_PER_TEAM, MatchState

TeamSide = Literal["home", "away"]

#: Number of nearest teammates and nearest opponents encoded per player.
K_NEIGHBOURS = 3
#: Relative offsets are divided by this before the standardizer sees them, so a
#: teammate 20 m away lands near 1.0 and the raw feature scales start sane.
REL_SCALE_M = 20.0
#: A player this close to the ball is treated as carrying it when the source
#: frame carries no explicit possession column.
POSSESSION_RADIUS_M = 2.0

_POS_SCALE = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float64)


def feature_dim(k: int = K_NEIGHBOURS) -> int:
    """Own pos/vel, ball pos/vel relative, k mates, k opponents, possession."""
    return 9 + 4 * k


def _pad_neighbours(rel: np.ndarray, k: int) -> np.ndarray:
    """Take the k nearest rows of ``rel`` (M, 2), zero-padded to exactly k."""
    out = np.zeros((k, 2), dtype=np.float64)
    if rel.shape[0] == 0:
        return out
    order = np.argsort(np.linalg.norm(rel, axis=1), kind="stable")[:k]
    take = rel[order]
    out[: take.shape[0]] = take
    return out


def build_features(
    own_pos: np.ndarray,
    own_vel: np.ndarray,
    opp_pos: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    possession: np.ndarray,
    *,
    k: int = K_NEIGHBOURS,
) -> np.ndarray:
    """Per-player feature rows, (n_own, feature_dim(k)).

    Everything except the player's own position is expressed relative to the
    player, so the network sees a local view and does not have to memorise
    absolute pitch coordinates for each shirt number.
    """
    own_pos = np.asarray(own_pos, dtype=np.float64).reshape(-1, 2)
    own_vel = np.asarray(own_vel, dtype=np.float64).reshape(-1, 2)
    opp_pos = np.asarray(opp_pos, dtype=np.float64).reshape(-1, 2)
    ball_pos = np.asarray(ball_pos, dtype=np.float64).reshape(2)
    ball_vel = np.asarray(ball_vel, dtype=np.float64).reshape(2)
    possession = np.asarray(possession, dtype=np.float64).reshape(-1)

    n = own_pos.shape[0]
    feats = np.zeros((n, feature_dim(k)), dtype=np.float64)
    for i in range(n):
        p = own_pos[i]
        mates = np.delete(own_pos, i, axis=0) - p
        opps = opp_pos - p if opp_pos.shape[0] else opp_pos.reshape(0, 2)
        row = np.concatenate(
            [
                p / _POS_SCALE,
                own_vel[i] / MAX_SPEED,
                (ball_pos - p) / REL_SCALE_M,
                (ball_vel - own_vel[i]) / MAX_SPEED,
                _pad_neighbours(mates, k).reshape(-1) / REL_SCALE_M,
                _pad_neighbours(opps, k).reshape(-1) / REL_SCALE_M,
                np.array([possession[i]]),
            ]
        )
        feats[i] = row
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def possession_flags(
    own_pos: np.ndarray, opp_pos: np.ndarray, ball_pos: np.ndarray
) -> np.ndarray:
    """1.0 for the carrier, 0.5 for his teammates, 0.0 when out of possession."""
    own_pos = np.asarray(own_pos, dtype=np.float64).reshape(-1, 2)
    opp_pos = np.asarray(opp_pos, dtype=np.float64).reshape(-1, 2)
    ball_pos = np.asarray(ball_pos, dtype=np.float64).reshape(2)
    flags = np.zeros(own_pos.shape[0], dtype=np.float64)
    if own_pos.shape[0] == 0 or not np.isfinite(ball_pos).all():
        return flags

    own_d = np.linalg.norm(own_pos - ball_pos, axis=1)
    opp_d = (
        np.linalg.norm(opp_pos - ball_pos, axis=1)
        if opp_pos.shape[0]
        else np.array([np.inf])
    )
    i = int(np.argmin(own_d))
    if own_d[i] <= POSSESSION_RADIUS_M and own_d[i] <= float(np.min(opp_d)):
        flags[:] = 0.5
        flags[i] = 1.0
    return flags


def _side_frame(group: pd.DataFrame, team: str) -> pd.DataFrame:
    sub = group.loc[group["team"] == team]
    return sub.sort_values("player_id", kind="stable")


def _xy(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros((0, 2), dtype=np.float64)
    return frame.loc[:, ["x", "y"]].to_numpy(dtype=np.float64)


def _vxy(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros((0, 2), dtype=np.float64)
    return frame.loc[:, ["vx", "vy"]].to_numpy(dtype=np.float64)


def build_dataset(
    tracking: pd.DataFrame, *, k: int = K_NEIGHBOURS
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a canonical tracking frame into (X, Y) for supervised training.

    Y is the player's velocity one tick later, in units of ``MAX_SPEED``. Where
    the frame's ``vx``/``vy`` are missing the velocity is finite-differenced
    from consecutive positions, which is what the tracking pipeline would have
    done anyway.
    """
    required = {"frame", "team", "player_id", "x", "y", "vx", "vy", "t"}
    missing = required - set(tracking.columns)
    if missing:
        raise ValueError(f"tracking frame is missing columns needed for BC: {sorted(missing)}")

    groups = {int(f): g for f, g in tracking.groupby("frame", sort=True)}
    frames = sorted(groups)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for f0, f1 in zip(frames[:-1], frames[1:], strict=False):
        g0, g1 = groups[f0], groups[f1]
        dt = float(np.nanmean(g1["t"].to_numpy(dtype=np.float64))) - float(
            np.nanmean(g0["t"].to_numpy(dtype=np.float64))
        )
        if not np.isfinite(dt) or dt <= 0.0:
            continue
        ball0 = _xy(_side_frame(g0, "ball"))
        ball1 = _xy(_side_frame(g1, "ball"))
        if ball0.shape[0] == 0:
            continue
        ball_pos = ball0[0]
        ball_vel = (ball1[0] - ball0[0]) / dt if ball1.shape[0] else np.zeros(2)

        for team, other in (("home", "away"), ("away", "home")):
            cur = _side_frame(g0, team)
            nxt = _side_frame(g1, team)
            if cur.empty or nxt.empty:
                continue
            ids = cur["player_id"].astype("string")
            nxt_ids = list(nxt["player_id"].astype("string"))
            keep = [i for i, pid in enumerate(ids) if pid in nxt_ids]
            if not keep:
                continue
            nxt_index = {pid: j for j, pid in enumerate(nxt_ids)}

            own_pos = _xy(cur)[keep]
            own_next = _xy(nxt)[[nxt_index[ids.iloc[i]] for i in keep]]
            raw_vel = _vxy(cur)[keep]
            fd_vel = (own_next - own_pos) / dt
            own_vel = np.where(np.isfinite(raw_vel), raw_vel, fd_vel)

            raw_next_vel = _vxy(nxt)[[nxt_index[ids.iloc[i]] for i in keep]]
            target = np.where(np.isfinite(raw_next_vel), raw_next_vel, fd_vel)

            opp_pos = _xy(_side_frame(g0, other))
            flags = possession_flags(own_pos, opp_pos, ball_pos)
            feats = build_features(
                own_pos, own_vel, opp_pos, ball_pos, ball_vel, flags, k=k
            )
            good = np.isfinite(feats).all(axis=1) & np.isfinite(target).all(axis=1)
            if good.any():
                xs.append(feats[good])
                ys.append(target[good] / MAX_SPEED)

    if not xs:
        return np.zeros((0, feature_dim(k))), np.zeros((0, 2))
    return np.vstack(xs), np.vstack(ys)


class BCPolicy:
    """One hidden tanh layer, trained by Adam on hand-derived gradients.

    See the module docstring for what this is and is not capable of.
    """

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        *,
        k: int = K_NEIGHBOURS,
        loss_history: np.ndarray | None = None,
    ) -> None:
        self.w1 = np.asarray(w1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.w2 = np.asarray(w2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)
        self.feat_mean = np.asarray(feat_mean, dtype=np.float64)
        self.feat_std = np.asarray(feat_std, dtype=np.float64)
        self.k = int(k)
        self.loss_history = (
            np.zeros(0, dtype=np.float64)
            if loss_history is None
            else np.asarray(loss_history, dtype=np.float64)
        )

    # -- inference --------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalized-feature forward pass, returns velocities / MAX_SPEED."""
        z = (np.asarray(x, dtype=np.float64) - self.feat_mean) / self.feat_std
        h = np.tanh(z @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def desired_velocity(
        self, state: MatchState, team: TeamSide, rng: np.random.Generator
    ) -> np.ndarray:
        """(11, 2) desired velocities. ``rng`` is unused but drawn from anyway.

        The draw keeps the random stream aligned with ``RuleBasedPolicy`` so
        paired baseline/scenario rollouts stay comparable when a scenario swaps
        one policy for the other.
        """
        rng.normal(size=(N_PLAYERS_PER_TEAM, 2))
        other: TeamSide = "away" if team == "home" else "home"
        own_pos = np.asarray(state.pos[team_slice(team)], dtype=np.float64)
        own_vel = np.asarray(state.vel[team_slice(team)], dtype=np.float64)
        opp_pos = np.asarray(state.pos[team_slice(other)], dtype=np.float64)
        ball_pos = np.asarray(state.ball, dtype=np.float64).reshape(2)
        ball_vel = np.asarray(state.vel[-1], dtype=np.float64).reshape(2)

        flags = possession_flags(own_pos, opp_pos, ball_pos)
        if state.possession is not None and state.possession >= 0:
            sl = team_slice(team)
            flags[:] = 0.0
            if sl.start <= state.possession < sl.stop:
                flags[:] = 0.5
                flags[int(state.possession) - sl.start] = 1.0

        feats = build_features(
            own_pos, own_vel, opp_pos, ball_pos, ball_vel, flags, k=self.k
        )
        return cap_speed(self.forward(feats) * MAX_SPEED)

    # -- training ---------------------------------------------------------

    @classmethod
    def train(
        cls,
        tracking: pd.DataFrame,
        *,
        hidden: int = 64,
        epochs: int = 40,
        seed: int = 0,
        k: int = K_NEIGHBOURS,
        batch_size: int = 64,
        lr: float = 1e-2,
    ) -> BCPolicy:
        x, y = build_dataset(tracking, k=k)
        if x.shape[0] == 0:
            raise ValueError(
                "no usable transitions in the tracking frame: BC needs at least two "
                "consecutive frames with a ball row and players on both teams"
            )
        return cls.fit(x, y, hidden=hidden, epochs=epochs, seed=seed, k=k,
                       batch_size=batch_size, lr=lr)

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        hidden: int = 64,
        epochs: int = 40,
        seed: int = 0,
        k: int = K_NEIGHBOURS,
        batch_size: int = 64,
        lr: float = 1e-2,
    ) -> BCPolicy:
        """Adam on mini-batches with an explicit backward pass."""
        rng = np.random.default_rng(seed)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = x.shape

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        xn = (x - mean) / std

        w1 = rng.normal(scale=np.sqrt(1.0 / d), size=(d, hidden))
        b1 = np.zeros(hidden)
        w2 = rng.normal(scale=np.sqrt(1.0 / hidden), size=(hidden, 2))
        b2 = np.zeros(2)

        params = [w1, b1, w2, b2]
        m_state = [np.zeros_like(p) for p in params]
        v_state = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        losses = np.zeros(epochs, dtype=np.float64)
        bs = max(1, min(batch_size, n))

        for epoch in range(epochs):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, bs):
                idx = order[start : start + bs]
                xb, yb = xn[idx], y[idx]
                m = xb.shape[0]

                pre = xb @ params[0] + params[1]
                h = np.tanh(pre)
                out = h @ params[2] + params[3]
                diff = out - yb
                total += float((diff * diff).sum())

                d_out = 2.0 * diff / (m * 2)
                g_w2 = h.T @ d_out
                g_b2 = d_out.sum(axis=0)
                d_h = d_out @ params[2].T
                d_pre = d_h * (1.0 - h * h)
                g_w1 = xb.T @ d_pre
                g_b1 = d_pre.sum(axis=0)

                step += 1
                for p, g, ms, vs in zip(
                    params, [g_w1, g_b1, g_w2, g_b2], m_state, v_state, strict=True
                ):
                    ms *= beta1
                    ms += (1.0 - beta1) * g
                    vs *= beta2
                    vs += (1.0 - beta2) * g * g
                    m_hat = ms / (1.0 - beta1**step)
                    v_hat = vs / (1.0 - beta2**step)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)

            losses[epoch] = total / (n * 2)

        return cls(
            params[0], params[1], params[2], params[3], mean, std, k=k, loss_history=losses
        )

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
            k=np.array(self.k),
            loss_history=self.loss_history,
        )
        # np.savez appends .npz when the name has no suffix.
        return out if out.suffix == ".npz" else out.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> BCPolicy:
        p = Path(path)
        if not p.exists() and p.suffix != ".npz":
            p = p.with_suffix(".npz")
        with np.load(p) as data:
            return cls(
                data["w1"],
                data["b1"],
                data["w2"],
                data["b2"],
                data["feat_mean"],
                data["feat_std"],
                k=int(data["k"]),
                loss_history=data["loss_history"],
            )
