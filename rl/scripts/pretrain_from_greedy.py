"""Behavioral cloning pretraining from GreedyAgent trajectories.

This is a pragmatic bootstrap for milestone-3: it learns a strong
policy quickly (typically winning solo_race < 600 moves) and then can
optionally be fine-tuned further with DQN RL.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.agents import DQNAgent, GreedyAgent
from rl.agents.dqn_agent import DQNConfig
from rl.encoding import NUM_ACTIONS, NUM_CELLS, OBS_CHANNELS
from rl.env import CheckersEnv, EnvConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="rl/checkpoints/dqn_best.pt")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CheckersEnv(EnvConfig(mode="solo_race", my_colour="red", max_steps=600, seed=args.seed))

    print(f"[pretrain] collecting dataset from greedy ({args.episodes} episodes)...")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ms: list[np.ndarray] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        greedy = GreedyAgent(env.board, "red", epsilon=0.0, seed=args.seed + ep)
        for _ in range(600):
            mask = env.action_mask()
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            greedy.set_pin_sources([p.axialindex for p in env.pins_by_colour["red"]])
            a = greedy.act(obs, mask)
            xs.append(obs.reshape(-1).astype(np.float32))
            ys.append(int(a))
            ms.append(mask.astype(np.bool_))
            res = env.step(a)
            obs = res.obs
            if res.terminated or res.truncated:
                break

    X = torch.from_numpy(np.stack(xs, axis=0)).to(device)
    y = torch.from_numpy(np.array(ys, dtype=np.int64)).to(device)
    M = torch.from_numpy(np.stack(ms, axis=0)).to(device)
    n = X.shape[0]
    print(f"[pretrain] dataset size: {n}")

    cfg = DQNConfig(
        obs_dim=OBS_CHANNELS * NUM_CELLS,
        num_actions=NUM_ACTIONS,
        hidden_sizes=(256, 256),
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    agent = DQNAgent(cfg)
    model = agent.online
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    idx = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(idx)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            b = idx[i : i + args.batch_size]
            logits = model(X[b])
            # Train only over legal actions for each sample.
            logits = logits.masked_fill(~M[b], -1e9)
            loss = criterion(logits, y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += float(loss.item()) * len(b)
        print(f"[pretrain] epoch {epoch+1}/{args.epochs} loss={total_loss/n:.4f}")

    agent.target.load_state_dict(agent.online.state_dict())
    out = (REPO_ROOT / args.out).resolve()
    os.makedirs(out.parent, exist_ok=True)
    agent.save(str(out))
    print(f"[pretrain] saved checkpoint -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

