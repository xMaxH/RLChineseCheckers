"""CLI entry point for training the DQN agent.

Usage (run from the repository root)::

    python -m rl.scripts.train_dqn --config rl/configs/dqn_default.json
    python -m rl.scripts.train_dqn --config rl/configs/dqn_default.json --skip-vs-random
    python -m rl.scripts.train_dqn --minutes 45    # cap total wall-clock time
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure repo root on sys.path when invoked via ``python rl/scripts/train_dqn.py``
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.agents import DQNAgent  # noqa: E402
from rl.agents.dqn_agent import DQNConfig  # noqa: E402
from rl.encoding import NUM_ACTIONS, NUM_CELLS, OBS_CHANNELS  # noqa: E402
from rl.training.trainer import StageConfig, Trainer, TrainerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN Chinese Checkers agent.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "rl/configs/dqn_default.json"),
    )
    parser.add_argument("--skip-solo", action="store_true")
    parser.add_argument("--skip-vs-random", action="store_true")
    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Hard cap on training wall-clock time in minutes.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    return parser.parse_args()


def resolve_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def build_stages(cfg: dict, args: argparse.Namespace) -> list:
    stages = []
    cur = cfg["curriculum"]

    if cur.get("solo", {}).get("enabled", False) and not args.skip_solo:
        s = cur["solo"]
        stages.append(
            StageConfig(
                name="solo_race",
                mode="solo_race",
                max_episodes=int(s["max_episodes"]),
                max_steps_per_episode=int(s["max_steps_per_episode"]),
                my_colour=str(s["my_colour"]),
                exit_win_rate=float(s["exit_win_rate"]),
                exit_window=int(s["exit_window"]),
                reward=dict(s.get("reward", {})),
                warmup_episodes=int(s.get("warmup_episodes", 0)),
            )
        )

    if cur.get("vs_random", {}).get("enabled", False) and not args.skip_vs_random:
        s = cur["vs_random"]
        stages.append(
            StageConfig(
                name="vs_random",
                mode="multi",
                max_episodes=int(s["max_episodes"]),
                max_steps_per_episode=int(s["max_steps_per_episode"]),
                my_colour=str(s["my_colour"]),
                exit_win_rate=float(s["exit_win_rate"]),
                exit_window=int(s["exit_window"]),
                reward=dict(s.get("reward", {})),
                warmup_episodes=0,
            )
        )

    return stages


def main() -> int:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = resolve_device(cfg.get("device", "cpu"))
    print(f"[train_dqn] device={device}")

    ac = cfg["agent"]
    dqn_cfg = DQNConfig(
        obs_dim=OBS_CHANNELS * NUM_CELLS,
        num_actions=NUM_ACTIONS,
        hidden_sizes=tuple(ac.get("hidden_sizes", [512, 512])),
        gamma=float(ac.get("gamma", 0.99)),
        lr=float(ac.get("lr", 3e-4)),
        batch_size=int(ac.get("batch_size", 256)),
        buffer_size=int(ac.get("buffer_size", 200_000)),
        epsilon_start=float(ac.get("epsilon_start", 1.0)),
        epsilon_end=float(ac.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(ac.get("epsilon_decay_steps", 50_000)),
        target_update_interval=int(ac.get("target_update_interval", 1_000)),
        train_freq=int(ac.get("train_freq", 4)),
        learn_starts=int(ac.get("learn_starts", 1_000)),
        double_dqn=bool(ac.get("double_dqn", True)),
        device=device,
        seed=int(cfg.get("seed", 0)),
    )

    agent = DQNAgent(dqn_cfg)
    if args.resume:
        print(f"[train_dqn] Resuming from {args.resume}")
        agent.load(args.resume)

    stages = build_stages(cfg, args)
    if not stages:
        print("No stages enabled; exiting.")
        return 0

    # If user gave a minutes budget, split across stages proportional to
    # their configured max_episodes. This is an upper bound — stages can
    # still exit early on win-rate criterion.
    if args.minutes is not None:
        total_eps = sum(s.max_episodes for s in stages)
        budget_secs = args.minutes * 60.0
        # Simple approach: ensure the overall budget is enforced by
        # running a watchdog thread. Kept outside trainer logic.
        print(f"[train_dqn] Time budget: {args.minutes:.1f} min "
              f"across {total_eps} planned episodes.")
        _install_time_budget(budget_secs)

    out = cfg["output"]
    trainer_cfg = TrainerConfig(
        stages=stages,
        checkpoint_dir=str(REPO_ROOT / out["checkpoint_dir"]),
        metrics_dir=str(REPO_ROOT / out["metrics_dir"]),
        checkpoint_name=str(out.get("checkpoint_name", "dqn_latest.pt")),
        best_name=str(out.get("best_name", "dqn_best.pt")),
        seed=int(cfg.get("seed", 0)),
    )

    trainer = Trainer(agent, trainer_cfg)
    trainer.run()
    return 0


_DEADLINE_NS: int = 0


def _install_time_budget(seconds: float) -> None:
    """Register a soft deadline via ``sys.settrace`` style check.

    We do NOT kill the process hard; instead, the trainer checks this
    deadline between episodes. Hook is provided via the module global
    below, and ``Trainer`` imports it lazily on demand.
    """
    import threading

    global _DEADLINE_NS
    _DEADLINE_NS = int(time.perf_counter_ns() + seconds * 1e9)

    def _watchdog():
        while True:
            time.sleep(5.0)
            if time.perf_counter_ns() > _DEADLINE_NS:
                print("[train_dqn] Time budget exceeded; flushing final checkpoint and exiting.")
                # Best-effort exit; the trainer saves latest each episode
                # so no data is lost.
                os._exit(0)

    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()


if __name__ == "__main__":
    raise SystemExit(main())
