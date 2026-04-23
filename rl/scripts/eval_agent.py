"""Evaluate a trained agent in the in-process env.

Usage::

    python -m rl.scripts.eval_agent --checkpoint rl/checkpoints/dqn_best.pt --mode solo_race --episodes 50
    python -m rl.scripts.eval_agent --checkpoint rl/checkpoints/dqn_best.pt --mode vs_random --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.agents import DQNAgent, RandomAgent  # noqa: E402
from rl.agents.dqn_agent import DQNConfig  # noqa: E402
from rl.encoding import NUM_ACTIONS, NUM_CELLS, OBS_CHANNELS  # noqa: E402
from rl.env import CheckersEnv, EnvConfig  # noqa: E402
from rl.env.checkers_env import _random_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="rl/checkpoints/dqn_best.pt")
    p.add_argument("--mode", type=str, default="solo_race",
                   choices=["solo_race", "vs_random"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--my-colour", type=str, default="red")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--render-first", action="store_true")
    p.add_argument("--random-baseline", action="store_true")
    return p.parse_args()


def resolve_device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return x


def build_agent(args, device) -> DQNAgent:
    cfg = DQNConfig(
        obs_dim=OBS_CHANNELS * NUM_CELLS,
        num_actions=NUM_ACTIONS,
        device=device,
    )
    agent = DQNAgent(cfg)
    agent.load(str((REPO_ROOT / args.checkpoint).resolve()))
    return agent


def run_episode(env: CheckersEnv, agent, max_steps, opp_policies=None, render=False):
    obs, info = env.reset()
    mask = info["legal_mask"]
    total = 0.0
    for _ in range(max_steps):
        action = agent.act(obs, mask, training=False)
        if action < 0:
            break
        res = env.step(action, opponent_policies=opp_policies or {})
        total += float(res.reward)
        obs = res.obs
        mask = env.action_mask()
        if res.terminated or res.truncated:
            if render:
                print(env.render_ascii())
            return {
                "return": total,
                "outcome": str(res.info.get("outcome", "truncated")),
                "pins_home": int(res.info.get("pins_home", 0)),
                "agent_moves": env.agent_move_count,
                "steps": env.step_count,
            }
    return {
        "return": total,
        "outcome": "truncated",
        "pins_home": int(env._pins_in_goal(env.cfg.my_colour)),
        "agent_moves": env.agent_move_count,
        "steps": env.step_count,
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[eval] device={device} mode={args.mode} episodes={args.episodes}")

    if args.random_baseline:
        agent = RandomAgent(seed=0)
    else:
        agent = build_agent(args, device)

    mode = "solo_race" if args.mode == "solo_race" else "multi"
    env_cfg = EnvConfig(mode=mode, my_colour=args.my_colour, max_steps=args.max_steps, seed=0)
    env = CheckersEnv(env_cfg)

    opp_policies = None
    if mode == "multi":
        opp_policies = {c: _random_policy for c in env.cfg.opponent_colours}

    results = []
    for ep in range(args.episodes):
        # Re-seed each episode for variety.
        env.rng = __import__("random").Random(ep + 1)
        results.append(
            run_episode(
                env,
                agent,
                args.max_steps,
                opp_policies=opp_policies,
                render=(args.render_first and ep == 0),
            )
        )

    wins = sum(1 for r in results if r["outcome"] == "win")
    win_moves = [r["agent_moves"] for r in results if r["outcome"] == "win"]
    returns = [r["return"] for r in results]
    pins = [r["pins_home"] for r in results]

    print("\n=== RESULTS ===")
    print(f"episodes: {len(results)}")
    print(f"win rate: {wins / len(results) * 100:.1f}% ({wins}/{len(results)})")
    if win_moves:
        print(f"avg moves (when won): {mean(win_moves):.1f} "
              f"+/- {stdev(win_moves) if len(win_moves) > 1 else 0:.1f} "
              f"(min {min(win_moves)}, max {max(win_moves)})")
    print(f"avg pins_home: {mean(pins):.2f} / 10")
    print(f"avg return: {mean(returns):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
