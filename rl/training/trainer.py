"""Curriculum trainer for Chinese Checkers.

Stage 1 (solo_race): agent learns to race all 10 pins to the goal.
Stage 2 (vs_random) : agent plays 2-player vs a random opponent.

Metrics are logged per episode to CSV (via MetricsLogger) and the best
model (by rolling win-rate) is checkpointed automatically.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..agents.base import Agent, Transition
from ..agents.dqn_agent import DQNAgent
from ..agents.greedy_agent import GreedyAgent
from ..agents.random_agent import RandomAgent
from ..encoding import ActionEncoder
from ..env import CheckersEnv, EnvConfig
from .metrics import EpisodeStats, MetricsLogger


@dataclass
class StageConfig:
    name: str
    mode: str
    max_episodes: int
    max_steps_per_episode: int
    my_colour: str
    exit_win_rate: float
    exit_window: int
    reward: Dict[str, float] = field(default_factory=dict)
    warmup_episodes: int = 0


@dataclass
class TrainerConfig:
    stages: list
    checkpoint_dir: str
    metrics_dir: str
    checkpoint_name: str = "dqn_latest.pt"
    best_name: str = "dqn_best.pt"
    log_every: int = 20
    seed: int = 0


class Trainer:
    def __init__(self, agent: DQNAgent, cfg: TrainerConfig):
        self.agent = agent
        self.cfg = cfg
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.metrics = MetricsLogger(cfg.metrics_dir, window=200)
        self._best_win_rate = -1.0
        self.episode_counter = 0

    # ------------------------------------------------------------------
    def _build_env(self, stage: StageConfig, seed: int) -> CheckersEnv:
        r = stage.reward
        env_cfg = EnvConfig(
            mode=stage.mode,
            my_colour=stage.my_colour,
            max_steps=stage.max_steps_per_episode,
            seed=seed,
            reward_per_pin_home=r.get("per_pin_home", 1.0),
            reward_step_penalty=r.get("step_penalty", 0.01),
            reward_all_home_bonus=r.get("all_home_bonus", 10.0),
            reward_distance_shaping=r.get("distance_shaping", 0.05),
            reward_win_bonus=r.get("win_bonus", 10.0),
            reward_lose_penalty=r.get("lose_penalty", -5.0),
            anti_loop_enabled=bool(r.get("anti_loop_enabled", False)),
            anti_loop_window=int(r.get("anti_loop_window", 12)),
            anti_loop_revisit_penalty=float(r.get("anti_loop_revisit_penalty", 0.03)),
            anti_loop_aba_penalty=float(r.get("anti_loop_aba_penalty", 0.06)),
        )
        return CheckersEnv(env_cfg)

    # ------------------------------------------------------------------
    def _run_warmup(self, env: CheckersEnv, stage: StageConfig) -> None:
        """Fill the replay buffer with transitions from a greedy policy."""
        if stage.warmup_episodes <= 0:
            return
        print(f"[warmup] Priming buffer with greedy policy for {stage.warmup_episodes} episodes...")
        greedy = GreedyAgent(env.board, stage.my_colour, epsilon=0.1, seed=self.cfg.seed)

        opp_policies = self._opponent_policies(stage)

        for ep in range(stage.warmup_episodes):
            obs, info = env.reset(seed=self.cfg.seed + 10_000 + ep)
            mask = info["legal_mask"]
            done = False
            while not done:
                greedy.set_pin_sources(
                    [p.axialindex for p in env.pins_by_colour[stage.my_colour]]
                )
                action = greedy.act(obs, mask)
                if action < 0:
                    break
                res = env.step(action, opponent_policies=opp_policies)
                next_mask = env.action_mask()
                self.agent.buffer.push(
                    obs=obs,
                    action=action,
                    reward=res.reward,
                    next_obs=res.obs,
                    next_mask=next_mask,
                    done=bool(res.terminated or res.truncated),
                )
                obs = res.obs
                mask = next_mask
                done = res.terminated or res.truncated

    # ------------------------------------------------------------------
    def _opponent_policies(self, stage: StageConfig):
        if stage.mode != "multi":
            return {}
        # Return a closure that plays uniformly at random for every
        # non-agent colour currently on the board.
        from ..env.checkers_env import _random_policy
        return {"__default__": _random_policy}

    # ------------------------------------------------------------------
    def _pick_opponent_action(self, env: CheckersEnv, colour: str) -> int:
        from ..env.checkers_env import _random_policy
        return _random_policy(env, colour)

    # ------------------------------------------------------------------
    def _run_stage(self, stage: StageConfig) -> None:
        print(f"\n=== Stage: {stage.name} | mode={stage.mode} "
              f"| episodes<= {stage.max_episodes} ===")
        env = self._build_env(stage, seed=self.cfg.seed)

        # Warmup (greedy-policy priming).
        self._run_warmup(env, stage)

        # Opponent policy dict keyed by actual colour names.
        opp_policies: Dict[str, Any] = {}
        if stage.mode == "multi":
            for c in env.cfg.opponent_colours:
                opp_policies[c] = self._pick_opponent_action

        t0 = time.time()

        for local_ep in range(stage.max_episodes):
            self.episode_counter += 1
            obs, info = env.reset(seed=self.cfg.seed + self.episode_counter)
            mask = info["legal_mask"]
            ep_return = 0.0
            done = False
            outcome = "truncated"
            pins_home = 0
            loop_revisit_events = 0
            loop_aba_events = 0

            while not done:
                action = self.agent.act(obs, mask, training=True)
                if action < 0:
                    outcome = "no_legal_moves"
                    break
                res = env.step(action, opponent_policies=opp_policies)
                next_mask = env.action_mask()
                self.agent.observe(
                    Transition(
                        obs=obs,
                        action=action,
                        reward=res.reward,
                        next_obs=res.obs,
                        next_mask=next_mask,
                        done=bool(res.terminated or res.truncated),
                    )
                )
                ep_return += float(res.reward)
                obs = res.obs
                mask = next_mask
                if res.terminated or res.truncated:
                    outcome = str(res.info.get("outcome", "truncated"))
                    pins_home = int(res.info.get("pins_home", 0))
                    done = True
                if bool(res.info.get("loop_revisit", False)):
                    loop_revisit_events += 1
                if bool(res.info.get("loop_aba", False)):
                    loop_aba_events += 1

            self.agent.on_episode_end(info=None)
            stats = EpisodeStats(
                episode=self.episode_counter,
                stage=stage.name,
                steps=env.step_count,
                agent_moves=env.agent_move_count,
                return_sum=ep_return,
                pins_home=pins_home,
                outcome=outcome,
                epsilon=self.agent.epsilon(),
                loss=self.agent.last_loss,
                loop_revisit_events=loop_revisit_events,
                loop_aba_events=loop_aba_events,
            )
            self.metrics.log(stats)

            # Progress print
            if local_ep % self.cfg.log_every == 0 or (local_ep + 1) == stage.max_episodes:
                roll = self.metrics.rolling()
                elapsed = time.time() - t0
                print(
                    f"[{stage.name}] ep={self.episode_counter:5d} "
                    f"ret={ep_return:7.2f} out={outcome:<18s} "
                    f"pins={pins_home}/10 moves={env.agent_move_count:4d} "
                    f"eps={self.agent.epsilon():.3f} "
                    f"loss={self.agent.last_loss if self.agent.last_loss else 0.0:.4f} "
                    f"| win%={roll.get('win_rate', 0)*100:5.1f} "
                    f"avgmv={roll.get('avg_moves_if_win', float('nan')):.1f} "
                    f"loopR={roll.get('avg_loop_revisit_events', 0):.2f} "
                    f"loopABA={roll.get('avg_loop_aba_events', 0):.2f} "
                    f"| {elapsed:5.0f}s"
                )

            # Checkpoint + best
            self._save_latest()
            self._maybe_save_best(stage)

            # Early stop on exit criterion
            roll = self.metrics.rolling()
            if (
                len(self.metrics._recent) >= stage.exit_window
                and roll.get("win_rate", 0.0) >= stage.exit_win_rate
            ):
                print(
                    f"[{stage.name}] Exit criterion met "
                    f"(win_rate={roll['win_rate']*100:.1f}% >= "
                    f"{stage.exit_win_rate*100:.0f}% over {stage.exit_window} eps). Stopping stage."
                )
                break

    # ------------------------------------------------------------------
    def _save_latest(self) -> None:
        path = os.path.join(self.cfg.checkpoint_dir, self.cfg.checkpoint_name)
        self.agent.save(path)

    def _maybe_save_best(self, stage: StageConfig) -> None:
        roll = self.metrics.rolling()
        if len(self.metrics._recent) < min(50, stage.exit_window // 2 or 50):
            return
        wr = roll.get("win_rate", 0.0)
        if wr > self._best_win_rate:
            self._best_win_rate = wr
            path = os.path.join(self.cfg.checkpoint_dir, self.cfg.best_name)
            self.agent.save(path)

    # ------------------------------------------------------------------
    def run(self) -> None:
        for stage in self.cfg.stages:
            self._run_stage(stage)
        self.metrics.plot()
        self.metrics.close()
        print("Training complete.")
