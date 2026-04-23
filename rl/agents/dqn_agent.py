"""Double-DQN agent with action masking.

Designed for discrete, masked action spaces (Chinese Checkers action
space = pins_per_player * num_cells with most actions illegal at any
time).

Training workflow (driven by ``Trainer``):
    1. ``act(obs, mask, training=True)``  -> epsilon-greedy action.
    2. env.step(action) -> transition.
    3. ``observe(Transition)``            -> pushes to replay buffer,
                                             and periodically ``learn()``.
    4. ``on_episode_end()``               -> bookkeeping only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn, optim

from ..encoding import NUM_ACTIONS, NUM_CELLS, OBS_CHANNELS
from ..networks import DQNNet
from ..training.replay_buffer import ReplayBuffer
from .base import Agent, Transition


@dataclass
class DQNConfig:
    obs_dim: int = OBS_CHANNELS * NUM_CELLS
    num_actions: int = NUM_ACTIONS
    hidden_sizes: Sequence[int] = (512, 512)
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    target_update_interval: int = 1_000
    train_freq: int = 4
    learn_starts: int = 1_000
    double_dqn: bool = True
    device: str = "cpu"
    seed: int = 0
    # Eval-time policy: "argmax" or "epsilon_greedy" or "boltzmann".
    # Small-epsilon greedy at eval breaks deterministic loops that
    # plague purely deterministic argmax in static environments like
    # solo_race (there's no opponent to change state).
    eval_policy: str = "epsilon_greedy"
    eval_epsilon: float = 0.05
    eval_boltzmann_temperature: float = 0.5


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)

        self.online = DQNNet(
            in_features=cfg.obs_dim,
            num_actions=cfg.num_actions,
            hidden_sizes=cfg.hidden_sizes,
        ).to(self.device)
        self.target = DQNNet(
            in_features=cfg.obs_dim,
            num_actions=cfg.num_actions,
            hidden_sizes=cfg.hidden_sizes,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(
            capacity=cfg.buffer_size,
            obs_dim=cfg.obs_dim,
            mask_dim=cfg.num_actions,
            seed=cfg.seed,
        )

        self.total_steps = 0
        self.train_steps = 0
        self.last_loss: Optional[float] = None

    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, training: bool = False) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return -1

        if training:
            self.total_steps += 1
            if self.rng.random() < self.epsilon():
                return int(self.rng.choice(legal.tolist()))

        t_obs = torch.from_numpy(obs.reshape(1, -1).astype(np.float32)).to(self.device)
        q = self.online(t_obs).cpu().numpy().reshape(-1)

        policy = getattr(self.cfg, "eval_policy", "argmax")
        if not training and policy == "epsilon_greedy":
            if self.rng.random() < self.cfg.eval_epsilon:
                return int(self.rng.choice(legal.tolist()))
        elif not training and policy == "boltzmann":
            q_masked = np.where(mask, q, -np.inf)
            q_legal = q_masked[legal]
            logits = q_legal / max(self.cfg.eval_boltzmann_temperature, 1e-6)
            logits -= logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            return int(self.rng.choices(legal.tolist(), weights=probs.tolist(), k=1)[0])

        q_masked = np.where(mask, q, -np.inf)
        return int(np.argmax(q_masked))

    # ------------------------------------------------------------------
    def observe(self, transition: Transition) -> None:
        self.buffer.push(
            obs=transition.obs,
            action=transition.action,
            reward=transition.reward,
            next_obs=transition.next_obs,
            next_mask=transition.next_mask,
            done=transition.done,
        )
        if (
            len(self.buffer) >= max(self.cfg.learn_starts, self.cfg.batch_size)
            and self.total_steps % self.cfg.train_freq == 0
        ):
            self._learn()

    # ------------------------------------------------------------------
    def _learn(self) -> None:
        obs, actions, rewards, next_obs, next_masks, dones = self.buffer.sample(
            self.cfg.batch_size
        )

        obs_t = torch.from_numpy(obs).to(self.device)
        act_t = torch.from_numpy(actions).to(self.device)
        rew_t = torch.from_numpy(rewards).to(self.device)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        mask_t = torch.from_numpy(next_masks).to(self.device)
        done_t = torch.from_numpy(dones).to(self.device)

        # Current Q(s, a)
        q_vals = self.online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                # Online chooses best action, target evaluates it.
                q_next_online = self.online(next_obs_t)
                q_next_online = q_next_online.masked_fill(~mask_t, -1e9)
                best_actions = q_next_online.argmax(dim=1, keepdim=True)
                q_next_target = self.target(next_obs_t).gather(1, best_actions).squeeze(1)
            else:
                q_next = self.target(next_obs_t)
                q_next = q_next.masked_fill(~mask_t, -1e9)
                q_next_target = q_next.max(dim=1).values

            # If mask is all-False (no legal moves, terminal state), zero the bootstrap.
            any_legal = mask_t.any(dim=1).float()
            q_next_target = q_next_target * any_legal

            target = rew_t + self.cfg.gamma * (1.0 - done_t) * q_next_target

        loss = nn.functional.smooth_l1_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        self.last_loss = float(loss.item())

        if self.train_steps % max(1, self.cfg.target_update_interval) == 0:
            self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "train_steps": self.train_steps,
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)

        saved_cfg = ckpt.get("cfg", {})
        saved_hidden = saved_cfg.get("hidden_sizes") if isinstance(saved_cfg, dict) else None
        cur_hidden = tuple(self.cfg.hidden_sizes)
        if saved_hidden is not None and tuple(saved_hidden) != cur_hidden:
            print(f"[DQNAgent] Rebuilding network with saved hidden_sizes={tuple(saved_hidden)} "
                  f"(was {cur_hidden}).")
            self.cfg.hidden_sizes = tuple(saved_hidden)
            self.online = DQNNet(
                in_features=self.cfg.obs_dim,
                num_actions=self.cfg.num_actions,
                hidden_sizes=self.cfg.hidden_sizes,
            ).to(self.device)
            self.target = DQNNet(
                in_features=self.cfg.obs_dim,
                num_actions=self.cfg.num_actions,
                hidden_sizes=self.cfg.hidden_sizes,
            ).to(self.device)
            self.optimizer = optim.Adam(self.online.parameters(), lr=self.cfg.lr)

        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt.get("target", ckpt["online"]))
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.train_steps = int(ckpt.get("train_steps", 0))
