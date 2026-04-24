"""Replay buffer with optional Prioritized Experience Replay (PER).

Stores transitions as packed numpy arrays. Masks are bit-packed to keep
RAM usage bounded when the buffer is large.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        mask_dim: int,
        prioritized: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        per_eps: float = 1e-6,
        seed: Optional[int] = None,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.mask_dim = int(mask_dim)
        self._packed_bytes = (self.mask_dim + 7) // 8
        self.prioritized = bool(prioritized)
        self.per_alpha = float(per_alpha)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_frames = int(per_beta_frames)
        self.per_eps = float(per_eps)

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_masks_packed = np.zeros(
            (self.capacity, self._packed_bytes), dtype=np.uint8
        )
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.max_priority = 1.0

        self.size = 0
        self.ptr = 0
        self.frame = 1
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs.reshape(-1)
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_obs[self.ptr] = next_obs.reshape(-1)
        packed = np.packbits(next_mask.astype(np.uint8))
        # np.packbits may return shorter array if mask_dim not multiple of 8; pad.
        if packed.size < self._packed_bytes:
            padded = np.zeros(self._packed_bytes, dtype=np.uint8)
            padded[: packed.size] = packed
            packed = padded
        self.next_masks_packed[self.ptr] = packed[: self._packed_bytes]
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if not self.prioritized:
            idxs = np.array(
                [self.rng.randrange(self.size) for _ in range(batch_size)],
                dtype=np.int64,
            )
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            p = self.priorities[: self.size].copy()
            p = np.power(np.maximum(p, self.per_eps), self.per_alpha)
            p_sum = float(p.sum())
            if p_sum <= 0:
                p = np.ones_like(p) / float(len(p))
            else:
                p = p / p_sum
            idxs = np.random.choice(self.size, size=batch_size, p=p, replace=True).astype(
                np.int64
            )
            beta = self._per_beta()
            weights = np.power(self.size * p[idxs], -beta)
            weights /= np.max(weights) if np.max(weights) > 0 else 1.0
            weights = weights.astype(np.float32)
            self.frame += 1
        packed = self.next_masks_packed[idxs]
        masks = np.unpackbits(packed, axis=1).astype(bool)[:, : self.mask_dim]
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            masks,
            self.dones[idxs],
            idxs,
            weights,
        )

    def _per_beta(self) -> float:
        if self.per_beta_frames <= 1:
            return 1.0
        frac = min(1.0, self.frame / float(self.per_beta_frames))
        return self.per_beta_start + frac * (1.0 - self.per_beta_start)

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        if not self.prioritized:
            return
        vals = np.abs(td_errors).astype(np.float32) + self.per_eps
        self.priorities[idxs] = vals
        self.max_priority = max(self.max_priority, float(np.max(vals)))

    def __len__(self) -> int:
        return self.size
