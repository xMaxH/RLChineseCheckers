"""Uniform experience replay buffer.

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
        seed: Optional[int] = None,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.mask_dim = int(mask_dim)
        self._packed_bytes = (self.mask_dim + 7) // 8

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_masks_packed = np.zeros(
            (self.capacity, self._packed_bytes), dtype=np.uint8
        )
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.size = 0
        self.ptr = 0
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

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idxs = np.array(
            [self.rng.randrange(self.size) for _ in range(batch_size)],
            dtype=np.int64,
        )
        packed = self.next_masks_packed[idxs]
        masks = np.unpackbits(packed, axis=1).astype(bool)[:, : self.mask_dim]
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            masks,
            self.dones[idxs],
        )

    def __len__(self) -> int:
        return self.size
