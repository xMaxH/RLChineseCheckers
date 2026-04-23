"""Uniformly random agent over legal actions."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray, training: bool = False) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return -1
        return int(self.rng.choice(legal.tolist()))
