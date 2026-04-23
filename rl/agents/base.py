"""Agent ABC and supporting dataclasses.

All agent implementations (Random, Greedy, DQN, later PPO/AlphaZero)
conform to this interface so trainers and the player.py integration
can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    next_mask: np.ndarray
    done: bool


class Agent(ABC):
    """Generic agent interface.

    ``act`` returns a flat action index in ``[0, NUM_ACTIONS)``.
    ``observe`` is optional (for learning agents).
    ``save``/``load`` handle checkpoint persistence.
    """

    name: str = "agent"

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Called at the start of each episode. Default: no-op."""

    @abstractmethod
    def act(self, obs: np.ndarray, mask: np.ndarray, training: bool = False) -> int:
        """Select an action given observation and legal-action mask."""

    # Learning-specific hooks (no-op for scripted agents).
    def observe(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, info: Optional[Dict[str, Any]] = None) -> None:
        pass

    # Persistence
    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support load()")
