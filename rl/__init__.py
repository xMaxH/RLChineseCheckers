"""RL framework for Chinese Checkers.

Provides an in-process Gym-style environment that reuses the existing
HexBoard/Pin game logic, a generic Agent abstraction, and a DQN
implementation suitable for curriculum training.
"""

from . import game_bridge  # noqa: F401  (side effect: registers sys.path)
