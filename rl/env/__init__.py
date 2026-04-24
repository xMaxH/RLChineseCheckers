"""RL in-process environment package."""

from .checkers_env import CheckersEnv, EnvConfig, StepResult, _random_policy

__all__ = ["CheckersEnv", "EnvConfig", "StepResult", "_random_policy"]
"""RL environments for Chinese Checkers."""

from .checkers_env import CheckersEnv, EnvConfig, StepResult

__all__ = ["CheckersEnv", "EnvConfig", "StepResult"]
