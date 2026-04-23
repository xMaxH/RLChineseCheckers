"""DQN move method for player.py.

Mirrors the ``alphazero_method.py`` interface:
    choose_move_dqn(legal_moves, state, player_context) -> (pin_id, to_index, delay)

Loads the trained DQN checkpoint lazily on first call and reuses the
same state/action encoders used during training, so inference-time
behaviour matches training distribution.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# Ensure repo root is importable so ``import rl`` works from any CWD.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_CHECKPOINT = _REPO_ROOT / "rl" / "checkpoints" / "dqn_best.pt"
_CHECKPOINT_ENV = "DQN_CHECKPOINT"


_AGENT = None  # type: ignore[var-annotated]
_ENCODER = None  # type: ignore[var-annotated]
_SYMMETRY = None  # type: ignore[var-annotated]
_LOG_ONCE = False


def _lazy_init() -> bool:
    """Load model + encoder on first use. Returns False if unavailable."""
    global _AGENT, _ENCODER, _SYMMETRY, _LOG_ONCE

    if _AGENT is not None:
        return True

    try:
        import numpy as np  # noqa: F401  (ensures numpy available)
        import torch

        from rl.agents.dqn_agent import DQNAgent, DQNConfig
        from rl.encoding import (
            NUM_ACTIONS,
            NUM_CELLS,
            OBS_CHANNELS,
            BoardSymmetry,
            StateEncoder,
        )
        from rl.game_bridge import new_board
    except Exception as e:
        if not _LOG_ONCE:
            print(f"[dqn_method] Unable to load RL dependencies: {e}")
            _LOG_ONCE = True
        return False

    ckpt = os.environ.get(_CHECKPOINT_ENV) or str(_DEFAULT_CHECKPOINT)
    if not os.path.isfile(ckpt):
        if not _LOG_ONCE:
            print(f"[dqn_method] No checkpoint found at {ckpt} — will fall back to random.")
            _LOG_ONCE = True
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DQNConfig(
        obs_dim=OBS_CHANNELS * NUM_CELLS,
        num_actions=NUM_ACTIONS,
        device=device,
    )
    try:
        agent = DQNAgent(cfg)
        agent.load(ckpt)
    except Exception as e:
        if not _LOG_ONCE:
            print(f"[dqn_method] Failed to load checkpoint ({e}) — falling back to random.")
            _LOG_ONCE = True
        return False

    board = new_board()
    symmetry = BoardSymmetry(board)
    _AGENT = agent
    _SYMMETRY = symmetry
    _ENCODER = StateEncoder(board, symmetry=symmetry)
    print(f"[dqn_method] Loaded DQN checkpoint from {ckpt} on {device}.")
    return True


def choose_move_dqn(
    legal_moves: Mapping[object, Sequence[int]],
    state: Mapping[str, Any],
    player_context: Dict[str, Any],
) -> Tuple[int, int, float]:
    """Return (pin_id, to_index, delay_seconds).

    ``state`` has the same shape the server returns via ``get_state``.
    ``player_context`` includes ``colour`` (the agent's colour).
    """
    if not _lazy_init():
        return _fallback_random(legal_moves)

    from rl.encoding import ActionEncoder
    import numpy as np
    import torch

    colour = str(player_context.get("colour"))
    pins_by_colour = state.get("pins", {}) or {}
    move_count = int(state.get("move_count", 0) or 0)

    # Normalise legal_moves keys (server sends them as JSON ints; player.py
    # iterates the dict with str keys in some paths).
    legal_norm: Dict[int, Sequence[int]] = {}
    for k, v in legal_moves.items():
        try:
            legal_norm[int(k)] = v
        except (TypeError, ValueError):
            continue

    if not any(len(v) > 0 for v in legal_norm.values()):
        raise ValueError("No legal moves available for DQN method.")

    obs = _ENCODER.encode(pins_by_colour, colour, move_count=move_count)
    a2c = _SYMMETRY.actual_to_canonical(colour)
    c2a = _SYMMETRY.canonical_to_actual(colour)
    pin_a2c = _SYMMETRY.pin_actual_to_canonical(colour)
    pin_c2a = _SYMMETRY.pin_canonical_to_actual(colour)
    mask = ActionEncoder.build_mask(
        legal_norm, actual_to_canon=a2c, pin_actual_to_canon=pin_a2c
    )

    with torch.no_grad():
        flat = _AGENT.act(obs, mask, training=False)

    if flat < 0:
        return _fallback_random(legal_moves)

    pin_id, to_index = ActionEncoder.from_flat(
        int(flat), canon_to_actual=c2a, pin_canon_to_actual=pin_c2a
    )
    delay = random.uniform(0.05, 0.15)
    return pin_id, to_index, delay


def _fallback_random(legal_moves: Mapping[object, Sequence[int]]) -> Tuple[int, int, float]:
    movable = [(int(k), list(v)) for k, v in legal_moves.items() if v]
    if not movable:
        raise ValueError("No legal moves available for DQN fallback.")
    pid, moves = random.choice(movable)
    to_index = random.choice(moves)
    return pid, to_index, random.uniform(0.05, 0.15)
