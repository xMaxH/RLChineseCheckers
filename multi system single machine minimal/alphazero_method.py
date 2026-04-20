"""AlphaZero move method interface for player.py.

Replace this placeholder logic with a real AlphaZero policy/value model.
"""

import random
from typing import Any, Dict, Mapping, Sequence, Tuple


def choose_move_alphazero(
    legal_moves: Mapping[str, Sequence[int]],
    state: Mapping[str, Any],
    player_context: Dict[str, Any],
) -> Tuple[str, int, float]:
    """Return (pin_id, to_index, delay_seconds).

    Current placeholder implementation is random until AlphaZero integration is added.
    """
    _ = state
    _ = player_context

    movable = [(pid, moves) for pid, moves in legal_moves.items() if moves]
    if not movable:
        raise ValueError("No legal moves available for AlphaZero method.")

    pid, moves = random.choice(movable)
    to_index = random.choice(list(moves))
    delay = random.uniform(0.1, 0.2)
    return pid, to_index, delay
