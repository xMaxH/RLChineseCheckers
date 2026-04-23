"""Action encoder: (pin_id, to_index) <-> flat action index with mask.

Action space = PINS_PER_PLAYER * NUM_CELLS. The network predicts Q-values
over this flat space and we mask illegal actions to ``-inf`` before
argmax. Building the mask from the server-style ``legal_moves`` dict
keeps train- and inference-time paths identical.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from ..game_bridge import PINS_PER_PLAYER, num_cells as _num_cells

NUM_CELLS = _num_cells()
NUM_ACTIONS = PINS_PER_PLAYER * NUM_CELLS


class ActionEncoder:
    """Flatten and mask actions. Stateless.

    The ``actual_to_canon`` / ``canon_to_actual`` permutations let the
    class operate either in actual board coordinates (default) or in the
    canonical (red-perspective) frame used by the trained network. For
    red the permutations are the identity, so call sites that don't pass
    them behave exactly as before.
    """

    pins_per_player: int = PINS_PER_PLAYER
    num_cells: int = NUM_CELLS
    num_actions: int = NUM_ACTIONS

    # ------------------------------------------------------------------
    @staticmethod
    def to_flat(pin_id: int, to_index: int) -> int:
        if not (0 <= pin_id < PINS_PER_PLAYER):
            raise ValueError(f"pin_id {pin_id} out of range")
        if not (0 <= to_index < NUM_CELLS):
            raise ValueError(f"to_index {to_index} out of range")
        return pin_id * NUM_CELLS + to_index

    @staticmethod
    def from_flat(
        flat: int,
        canon_to_actual: Optional[np.ndarray] = None,
        pin_canon_to_actual: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """Decode a flat action into ``(pin_id, to_index)``.

        When ``canon_to_actual`` / ``pin_canon_to_actual`` are provided,
        the decoded fields are translated from canonical back to actual
        board / pin coordinates.
        """
        if not (0 <= flat < NUM_ACTIONS):
            raise ValueError(f"flat action {flat} out of range")
        pid, to_idx = divmod(flat, NUM_CELLS)
        if canon_to_actual is not None:
            to_idx = int(canon_to_actual[to_idx])
        if pin_canon_to_actual is not None:
            pid = int(pin_canon_to_actual[pid])
        return pid, to_idx

    # ------------------------------------------------------------------
    @staticmethod
    def build_mask(
        legal_moves: Mapping[object, Sequence[int]],
        actual_to_canon: Optional[np.ndarray] = None,
        pin_actual_to_canon: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build a boolean mask of shape (NUM_ACTIONS,).

        The server/env returns legal_moves keyed by pin_id (int or str
        depending on whether it has been round-tripped through JSON).
        We accept both. When ``actual_to_canon`` / ``pin_actual_to_canon``
        are provided, each ``to_index`` / ``pin_id`` is translated to its
        canonical slot before the corresponding mask bit is set.
        """
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for pid_key, targets in legal_moves.items():
            try:
                pid = int(pid_key)
            except (TypeError, ValueError):
                continue
            if not (0 <= pid < PINS_PER_PLAYER):
                continue
            if pin_actual_to_canon is not None:
                pid = int(pin_actual_to_canon[pid])
            base = pid * NUM_CELLS
            for t in targets:
                ti = int(t)
                if not (0 <= ti < NUM_CELLS):
                    continue
                if actual_to_canon is not None:
                    ti = int(actual_to_canon[ti])
                mask[base + ti] = True
        return mask

    @staticmethod
    def any_legal(mask: np.ndarray) -> bool:
        return bool(mask.any())
