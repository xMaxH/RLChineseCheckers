"""State encoder: board snapshot -> fixed-size tensor.

Produces a multi-channel representation over all cells (121 for the
default HexBoard). Always encoded from the *agent's* perspective so the
network is colour-agnostic.

Channels:
    0: my_pins            1 where ANY of my pins sits (aggregated)
    1: opponent_pins      1 where any opponent pin sits (aggregated)
    2: my_goal_zone       1 where cell.postype == my opposite colour
    3: my_start_zone      1 where cell.postype == my colour
    4: empty              1 where no pin sits on this cell
    5: move_progress      constant = move_count / max_moves broadcast
    6..15: per-pin-id     channel i+6 has 1 at my pin-id i's cell. This
                          lets the action head ``(pin_id, to_idx)`` relate
                          pin_id to a spatial location, which is otherwise
                          hidden (pins of the same colour are
                          interchangeable from the game rules' perspective
                          but the action indexes them).

The encoder is stateless; instances are cheap to build. A single instance
is bound to a specific HexBoard to cache static masks (goal/start zones).
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..game_bridge import HexBoard, OPPOSITES, PINS_PER_PLAYER
from .board_symmetry import BoardSymmetry

BASE_CHANNELS = 6
OBS_CHANNELS = BASE_CHANNELS + PINS_PER_PLAYER  # 16


class StateEncoder:
    """Encode a board state to a (OBS_CHANNELS, num_cells) float32 tensor.

    The returned tensor is always in the *canonical* (red-perspective)
    orientation: regardless of which colour the agent plays, the output
    is rotated so that "my pins" live where red's pins would live if the
    agent were playing red. This makes the network colour-invariant.
    For my_colour == "red" the rotation is identity.
    """

    def __init__(
        self,
        board: HexBoard,
        max_moves: int = 600,
        symmetry: Optional[BoardSymmetry] = None,
    ):
        self.board = board
        self.num_cells = len(board.cells)
        self.max_moves = max(1, int(max_moves))
        self.symmetry = symmetry if symmetry is not None else BoardSymmetry(board)

        # Pre-computed static masks keyed by colour.
        self._goal_masks: Dict[str, np.ndarray] = {}
        self._start_masks: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _zone_mask(self, postype: str) -> np.ndarray:
        mask = np.zeros(self.num_cells, dtype=np.float32)
        for i, cell in enumerate(self.board.cells):
            if cell.postype == postype:
                mask[i] = 1.0
        return mask

    def goal_mask(self, my_colour: str) -> np.ndarray:
        opposite = OPPOSITES[my_colour]
        if opposite not in self._goal_masks:
            self._goal_masks[opposite] = self._zone_mask(opposite)
        return self._goal_masks[opposite]

    def start_mask(self, my_colour: str) -> np.ndarray:
        if my_colour not in self._start_masks:
            self._start_masks[my_colour] = self._zone_mask(my_colour)
        return self._start_masks[my_colour]

    # ------------------------------------------------------------------
    def encode(
        self,
        pins_by_colour: Mapping[str, Sequence[int]],
        my_colour: str,
        move_count: int = 0,
    ) -> np.ndarray:
        """Encode from a dict of {colour: [cell_index, ...]}.

        This matches the JSON format the server returns via ``get_state``
        (``state["pins"]``), so the same encoder works in training and at
        inference time against the real server.
        """
        obs = np.zeros((OBS_CHANNELS, self.num_cells), dtype=np.float32)

        my_indices = list(pins_by_colour.get(my_colour, ()))
        for idx in my_indices:
            obs[0, idx] = 1.0

        for colour, indices in pins_by_colour.items():
            if colour == my_colour:
                continue
            for idx in indices:
                obs[1, idx] = 1.0

        obs[2] = self.goal_mask(my_colour)
        obs[3] = self.start_mask(my_colour)
        obs[4] = 1.0 - (obs[0] + obs[1])
        obs[5] = float(move_count) / float(self.max_moves)

        # Per-pin-id channels (6 .. 6+PINS-1). Gives the network an
        # explicit link between action pin_id and spatial position.
        # We store these in CANONICAL pin-id order so the network sees
        # the same pin ids across all colours.
        pin_a2c = self.symmetry.pin_actual_to_canonical(my_colour)
        for actual_pid, idx in enumerate(my_indices):
            canon_pid = int(pin_a2c[actual_pid])
            if canon_pid >= (OBS_CHANNELS - BASE_CHANNELS):
                continue
            obs[BASE_CHANNELS + canon_pid, idx] = 1.0

        # Canonicalize so the network sees every colour as if it were red.
        c2a = self.symmetry.canonical_to_actual(my_colour)
        return obs[:, c2a]

    # ------------------------------------------------------------------
    def flat_size(self) -> int:
        return OBS_CHANNELS * self.num_cells

    def encode_flat(
        self,
        pins_by_colour: Mapping[str, Sequence[int]],
        my_colour: str,
        move_count: int = 0,
    ) -> np.ndarray:
        return self.encode(pins_by_colour, my_colour, move_count).reshape(-1)
