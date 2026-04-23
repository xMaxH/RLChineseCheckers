"""Canonical-orientation permutations for the hex star board.

The Chinese Checkers star is symmetric under the dihedral group D6. For
every supported colour, we precompute the element of D6 that maps the
canonical "red" perspective onto that colour's actual orientation, plus
the induced permutation on the 121 cell indices. The encoder and action
encoder use these permutations so the network always sees and outputs
actions as if it were playing red, regardless of the colour assigned at
runtime.

For ``my_colour == "red"`` both permutations are the identity, so
training and any red-as-agent code path is unaffected.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from ..game_bridge import HexBoard, OPPOSITES, PINS_PER_PLAYER

# D6 acting on axial hex coordinates (q, r). Rotations are CCW in
# multiples of 60 degrees; the last six are reflections composed with
# those rotations. Only one element is needed per colour but we search
# over all 12 to stay independent of board orientation conventions.
_D6: List[Callable[[int, int], Tuple[int, int]]] = [
    lambda q, r: (q, r),
    lambda q, r: (-r, q + r),
    lambda q, r: (-q - r, q),
    lambda q, r: (-q, -r),
    lambda q, r: (r, -q - r),
    lambda q, r: (q + r, -q),
    lambda q, r: (q + r, -r),
    lambda q, r: (-r, -q),
    lambda q, r: (-q, q + r),
    lambda q, r: (-q - r, r),
    lambda q, r: (r, q),
    lambda q, r: (q, -q - r),
]

CANONICAL_COLOUR = "red"
CANONICAL_OPPOSITE = OPPOSITES[CANONICAL_COLOUR]


def _postype_coords(board: HexBoard, postype: str) -> set:
    return {(c.q, c.r) for c in board.cells if c.postype == postype}


def _find_transform(board: HexBoard, my_colour: str):
    """Return R in D6 s.t. R maps canonical-red cells onto my_colour cells
    and canonical-blue cells onto OPPOSITES[my_colour] cells.
    """
    if my_colour == CANONICAL_COLOUR:
        return _D6[0]
    red = _postype_coords(board, CANONICAL_COLOUR)
    blue = _postype_coords(board, CANONICAL_OPPOSITE)
    mine = _postype_coords(board, my_colour)
    opp = _postype_coords(board, OPPOSITES[my_colour])
    for R in _D6:
        if {R(q, r) for (q, r) in red} == mine and {R(q, r) for (q, r) in blue} == opp:
            return R
    raise RuntimeError(
        f"No D6 element maps canonical '{CANONICAL_COLOUR}' to '{my_colour}'"
    )


class BoardSymmetry:
    """Cell-index and pin-id permutations between canonical
    (red-perspective) and actual board coordinates, per colour.

    Cell-level (arrays of length ``num_cells``):
      * ``canonical_to_actual(c)[j]``: the actual cell index that holds,
        in the colour-``c`` view, what canonical cell ``j`` holds in the
        "played as red" view.
      * ``actual_to_canonical(c)[i]``: the inverse.

    Pin-level (arrays of length ``PINS_PER_PLAYER``): pin ids are
    assigned by the underlying engine in actual-cell-index order, which
    means the same physical (canonical) pin has different ids across
    colours. These permutations remap pin ids so that canonical pin 0
    always corresponds to the pin starting at red's pin-0 cell, etc.
      * ``pin_actual_to_canonical(c)[actual_pid] -> canonical_pid``
      * ``pin_canonical_to_actual(c)[canonical_pid] -> actual_pid``

    Usage:
      * Canonicalize an obs tensor shaped ``(..., num_cells)``::
            obs_canon = obs_actual[..., c2a]
      * Convert an actual destination cell to canonical::
            canon_to = a2c[actual_to]
      * Convert a canonical destination cell to actual::
            actual_to = c2a[canon_to]
    """

    def __init__(self, board: HexBoard):
        self.board = board
        self.num_cells = len(board.cells)
        index_of = {(c.q, c.r): i for i, c in enumerate(board.cells)}

        self._c2a: Dict[str, np.ndarray] = {}
        self._a2c: Dict[str, np.ndarray] = {}
        self._pin_a2c: Dict[str, np.ndarray] = {}
        self._pin_c2a: Dict[str, np.ndarray] = {}

        red_start = list(board.axial_of_colour(CANONICAL_COLOUR))[:PINS_PER_PLAYER]
        red_start_pos = {cell_idx: i for i, cell_idx in enumerate(red_start)}

        for colour in OPPOSITES.keys():
            R = _find_transform(board, colour)
            c2a = np.empty(self.num_cells, dtype=np.int64)
            for canon_i, cell in enumerate(board.cells):
                q2, r2 = R(cell.q, cell.r)
                c2a[canon_i] = index_of[(q2, r2)]
            a2c = np.empty_like(c2a)
            a2c[c2a] = np.arange(self.num_cells, dtype=np.int64)
            self._c2a[colour] = c2a
            self._a2c[colour] = a2c

            colour_start = list(board.axial_of_colour(colour))[:PINS_PER_PLAYER]
            pin_a2c = np.empty(PINS_PER_PLAYER, dtype=np.int64)
            for actual_pid, cell_idx in enumerate(colour_start):
                canon_cell = int(a2c[cell_idx])
                pin_a2c[actual_pid] = red_start_pos[canon_cell]
            pin_c2a = np.empty_like(pin_a2c)
            pin_c2a[pin_a2c] = np.arange(PINS_PER_PLAYER, dtype=np.int64)
            self._pin_a2c[colour] = pin_a2c
            self._pin_c2a[colour] = pin_c2a

    def canonical_to_actual(self, colour: str) -> np.ndarray:
        return self._c2a[colour]

    def actual_to_canonical(self, colour: str) -> np.ndarray:
        return self._a2c[colour]

    def pin_actual_to_canonical(self, colour: str) -> np.ndarray:
        return self._pin_a2c[colour]

    def pin_canonical_to_actual(self, colour: str) -> np.ndarray:
        return self._pin_c2a[colour]


__all__ = ["BoardSymmetry", "CANONICAL_COLOUR"]
