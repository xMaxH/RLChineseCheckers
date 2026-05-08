"""Precomputed Chinese Checkers board topology and lightweight state.

Builds all expensive lookups (neighbors, hop pairs, zone indices, hex
distances to each goal zone, encoding masks) once at import time, so
that simulation and MCTS clones are O(1) numpy copies instead of
re-instantiating HexBoard + 10 Pin objects.

Loaded once per process. Topology is read from the canonical HexBoard
so behavior matches game.py exactly.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from checkers_board import HexBoard


DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
)

_HEX_BOARD = HexBoard()
N_CELLS: int = len(_HEX_BOARD.cells)

_QR = np.zeros((N_CELLS, 2), dtype=np.int32)
for _i, _c in enumerate(_HEX_BOARD.cells):
    _QR[_i, 0] = _c.q
    _QR[_i, 1] = _c.r
QR = _QR

# NEIGHBORS[i, d] = neighbor cell index in direction d, or -1 if off-board.
NEIGHBORS = np.full((N_CELLS, 6), -1, dtype=np.int32)
# HOPS[i, d] = (over_idx, land_idx) when stepping 2 cells in direction d, both -1 if off-board.
HOPS = np.full((N_CELLS, 6, 2), -1, dtype=np.int32)

for _i in range(N_CELLS):
    _q = int(QR[_i, 0])
    _r = int(QR[_i, 1])
    for _d, (_dq, _dr) in enumerate(DIRECTIONS):
        _n = _HEX_BOARD.index_of.get((_q + _dq, _r + _dr))
        if _n is not None:
            NEIGHBORS[_i, _d] = _n
        _o = _HEX_BOARD.index_of.get((_q + _dq, _r + _dr))
        _l = _HEX_BOARD.index_of.get((_q + 2 * _dq, _r + 2 * _dr))
        if _o is not None and _l is not None:
            HOPS[_i, _d, 0] = _o
            HOPS[_i, _d, 1] = _l

# Convert to nested Python lists for tight inner loops (Python int access is
# ~2x faster than indexing into a numpy array element-by-element).
_NEIGHBORS_LIST: List[List[int]] = [[int(NEIGHBORS[i, d]) for d in range(6)] for i in range(N_CELLS)]
_HOPS_LIST: List[List[Tuple[int, int]]] = [
    [(int(HOPS[i, d, 0]), int(HOPS[i, d, 1])) for d in range(6)]
    for i in range(N_CELLS)
]


COLOUR_LIST: Tuple[str, ...] = ("red", "lawn green", "yellow", "blue", "gray0", "purple")
COLOUR_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(COLOUR_LIST)}
COLOUR_OPPOSITES: Dict[str, str] = dict(_HEX_BOARD.colour_opposites)

ZONE_INDICES: Dict[str, np.ndarray] = {}   # cells where each colour starts
GOAL_INDICES: Dict[str, np.ndarray] = {}   # cells of opposite zone (target)
IS_IN_ZONE: Dict[str, np.ndarray] = {}     # bool mask of own start zone
IS_IN_GOAL: Dict[str, np.ndarray] = {}     # bool mask of opposite/goal zone

for _colour in COLOUR_LIST:
    _zone = np.array(_HEX_BOARD.axial_of_colour(_colour), dtype=np.int32)
    ZONE_INDICES[_colour] = _zone
    _is_in = np.zeros(N_CELLS, dtype=bool)
    _is_in[_zone] = True
    IS_IN_ZONE[_colour] = _is_in

for _colour in COLOUR_LIST:
    GOAL_INDICES[_colour] = ZONE_INDICES[COLOUR_OPPOSITES[_colour]]
    IS_IN_GOAL[_colour] = IS_IN_ZONE[COLOUR_OPPOSITES[_colour]]


def _hex_dist(a: int, b: int) -> int:
    aq, ar = int(QR[a, 0]), int(QR[a, 1])
    bq, br = int(QR[b, 0]), int(QR[b, 1])
    return (abs(aq - bq) + abs(ar - br) + abs((-aq - ar) - (-bq - br))) // 2


# DIST_TO_GOAL[colour][i] = min hex distance from cell i to nearest cell in colour's GOAL zone.
DIST_TO_GOAL: Dict[str, np.ndarray] = {}
for _colour in COLOUR_LIST:
    _goal = GOAL_INDICES[_colour]
    _dist = np.zeros(N_CELLS, dtype=np.int32)
    for _i in range(N_CELLS):
        _dist[_i] = min(_hex_dist(_i, int(g)) for g in _goal)
    DIST_TO_GOAL[_colour] = _dist

INITIAL_POSITIONS: Dict[str, np.ndarray] = {
    c: ZONE_INDICES[c][:10].copy() for c in COLOUR_LIST
}


# ---- pre-built torch encoding masks (one-hot zone channels) ----
# All zone channels are constant per colour; cache as float32 tensors so
# encode() is a few buffer writes instead of recomputing axial_of_colour.
_ZONE_TENSOR: Dict[str, torch.Tensor] = {
    c: torch.from_numpy(IS_IN_ZONE[c].astype(np.float32)) for c in COLOUR_LIST
}
_GOAL_TENSOR: Dict[str, torch.Tensor] = {
    c: torch.from_numpy(IS_IN_GOAL[c].astype(np.float32)) for c in COLOUR_LIST
}


def zone_mask(colour: str) -> torch.Tensor:
    return _ZONE_TENSOR[colour]


def goal_mask(colour: str) -> torch.Tensor:
    return _GOAL_TENSOR[colour]


def get_possible_moves(start_idx: int, occupied: np.ndarray) -> List[int]:
    """Legal target cells for a pin at start_idx given board occupancy.

    Matches Pin.getPossibleMoves semantics exactly:
      single-step to any empty neighbor; OR
      one-or-more chained hops (jump over occupied, land on empty).
    """
    possible = set()
    nbrs = _NEIGHBORS_LIST[start_idx]
    for d in range(6):
        n = nbrs[d]
        if n >= 0 and not occupied[n]:
            possible.add(n)

    visited = {start_idx}
    stack = [start_idx]
    while stack:
        cur = stack.pop()
        cur_hops = _HOPS_LIST[cur]
        for d in range(6):
            over, land = cur_hops[d]
            if over < 0 or land < 0:
                continue
            if occupied[over] and not occupied[land] and land not in visited:
                possible.add(land)
                visited.add(land)
                stack.append(land)

    return sorted(possible)


def total_distance_to_goal(positions: np.ndarray, colour: str) -> int:
    """Sum of min hex distances to goal across pins not already in goal."""
    dist = DIST_TO_GOAL[colour]
    in_goal = IS_IN_GOAL[colour]
    total = 0
    for p in positions:
        pi = int(p)
        if not in_goal[pi]:
            total += int(dist[pi])
    return total


def all_in_goal(positions: np.ndarray, colour: str) -> bool:
    return bool(IS_IN_GOAL[colour][positions].all())


# Pin id -> board cell hex distance lookup, used by pure-numpy heuristic.
def hex_distance(a_idx: int, b_idx: int) -> int:
    return _hex_dist(a_idx, b_idx)
