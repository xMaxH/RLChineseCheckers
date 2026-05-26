"""Board <-> tensor encoding with hex 60-rotation canonicalization.

Canonicalization: rotate the board so the to-move player's home zone is
always at the "red" corner. This means the policy + value head only have
to learn one orientation rather than six.

Cycle order of colours under 60deg CCW rotation R: (q,r) -> (-r, q+r):
    red -> gray0 -> yellow -> blue -> lawn green -> purple -> red

Slot assignment in canonical frame:
    slot 0 = to-move (always at "red" corner after rotation)
    slot k = colour at (cycle_pos(c) - cycle_pos(to_move)) mod 6
    Complement of to-move always lands at slot 3 ("blue" corner).
"""

import numpy as np
from typing import Dict, List, Tuple

from checkers_board import HexBoard
from .config import (
    NUM_CELLS, PINS_PER_PLAYER, BOARD_CHANNELS, NUM_ACTIONS, MAX_PLAYERS,
    COLOUR_OPPOSITES,
)

# 60deg-rotation cycle order. Verified by tracing corner cells:
#   red corner    (-4, 8) --R--> (-8, 4)  = gray0
#   gray0 corner  (-8, 4) --R--> (-4,-4)  = yellow
#   yellow corner (-4,-4) --R--> ( 4,-8)  = blue
#   blue corner   ( 4,-8) --R--> ( 8,-4)  = lawn green
#   green corner  ( 8,-4) --R--> ( 4, 4)  = purple
#   purple corner ( 4, 4) --R--> (-4, 8)  = red
CYCLE = ['red', 'gray0', 'yellow', 'blue', 'lawn green', 'purple']
CYCLE_POS = {c: i for i, c in enumerate(CYCLE)}

# Build a singleton board for index <-> (q,r) lookups.
_BOARD = HexBoard()
assert len(_BOARD.cells) == NUM_CELLS, f"expected {NUM_CELLS} cells, got {len(_BOARD.cells)}"


def _rotate60(q: int, r: int) -> Tuple[int, int]:
    """One 60deg CCW step on pointy-top axial coords."""
    return (-r, q + r)


def _build_rot_perm(k: int) -> np.ndarray:
    """perm[orig_idx] = canonical_idx after applying R^k."""
    perm = np.zeros(NUM_CELLS, dtype=np.int64)
    for orig_idx, cell in enumerate(_BOARD.cells):
        q, r = cell.q, cell.r
        for _ in range(k):
            q, r = _rotate60(q, r)
        perm[orig_idx] = _BOARD.index_of[(q, r)]
    return perm


ROT_PERMS = np.stack([_build_rot_perm(k) for k in range(6)], axis=0)  # (6, 121)
INV_ROT_PERMS = np.zeros_like(ROT_PERMS)
for _k in range(6):
    INV_ROT_PERMS[_k, ROT_PERMS[_k]] = np.arange(NUM_CELLS, dtype=np.int64)


def rotation_for_to_move(to_move: str) -> int:
    """k such that R^k maps to_move's home corner to red's corner."""
    return (6 - CYCLE_POS[to_move]) % 6


def slot_of(colour: str, to_move: str) -> int:
    """Canonical slot a given colour occupies when to_move is canonicalized to slot 0."""
    return (CYCLE_POS[colour] - CYCLE_POS[to_move]) % 6


# Channel layout:
#   0..5    colour occupancy by canonical slot
#   6..11   static home-zone templates by canonical slot
#   12      empty-cell flag
#   13      to-move pins-in-goal scalar broadcast
#   14..23  to-move pin-id planes, one plane per actionable pin id
PIN_ID_CHANNEL_START = 14


# --- Static home-zone templates (one per slot, in canonical frame) ----------
# Slot s "home zone" = original cells with postype == CYCLE[s].
# After rotation R^k (k = rotation_for_to_move), these cells land at canonical
# positions ROT_PERMS[k][orig_indices]. But we precompute templates for the
# canonical frame *as-if no rotation needed*, because slot s in canonical frame
# corresponds to the same canonical cells regardless of which to-move we picked
# (rotation moves *pins* to slot positions; the slot positions themselves are
#  fixed once we say "slot 0 = red corner in canonical frame").
HOME_ZONE_TEMPLATE = np.zeros((MAX_PLAYERS, NUM_CELLS), dtype=np.float32)
for _s, _c in enumerate(CYCLE):
    for _i, _cell in enumerate(_BOARD.cells):
        if _cell.postype == _c:
            HOME_ZONE_TEMPLATE[_s, _i] = 1.0


def encode_state(
    pins_by_colour: Dict[str, List[int]],
    to_move: str,
    turn_order: List[str],
    move_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode board state to (board (24, 121) float32, global (8,) float32).

    `pins_by_colour` maps absolute colour strings to lists of original axial indices.
    Returns canonical-frame tensors with to_move at slot 0.
    """
    k = rotation_for_to_move(to_move)
    rot = ROT_PERMS[k]

    board = np.zeros((BOARD_CHANNELS, NUM_CELLS), dtype=np.float32)

    pins_in_goal_to_move = 0
    for colour, idxs in pins_by_colour.items():
        if not idxs:
            continue
        s = slot_of(colour, to_move)
        canonical_idxs = rot[np.asarray(idxs, dtype=np.int64)]
        board[s, canonical_idxs] = 1.0
        if colour == to_move:
            # The policy action is pin_id x destination. Occupancy alone loses
            # pin identity, so expose the current player's pin ids explicitly.
            for pid, orig_i in enumerate(idxs[:PINS_PER_PLAYER]):
                board[PIN_ID_CHANNEL_START + pid, rot[orig_i]] = 1.0

            # count pins in opposite zone (= the to_move's goal zone)
            opp = COLOUR_OPPOSITES[to_move]
            for orig_i in idxs:
                if _BOARD.cells[orig_i].postype == opp:
                    pins_in_goal_to_move += 1

    # Channels 6..11: home-zone templates per slot (static)
    board[6:12] = HOME_ZONE_TEMPLATE

    # Channel 12: empty flag (cells with no pin of any colour)
    occupied = board[0:6].sum(axis=0)
    board[12] = (occupied == 0).astype(np.float32)

    # Channel 13: broadcast scalar — fraction of pins-in-goal for to-move
    board[13] = pins_in_goal_to_move / float(PINS_PER_PLAYER)

    # Global vec: [N_players_one_hot(5), move_count_norm, my_pins_in_goal_norm, my_turn_idx_norm]
    n_players = len(turn_order)
    glob = np.zeros(8, dtype=np.float32)
    if 2 <= n_players <= 6:
        glob[n_players - 2] = 1.0
    glob[5] = min(move_count, 300) / 300.0
    glob[6] = pins_in_goal_to_move / float(PINS_PER_PLAYER)
    glob[7] = turn_order.index(to_move) / float(max(1, n_players)) if to_move in turn_order else 0.0

    return board, glob


def encode_legal_mask(
    legal_moves: Dict[int, List[int]],
    to_move: str,
) -> np.ndarray:
    """Build (NUM_ACTIONS=1210,) bool mask in canonical frame.

    legal_moves keys are pin_ids 0..9; values are original to-indices.
    """
    k = rotation_for_to_move(to_move)
    rot = ROT_PERMS[k]
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for pid, dests in legal_moves.items():
        if not dests:
            continue
        canon = rot[np.asarray(dests, dtype=np.int64)]
        base = int(pid) * NUM_CELLS
        mask[base + canon] = True
    return mask


def decode_action(action_idx: int, to_move: str) -> Tuple[int, int]:
    """Map canonical action index -> (pin_id, original_to_index)."""
    pid = action_idx // NUM_CELLS
    canon_to = action_idx % NUM_CELLS
    k = rotation_for_to_move(to_move)
    orig_to = int(INV_ROT_PERMS[k][canon_to])
    return pid, orig_to


def encode_action(pin_id: int, orig_to_index: int, to_move: str) -> int:
    """Map (pin_id, original to_index) -> canonical action index."""
    k = rotation_for_to_move(to_move)
    canon_to = int(ROT_PERMS[k][orig_to_index])
    return pin_id * NUM_CELLS + canon_to


def value_target_to_canonical(
    outcomes_by_colour: Dict[str, float],
    to_move: str,
) -> np.ndarray:
    """Convert {colour: +1/-1/0} to a canonical (6,) value target vector.

    Index s holds the outcome for the colour that occupies canonical slot s
    when to_move is the to-move player. Absent slots get a NaN sentinel
    that the loss function will mask out.
    """
    out = np.full(MAX_PLAYERS, np.nan, dtype=np.float32)
    for colour, v in outcomes_by_colour.items():
        s = slot_of(colour, to_move)
        out[s] = v
    return out
