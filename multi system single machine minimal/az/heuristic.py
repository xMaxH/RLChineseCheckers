"""Greedy pin-racer baseline.

Score each legal move as decrease in sum of axial distances of own pins
to nearest cell of opposite-colour zone. Strategy:
  1. If any move has positive gain, restrict to those with the maximum gain.
  2. Else fall back to gain >= 0 (any non-regressing move).
  3. Tiebreak by pin's current distance to goal (descending) so the
     furthest-from-goal pins get prioritized — prevents single-pin shuffle.
  4. Final tiebreak: deterministic by (pin_id, to_idx).
"""

from typing import Dict, List, Tuple, Optional
import random

from .config import COLOUR_OPPOSITES
from .sim import Sim


def _axial_dist_q_r(a_q: int, a_r: int, b_q: int, b_r: int) -> int:
    dq = abs(a_q - b_q)
    dr = abs(a_r - b_r)
    ds = abs((-a_q - a_r) - (-b_q - b_r))
    return max(dq, dr, ds)


def _nearest_target_dist(sim: Sim, q: int, r: int, target_cells) -> int:
    best = 10**9
    for t in target_cells:
        d = _axial_dist_q_r(q, r, t.q, t.r)
        if d < best:
            best = d
    return best


def heuristic_choose_move(
    sim: Sim,
    colour: str,
    legal: Dict[int, List[int]],
    rng: Optional[random.Random] = None,
) -> Tuple[int, int]:
    pool = heuristic_move_pool(sim, colour, legal)
    rng = rng or random.Random(sim.move_count * 9301 + 49297)
    chosen = rng.choice(pool)
    return chosen[2], chosen[3]


def heuristic_move_pool(
    sim: Sim,
    colour: str,
    legal: Dict[int, List[int]],
) -> List[Tuple[int, int, int, int]]:
    """Return all moves tied under the greedy heuristic.

    Tuple format is (gain, current_distance, pin_id, to_idx). Training should
    treat this whole pool as the policy target; picking one random tied move as
    a one-hot label makes equally good actions look wrong.
    """
    opp = COLOUR_OPPOSITES[colour]
    target_indices = sim.board.axial_of_colour(opp)
    target_cells = [sim.board.cells[i] for i in target_indices]

    pins = sim.pins_by_colour[colour]
    cur_dist = {}
    for p in pins:
        cell = sim.board.cells[p.axialindex]
        if cell.postype == opp:
            cur_dist[p.id] = 0
        else:
            cur_dist[p.id] = _nearest_target_dist(sim, cell.q, cell.r, target_cells)

    candidates: List[Tuple[int, int, int, int]] = []  # (gain, cur_dist, pid, to_idx)
    for pid, dests in legal.items():
        if not dests:
            continue
        cur = cur_dist[pid]
        for to_idx in dests:
            cell = sim.board.cells[to_idx]
            if cell.postype == opp:
                new = 0
            else:
                new = _nearest_target_dist(sim, cell.q, cell.r, target_cells)
            gain = cur - new
            candidates.append((gain, cur, pid, to_idx))

    if not candidates:
        raise RuntimeError("heuristic_choose_move with no legal moves")

    max_gain = max(c[0] for c in candidates)
    # Always prefer the largest-gain moves available. If max_gain < 1,
    # accept any non-regressing move (gain >= 0); if those don't exist,
    # any move at all (rare — implies all options regress).
    if max_gain >= 1:
        pool = [c for c in candidates if c[0] == max_gain]
    elif max_gain == 0:
        pool = [c for c in candidates if c[0] == 0]
    else:
        pool = [c for c in candidates if c[0] == max_gain]

    return pool
