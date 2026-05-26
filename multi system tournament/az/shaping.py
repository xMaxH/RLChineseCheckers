"""Potential-based progress shaping for genuine-RL value targets.

The value head used to train on a flat terminal +-1, which collapses in
5-6p games: the candidate almost always loses, so every target is -1 and
there is no gradient to rank positions. This module supplies a bounded
progress potential Phi(colour, state) built from the same goal-distance
metric the heuristic uses, plus pins-in-goal.

Phi is used two ways (see selfplay.py / mcts.py):

  * value target  v[c] = T_c + Phi_c(end) - Phi_c(now)
        the telescoped potential-shaped Monte-Carlo return. Total shaping
        over a game telescopes to Phi(end) - Phi(start) and is therefore
        independent of game length -- it cannot reward stalling, it only
        grades positions by genuine progress.

  * MCTS leaf     leaf_value[c] += Phi_c(leaf)
        the head learns V_head ≈ E[T + Phi_final] - Phi(s); adding Phi(leaf)
        back recovers the unbiased E[T + Phi_final] for MaxN backup, so
        search is not biased toward low-progress leaves.

Phi lies in [0, scale]; with scale <= 0.15 and terminal weight ~0.8 every
value target stays inside the tanh value head's (-1, 1) range.
"""

from __future__ import annotations

from typing import Dict, List

from checkers_board import HexBoard

from .config import COLOUR_ORDER, COLOUR_OPPOSITES, PINS_PER_PLAYER


def _axial_dist(aq: int, ar: int, bq: int, br: int) -> int:
    return max(abs(aq - bq), abs(ar - br), abs((-aq - ar) - (-bq - br)))


# Singleton board for static lookups (cell coords / colour zones never change).
_BOARD = HexBoard()

# Per-colour nearest-goal distance for every board cell index. A cell with
# distance 0 is itself a goal-zone cell, i.e. a pin on it is home.
_DIST_TO_GOAL: Dict[str, List[int]] = {}
# Per-colour sum-distance of the 10 home start cells -- the Phi=0 reference.
_D0: Dict[str, float] = {}

for _c in COLOUR_ORDER:
    _goal_cells = [
        (_BOARD.cells[i].q, _BOARD.cells[i].r)
        for i in _BOARD.axial_of_colour(COLOUR_OPPOSITES[_c])
    ]
    _DIST_TO_GOAL[_c] = [
        min(_axial_dist(cell.q, cell.r, gq, gr) for gq, gr in _goal_cells)
        for cell in _BOARD.cells
    ]
    _home = _BOARD.axial_of_colour(_c)[:PINS_PER_PLAYER]
    _D0[_c] = float(max(1, sum(_DIST_TO_GOAL[_c][i] for i in _home)))


def _pins_and_distance(sim, colour: str):
    """Return (pins_in_goal, sum_axial_distance_to_goal) for `colour`."""
    table = _DIST_TO_GOAL[colour]
    pins_in_goal = 0
    total = 0
    for p in sim.pins_by_colour[colour]:
        d = table[p.axialindex]
        if d == 0:
            pins_in_goal += 1
        else:
            total += d
    return pins_in_goal, total


def progress(sim, colour: str, goal_weight: float) -> float:
    """Fraction-of-the-way-home in [0, 1]: ~0 at game start, 1 at a full win."""
    pins_in_goal, total = _pins_and_distance(sim, colour)
    goal_term = pins_in_goal / float(PINS_PER_PLAYER)
    dist_term = (_D0[colour] - total) / _D0[colour]
    if dist_term < 0.0:
        dist_term = 0.0
    elif dist_term > 1.0:
        dist_term = 1.0
    return goal_weight * goal_term + (1.0 - goal_weight) * dist_term


def potential(sim, colour: str, scale: float, goal_weight: float) -> float:
    """Shaping potential Phi(colour, state) in [0, scale]."""
    return scale * progress(sim, colour, goal_weight)


def potentials(sim, scale: float, goal_weight: float) -> Dict[str, float]:
    """Phi for every colour currently in the game."""
    return {c: potential(sim, c, scale, goal_weight) for c in sim.colours}
