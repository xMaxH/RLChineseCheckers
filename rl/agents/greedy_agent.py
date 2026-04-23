"""Greedy heuristic agent.

Picks the legal action that maximises the reduction in axial distance
to the goal zone for the moved pin. Uses a HexBoard for geometry and
a live mapping ``pin_id -> current cell index`` supplied by the caller
(via ``set_pin_sources``) before each ``act`` call.

Used as a baseline and for priming the DQN replay buffer.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..encoding import ActionEncoder
from ..game_bridge import HexBoard, OPPOSITES
from .base import Agent


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(
        self,
        board: HexBoard,
        my_colour: str,
        epsilon: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.board = board
        self.my_colour = my_colour
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        opp = OPPOSITES[my_colour]
        self._goal_cells = [board.cells[i] for i in board.axial_of_colour(opp)]

        self._pin_sources: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def set_pin_sources(self, pin_sources: Sequence[int]) -> None:
        """Update current ``pin_id -> cell_idx`` mapping."""
        self._pin_sources = {i: int(src) for i, src in enumerate(pin_sources)}

    # ------------------------------------------------------------------
    @staticmethod
    def _axial_dist(a, b) -> int:
        dq = abs(a.q - b.q)
        dr = abs(a.r - b.r)
        ds = abs((-a.q - a.r) - (-b.q - b.r))
        return max(dq, dr, ds)

    def _min_goal_dist(self, cell_idx: int) -> int:
        cell = self.board.cells[cell_idx]
        if cell.postype == OPPOSITES[self.my_colour]:
            return 0
        return min(self._axial_dist(cell, g) for g in self._goal_cells)

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, mask: np.ndarray, training: bool = False) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return -1
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal.tolist()))

        best_score = -float("inf")
        best_actions: List[int] = []
        for flat in legal.tolist():
            pin_id, to_idx = ActionEncoder.from_flat(int(flat))
            src = self._pin_sources.get(pin_id)
            if src is None:
                # No context set; behave randomly on this action.
                score = 0.0
            else:
                score = self._min_goal_dist(src) - self._min_goal_dist(to_idx)
            if score > best_score + 1e-9:
                best_score = score
                best_actions = [int(flat)]
            elif score > best_score - 1e-9:
                best_actions.append(int(flat))

        return int(self.rng.choice(best_actions))
