"""Gym-style in-process Chinese Checkers environment.

Wraps the existing HexBoard/Pin engine so training runs faster than the
TCP server loop while keeping observation/action interfaces aligned.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..encoding import ActionEncoder, BoardSymmetry, StateEncoder
from ..game_bridge import COLOUR_ORDER, OPPOSITES, PINS_PER_PLAYER, HexBoard, Pin

OpponentPolicy = Callable[["CheckersEnv", str], int]


@dataclass
class EnvConfig:
    mode: str = "solo_race"
    my_colour: str = "red"
    opponent_colours: Sequence[str] = field(default_factory=list)
    max_steps: int = 600
    seed: Optional[int] = None
    reward_per_pin_home: float = 1.0
    reward_step_penalty: float = 0.01
    reward_all_home_bonus: float = 10.0
    reward_distance_shaping: float = 0.05
    reward_win_bonus: float = 10.0
    reward_lose_penalty: float = -5.0
    anti_loop_enabled: bool = False
    anti_loop_window: int = 12
    anti_loop_revisit_penalty: float = 0.03
    anti_loop_aba_penalty: float = 0.06

    def __post_init__(self) -> None:
        if self.mode not in ("solo_race", "multi"):
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.my_colour not in OPPOSITES:
            raise ValueError(f"Unknown my_colour: {self.my_colour}")
        if self.mode == "multi" and not self.opponent_colours:
            self.opponent_colours = [OPPOSITES[self.my_colour]]


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]


class CheckersEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.board = HexBoard()
        self.symmetry = BoardSymmetry(self.board)
        self.encoder = StateEncoder(self.board, max_moves=cfg.max_steps, symmetry=self.symmetry)
        self.pins_by_colour: Dict[str, List[Pin]] = {}
        self.turn_order: List[str] = []
        self.current_turn_index: int = 0
        self.step_count: int = 0
        self.agent_move_count: int = 0
        self.rng = random.Random(cfg.seed)
        self._goal_cells: Dict[str, List] = {}
        self._last_my_distance: float = 0.0
        self._last_pins_home: int = 0
        self._recent_state_sigs: deque[tuple] = deque(maxlen=max(1, int(cfg.anti_loop_window)))
        self._pin_recent_positions: Dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=3))

    def _clear_board(self) -> None:
        for cell in self.board.cells:
            cell.occupied = False

    def _spawn_colour(self, colour: str) -> List[Pin]:
        idxs = self.board.axial_of_colour(colour)[:PINS_PER_PLAYER]
        return [Pin(self.board, idxs[i], id=i, color=colour) for i in range(len(idxs))]

    def _compute_turn_order(self, colours: List[str]) -> List[str]:
        first = colours[0]
        rotated = list(COLOUR_ORDER)
        if first in COLOUR_ORDER:
            idx = COLOUR_ORDER.index(first)
            rotated = COLOUR_ORDER[idx:] + COLOUR_ORDER[:idx]
        present = set(colours)
        return [c for c in rotated if c in present]

    @staticmethod
    def _axial_dist(a, b) -> int:
        dq = abs(a.q - b.q)
        dr = abs(a.r - b.r)
        ds = abs((-a.q - a.r) - (-b.q - b.r))
        return max(dq, dr, ds)

    def _goal_for(self, colour: str):
        if colour not in self._goal_cells:
            opp = OPPOSITES[colour]
            idxs = self.board.axial_of_colour(opp)
            self._goal_cells[colour] = [self.board.cells[i] for i in idxs]
        return self._goal_cells[colour]

    def _total_distance(self, colour: str) -> float:
        goals = self._goal_for(colour)
        opp = OPPOSITES[colour]
        total = 0
        for p in self.pins_by_colour[colour]:
            cell = self.board.cells[p.axialindex]
            if cell.postype == opp:
                continue
            total += min(self._axial_dist(cell, tgt) for tgt in goals)
        return float(total)

    def _pins_in_goal(self, colour: str) -> int:
        opp = OPPOSITES[colour]
        return sum(1 for p in self.pins_by_colour[colour] if self.board.cells[p.axialindex].postype == opp)

    def _state_signature(self) -> tuple:
        by_colour = []
        for colour in sorted(self.pins_by_colour.keys()):
            by_colour.append((colour, tuple(sorted(p.axialindex for p in self.pins_by_colour[colour]))))
        return (tuple(by_colour), self.current_turn_index)

    def _pins_as_indices(self) -> Dict[str, List[int]]:
        return {c: [p.axialindex for p in pins] for c, pins in self.pins_by_colour.items()}

    def _build_obs(self) -> np.ndarray:
        return self.encoder.encode(self._pins_as_indices(), self.cfg.my_colour, move_count=self.step_count)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, object]]:
        if seed is not None:
            self.rng = random.Random(seed)
        self._clear_board()
        self.pins_by_colour = {}
        colours = [self.cfg.my_colour] if self.cfg.mode == "solo_race" else [self.cfg.my_colour] + list(self.cfg.opponent_colours)
        for c in colours:
            self.pins_by_colour[c] = self._spawn_colour(c)
            self._goal_for(c)
        self.turn_order = self._compute_turn_order(colours)
        self.current_turn_index = 0
        self.step_count = 0
        self.agent_move_count = 0
        self._last_my_distance = self._total_distance(self.cfg.my_colour)
        self._last_pins_home = self._pins_in_goal(self.cfg.my_colour)
        self._recent_state_sigs = deque(maxlen=max(1, int(self.cfg.anti_loop_window)))
        self._pin_recent_positions = defaultdict(lambda: deque(maxlen=3))
        for pid, p in enumerate(self.pins_by_colour[self.cfg.my_colour]):
            self._pin_recent_positions[pid].append(int(p.axialindex))
        self._recent_state_sigs.append(self._state_signature())
        return self._build_obs(), {"legal_mask": self.action_mask()}

    def legal_moves_for(self, colour: str) -> Dict[int, List[int]]:
        return {i: pin.getPossibleMoves() for i, pin in enumerate(self.pins_by_colour.get(colour, []))}

    def action_mask(self, colour: Optional[str] = None) -> np.ndarray:
        colour = colour or self.cfg.my_colour
        legal = self.legal_moves_for(colour)
        a2c = self.symmetry.actual_to_canonical(colour)
        p2c = self.symmetry.pin_actual_to_canonical(colour)
        return ActionEncoder.build_mask(legal, actual_to_canon=a2c, pin_actual_to_canon=p2c)

    def _apply_move(self, colour: str, pin_id: int, to_index: int) -> bool:
        pins = self.pins_by_colour[colour]
        if not (0 <= pin_id < len(pins)):
            return False
        pin = pins[pin_id]
        if to_index < 0 or to_index >= len(self.board.cells):
            return False
        if self.board.cells[to_index].occupied:
            return False
        if to_index not in pin.getPossibleMoves():
            return False
        self.board.cells[pin.axialindex].occupied = False
        pin.axialindex = int(to_index)
        self.board.cells[int(to_index)].occupied = True
        return True

    def _advance_turn(self) -> None:
        self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)

    def current_turn_colour(self) -> str:
        return self.turn_order[self.current_turn_index]

    def _is_colour_winner(self, colour: str) -> bool:
        opp = OPPOSITES[colour]
        return all(self.board.cells[p.axialindex].postype == opp for p in self.pins_by_colour[colour])

    def _run_opponents(self, opponent_policies: Mapping[str, OpponentPolicy]) -> Optional[str]:
        while self.current_turn_colour() != self.cfg.my_colour:
            colour = self.current_turn_colour()
            policy = opponent_policies.get(colour, _random_policy)
            try:
                flat = policy(self, colour)
            except Exception:
                flat = _random_policy(self, colour)
            if flat >= 0:
                c2a = self.symmetry.canonical_to_actual(colour)
                p2a = self.symmetry.pin_canonical_to_actual(colour)
                pid, to_idx = ActionEncoder.from_flat(flat, canon_to_actual=c2a, pin_canon_to_actual=p2a)
                if not self._apply_move(colour, pid, to_idx):
                    flat = _random_policy(self, colour)
                    if flat >= 0:
                        pid, to_idx = ActionEncoder.from_flat(flat, canon_to_actual=c2a, pin_canon_to_actual=p2a)
                        self._apply_move(colour, pid, to_idx)
            if self._is_colour_winner(colour):
                return f"opponent_win:{colour}"
            self._advance_turn()
            self.step_count += 1
            if self.step_count >= self.cfg.max_steps:
                return "truncated"
        return None

    def step(self, action: int, opponent_policies: Optional[Mapping[str, OpponentPolicy]] = None) -> StepResult:
        if self.current_turn_colour() != self.cfg.my_colour:
            raise RuntimeError("step() called when it is not agent turn")
        c2a = self.symmetry.canonical_to_actual(self.cfg.my_colour)
        p2a = self.symmetry.pin_canonical_to_actual(self.cfg.my_colour)
        pid, to_idx = ActionEncoder.from_flat(int(action), canon_to_actual=c2a, pin_canon_to_actual=p2a)
        if not self._apply_move(self.cfg.my_colour, pid, to_idx):
            return StepResult(self._build_obs(), -1.0, True, False, {"error": "illegal_action"})

        self.step_count += 1
        self.agent_move_count += 1
        reward = -self.cfg.reward_step_penalty
        loop_revisit = False
        loop_aba = False

        new_dist = self._total_distance(self.cfg.my_colour)
        reward += self.cfg.reward_distance_shaping * float(self._last_my_distance - new_dist)
        self._last_my_distance = new_dist
        pins_home = self._pins_in_goal(self.cfg.my_colour)
        if pins_home > self._last_pins_home:
            reward += self.cfg.reward_per_pin_home * (pins_home - self._last_pins_home)
            self._last_pins_home = pins_home

        if self.cfg.anti_loop_enabled:
            sig = self._state_signature()
            if sig in self._recent_state_sigs:
                reward -= float(self.cfg.anti_loop_revisit_penalty)
                loop_revisit = True
            self._recent_state_sigs.append(sig)
            hist = self._pin_recent_positions[pid]
            hist.append(int(to_idx))
            if len(hist) == 3 and hist[0] == hist[2] and hist[0] != hist[1]:
                reward -= float(self.cfg.anti_loop_aba_penalty)
                loop_aba = True

        info: Dict[str, object] = {
            "pins_home": pins_home,
            "distance": new_dist,
            "agent_move_count": self.agent_move_count,
            "loop_revisit": loop_revisit,
            "loop_aba": loop_aba,
        }

        if self._is_colour_winner(self.cfg.my_colour):
            reward += self.cfg.reward_all_home_bonus
            if self.cfg.mode == "multi":
                reward += self.cfg.reward_win_bonus
            info["outcome"] = "win"
            return StepResult(self._build_obs(), reward, True, False, info)

        if self.cfg.mode == "solo_race":
            self._advance_turn()
            if self.step_count >= self.cfg.max_steps:
                info["outcome"] = "truncated"
                return StepResult(self._build_obs(), reward, False, True, info)
            if not self.action_mask().any():
                info["outcome"] = "no_legal_moves"
                return StepResult(self._build_obs(), reward, True, False, info)
            return StepResult(self._build_obs(), reward, False, False, info)

        self._advance_turn()
        reason = self._run_opponents(opponent_policies or {})
        if reason is not None:
            if reason.startswith("opponent_win"):
                reward += self.cfg.reward_lose_penalty
                info["outcome"] = reason
                return StepResult(self._build_obs(), reward, True, False, info)
            if reason == "truncated":
                info["outcome"] = "truncated"
                return StepResult(self._build_obs(), reward, False, True, info)
        if self.step_count >= self.cfg.max_steps:
            info["outcome"] = "truncated"
            return StepResult(self._build_obs(), reward, False, True, info)
        if not self.action_mask().any():
            reward += self.cfg.reward_lose_penalty * 0.5
            info["outcome"] = "draw_no_moves"
            return StepResult(self._build_obs(), reward, True, False, info)
        return StepResult(self._build_obs(), reward, False, False, info)

    def render_ascii(self) -> str:
        pin_map = {}
        for colour, pins in self.pins_by_colour.items():
            letter = colour[:1].upper()
            for p in pins:
                cell = self.board.cells[p.axialindex]
                pin_map[(cell.q, cell.r)] = letter
        out_lines: List[str] = []
        rows = self.board._rows
        max_width = max(len(row) for row in rows)
        for row in rows:
            pad = " " * (max_width - len(row))
            parts = [pin_map.get((q, r), "." if t == "board" else t[:1].lower()) for (q, r, t) in row]
            out_lines.append(pad + " ".join(parts))
        return "\n".join(out_lines)


def _random_policy(env: CheckersEnv, colour: str) -> int:
    mask = env.action_mask(colour)
    if not mask.any():
        return -1
    return int(env.rng.choice(np.flatnonzero(mask).tolist()))
