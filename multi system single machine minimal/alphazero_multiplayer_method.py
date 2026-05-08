"""AlphaZero-style method for 3-6 player Chinese Checkers.

This module is a multiplayer companion to alphazero_method.py.
It keeps the same runtime interface for player.py but removes the
strict 2-player assumptions by using a root-perspective paranoid MCTS.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import multiprocessing
import os
import queue as _stdqueue
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fast_board
from alphazero_method import format_parameter_summary


MAX_CELLS = 121
ACTION_SIZE = MAX_CELLS * MAX_CELLS
INPUT_CHANNELS = 24
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "alphazero_multi.pt"
MODEL_CACHE: Optional["MultiAlphaZeroAgent"] = None
SELF_PLAY_WORKER_AGENT: Optional["MultiAlphaZeroAgent"] = None
COLOUR_ORDER = list(fast_board.COLOUR_LIST)

if fast_board.N_CELLS > MAX_CELLS:
    raise ValueError(f"Board has {fast_board.N_CELLS} cells, but MAX_CELLS={MAX_CELLS}.")


@dataclass(frozen=True)
class Action:
    """Move represented by logical pin id and target board index."""

    pin_id: int
    to_index: int


class MultiSimState:
    """Fast multiplayer simulator backed by precomputed board topology."""

    def __init__(
        self,
        pins_by_colour: Mapping[str, Sequence[int]],
        turn_order: Sequence[str],
        current_turn_colour: str,
        move_count: int = 0,
        move_counts_by_colour: Optional[Mapping[str, int]] = None,
    ):
        if len(turn_order) < 2:
            raise ValueError("AlphaZero multiplayer mode needs at least 2 players.")
        if current_turn_colour not in turn_order:
            raise ValueError("Current turn colour must be part of the turn order.")

        self.turn_order = list(turn_order)
        self.current_turn_index = self.turn_order.index(current_turn_colour)
        self.move_count = int(move_count)
        self.move_counts_by_colour: Dict[str, int] = {
            colour: int((move_counts_by_colour or {}).get(colour, 0))
            for colour in self.turn_order
        }

        self.pins_by_colour: Dict[str, np.ndarray] = {}
        self.occupied = np.zeros(fast_board.N_CELLS, dtype=bool)
        for colour in self.turn_order:
            positions = np.asarray(list(pins_by_colour.get(colour, [])), dtype=np.int32)
            if positions.size == 0:
                raise ValueError(f"No pin positions supplied for colour {colour}.")
            if np.any((positions < 0) | (positions >= fast_board.N_CELLS)):
                raise ValueError(f"Pin position out of bounds for colour {colour}.")
            self.pins_by_colour[colour] = positions.copy()
            self.occupied[positions] = True

    @classmethod
    def _from_arrays(
        cls,
        pins_by_colour: Mapping[str, np.ndarray],
        occupied: np.ndarray,
        turn_order: Sequence[str],
        current_turn_index: int,
        move_count: int,
        move_counts_by_colour: Mapping[str, int],
    ) -> "MultiSimState":
        state = cls.__new__(cls)
        state.turn_order = list(turn_order)
        state.current_turn_index = int(current_turn_index)
        state.move_count = int(move_count)
        state.move_counts_by_colour = dict(move_counts_by_colour)
        state.pins_by_colour = {
            colour: positions.copy()
            for colour, positions in pins_by_colour.items()
        }
        state.occupied = occupied.copy()
        return state

    def clone(self) -> "MultiSimState":
        return self._from_arrays(
            pins_by_colour=self.pins_by_colour,
            occupied=self.occupied,
            turn_order=self.turn_order,
            current_turn_index=self.current_turn_index,
            move_count=self.move_count,
            move_counts_by_colour=self.move_counts_by_colour,
        )

    def current_turn_colour(self) -> str:
        return self.turn_order[self.current_turn_index]

    def pin_position(self, colour: str, pin_id: int) -> int:
        return int(self.pins_by_colour[colour][int(pin_id)])

    def legal_actions(self, colour: Optional[str] = None) -> List[Action]:
        chosen_colour = colour or self.current_turn_colour()
        actions: List[Action] = []
        for pin_id, start_idx in enumerate(self.pins_by_colour[chosen_colour]):
            for target in fast_board.get_possible_moves(int(start_idx), self.occupied):
                actions.append(Action(pin_id=pin_id, to_index=int(target)))
        return actions

    def apply_action(self, action: Action) -> None:
        colour = self.current_turn_colour()
        pin_id = int(action.pin_id)
        to_index = int(action.to_index)
        old_idx = int(self.pins_by_colour[colour][pin_id])

        self.occupied[old_idx] = False
        self.pins_by_colour[colour][pin_id] = to_index
        self.occupied[to_index] = True

        self.move_count += 1
        self.move_counts_by_colour[colour] = self.move_counts_by_colour.get(colour, 0) + 1
        self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)

    def next_state(self, action: Action) -> "MultiSimState":
        nxt = self.clone()
        nxt.apply_action(action)
        return nxt

    def winner(self) -> Optional[str]:
        for colour in self.turn_order:
            if fast_board.all_in_goal(self.pins_by_colour[colour], colour):
                return colour
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or not self.legal_actions()

    def distance_to_goal(self, colour: str) -> float:
        return float(fast_board.total_distance_to_goal(self.pins_by_colour[colour], colour))

    def pins_in_goal(self, colour: str) -> int:
        return int(fast_board.IS_IN_GOAL[colour][self.pins_by_colour[colour]].sum())

    def score_proxy(self, colour: str) -> float:
        """Approximate game.py final score without unavailable wall-clock time."""
        moves = int(self.move_counts_by_colour.get(colour, 0))
        pins_in_goal = self.pins_in_goal(colour)
        pin_goal_score = pins_in_goal * 100.0

        total_dist = self.distance_to_goal(colour)
        distance_score = max(0.0, 200.0 - total_dist) if moves > 0 else 0.0
        if moves > 0:
            sigma = 4.0 if moves < 45 else 18.0
            move_score = math.exp(-((moves - 45) ** 2) / (2.0 * sigma * sigma))
        else:
            move_score = 0.0
        return pin_goal_score + distance_score + move_score

    def position_score_proxy(self, colour: str) -> float:
        """Score proxy for board progress only, excluding move-count shaping."""
        pins_in_goal = self.pins_in_goal(colour)
        pin_goal_score = pins_in_goal * 100.0
        distance_score = max(0.0, 200.0 - self.distance_to_goal(colour))
        return pin_goal_score + distance_score

    def heuristic_value(self, root_colour: str) -> float:
        win = self.winner()
        if win == root_colour:
            return 1.0
        if win is not None:
            return -1.0

        root_score = self.score_proxy(root_colour)
        opp_scores = [
            self.score_proxy(colour)
            for colour in self.turn_order
            if colour != root_colour
        ]
        strongest_opp = max(opp_scores) if opp_scores else 0.0
        raw = (root_score - strongest_opp) / 400.0
        return float(max(-0.99, min(0.99, math.tanh(raw))))

    def action_quality(self, action: Action, colour: Optional[str] = None) -> float:
        """Positive prior shaping; heavily discourages leaving a goal cell."""
        mover = colour or self.current_turn_colour()
        old_idx = self.pin_position(mover, action.pin_id)
        old_in_goal = bool(fast_board.IS_IN_GOAL[mover][old_idx])
        new_in_goal = bool(fast_board.IS_IN_GOAL[mover][int(action.to_index)])

        before_score = self.score_proxy(mover)
        before_dist = self.distance_to_goal(mover)
        nxt = self.next_state(action)
        after_score = nxt.score_proxy(mover)
        after_dist = nxt.distance_to_goal(mover)

        score_delta = after_score - before_score
        dist_gain = before_dist - after_dist
        quality = 1e-3 + max(0.0, score_delta) / 100.0 + max(0.0, dist_gain) * 0.15
        if new_in_goal and not old_in_goal:
            quality += 1.0
        if old_in_goal and not new_in_goal and score_delta < 50.0:
            quality *= 0.01
        return float(max(1e-5, quality))

    def encode(self, root_colour: str) -> torch.Tensor:
        """Encode as [24, MAX_CELLS] tensor for the policy-value net."""
        cur = self.current_turn_colour()
        n = fast_board.N_CELLS
        x = torch.zeros((INPUT_CHANNELS, MAX_CELLS), dtype=torch.float32)

        root_idx = fast_board.COLOUR_TO_IDX[root_colour]
        cur_idx = fast_board.COLOUR_TO_IDX[cur]

        for colour, idx in fast_board.COLOUR_TO_IDX.items():
            positions = self.pins_by_colour.get(colour)
            if positions is not None:
                x[idx, torch.as_tensor(positions, dtype=torch.long)] = 1.0

        x[6 + root_idx, :n] = 1.0
        x[12 + cur_idx, :n] = 1.0
        for colour, idx in fast_board.COLOUR_TO_IDX.items():
            x[18 + idx, :n] = fast_board.zone_mask(colour)

        return x


class MultiPolicyValueNet(nn.Module):
    """Compact MLP policy-value network for multiplayer play."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        input_dim = INPUT_CHANNELS * MAX_CELLS
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, ACTION_SIZE)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        policy_logits = self.policy_head(z)
        value = torch.tanh(self.value_head(z)).squeeze(-1)
        return policy_logits, value


class MultiNode:
    def __init__(
        self,
        state: MultiSimState,
        prior: float = 1.0,
        legal_override: Optional[Mapping[int, Sequence[int]]] = None,
    ):
        self.state = state
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Action, "MultiNode"] = {}
        self.legal_override = legal_override
        self._legal_actions: Optional[List[Action]] = None
        self._winner: Optional[str] = None
        self._winner_checked = False
        self._is_terminal: Optional[bool] = None

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def legal_actions(self) -> List[Action]:
        if self._legal_actions is not None:
            return self._legal_actions
        if self.legal_override is None:
            self._legal_actions = self.state.legal_actions()
        else:
            actions: List[Action] = []
            cur_colour = self.state.current_turn_colour()
            pin_count = len(self.state.pins_by_colour[cur_colour])
            for raw_pin_id, targets in self.legal_override.items():
                pin_id = int(raw_pin_id)
                if pin_id < 0 or pin_id >= pin_count:
                    continue
                for target in targets:
                    target_idx = int(target)
                    if 0 <= target_idx < fast_board.N_CELLS:
                        actions.append(Action(pin_id=pin_id, to_index=target_idx))
            self._legal_actions = actions
        return self._legal_actions

    def winner(self) -> Optional[str]:
        if not self._winner_checked:
            self._winner = self.state.winner()
            self._winner_checked = True
        return self._winner

    def is_terminal(self) -> bool:
        if self._is_terminal is None:
            self._is_terminal = self.winner() is not None or not self.legal_actions()
        return self._is_terminal


@dataclass
class TrainingExample:
    state_tensor: torch.Tensor
    policy: torch.Tensor
    value: float


@dataclass
class EpisodeStats:
    player_count: int
    plies: int
    max_plies: int
    winner: Optional[str]
    terminal_reason: str
    cutoff_hit: bool
    adjudicated: bool
    samples: int
    score_proxy_by_colour: Dict[str, float]
    position_score_by_colour: Dict[str, float]
    pins_in_goal_by_colour: Dict[str, int]
    distance_to_goal_by_colour: Dict[str, float]
    move_counts_by_colour: Dict[str, int]
    episode_number: int = 0
    elapsed_sec: float = 0.0


@dataclass
class SelfPlayResult:
    samples: List[TrainingExample]
    stats: EpisodeStats


@dataclass
class EvaluationGameStats:
    baseline: str
    player_count: int
    candidate_colour: str
    plies: int
    max_plies: int
    winner: Optional[str]
    terminal_reason: str
    cutoff_hit: bool
    adjudicated: bool
    candidate_score: float
    best_opponent_score: float
    score_margin: float
    candidate_advantage: bool
    candidate_win: bool
    score_proxy_by_colour: Dict[str, float]
    position_score_by_colour: Dict[str, float]


class MultiAlphaZeroAgent:
    def __init__(
        self,
        model: Optional[MultiPolicyValueNet] = None,
        device: Optional[str] = None,
        remote_evaluator: Optional["RemoteEvaluator"] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device) if model is not None else None
        if self.model is not None:
            self.model.eval()
        self.remote_evaluator = remote_evaluator

    @staticmethod
    def _root_dirichlet(alpha: float, size: int) -> List[float]:
        if size <= 0:
            return []
        return list(torch.distributions.Dirichlet(torch.full((size,), alpha)).sample().tolist())

    def _evaluate_policy_and_value(
        self,
        state: MultiSimState,
        root_colour: str,
        legal_actions: Sequence[Action],
    ) -> Tuple[Dict[Action, float], float]:
        if not legal_actions:
            return {}, state.heuristic_value(root_colour)

        if self.remote_evaluator is not None:
            cur_colour = state.current_turn_colour()
            action_ids = [
                state.pin_position(cur_colour, action.pin_id) * MAX_CELLS + int(action.to_index)
                for action in legal_actions
            ]
            x_flat = state.encode(root_colour).flatten()
            legal_logits, net_value = self.remote_evaluator.evaluate(x_flat, action_ids)
            policy_probs = F.softmax(legal_logits, dim=0)
            shaped = []
            for action, prob in zip(legal_actions, policy_probs.tolist()):
                shaped.append(float(prob) * (0.25 + state.action_quality(action, cur_colour)))
            total = sum(shaped)
            if not math.isfinite(total) or total <= 0:
                probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
            else:
                probs = {a: p / total for a, p in zip(legal_actions, shaped)}
            heuristic = state.heuristic_value(root_colour)
            return probs, float(max(-0.99, min(0.99, 0.85 * net_value + 0.15 * heuristic)))

        if self.model is None:
            mover = state.current_turn_colour()
            raw = {action: state.action_quality(action, mover) for action in legal_actions}
            total = sum(raw.values())
            priors = (
                {a: p / total for a, p in raw.items()}
                if total > 0
                else {a: 1.0 / len(legal_actions) for a in legal_actions}
            )
            return priors, state.heuristic_value(root_colour)

        with torch.no_grad():
            x = state.encode(root_colour).flatten().unsqueeze(0).to(self.device)
            policy_logits, value = self.model(x)
            logits = policy_logits[0]
            cur_colour = state.current_turn_colour()
            action_ids = [
                state.pin_position(cur_colour, action.pin_id) * MAX_CELLS + int(action.to_index)
                for action in legal_actions
            ]
            action_id_tensor = torch.tensor(action_ids, dtype=torch.long, device=self.device)
            legal_logits = logits.index_select(0, action_id_tensor)
            policy_probs = F.softmax(legal_logits, dim=0).detach().cpu()

            shaped = []
            for action, prob in zip(legal_actions, policy_probs.tolist()):
                shaped.append(float(prob) * (0.25 + state.action_quality(action, cur_colour)))
            total = sum(shaped)
            if not math.isfinite(total) or total <= 0:
                probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
            else:
                probs = {a: p / total for a, p in zip(legal_actions, shaped)}
            net_value = float(value.item())
            heuristic = state.heuristic_value(root_colour)
            return probs, float(max(-0.99, min(0.99, 0.85 * net_value + 0.15 * heuristic)))

    def _select_child(self, node: MultiNode, root_colour: str, c_puct: float) -> Tuple[Action, MultiNode]:
        best_score = -float("inf")
        best_action = None
        best_child = None
        parent_sqrt = math.sqrt(max(1, node.visit_count))
        minimizing_turn = node.state.current_turn_colour() != root_colour

        for action, child in node.children.items():
            q = child.value()
            if minimizing_turn:
                q = -q
            u = c_puct * child.prior * parent_sqrt / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_action is None or best_child is None:
            raise RuntimeError("MCTS selection failed: node has no children.")
        return best_action, best_child

    def run_mcts(
        self,
        root_state: MultiSimState,
        root_colour: str,
        num_simulations: int = 96,
        c_puct: float = 1.5,
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_frac: float = 0.25,
        root_legal_override: Optional[Mapping[int, Sequence[int]]] = None,
    ) -> Dict[Action, int]:
        root = MultiNode(root_state.clone(), prior=1.0, legal_override=root_legal_override)

        root_legal = root.legal_actions()
        root_priors, root_value = self._evaluate_policy_and_value(root.state, root_colour, root_legal)
        for action in root_legal:
            child_state = root.state.next_state(action)
            root.children[action] = MultiNode(child_state, prior=root_priors[action])

        if root_dirichlet_frac > 0 and root_legal:
            noise = self._root_dirichlet(root_dirichlet_alpha, len(root_legal))
            for i, action in enumerate(root_legal):
                child = root.children[action]
                child.prior = (1 - root_dirichlet_frac) * child.prior + root_dirichlet_frac * noise[i]

        root.visit_count += 1
        root.value_sum += root_value

        for _ in range(num_simulations):
            node = root
            path = [node]

            while node.expanded() and not node.is_terminal():
                _, node = self._select_child(node, root_colour, c_puct)
                path.append(node)

            if node.is_terminal():
                winner = node.winner()
                if winner is None:
                    value = node.state.heuristic_value(root_colour)
                else:
                    value = 1.0 if winner == root_colour else -1.0
            else:
                legal = node.legal_actions()
                priors, value = self._evaluate_policy_and_value(node.state, root_colour, legal)
                for action in legal:
                    child_state = node.state.next_state(action)
                    node.children[action] = MultiNode(child_state, prior=priors[action])

            for p in reversed(path):
                p.visit_count += 1
                p.value_sum += value

        return {action: child.visit_count for action, child in root.children.items()}


def _sample_action_from_visits(visits: Mapping[Action, int], temperature: float) -> Action:
    if not visits:
        raise ValueError("No actions from MCTS.")

    actions = list(visits.keys())
    counts = torch.tensor([max(1, int(visits[a])) for a in actions], dtype=torch.float32)

    if temperature <= 1e-6:
        return actions[int(torch.argmax(counts).item())]

    probs = torch.pow(counts, 1.0 / temperature)
    probs = probs / probs.sum()
    idx = int(torch.multinomial(probs, 1).item())
    return actions[idx]


def _extract_turn_order(state: Mapping[str, Any]) -> List[str]:
    players = state.get("players", [])
    if len(players) < 2:
        raise ValueError("Multiplayer AlphaZero requires at least 2 players.")
    order = state.get("turn_order") or [p["colour"] for p in players]
    if len(order) != len(players):
        return list(order)
    return list(order)


def _load_agent() -> MultiAlphaZeroAgent:
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE

    model_path = Path(os.getenv("AZ_MP_MODEL_PATH", os.getenv("AZ_MODEL_PATH", str(DEFAULT_MODEL_PATH))))
    if not model_path.exists():
        MODEL_CACHE = MultiAlphaZeroAgent(model=None)
        return MODEL_CACHE

    model = MultiPolicyValueNet()
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise ValueError("Unsupported checkpoint format for multiplayer AlphaZero model.")

    MODEL_CACHE = MultiAlphaZeroAgent(model=model)
    return MODEL_CACHE


def build_parameter_summary(entries: Sequence[Tuple[str, Any]]) -> str:
    return format_parameter_summary("AlphaZero Multiplayer Parameters", entries)


def print_parameter_summary(entries: Sequence[Tuple[str, Any]]) -> None:
    print("\n" + build_parameter_summary(entries), flush=True)


class TeeWriter:
    """Write terminal output to multiple streams."""

    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def choose_move_alphazero_multiplayer(
    legal_moves: Mapping[str, Sequence[int]],
    state: Mapping[str, Any],
    player_context: Dict[str, Any],
) -> Tuple[int, int, float]:
    """Choose a move with AlphaZero-style MCTS for multiplayer games."""
    my_colour = str(player_context.get("colour", ""))
    if not my_colour:
        raise ValueError("player_context must include colour for AlphaZero multiplayer mode.")

    current_turn = str(state.get("current_turn_colour", ""))
    if current_turn != my_colour:
        raise ValueError("AlphaZero multiplayer called when it is not this player's turn.")

    pins = state.get("pins", {})
    if my_colour not in pins:
        raise ValueError("Current player colour not present in state pins.")

    turn_order = _extract_turn_order(state)
    move_counts_by_colour = {}
    for player in state.get("players", []):
        colour = str(player.get("colour", ""))
        score = player.get("score") or {}
        if colour:
            move_counts_by_colour[colour] = int(score.get("moves", 0) or 0)
    root_state = MultiSimState(
        pins_by_colour={colour: list(map(int, pins.get(colour, []))) for colour in turn_order},
        turn_order=turn_order,
        current_turn_colour=current_turn,
        move_count=int(state.get("move_count", 0) or 0),
        move_counts_by_colour=move_counts_by_colour,
    )

    legal_override = {int(k): [int(x) for x in v] for k, v in legal_moves.items() if v}
    if not legal_override:
        raise ValueError("No legal moves available for AlphaZero multiplayer mode.")

    agent = _load_agent()
    sims = int(os.getenv("AZ_MP_MCTS_SIMS", os.getenv("AZ_MCTS_SIMS", "96")))
    c_puct = float(os.getenv("AZ_MP_C_PUCT", os.getenv("AZ_C_PUCT", "1.4")))
    temp_opening = float(os.getenv("AZ_MP_TEMP_OPENING", os.getenv("AZ_TEMP_OPENING", "0.9")))
    temp_late = float(os.getenv("AZ_MP_TEMP_LATE", os.getenv("AZ_TEMP_LATE", "0.10")))
    cutoff = int(os.getenv("AZ_MP_TEMP_CUTOFF_MOVE", os.getenv("AZ_TEMP_CUTOFF_MOVE", "22")))
    move_count = int(state.get("move_count", 0))
    temperature = temp_opening if move_count < cutoff else temp_late

    visits = agent.run_mcts(
        root_state=root_state,
        root_colour=my_colour,
        num_simulations=sims,
        c_puct=c_puct,
        root_legal_override=legal_override,
    )

    chosen = _sample_action_from_visits(visits, temperature)
    delay = 0.0
    return chosen.pin_id, chosen.to_index, delay


def create_initial_multiplayer_state(colours: Sequence[str]) -> MultiSimState:
    positions = {
        colour: fast_board.INITIAL_POSITIONS[colour].copy()
        for colour in colours
    }
    return MultiSimState(
        pins_by_colour=positions,
        turn_order=list(colours),
        current_turn_colour=colours[0],
    )


def _resolve_episode_player_count(
    player_count_min: int,
    player_count_max: int,
    fixed_player_count: Optional[int] = None,
) -> int:
    if fixed_player_count is not None:
        if fixed_player_count < 2 or fixed_player_count > 6:
            raise ValueError("fixed_player_count must be between 2 and 6.")
        return fixed_player_count

    if player_count_min < 2 or player_count_min > 6:
        raise ValueError("player_count_min must be between 2 and 6.")
    if player_count_max < 2 or player_count_max > 6:
        raise ValueError("player_count_max must be between 2 and 6.")
    if player_count_min > player_count_max:
        raise ValueError("player_count_min must be <= player_count_max.")

    return random.randint(player_count_min, player_count_max)


def _build_episode_player_counts(
    episodes: int,
    player_count_min: int,
    player_count_max: int,
    fixed_player_count: Optional[int] = None,
) -> List[int]:
    if episodes <= 0:
        return []
    if fixed_player_count is not None:
        _resolve_episode_player_count(player_count_min, player_count_max, fixed_player_count)
        return [int(fixed_player_count)] * episodes

    _resolve_episode_player_count(player_count_min, player_count_max, None)
    choices = list(range(player_count_min, player_count_max + 1))
    counts = [choices[i % len(choices)] for i in range(episodes)]
    random.shuffle(counts)
    return counts


def build_policy_target(state: MultiSimState, action_visits: Mapping[Action, int]) -> torch.Tensor:
    target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    colour = state.current_turn_colour()
    total = sum(max(0, int(v)) for v in action_visits.values())
    if total <= 0:
        return target
    for action, count in action_visits.items():
        from_idx = state.pin_position(colour, action.pin_id)
        action_id = int(from_idx) * MAX_CELLS + int(action.to_index)
        target[action_id] = float(count) / float(total)
    return target


def _position_score_by_colour(state: MultiSimState) -> Dict[str, float]:
    return {
        colour: float(state.position_score_proxy(colour))
        for colour in state.turn_order
    }


def _enemy_pins_in_goal_zone(state: MultiSimState, colour: str) -> int:
    """Count pins of OTHER colours sitting inside `colour`'s goal zone."""
    goal_mask = fast_board.IS_IN_GOAL[colour]
    total = 0
    for other in state.turn_order:
        if other == colour:
            continue
        positions = state.pins_by_colour[other]
        total += int(goal_mask[positions].sum())
    return total


def _blocker_winner(state: MultiSimState) -> Optional[str]:
    """Return a colour that is provably blocked from winning by enemy pins
    parked in its goal zone, or None.

    The blocked-from-winning condition: the colour already has 7+ of its own
    pins in its goal, and the remaining 1-3 empty slots are all occupied by
    enemy pins. Such a player cannot reach the 10-pins-in-goal win.
    """
    for colour in state.turn_order:
        own_in_goal = state.pins_in_goal(colour)
        if own_in_goal < 7:
            continue
        enemy_in_goal = _enemy_pins_in_goal_zone(state, colour)
        if enemy_in_goal == 0:
            continue
        if own_in_goal + enemy_in_goal == 10:
            return colour
    return None


def _should_blocker_adjudicate(
    state: MultiSimState,
    ply: int,
    blocker_since_ply: Optional[int],
    stale_plies: int,
    min_plies: int,
) -> bool:
    if blocker_since_ply is None:
        return False
    if ply < max(0, int(min_plies)):
        return False
    if state.winner() is not None:
        return False
    if stale_plies <= 0:
        return True
    return ply - int(blocker_since_ply) >= int(stale_plies)


def _terminal_reason(
    state: MultiSimState,
    plies: int,
    max_moves: int,
    adjudicated: bool = False,
) -> str:
    if state.winner() is not None:
        return "winner"
    if adjudicated:
        return "adjudicated_blocker"
    if plies >= max_moves:
        return "max_moves"
    if not state.legal_actions():
        return "no_legal_moves"
    return "unknown"


def _max_plies_for_player_count(player_count: int, max_moves_per_player: int) -> int:
    return max(1, int(player_count) * max(1, int(max_moves_per_player)))


def _build_episode_stats(
    state: MultiSimState,
    player_count: int,
    plies: int,
    max_plies: int,
    sample_count: int,
    adjudicated: bool = False,
    adjudicated_winner: Optional[str] = None,
) -> EpisodeStats:
    winner = state.winner() or adjudicated_winner
    terminal_reason = _terminal_reason(state, plies, max_plies, adjudicated=adjudicated)
    return EpisodeStats(
        player_count=int(player_count),
        plies=int(plies),
        max_plies=int(max_plies),
        winner=winner,
        terminal_reason=terminal_reason,
        cutoff_hit=terminal_reason == "max_moves",
        adjudicated=terminal_reason == "adjudicated_blocker",
        samples=int(sample_count),
        score_proxy_by_colour={
            colour: float(state.score_proxy(colour))
            for colour in state.turn_order
        },
        position_score_by_colour=_position_score_by_colour(state),
        pins_in_goal_by_colour={
            colour: int(state.pins_in_goal(colour))
            for colour in state.turn_order
        },
        distance_to_goal_by_colour={
            colour: float(state.distance_to_goal(colour))
            for colour in state.turn_order
        },
        move_counts_by_colour={
            colour: int(state.move_counts_by_colour.get(colour, 0))
            for colour in state.turn_order
        },
    )


def generate_self_play_game(
    agent: MultiAlphaZeroAgent,
    player_count: int = 6,
    num_simulations: int = 96,
    c_puct: float = 1.5,
    max_moves: int = 1000,
    temperature_cutoff: int = 20,
    temp_opening: float = 1.0,
    temp_late: float = 0.15,
    adjudicate_stale_moves: int = 160,
    adjudicate_min_moves: int = 80,
    heuristic_opponent: bool = False,
    heuristic_agent: Optional[MultiAlphaZeroAgent] = None,
) -> SelfPlayResult:
    if player_count < 2 or player_count > 6:
        raise ValueError("player_count must be between 2 and 6.")

    colours = random.sample(COLOUR_ORDER, k=player_count)
    state = create_initial_multiplayer_state(colours)
    raw_samples: List[Tuple[MultiSimState, torch.Tensor]] = []
    max_plies = _max_plies_for_player_count(player_count, max_moves)
    adjudicate_stale_plies = (
        int(player_count) * int(adjudicate_stale_moves)
        if int(adjudicate_stale_moves) > 0
        else 0
    )
    adjudicate_min_plies = int(player_count) * max(0, int(adjudicate_min_moves))
    blocker_since_ply: Optional[int] = None
    blocker_colour: Optional[str] = None
    adjudicated = False
    adjudicated_winner: Optional[str] = None
    total_plies = 0

    # Heuristic-mixed self-play: pick one random colour to be controlled by
    # the heuristic agent. This exposes the candidate to a real pin-racing
    # opponent so the policy can't collapse into the degenerate self-play
    # equilibria that destroyed v5 ("move one pin then forget the rest").
    heuristic_colour: Optional[str] = None
    if heuristic_opponent:
        heuristic_colour = random.choice(colours)
        if heuristic_agent is None:
            heuristic_agent = MultiAlphaZeroAgent(model=None)

    for ply in range(max_plies):
        if state.is_terminal():
            break

        to_play = state.current_turn_colour()
        legal = state.legal_actions(to_play)
        if not legal:
            break

        is_heuristic_turn = heuristic_colour is not None and to_play == heuristic_colour
        active_agent = heuristic_agent if is_heuristic_turn else agent

        visits = active_agent.run_mcts(
            root_state=state,
            root_colour=to_play,
            num_simulations=num_simulations,
            c_puct=c_puct,
        )

        # Only record candidate-to-play states as training samples. We don't
        # want to train the policy head to imitate heuristic move choices —
        # the heuristic is here as an opponent, not as a teacher.
        if not is_heuristic_turn:
            policy_target = build_policy_target(state, visits)
            raw_samples.append((state.clone(), policy_target))

        temperature = temp_opening if ply < temperature_cutoff else temp_late
        action = _sample_action_from_visits(visits, temperature)
        state.apply_action(action)
        current_ply = ply + 1
        total_plies = current_ply
        if state.winner() is not None:
            break
        candidate_blocker = _blocker_winner(state)
        if candidate_blocker is None:
            blocker_since_ply = None
            blocker_colour = None
        else:
            if blocker_colour != candidate_blocker:
                blocker_colour = candidate_blocker
                blocker_since_ply = current_ply
            if _should_blocker_adjudicate(
                state=state,
                ply=current_ply,
                blocker_since_ply=blocker_since_ply,
                stale_plies=adjudicate_stale_plies,
                min_plies=adjudicate_min_plies,
            ):
                adjudicated = True
                adjudicated_winner = candidate_blocker
                break

    winner = state.winner() or adjudicated_winner
    samples: List[TrainingExample] = []
    # Only train on games with a clean ±1 outcome: real winners or
    # blocker-adjudicated winners. max_moves games would otherwise contribute
    # mushy `tanh(margin/400)` value targets that cause the policy to converge
    # on stalling behaviour (the failure mode that destroyed the overnight run
    # and v3). Discarding them is standard AlphaZero practice.
    if winner is not None:
        final_values = {
            colour: 1.0 if winner == colour else -1.0
            for colour in state.turn_order
        }
        for sample_state, policy_target in raw_samples:
            for root_colour in sample_state.turn_order:
                samples.append(
                    TrainingExample(
                        state_tensor=sample_state.encode(root_colour).flatten(),
                        policy=policy_target,
                        value=final_values[root_colour],
                    )
                )
    stats = _build_episode_stats(
        state=state,
        player_count=player_count,
        plies=total_plies,
        max_plies=max_plies,
        sample_count=len(samples),
        adjudicated=adjudicated,
        adjudicated_winner=adjudicated_winner,
    )
    return SelfPlayResult(samples=samples, stats=stats)


def _cpu_state_dict(model: MultiPolicyValueNet) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


# ----- Centralised GPU inference server ------------------------------------
#
# The legacy path runs each MCTS forward on CPU at batch size 1. With many
# workers and a small MLP, the GPU sits idle while CPUs saturate. To use the
# whole machine we route every NN evaluation through a single server process
# that owns the model on CUDA, batches outstanding requests from all workers,
# and ships logits + value back via per-worker response queues. The main
# process can hot-swap weights into the server between training updates.

class RemoteEvaluator:
    """Worker-side client that sends one NN request at a time to the server.

    Each request is `(req_id, x_flat, action_ids, worker_rank)`. The server
    looks up the per-worker response queue by rank, so queue objects never
    have to travel inside another queue (Python 3.12 forbids that).
    """

    def __init__(self, request_queue, response_queue, rank: int):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.rank = int(rank)
        self._req_id = 0

    def evaluate(
        self, x_flat: torch.Tensor, action_ids: Sequence[int]
    ) -> Tuple[torch.Tensor, float]:
        self._req_id += 1
        req_id = self._req_id
        ids = torch.as_tensor(action_ids, dtype=torch.long)
        self.request_queue.put((req_id, x_flat.detach().contiguous(), ids, self.rank))
        rid, legal_logits, value = self.response_queue.get()
        if rid != req_id:
            raise RuntimeError(f"RemoteEvaluator response id mismatch: got {rid} expected {req_id}")
        return legal_logits, value


def _inference_server_loop(
    request_queue,
    response_queues: Sequence[Any],
    control_queue,
    initial_state_dict: Mapping[str, torch.Tensor],
    device: str,
    max_batch: int,
    max_wait_ms: float,
) -> None:
    """Run on its own process. Owns the GPU model, batches inferences."""
    torch.set_num_threads(1)
    dev = torch.device(device)
    model = MultiPolicyValueNet()
    model.load_state_dict(initial_state_dict)
    model.to(dev)
    model.eval()
    use_amp = dev.type == "cuda"

    def _drain_control() -> bool:
        while True:
            try:
                ctrl = control_queue.get_nowait()
            except _stdqueue.Empty:
                return False
            if ctrl is None:
                return True
            if isinstance(ctrl, dict):
                model.load_state_dict({k: v.to(dev) for k, v in ctrl.items()})

    while True:
        if _drain_control():
            return
        try:
            first = request_queue.get(timeout=0.05)
        except _stdqueue.Empty:
            continue
        if first is None:
            return

        batch = [first]
        if max_wait_ms > 0:
            deadline = time.monotonic() + max_wait_ms / 1000.0
        else:
            deadline = 0.0
        while len(batch) < max_batch:
            if max_wait_ms > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = request_queue.get(timeout=remaining)
                except _stdqueue.Empty:
                    break
            else:
                try:
                    req = request_queue.get_nowait()
                except _stdqueue.Empty:
                    break
            if req is None:
                batch.append(None)
                break
            batch.append(req)

        shutdown_after = batch and batch[-1] is None
        if shutdown_after:
            batch = batch[:-1]
        if not batch:
            if shutdown_after:
                return
            continue

        x = torch.stack([b[1] for b in batch], dim=0).to(dev, non_blocking=True)
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    policy_logits, values = model(x)
                policy_logits = policy_logits.float()
                values = values.float()
            else:
                policy_logits, values = model(x)
        out_legal: List[torch.Tensor] = []
        for i, (_rid, _x, ids, _rank) in enumerate(batch):
            ids_dev = ids.to(dev, non_blocking=True)
            out_legal.append(policy_logits[i].index_select(0, ids_dev))
        if dev.type == "cuda":
            out_legal_cpu = [t.detach().cpu() for t in out_legal]
            values_cpu = values.detach().cpu().tolist()
        else:
            out_legal_cpu = [t.detach() for t in out_legal]
            values_cpu = values.detach().tolist()

        for (rid, _x, _ids, rank), legal_cpu, val in zip(batch, out_legal_cpu, values_cpu):
            response_queues[rank].put((rid, legal_cpu, float(val)))

        if shutdown_after:
            return


def _start_inference_server(
    state_dict: Mapping[str, torch.Tensor],
    device: str,
    max_batch: int,
    max_wait_ms: float,
    mp_context,
    num_response_queues: int,
) -> Tuple[Any, Any, List[Any], Any, Any]:
    request_queue = mp_context.Queue()
    response_queues = [mp_context.Queue() for _ in range(int(num_response_queues))]
    control_queue = mp_context.Queue()
    slot_counter = mp_context.Value("i", 0)
    proc = mp_context.Process(
        target=_inference_server_loop,
        args=(request_queue, response_queues, control_queue, state_dict, device, max_batch, max_wait_ms),
        daemon=False,
    )
    proc.start()
    return proc, request_queue, response_queues, control_queue, slot_counter


def _stop_inference_server(proc, request_queue, control_queue) -> None:
    try:
        control_queue.put(None)
    except Exception:
        pass
    try:
        request_queue.put(None)
    except Exception:
        pass
    proc.join(timeout=10.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)


def _push_weights_to_server(control_queue, model: MultiPolicyValueNet) -> None:
    control_queue.put(_cpu_state_dict(model))


def _init_self_play_worker(
    model_state_dict: Mapping[str, torch.Tensor],
    device: str,
    torch_threads: int,
) -> None:
    global SELF_PLAY_WORKER_AGENT
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    model = MultiPolicyValueNet()
    model.load_state_dict(model_state_dict)
    model.eval()
    SELF_PLAY_WORKER_AGENT = MultiAlphaZeroAgent(model=model, device=device)


def _init_self_play_worker_remote(
    request_queue,
    response_queues: Sequence[Any],
    slot_counter,
    torch_threads: int,
) -> None:
    global SELF_PLAY_WORKER_AGENT
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    with slot_counter.get_lock():
        rank = slot_counter.value
        slot_counter.value += 1
    if rank >= len(response_queues):
        raise RuntimeError(
            f"Remote worker rank {rank} exceeds response_queues={len(response_queues)}"
        )
    evaluator = RemoteEvaluator(request_queue, response_queues[rank], rank)
    SELF_PLAY_WORKER_AGENT = MultiAlphaZeroAgent(remote_evaluator=evaluator)


def _generate_self_play_worker_task(args: Tuple[int, int, Dict[str, Any]]) -> Tuple[int, SelfPlayResult, float]:
    episode_number, player_count, config = args
    seed = int(config["seed_base"]) + int(episode_number)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if SELF_PLAY_WORKER_AGENT is None:
        raise RuntimeError("Self-play worker was not initialized.")

    started = time.time()
    heuristic_fraction = float(config.get("heuristic_fraction", 0.0))
    use_heuristic = heuristic_fraction > 0.0 and random.random() < heuristic_fraction
    result = generate_self_play_game(
        agent=SELF_PLAY_WORKER_AGENT,
        player_count=player_count,
        num_simulations=int(config["num_simulations"]),
        c_puct=float(config["c_puct"]),
        max_moves=int(config["max_moves"]),
        temperature_cutoff=int(config["temperature_cutoff"]),
        temp_opening=float(config["temp_opening"]),
        temp_late=float(config["temp_late"]),
        adjudicate_stale_moves=int(config["adjudicate_stale_moves"]),
        adjudicate_min_moves=int(config["adjudicate_min_moves"]),
        heuristic_opponent=use_heuristic,
    )
    return episode_number, result, time.time() - started


def train_step(
    model: MultiPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[TrainingExample],
    device: str,
) -> Dict[str, float]:
    model.train()
    x = torch.stack([b.state_tensor for b in batch], dim=0).to(device)
    y_policy = torch.stack([b.policy for b in batch], dim=0).to(device)
    y_value = torch.tensor([b.value for b in batch], dtype=torch.float32, device=device)

    logits, values = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(y_policy * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, y_value)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }


def _load_model_weights_if_available(
    model: MultiPolicyValueNet,
    model_path: Path,
    verbose: bool = False,
) -> bool:
    if not model_path.exists():
        return False

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise ValueError(f"Unsupported checkpoint format: {model_path}")

    if verbose:
        print(f"[model] Loaded existing weights from {model_path}", flush=True)
    return True


def _trim_replay(replay: List[TrainingExample], replay_cap: int) -> None:
    if replay_cap > 0 and len(replay) > replay_cap:
        del replay[: len(replay) - replay_cap]


def _train_replay_epochs(
    model: MultiPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: Sequence[TrainingExample],
    batch_size: int,
    device: str,
    epochs: int,
    train_sample_size: int,
    verbose: bool,
    label: str,
    epoch_offset: int = 0,
    total_epochs_label: Optional[str] = None,
) -> Tuple[Dict[str, float], List[float]]:
    last_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    epoch_times: List[float] = []
    if epochs <= 0 or not replay:
        return last_metrics, epoch_times

    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        if train_sample_size > 0 and len(replay) > train_sample_size:
            epoch_data = random.sample(replay, train_sample_size)
        else:
            epoch_data = list(replay)
        random.shuffle(epoch_data)

        epoch_loss = 0.0
        epoch_policy = 0.0
        epoch_value = 0.0
        batches = 0

        for i in range(0, len(epoch_data), batch_size):
            batch = epoch_data[i : i + batch_size]
            if not batch:
                continue
            last_metrics = train_step(model, optimizer, batch, device)
            epoch_loss += last_metrics["loss"]
            epoch_policy += last_metrics["policy_loss"]
            epoch_value += last_metrics["value_loss"]
            batches += 1

        epoch_elapsed = time.time() - epoch_started
        epoch_times.append(epoch_elapsed)

        if batches > 0:
            last_metrics = {
                "loss": epoch_loss / batches,
                "policy_loss": epoch_policy / batches,
                "value_loss": epoch_value / batches,
            }

        if verbose and batches > 0:
            if epoch_offset == 0 and total_epochs_label is None:
                epoch_text = f"{epoch}/{epochs}"
            else:
                display_epoch = epoch_offset + epoch
                display_total = total_epochs_label or str(epoch_offset + epochs)
                epoch_text = f"{display_epoch}/{display_total}"
            print(
                f"[train:{label}] epoch {epoch_text} "
                f"samples={len(epoch_data)} loss={last_metrics['loss']:.4f} "
                f"policy={last_metrics['policy_loss']:.4f} "
                f"value={last_metrics['value_loss']:.4f} time={epoch_elapsed:.2f}s",
                flush=True,
            )

    model.eval()
    return last_metrics, epoch_times


def _generate_self_play_chunk(
    model: MultiPolicyValueNet,
    episode_numbers: Sequence[int],
    episode_player_counts: Sequence[int],
    num_simulations: int,
    c_puct: float,
    max_moves: int,
    temperature_cutoff: int,
    temp_opening: float,
    temp_late: float,
    adjudicate_stale_moves: int,
    adjudicate_min_moves: int,
    self_play_workers: int,
    self_play_device: str,
    worker_torch_threads: int,
    chunk_heartbeat_sec: float,
    verbose: bool,
    heuristic_fraction: float = 0.0,
    inference_request_queue: Optional[Any] = None,
    inference_response_queues: Optional[Sequence[Any]] = None,
    inference_slot_counter: Optional[Any] = None,
) -> List[Tuple[int, SelfPlayResult, float]]:
    config = {
        "num_simulations": int(num_simulations),
        "c_puct": float(c_puct),
        "max_moves": int(max_moves),
        "temperature_cutoff": int(temperature_cutoff),
        "temp_opening": float(temp_opening),
        "temp_late": float(temp_late),
        "adjudicate_stale_moves": int(adjudicate_stale_moves),
        "adjudicate_min_moves": int(adjudicate_min_moves),
        "heuristic_fraction": float(heuristic_fraction),
        "seed_base": random.randint(1, 2_000_000_000),
    }

    tasks = [
        (int(ep), int(player_count), config)
        for ep, player_count in zip(episode_numbers, episode_player_counts)
    ]

    if self_play_workers <= 1 or len(tasks) <= 1:
        local_model = MultiPolicyValueNet()
        local_model.load_state_dict(_cpu_state_dict(model))
        agent = MultiAlphaZeroAgent(model=local_model, device=self_play_device)
        results = []
        for episode_number, player_count, task_config in tasks:
            started = time.time()
            use_heuristic = (
                float(task_config.get("heuristic_fraction", 0.0)) > 0.0
                and random.random() < float(task_config.get("heuristic_fraction", 0.0))
            )
            result = generate_self_play_game(
                agent=agent,
                player_count=player_count,
                num_simulations=int(task_config["num_simulations"]),
                c_puct=float(task_config["c_puct"]),
                max_moves=int(task_config["max_moves"]),
                temperature_cutoff=int(task_config["temperature_cutoff"]),
                temp_opening=float(task_config["temp_opening"]),
                temp_late=float(task_config["temp_late"]),
                adjudicate_stale_moves=int(task_config["adjudicate_stale_moves"]),
                adjudicate_min_moves=int(task_config["adjudicate_min_moves"]),
                heuristic_opponent=use_heuristic,
            )
            results.append((episode_number, result, time.time() - started))
        return results

    max_workers = min(int(self_play_workers), len(tasks))
    using_remote = inference_request_queue is not None
    if verbose:
        device_label = "gpu-server" if using_remote else self_play_device
        print(
            f"[self-play] parallel chunk episodes={episode_numbers[0]}-{episode_numbers[-1]} "
            f"workers={max_workers} device={device_label}",
            flush=True,
        )

    mp_context = multiprocessing.get_context("spawn")
    if using_remote:
        if inference_response_queues is None or inference_slot_counter is None:
            raise RuntimeError("Remote inference requires response_queues and slot_counter")
        with inference_slot_counter.get_lock():
            inference_slot_counter.value = 0
        initializer = _init_self_play_worker_remote
        initargs = (
            inference_request_queue,
            inference_response_queues,
            inference_slot_counter,
            worker_torch_threads,
        )
    else:
        state_dict = _cpu_state_dict(model)
        initializer = _init_self_play_worker
        initargs = (state_dict, self_play_device, worker_torch_threads)
    results: List[Tuple[int, SelfPlayResult, float]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        future_to_episode = {
            executor.submit(_generate_self_play_worker_task, task): task[0]
            for task in tasks
        }
        results.extend(
            _collect_self_play_results(
                future_to_episode=future_to_episode,
                label=f"chunk{episode_numbers[0]}-{episode_numbers[-1]}",
                heartbeat_sec=chunk_heartbeat_sec,
                verbose=verbose,
            )
        )

    results.sort(key=lambda item: item[0])
    return results


def _generate_parallel_self_play_with_background_training(
    model: MultiPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: Sequence[TrainingExample],
    batch_size: int,
    train_device: str,
    train_sample_size: int,
    background_train_epochs: int,
    background_train_until_self_play_done: bool,
    background_train_max_epochs: int,
    episode_numbers: Sequence[int],
    episode_player_counts: Sequence[int],
    num_simulations: int,
    c_puct: float,
    max_moves: int,
    temperature_cutoff: int,
    temp_opening: float,
    temp_late: float,
    adjudicate_stale_moves: int,
    adjudicate_min_moves: int,
    self_play_workers: int,
    self_play_device: str,
    worker_torch_threads: int,
    chunk_heartbeat_sec: float,
    verbose: bool,
    heuristic_fraction: float = 0.0,
    inference_request_queue: Optional[Any] = None,
    inference_response_queues: Optional[Sequence[Any]] = None,
    inference_slot_counter: Optional[Any] = None,
    inference_control_queue: Optional[Any] = None,
) -> Tuple[List[Tuple[int, SelfPlayResult, float]], Dict[str, float], List[float]]:
    config = {
        "num_simulations": int(num_simulations),
        "c_puct": float(c_puct),
        "max_moves": int(max_moves),
        "temperature_cutoff": int(temperature_cutoff),
        "temp_opening": float(temp_opening),
        "temp_late": float(temp_late),
        "adjudicate_stale_moves": int(adjudicate_stale_moves),
        "adjudicate_min_moves": int(adjudicate_min_moves),
        "heuristic_fraction": float(heuristic_fraction),
        "seed_base": random.randint(1, 2_000_000_000),
    }
    tasks = [
        (int(ep), int(player_count), config)
        for ep, player_count in zip(episode_numbers, episode_player_counts)
    ]
    max_workers = min(int(self_play_workers), len(tasks))
    using_remote = inference_request_queue is not None
    if verbose:
        device_label = "gpu-server" if using_remote else self_play_device
        print(
            f"[self-play] parallel chunk episodes={episode_numbers[0]}-{episode_numbers[-1]} "
            f"workers={max_workers} device={device_label}",
            flush=True,
        )

    mp_context = multiprocessing.get_context("spawn")
    results: List[Tuple[int, SelfPlayResult, float]] = []
    bg_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    bg_times: List[float] = []

    if using_remote:
        if inference_response_queues is None or inference_slot_counter is None:
            raise RuntimeError("Remote inference requires response_queues and slot_counter")
        with inference_slot_counter.get_lock():
            inference_slot_counter.value = 0
        initializer = _init_self_play_worker_remote
        initargs = (
            inference_request_queue,
            inference_response_queues,
            inference_slot_counter,
            worker_torch_threads,
        )
    else:
        state_dict = _cpu_state_dict(model)
        initializer = _init_self_play_worker
        initargs = (state_dict, self_play_device, worker_torch_threads)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        future_to_episode = {
            executor.submit(_generate_self_play_worker_task, task): task[0]
            for task in tasks
        }

        bg_epochs_run = 0
        bg_label = f"bg{episode_numbers[0]}-{episode_numbers[-1]}"
        while replay:
            futures_done = all(future.done() for future in future_to_episode)
            needs_min_epochs = bg_epochs_run < background_train_epochs
            needs_adaptive_epochs = background_train_until_self_play_done and not futures_done
            if not needs_min_epochs and not needs_adaptive_epochs:
                break
            if background_train_max_epochs > 0 and bg_epochs_run >= background_train_max_epochs:
                if verbose and background_train_until_self_play_done and not futures_done:
                    print(
                        f"[train:{bg_label}] stopped at adaptive cap "
                        f"{background_train_max_epochs} while self-play is still running",
                        flush=True,
                    )
                break

            total_label = (
                "auto"
                if background_train_until_self_play_done
                else str(max(1, background_train_epochs))
            )
            next_metrics, next_times = _train_replay_epochs(
                model=model,
                optimizer=optimizer,
                replay=replay,
                batch_size=batch_size,
                device=train_device,
                epochs=1,
                train_sample_size=train_sample_size,
                verbose=verbose,
                label=bg_label,
                epoch_offset=bg_epochs_run,
                total_epochs_label=total_label,
            )
            if not next_times:
                break
            bg_epochs_run += len(next_times)
            bg_metrics = next_metrics
            bg_times.extend(next_times)

        results.extend(
            _collect_self_play_results(
                future_to_episode=future_to_episode,
                label=bg_label,
                heartbeat_sec=chunk_heartbeat_sec,
                verbose=verbose,
            )
        )

    results.sort(key=lambda item: item[0])
    return results, bg_metrics, bg_times


def _collect_self_play_results(
    future_to_episode: Mapping[Any, int],
    label: str,
    heartbeat_sec: float,
    verbose: bool,
) -> List[Tuple[int, SelfPlayResult, float]]:
    pending = set(future_to_episode.keys())
    total = len(pending)
    results: List[Tuple[int, SelfPlayResult, float]] = []
    started = time.time()
    heartbeat = max(1.0, float(heartbeat_sec)) if heartbeat_sec > 0 else 0.0

    while pending:
        timeout = heartbeat if heartbeat > 0 else None
        done, pending = concurrent.futures.wait(
            pending,
            timeout=timeout,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            if verbose and heartbeat > 0:
                print(
                    f"[self-play:{label}] heartbeat "
                    f"completed={total - len(pending)}/{total} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
            continue

        for future in done:
            episode_number = future_to_episode[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"Self-play worker failed for episode {episode_number}: {exc}") from exc

            results.append(result)
            if verbose:
                completed_ep, episode_result, episode_seconds = result
                stats = episode_result.stats
                print(
                    f"[self-play:{label}] finished episode={completed_ep} "
                    f"players={stats.player_count} plies={stats.plies}/{stats.max_plies} "
                    f"terminal_reason={stats.terminal_reason} "
                    f"adjudicated={getattr(stats, 'adjudicated', False)} "
                    f"completed={len(results)}/{total} time={episode_seconds:.2f}s",
                    flush=True,
                )

    return results


def _ordered_colours(mapping: Mapping[str, Any]) -> List[str]:
    known = [colour for colour in COLOUR_ORDER if colour in mapping]
    unknown = sorted(colour for colour in mapping if colour not in COLOUR_ORDER)
    return known + unknown


def _format_metric_map(mapping: Mapping[str, Any], precision: int = 1) -> str:
    parts = []
    for colour in _ordered_colours(mapping):
        key = colour.replace(" ", "_")
        value = mapping[colour]
        if isinstance(value, float):
            parts.append(f"{key}={value:.{precision}f}")
        else:
            parts.append(f"{key}={value}")
    return "{" + ", ".join(parts) + "}"


def _score_margin(scores: Mapping[str, float]) -> float:
    values = sorted((float(value) for value in scores.values()), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return values[0] - values[1]


def _avg(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summarize_episode_stats(stats: Sequence[EpisodeStats]) -> Dict[str, Any]:
    total = len(stats)
    if total == 0:
        return {
            "episodes": 0,
            "cutoff_rate": 0.0,
            "adjudication_rate": 0.0,
            "natural_finish_rate": 0.0,
            "avg_plies": 0.0,
            "avg_score_margin": 0.0,
            "winner_counts": {},
            "by_player_count": {},
        }

    winner_counts = Counter(s.winner for s in stats if s.winner is not None)
    grouped: Dict[int, List[EpisodeStats]] = defaultdict(list)
    for item in stats:
        grouped[int(item.player_count)].append(item)

    def group_summary(items: Sequence[EpisodeStats]) -> Dict[str, Any]:
        return {
            "episodes": len(items),
            "cutoff_rate": sum(1 for s in items if s.cutoff_hit) / len(items),
            "adjudication_rate": sum(1 for s in items if getattr(s, "adjudicated", False)) / len(items),
            "natural_finish_rate": sum(
                1
                for s in items
                if not s.cutoff_hit and not getattr(s, "adjudicated", False)
            ) / len(items),
            "avg_plies": _avg([float(s.plies) for s in items]),
            "avg_score_margin": _avg([_score_margin(s.score_proxy_by_colour) for s in items]),
            "winner_counts": dict(Counter(s.winner for s in items if s.winner is not None)),
        }

    return {
        "episodes": total,
        "cutoff_rate": sum(1 for s in stats if s.cutoff_hit) / total,
        "adjudication_rate": sum(1 for s in stats if getattr(s, "adjudicated", False)) / total,
        "natural_finish_rate": sum(
            1
            for s in stats
            if not s.cutoff_hit and not getattr(s, "adjudicated", False)
        ) / total,
        "avg_plies": _avg([float(s.plies) for s in stats]),
        "avg_score_margin": _avg([_score_margin(s.score_proxy_by_colour) for s in stats]),
        "winner_counts": dict(winner_counts),
        "by_player_count": {
            player_count: group_summary(items)
            for player_count, items in sorted(grouped.items())
        },
    }


def _print_self_play_summary(summary: Mapping[str, Any]) -> None:
    if int(summary.get("episodes", 0) or 0) <= 0:
        print("Self-play summary : n/a")
        return
    print(
        "Self-play summary : "
        f"episodes={int(summary['episodes'])} "
        f"cutoff_rate={float(summary['cutoff_rate']):.1%} "
        f"adjudication_rate={float(summary.get('adjudication_rate', 0.0)):.1%} "
        f"natural_finish_rate={float(summary['natural_finish_rate']):.1%} "
        f"avg_plies={float(summary['avg_plies']):.1f} "
        f"avg_score_margin={float(summary['avg_score_margin']):.1f} "
        f"winners={summary.get('winner_counts', {})}",
        flush=True,
    )
    for player_count, item in summary.get("by_player_count", {}).items():
        print(
            f"  {player_count}p: episodes={int(item['episodes'])} "
            f"cutoff_rate={float(item['cutoff_rate']):.1%} "
            f"adjudication_rate={float(item.get('adjudication_rate', 0.0)):.1%} "
            f"natural_finish_rate={float(item['natural_finish_rate']):.1%} "
            f"avg_plies={float(item['avg_plies']):.1f} "
            f"avg_score_margin={float(item['avg_score_margin']):.1f}",
            flush=True,
        )


def _log_episode_stats(
    completed_ep: int,
    episodes: int,
    stats: EpisodeStats,
    total_samples: int,
) -> None:
    winner = stats.winner or "none"
    print(
        f"[self-play] episode {completed_ep}/{episodes} "
        f"players={stats.player_count} plies={stats.plies}/{stats.max_plies} "
        f"terminal_reason={stats.terminal_reason} winner={winner} "
        f"cutoff_hit={stats.cutoff_hit} adjudicated={getattr(stats, 'adjudicated', False)} "
        f"samples={stats.samples} "
        f"total_samples={total_samples} score_margin={_score_margin(stats.score_proxy_by_colour):.2f} "
        f"time={stats.elapsed_sec:.2f}s",
        flush=True,
    )
    print(
        f"[self-play:stats] episode {completed_ep}/{episodes} "
        f"score_proxy={_format_metric_map(stats.score_proxy_by_colour)} "
        f"pins_in_goal={_format_metric_map(stats.pins_in_goal_by_colour, precision=0)} "
        f"distance_to_goal={_format_metric_map(stats.distance_to_goal_by_colour)} "
        f"moves={_format_metric_map(stats.move_counts_by_colour, precision=0)}",
        flush=True,
    )


def _load_policy_value_model(model_path: Path) -> MultiPolicyValueNet:
    model = MultiPolicyValueNet()
    _load_model_weights_if_available(model, model_path, verbose=False)
    model.eval()
    return model


def _checkpoint_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _existing_unique_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        key = _checkpoint_key(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _arena_snapshot_prefix(save_path: Path) -> str:
    return Path(save_path).stem


def _save_arena_snapshot(
    model: MultiPolicyValueNet,
    pool_dir: Path,
    save_path: Path,
    completed_episode: int,
    train_epoch_count: int,
    verbose: bool,
) -> Path:
    pool_dir.mkdir(parents=True, exist_ok=True)
    prefix = _arena_snapshot_prefix(save_path)
    snapshot_path = pool_dir / (
        f"{prefix}_ep{int(completed_episode):04d}_te{int(train_epoch_count):05d}.pt"
    )
    torch.save(
        {
            "model_state_dict": _cpu_state_dict(model),
            "completed_episodes": int(completed_episode),
            "train_epoch_count": int(train_epoch_count),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        snapshot_path,
    )
    if verbose:
        print(
            f"[arena] saved snapshot epoch={int(train_epoch_count)} "
            f"episode={int(completed_episode)} path={snapshot_path}",
            flush=True,
        )
    return snapshot_path


def _prune_arena_snapshots(
    pool_dir: Path,
    save_path: Path,
    pool_size: int,
    keep_paths: Sequence[Path] = (),
    verbose: bool = False,
) -> None:
    if int(pool_size) <= 0 or not pool_dir.exists():
        return
    prefix = _arena_snapshot_prefix(save_path)
    keep = {_checkpoint_key(path) for path in keep_paths}
    snapshots = sorted(
        pool_dir.glob(f"{prefix}_ep*_te*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in snapshots[int(pool_size):]:
        if _checkpoint_key(path) in keep:
            continue
        try:
            path.unlink()
            if verbose:
                print(f"[arena] pruned old snapshot {path}", flush=True)
        except OSError as exc:
            if verbose:
                print(f"[arena] warning: could not prune snapshot {path}: {exc}", flush=True)


def _resolve_checkpoint_baselines(
    explicit_paths: Sequence[Path],
    pool_dir: Optional[Path],
    pool_limit: int,
    exclude_paths: Sequence[Path] = (),
) -> List[Path]:
    excluded = {_checkpoint_key(path) for path in exclude_paths}
    chosen: List[Path] = []
    seen = set()

    def add(path: Path) -> bool:
        key = _checkpoint_key(path)
        if key in seen or key in excluded or not path.exists():
            return False
        seen.add(key)
        chosen.append(path)
        return True

    for path in explicit_paths:
        add(Path(path))

    if pool_dir is not None and int(pool_limit) != 0 and pool_dir.exists():
        pool_paths = sorted(
            pool_dir.glob("*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        added_from_pool = 0
        for path in pool_paths:
            if add(path):
                added_from_pool += 1
            if int(pool_limit) > 0 and added_from_pool >= int(pool_limit):
                break

    return chosen


def _evaluation_colours(player_count: int) -> List[str]:
    if player_count < 2 or player_count > 6:
        raise ValueError("Evaluation player counts must be between 2 and 6.")
    paired_join_order = ["red", "blue", "lawn green", "gray0", "yellow", "purple"]
    selected = set(paired_join_order[:player_count])
    return [colour for colour in COLOUR_ORDER if colour in selected]


def _run_evaluation_game(
    candidate_agent: MultiAlphaZeroAgent,
    baseline_agent: MultiAlphaZeroAgent,
    baseline: str,
    player_count: int,
    candidate_colour: str,
    num_simulations: int,
    c_puct: float,
    max_moves: int,
    adjudicate_stale_moves: int,
    adjudicate_min_moves: int,
) -> EvaluationGameStats:
    colours = _evaluation_colours(player_count)
    state = create_initial_multiplayer_state(colours)
    max_plies = _max_plies_for_player_count(player_count, max_moves)
    adjudicate_stale_plies = (
        int(player_count) * int(adjudicate_stale_moves)
        if int(adjudicate_stale_moves) > 0
        else 0
    )
    adjudicate_min_plies = int(player_count) * max(0, int(adjudicate_min_moves))
    blocker_since_ply: Optional[int] = None
    blocker_colour: Optional[str] = None
    adjudicated = False
    adjudicated_winner: Optional[str] = None

    for ply in range(max_plies):
        if state.is_terminal():
            break
        to_play = state.current_turn_colour()
        legal = state.legal_actions(to_play)
        if not legal:
            break
        agent = candidate_agent if to_play == candidate_colour else baseline_agent
        visits = agent.run_mcts(
            root_state=state,
            root_colour=to_play,
            num_simulations=num_simulations,
            c_puct=c_puct,
            root_dirichlet_frac=0.0,
        )
        action = _sample_action_from_visits(visits, 0.0)
        state.apply_action(action)
        current_ply = ply + 1
        if state.winner() is not None:
            break
        candidate_blocker = _blocker_winner(state)
        if candidate_blocker is None:
            blocker_since_ply = None
            blocker_colour = None
        else:
            if blocker_colour != candidate_blocker:
                blocker_colour = candidate_blocker
                blocker_since_ply = current_ply
            if _should_blocker_adjudicate(
                state=state,
                ply=current_ply,
                blocker_since_ply=blocker_since_ply,
                stale_plies=adjudicate_stale_plies,
                min_plies=adjudicate_min_plies,
            ):
                adjudicated = True
                adjudicated_winner = candidate_blocker
                break

    scores = {
        colour: float(state.score_proxy(colour))
        for colour in state.turn_order
    }
    position_scores = _position_score_by_colour(state)
    candidate_score = float(scores[candidate_colour])
    opponent_scores = [
        score for colour, score in scores.items()
        if colour != candidate_colour
    ]
    best_opponent_score = max(opponent_scores) if opponent_scores else 0.0
    winner = state.winner() or adjudicated_winner
    terminal_reason = _terminal_reason(state, state.move_count, max_plies, adjudicated=adjudicated)
    return EvaluationGameStats(
        baseline=baseline,
        player_count=int(player_count),
        candidate_colour=candidate_colour,
        plies=int(state.move_count),
        max_plies=int(max_plies),
        winner=winner,
        terminal_reason=terminal_reason,
        cutoff_hit=terminal_reason == "max_moves",
        adjudicated=terminal_reason == "adjudicated_blocker",
        candidate_score=candidate_score,
        best_opponent_score=float(best_opponent_score),
        score_margin=float(candidate_score - best_opponent_score),
        candidate_advantage=candidate_score > best_opponent_score,
        candidate_win=winner == candidate_colour,
        score_proxy_by_colour=scores,
        position_score_by_colour=position_scores,
    )


def _summarize_evaluation_stats(stats: Sequence[EvaluationGameStats]) -> Dict[str, Any]:
    grouped: Dict[str, List[EvaluationGameStats]] = defaultdict(list)
    for item in stats:
        grouped[item.baseline].append(item)

    def summarize_group(items: Sequence[EvaluationGameStats]) -> Dict[str, Any]:
        by_player_count: Dict[int, List[EvaluationGameStats]] = defaultdict(list)
        for item in items:
            by_player_count[item.player_count].append(item)

        def one(items_for_count: Sequence[EvaluationGameStats]) -> Dict[str, Any]:
            n = len(items_for_count)
            return {
                "games": n,
                "win_rate": sum(1 for s in items_for_count if s.candidate_win) / n,
                "advantage_rate": sum(1 for s in items_for_count if s.candidate_advantage) / n,
                "finish_rate": sum(
                    1
                    for s in items_for_count
                    if not s.cutoff_hit and not getattr(s, "adjudicated", False)
                ) / n,
                "cutoff_rate": sum(1 for s in items_for_count if s.cutoff_hit) / n,
                "adjudication_rate": sum(
                    1 for s in items_for_count if getattr(s, "adjudicated", False)
                ) / n,
                "avg_plies": _avg([float(s.plies) for s in items_for_count]),
                "avg_score_margin": _avg([s.score_margin for s in items_for_count]),
            }

        summary = one(items)
        summary["by_player_count"] = {
            player_count: one(items_for_count)
            for player_count, items_for_count in sorted(by_player_count.items())
        }
        return summary

    return {
        baseline: summarize_group(items)
        for baseline, items in sorted(grouped.items())
    }


def _print_evaluation_summary(summary: Mapping[str, Any]) -> None:
    if not summary:
        print("Evaluation summary: n/a")
        return
    print("Evaluation summary:", flush=True)
    for baseline, item in summary.items():
        print(
            f"  {baseline}: games={int(item['games'])} "
            f"win_rate={float(item['win_rate']):.1%} "
            f"advantage_rate={float(item['advantage_rate']):.1%} "
            f"finish_rate={float(item['finish_rate']):.1%} "
            f"cutoff_rate={float(item['cutoff_rate']):.1%} "
            f"adjudication_rate={float(item.get('adjudication_rate', 0.0)):.1%} "
            f"avg_plies={float(item['avg_plies']):.1f} "
            f"avg_score_margin={float(item['avg_score_margin']):.1f}",
            flush=True,
        )
        for player_count, by_count in item.get("by_player_count", {}).items():
            print(
                f"    {player_count}p: games={int(by_count['games'])} "
                f"win_rate={float(by_count['win_rate']):.1%} "
                f"advantage_rate={float(by_count['advantage_rate']):.1%} "
                f"finish_rate={float(by_count['finish_rate']):.1%} "
                f"cutoff_rate={float(by_count['cutoff_rate']):.1%} "
                f"adjudication_rate={float(by_count.get('adjudication_rate', 0.0)):.1%} "
                f"avg_plies={float(by_count['avg_plies']):.1f} "
                f"avg_score_margin={float(by_count['avg_score_margin']):.1f}",
                flush=True,
            )


def run_evaluation(
    candidate_model: MultiPolicyValueNet,
    baselines: Sequence[str],
    eval_episodes: int,
    player_counts: Sequence[int],
    max_moves: int,
    num_simulations: int,
    c_puct: float,
    probe_path: Path,
    device: Optional[str],
    adjudicate_stale_moves: int,
    adjudicate_min_moves: int,
    verbose: bool,
    checkpoint_paths: Sequence[Path] = (),
) -> Dict[str, Any]:
    if eval_episodes <= 0 or not baselines:
        return {}

    eval_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    candidate_agent = MultiAlphaZeroAgent(model=candidate_model, device=eval_device)
    game_stats: List[EvaluationGameStats] = []

    def run_baseline(baseline_label: str, baseline_agent: MultiAlphaZeroAgent) -> None:
        for player_count in player_counts:
            colours = _evaluation_colours(int(player_count))
            for game_idx in range(int(eval_episodes)):
                candidate_colour = colours[game_idx % len(colours)]
                started = time.time()
                stats = _run_evaluation_game(
                    candidate_agent=candidate_agent,
                    baseline_agent=baseline_agent,
                    baseline=baseline_label,
                    player_count=int(player_count),
                    candidate_colour=candidate_colour,
                    num_simulations=int(num_simulations),
                    c_puct=float(c_puct),
                    max_moves=int(max_moves),
                    adjudicate_stale_moves=int(adjudicate_stale_moves),
                    adjudicate_min_moves=int(adjudicate_min_moves),
                )
                elapsed = time.time() - started
                game_stats.append(stats)
                if verbose:
                    print(
                        f"[eval:{baseline_label}] players={player_count} "
                        f"game={game_idx + 1}/{eval_episodes} "
                        f"candidate_colour={candidate_colour.replace(' ', '_')} "
                        f"plies={stats.plies}/{stats.max_plies} terminal_reason={stats.terminal_reason} "
                        f"winner={stats.winner or 'none'} cutoff_hit={stats.cutoff_hit} "
                        f"adjudicated={stats.adjudicated} "
                        f"candidate_win={stats.candidate_win} "
                        f"candidate_advantage={stats.candidate_advantage} "
                        f"score_margin={stats.score_margin:.2f} time={elapsed:.2f}s "
                        f"score_proxy={_format_metric_map(stats.score_proxy_by_colour)}",
                        flush=True,
                    )

    for baseline in baselines:
        baseline_key = baseline.strip().lower()
        if baseline_key == "heuristic":
            run_baseline("heuristic", MultiAlphaZeroAgent(model=None, device=eval_device))
        elif baseline_key == "probe":
            if not probe_path.exists():
                if verbose:
                    print(f"[eval:probe] skipped missing probe path: {probe_path}", flush=True)
                continue
            probe_model = _load_policy_value_model(probe_path)
            run_baseline("probe", MultiAlphaZeroAgent(model=probe_model, device=eval_device))
        elif baseline_key in {"checkpoint", "checkpoints", "pool"}:
            existing_paths = _existing_unique_paths([Path(path) for path in checkpoint_paths])
            if not existing_paths and verbose:
                print("[eval:checkpoints] skipped; no checkpoint baselines found", flush=True)
            for checkpoint_path in existing_paths:
                try:
                    checkpoint_model = _load_policy_value_model(checkpoint_path)
                except Exception as exc:
                    if verbose:
                        print(
                            f"[eval:checkpoints] skipped {checkpoint_path}: {exc}",
                            flush=True,
                        )
                    continue
                label = f"checkpoint:{checkpoint_path.stem}"
                run_baseline(label, MultiAlphaZeroAgent(model=checkpoint_model, device=eval_device))
        else:
            if verbose:
                print(f"[eval] skipped unknown baseline: {baseline}", flush=True)

    return _summarize_evaluation_stats(game_stats)


def run_self_play_training(
    episodes: int = 100,
    train_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_simulations: int = 96,
    c_puct: float = 1.5,
    max_moves: int = 1000,
    player_count_min: int = 2,
    player_count_max: int = 6,
    fixed_player_count: Optional[int] = None,
    temp_opening: float = 1.0,
    temp_late: float = 0.15,
    temperature_cutoff: int = 20,
    adjudicate_stale_moves: int = 160,
    adjudicate_min_moves: int = 80,
    heuristic_fraction: float = 0.0,
    model_path: Optional[Path] = None,
    init_model_path: Optional[Path] = None,
    train_interval: int = 5,
    update_epochs: int = 1,
    replay_cap: int = 50000,
    train_sample_size: int = 4096,
    train_device: Optional[str] = None,
    background_train_epochs: int = 0,
    background_train_until_self_play_done: bool = False,
    background_train_max_epochs: int = 0,
    self_play_workers: int = 1,
    self_play_device: str = "cpu",
    worker_torch_threads: int = 1,
    chunk_heartbeat_sec: float = 60.0,
    inference_server: bool = False,
    inference_server_device: str = "cuda",
    inference_max_batch: int = 64,
    inference_max_wait_ms: float = 4.0,
    eval_episodes: int = 0,
    eval_player_counts: Sequence[int] = (),
    eval_max_moves: Optional[int] = None,
    eval_num_simulations: Optional[int] = None,
    eval_probe_path: Optional[Path] = None,
    eval_baselines: Sequence[str] = (),
    eval_checkpoint_paths: Sequence[Path] = (),
    eval_checkpoint_pool_size: int = 4,
    checkpoint_pool_dir: Optional[Path] = None,
    checkpoint_pool_size: int = 8,
    arena_interval_epochs: int = 0,
    arena_eval_episodes: int = 1,
    fresh: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    started = time.time()
    model = MultiPolicyValueNet()
    save_path = Path(model_path or os.getenv("AZ_MP_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_path.with_suffix(".checkpoint")

    if not fresh and not checkpoint_path.exists():
        _load_model_weights_if_available(model, Path(init_model_path or save_path), verbose=verbose)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    agent = MultiAlphaZeroAgent(model=model, device=train_device)

    replay: List[TrainingExample] = []
    start_episode = 1
    episode_player_counts: List[int] = []
    episode_stats: List[EpisodeStats] = []
    last_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    epoch_times: List[float] = []
    train_epoch_count = 0
    arena_evaluations: List[Dict[str, Any]] = []
    pool_dir = Path(checkpoint_pool_dir) if checkpoint_pool_dir is not None else save_path.parent / "arena"

    # Try to resume from checkpoint
    if checkpoint_path.exists() and not fresh:
        try:
            # Register TrainingExample as safe global for unpickling checkpoint
            torch.serialization.add_safe_globals([TrainingExample, EpisodeStats])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            replay = checkpoint["replay"]
            start_episode = checkpoint["completed_episodes"] + 1
            episode_player_counts = checkpoint.get("episode_player_counts", [])
            episode_stats = checkpoint.get("episode_stats", [])
            train_epoch_count = int(checkpoint.get("train_epoch_count", 0))
            arena_evaluations = list(checkpoint.get("arena_evaluations", []))
            if verbose:
                print(
                    f"[checkpoint] Resumed from episode {start_episode}/{episodes} with {len(replay)} samples",
                    flush=True,
                )
        except Exception as e:
            if verbose:
                print(f"[checkpoint] Could not load checkpoint: {e}. Starting fresh.", flush=True)
            start_episode = 1
    elif fresh and checkpoint_path.exists() and verbose:
        print(f"[checkpoint] Ignoring existing checkpoint because --fresh was set: {checkpoint_path}", flush=True)

    # Build shuffled player count distribution on first run
    if not episode_player_counts:
        episode_player_counts = _build_episode_player_counts(
            episodes=episodes,
            player_count_min=player_count_min,
            player_count_max=player_count_max,
            fixed_player_count=fixed_player_count,
        )
    elif len(episode_player_counts) < episodes:
        episode_player_counts.extend(
            _build_episode_player_counts(
                episodes=episodes - len(episode_player_counts),
                player_count_min=player_count_min,
                player_count_max=player_count_max,
                fixed_player_count=fixed_player_count,
            )
        )

    if verbose and start_episode == 1:
        present_counts = {p: episode_player_counts.count(p) for p in range(2, 7)}
        print(
            "[setup] Created shuffled distribution: "
            + ", ".join(f"{present_counts[p]} {p}p" for p in range(2, 7) if present_counts[p] > 0),
            flush=True,
        )

    inference_proc = None
    inference_request_queue = None
    inference_response_queues = None
    inference_slot_counter = None
    inference_control_queue = None
    if inference_server:
        mp_context = multiprocessing.get_context("spawn")
        (
            inference_proc,
            inference_request_queue,
            inference_response_queues,
            inference_control_queue,
            inference_slot_counter,
        ) = _start_inference_server(
            state_dict=_cpu_state_dict(model),
            device=inference_server_device,
            max_batch=int(inference_max_batch),
            max_wait_ms=float(inference_max_wait_ms),
            mp_context=mp_context,
            num_response_queues=int(self_play_workers),
        )
        if verbose:
            print(
                f"[infer-server] started device={inference_server_device} "
                f"max_batch={inference_max_batch} max_wait_ms={inference_max_wait_ms} "
                f"response_queues={self_play_workers}",
                flush=True,
            )

    arena_interval = max(0, int(arena_interval_epochs))
    next_arena_epoch = (
        ((train_epoch_count // arena_interval) + 1) * arena_interval
        if arena_interval > 0
        else 0
    )

    def maybe_run_arena_eval(completed_episode: int) -> None:
        nonlocal next_arena_epoch
        if arena_interval <= 0 or train_epoch_count < next_arena_epoch:
            return

        snapshot_path = _save_arena_snapshot(
            model=model,
            pool_dir=pool_dir,
            save_path=save_path,
            completed_episode=int(completed_episode),
            train_epoch_count=int(train_epoch_count),
            verbose=verbose,
        )
        _prune_arena_snapshots(
            pool_dir=pool_dir,
            save_path=save_path,
            pool_size=int(checkpoint_pool_size),
            keep_paths=[snapshot_path],
            verbose=verbose,
        )

        checkpoint_baselines = _resolve_checkpoint_baselines(
            explicit_paths=eval_checkpoint_paths,
            pool_dir=pool_dir,
            pool_limit=int(eval_checkpoint_pool_size),
            exclude_paths=[snapshot_path],
        )
        summary: Dict[str, Any] = {}
        if int(arena_eval_episodes) > 0 and eval_baselines and eval_player_counts:
            if verbose:
                print(
                    f"[arena] evaluating epoch={int(train_epoch_count)} "
                    f"episode={int(completed_episode)} "
                    f"checkpoint_baselines={len(checkpoint_baselines)}",
                    flush=True,
                )
            summary = run_evaluation(
                candidate_model=model,
                baselines=eval_baselines,
                eval_episodes=int(arena_eval_episodes),
                player_counts=eval_player_counts,
                max_moves=int(eval_max_moves if eval_max_moves is not None else max_moves),
                num_simulations=int(eval_num_simulations if eval_num_simulations is not None else num_simulations),
                c_puct=c_puct,
                probe_path=Path(eval_probe_path or save_path.parent / "alphazero_multi_probe.pt"),
                device=str(agent.device),
                adjudicate_stale_moves=adjudicate_stale_moves,
                adjudicate_min_moves=adjudicate_min_moves,
                verbose=verbose,
                checkpoint_paths=checkpoint_baselines,
            )
            if verbose:
                _print_evaluation_summary(summary)
        arena_evaluations.append(
            {
                "train_epoch_count": int(train_epoch_count),
                "completed_episode": int(completed_episode),
                "snapshot_path": str(snapshot_path),
                "evaluation_summary": summary,
            }
        )
        next_arena_epoch = ((train_epoch_count // arena_interval) + 1) * arena_interval

    try:
        ep = start_episode
        while ep <= episodes:
            if train_interval > 0:
                chunk_end = min(episodes, ((ep - 1) // train_interval + 1) * train_interval)
            else:
                chunk_end = min(episodes, ep + max(1, int(self_play_workers)) - 1)

            chunk_episodes = list(range(ep, chunk_end + 1))
            chunk_player_counts = [episode_player_counts[i - 1] for i in chunk_episodes]
            bg_times: List[float] = []

            if self_play_workers > 1 and len(chunk_episodes) > 1:
                chunk_results, bg_metrics, bg_times = _generate_parallel_self_play_with_background_training(
                    model=model,
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=batch_size,
                    train_device=str(agent.device),
                    train_sample_size=train_sample_size,
                    background_train_epochs=background_train_epochs,
                    background_train_until_self_play_done=background_train_until_self_play_done,
                    background_train_max_epochs=background_train_max_epochs,
                    episode_numbers=chunk_episodes,
                    episode_player_counts=chunk_player_counts,
                    num_simulations=num_simulations,
                    c_puct=c_puct,
                    max_moves=max_moves,
                    temperature_cutoff=temperature_cutoff,
                    temp_opening=temp_opening,
                    temp_late=temp_late,
                    adjudicate_stale_moves=adjudicate_stale_moves,
                    adjudicate_min_moves=adjudicate_min_moves,
                    self_play_workers=self_play_workers,
                    self_play_device=self_play_device,
                    worker_torch_threads=worker_torch_threads,
                    chunk_heartbeat_sec=chunk_heartbeat_sec,
                    verbose=verbose,
                    heuristic_fraction=heuristic_fraction,
                    inference_request_queue=inference_request_queue,
                    inference_response_queues=inference_response_queues,
                    inference_slot_counter=inference_slot_counter,
                    inference_control_queue=inference_control_queue,
                )
                if bg_times:
                    last_metrics = bg_metrics
                    epoch_times.extend(bg_times)
                    train_epoch_count += len(bg_times)
                    if inference_control_queue is not None:
                        _push_weights_to_server(inference_control_queue, model)
            else:
                chunk_results = []
                for episode_number, episode_player_count in zip(chunk_episodes, chunk_player_counts):
                    if verbose:
                        print(
                            f"[self-play] episode {episode_number}/{episodes} started "
                            f"(players={episode_player_count})",
                            flush=True,
                        )
                    started_ep = time.time()
                    use_heuristic = (
                        heuristic_fraction > 0.0 and random.random() < heuristic_fraction
                    )
                    episode_result = generate_self_play_game(
                        agent=agent,
                        player_count=episode_player_count,
                        num_simulations=num_simulations,
                        c_puct=c_puct,
                        max_moves=max_moves,
                        temperature_cutoff=temperature_cutoff,
                        temp_opening=temp_opening,
                        temp_late=temp_late,
                        adjudicate_stale_moves=adjudicate_stale_moves,
                        adjudicate_min_moves=adjudicate_min_moves,
                        heuristic_opponent=use_heuristic,
                    )
                    chunk_results.append((episode_number, episode_result, time.time() - started_ep))

            for completed_ep, episode_result, episode_seconds in chunk_results:
                episode_result.stats.episode_number = int(completed_ep)
                episode_result.stats.elapsed_sec = float(episode_seconds)
                replay.extend(episode_result.samples)
                _trim_replay(replay, replay_cap)
                episode_stats.append(episode_result.stats)
                if verbose:
                    _log_episode_stats(
                        completed_ep=completed_ep,
                        episodes=episodes,
                        stats=episode_result.stats,
                        total_samples=len(replay),
                    )

                # Save checkpoint after each completed episode.
                try:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "replay": replay,
                            "completed_episodes": completed_ep,
                            "episode_player_counts": episode_player_counts,
                            "episode_stats": episode_stats,
                            "train_epoch_count": train_epoch_count,
                            "arena_evaluations": arena_evaluations,
                        },
                        checkpoint_path,
                    )
                except Exception as e:
                    if verbose:
                        print(f"[checkpoint] Warning: Could not save checkpoint: {e}", flush=True)

            if bg_times:
                maybe_run_arena_eval(completed_episode=chunk_end)

            should_update = train_interval > 0 and chunk_end < episodes and chunk_end % train_interval == 0
            if should_update:
                update_metrics, update_times = _train_replay_epochs(
                    model=model,
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=batch_size,
                    device=str(agent.device),
                    epochs=update_epochs,
                    train_sample_size=train_sample_size,
                    verbose=verbose,
                    label=f"ep{chunk_end}",
                )
                last_metrics = update_metrics
                epoch_times.extend(update_times)
                train_epoch_count += len(update_times)
                if inference_control_queue is not None:
                    _push_weights_to_server(inference_control_queue, model)
                maybe_run_arena_eval(completed_episode=chunk_end)
            ep = chunk_end + 1

        if not replay:
            raise RuntimeError("No self-play samples generated.")

        final_metrics, final_times = _train_replay_epochs(
            model=model,
            optimizer=optimizer,
            replay=replay,
            batch_size=batch_size,
            device=str(agent.device),
            epochs=train_epochs,
            train_sample_size=train_sample_size,
            verbose=verbose,
            label="final",
        )
        if final_times:
            last_metrics = final_metrics
            epoch_times.extend(final_times)
            train_epoch_count += len(final_times)
            maybe_run_arena_eval(completed_episode=episodes)

        save_path = Path(model_path or os.getenv("AZ_MP_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, save_path)

        self_play_summary = _summarize_episode_stats(episode_stats)
        final_checkpoint_baselines = _resolve_checkpoint_baselines(
            explicit_paths=eval_checkpoint_paths,
            pool_dir=pool_dir,
            pool_limit=int(eval_checkpoint_pool_size),
            exclude_paths=[save_path],
        )
        evaluation_summary = run_evaluation(
            candidate_model=model,
            baselines=eval_baselines,
            eval_episodes=int(eval_episodes),
            player_counts=eval_player_counts,
            max_moves=int(eval_max_moves if eval_max_moves is not None else max_moves),
            num_simulations=int(eval_num_simulations if eval_num_simulations is not None else num_simulations),
            c_puct=c_puct,
            probe_path=Path(eval_probe_path or save_path.parent / "alphazero_multi_probe.pt"),
            device=str(agent.device),
            adjudicate_stale_moves=adjudicate_stale_moves,
            adjudicate_min_moves=adjudicate_min_moves,
            verbose=verbose,
            checkpoint_paths=final_checkpoint_baselines,
        )

        # Clean up checkpoint after successful training
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                if verbose:
                    print(f"[checkpoint] Cleaned up checkpoint file", flush=True)
            except Exception as e:
                if verbose:
                    print(f"[checkpoint] Could not delete checkpoint: {e}", flush=True)

        elapsed = time.time() - started
        avg_epoch_time = (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0

        return {
            "samples": float(len(replay)),
            "train_seconds": float(elapsed),
            "avg_epoch_seconds": float(avg_epoch_time),
            "model_path": str(save_path),
            "self_play_summary": self_play_summary,
            "evaluation_summary": evaluation_summary,
            "arena_evaluations": arena_evaluations,
            "train_epoch_count": float(train_epoch_count),
            **last_metrics,
        }
    finally:
        if inference_proc is not None:
            try:
                _stop_inference_server(inference_proc, inference_request_queue, inference_control_queue)
            except Exception as e:
                if verbose:
                    print(f"[infer-server] shutdown warning: {e}", flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a multiplayer AlphaZero-style model via self-play.",
    )
    parser.add_argument("--episodes", type=int, default=125, help="Number of self-play games.")
    parser.add_argument("--train-epochs", type=int, default=8, help="Epochs over replay data.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for optimizer steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--num-simulations", type=int, default=24, help="MCTS simulations per move.")
    parser.add_argument(
        "--max-moves",
        type=int,
        default=1000,
        help="Max moves per player for each self-play game. Total ply cap is players * this value.",
    )
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument("--temp-opening", type=float, default=1.0, help="Temperature for early moves.")
    parser.add_argument("--temp-late", type=float, default=0.15, help="Temperature for late moves.")
    parser.add_argument("--temp-cutoff-move", type=int, default=20, help="Move number to switch temperatures.")
    parser.add_argument(
        "--adjudicate-stale-moves",
        type=int,
        default=160,
        help="When a player has 7+ pins in goal blocked by enemy pins, wait this many moves per player before adjudicating that player as the winner. Use 0 to adjudicate immediately.",
    )
    parser.add_argument(
        "--adjudicate-min-moves",
        type=int,
        default=80,
        help="Minimum moves per player before blocker adjudication can stop a game.",
    )
    parser.add_argument(
        "--self-play-heuristic-fraction",
        type=float,
        default=0.0,
        help=(
            "Probability that a self-play game replaces ONE colour's policy with the heuristic agent. "
            "Mixing in a real pin-racing opponent prevents policy collapse into single-pin shuffling. "
            "Recommended: 0.25 - 0.5. 0.0 disables (pure self-play)."
        ),
    )
    parser.add_argument(
        "--train-interval",
        type=int,
        default=5,
        help="Run update training every N self-play episodes. Use 0 to train only at the end.",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=1,
        help="Training epochs to run at each train interval.",
    )
    parser.add_argument(
        "--replay-cap",
        type=int,
        default=50000,
        help="Maximum replay samples kept in memory/checkpoint. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=4096,
        help="Samples used per training epoch. Use 0 to train on all replay samples.",
    )
    parser.add_argument(
        "--train-device",
        type=str,
        default="",
        help="Torch device for replay training. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--background-train-epochs",
        type=int,
        default=0,
        help="Train on existing replay while parallel self-play workers generate the next chunk.",
    )
    parser.add_argument(
        "--background-train-until-self-play-done",
        action="store_true",
        help="Continue background replay training until the current parallel self-play chunk finishes.",
    )
    parser.add_argument(
        "--background-train-max-epochs",
        type=int,
        default=10,
        help="Optional cap for adaptive background training epochs per chunk. Use 0 for no cap.",
    )
    parser.add_argument(
        "--self-play-workers",
        type=int,
        default=5,
        help="Parallel CPU worker processes for self-play episode generation.",
    )
    parser.add_argument(
        "--self-play-device",
        type=str,
        default="cpu",
        help="Torch device used inside self-play workers. Default: cpu.",
    )
    parser.add_argument(
        "--worker-torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads per self-play worker. Keep 1 when using many workers.",
    )
    parser.add_argument(
        "--chunk-heartbeat-sec",
        type=float,
        default=60.0,
        help="Seconds between progress heartbeat prints while waiting for a self-play chunk.",
    )
    parser.add_argument(
        "--inference-server",
        dest="inference_server",
        action="store_true",
        default=True,
        help="Route every self-play NN evaluation through a single batched GPU server "
             "process. Enabled by default.",
    )
    parser.add_argument(
        "--no-inference-server",
        dest="inference_server",
        action="store_false",
        help="Disable the batched inference server and run NN inference in each worker.",
    )
    parser.add_argument(
        "--inference-server-device",
        type=str,
        default="cuda",
        help="Torch device for the batched inference server. Default: cuda.",
    )
    parser.add_argument(
        "--inference-max-batch",
        type=int,
        default=128,
        help="Maximum batch size assembled per server forward pass.",
    )
    parser.add_argument(
        "--inference-max-wait-ms",
        type=float,
        default=4.0,
        help="Server waits up to this many ms after the first request to fill a batch.",
    )
    parser.add_argument(
        "--player-count-min",
        type=int,
        default=2,
        help="Minimum player count sampled per self-play episode (2-6).",
    )
    parser.add_argument(
        "--player-count-max",
        type=int,
        default=6,
        help="Maximum player count sampled per self-play episode (2-6).",
    )
    parser.add_argument(
        "--fixed-player-count",
        type=int,
        default=None,
        help="Optional fixed player count for all episodes (2-6). Overrides the sampled range.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--init-model-path",
        type=str,
        default="",
        help="Optional existing model to initialize from when --model-path is a new output path.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="training.log",
        help="Append terminal training output to this log file. Default: training.log",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing training output to a log file.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing model/checkpoint and train from random initialization.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Evaluation games per player count and baseline after training. Use 0 to disable.",
    )
    parser.add_argument(
        "--eval-player-counts",
        type=str,
        default="2,3,4",
        help="Comma-separated player counts for final evaluation. Default: 2,3,4.",
    )
    parser.add_argument(
        "--eval-max-moves",
        type=int,
        default=0,
        help="Max moves per player for evaluation. Use 0 to reuse --max-moves.",
    )
    parser.add_argument(
        "--eval-num-simulations",
        type=int,
        default=0,
        help=(
            "MCTS simulations per evaluation move. Use 0 (default) to inherit from --num-simulations "
            "so eval reflects real play strength. Override only if eval needs to be faster/slower than training."
        ),
    )
    parser.add_argument(
        "--eval-probe-path",
        type=str,
        default="checkpoints/alphazero_multi_probe.pt",
        help="Probe model path for probe baseline evaluation.",
    )
    parser.add_argument(
        "--eval-baselines",
        type=str,
        default="probe,heuristic",
        help="Comma-separated final evaluation baselines. Supported: probe,heuristic,checkpoints.",
    )
    parser.add_argument(
        "--eval-checkpoint-paths",
        type=str,
        default="",
        help="Comma-separated checkpoint paths used when eval baselines include checkpoints.",
    )
    parser.add_argument(
        "--eval-checkpoint-pool-size",
        type=int,
        default=4,
        help="Newest checkpoint-pool snapshots to evaluate when baselines include checkpoints. Use 0 for explicit paths only.",
    )
    parser.add_argument(
        "--checkpoint-pool-dir",
        type=str,
        default="checkpoints/arena",
        help="Directory for arena snapshots and checkpoint-pool baselines.",
    )
    parser.add_argument(
        "--checkpoint-pool-size",
        type=int,
        default=8,
        help="Number of current-run arena snapshots to retain. Use 0 to keep all.",
    )
    parser.add_argument(
        "--arena-interval-epochs",
        type=int,
        default=0,
        help="Save an arena snapshot and run periodic eval after this many training epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--arena-eval-episodes",
        type=int,
        default=1,
        help="Periodic arena eval games per player count and baseline.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-episode and per-epoch progress prints.",
    )
    return parser


def _print_training_review(metrics: Mapping[str, Any]) -> None:
    print("\n=== TRAINING REVIEW ===")
    print(f"Samples generated : {int(float(metrics['samples']))}")
    print(f"Final loss        : {float(metrics['loss']):.4f}")
    print(f"Final policy loss : {float(metrics['policy_loss']):.4f}")
    print(f"Final value loss  : {float(metrics['value_loss']):.4f}")
    print(f"Train time (sec)  : {float(metrics['train_seconds']):.2f}")
    print(f"Avg epoch (sec)   : {float(metrics['avg_epoch_seconds']):.2f}")
    print(f"Train epochs      : {int(float(metrics.get('train_epoch_count', 0.0)))}")
    print(f"Model saved to    : {metrics['model_path']}")
    _print_self_play_summary(metrics.get("self_play_summary", {}))
    arena_evals = metrics.get("arena_evaluations", [])
    if arena_evals:
        latest = arena_evals[-1]
        print(
            "Arena snapshots   : "
            f"{len(arena_evals)} latest_epoch={int(latest.get('train_epoch_count', 0))} "
            f"latest_episode={int(latest.get('completed_episode', 0))} "
            f"latest_path={latest.get('snapshot_path', 'n/a')}",
            flush=True,
        )
    _print_evaluation_summary(metrics.get("evaluation_summary", {}))
    print("=======================")


def main() -> int:
    args = _build_arg_parser().parse_args()

    if not args.no_log and args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with log_path.open("a", encoding="utf-8") as log_file:
            tee = TeeWriter(original_stdout, log_file)
            err_tee = TeeWriter(original_stderr, log_file)
            sys.stdout = tee
            sys.stderr = err_tee
            try:
                print(
                    f"\n=== TRAINING RUN START {datetime.now().isoformat(timespec='seconds')} ===",
                    flush=True,
                )
                print("Command: " + " ".join(sys.argv), flush=True)
                return _run_training_from_args(args)
            finally:
                print(
                    f"=== TRAINING RUN END {datetime.now().isoformat(timespec='seconds')} ===",
                    flush=True,
                )
                sys.stdout = original_stdout
                sys.stderr = original_stderr

    return _run_training_from_args(args)


def _parse_int_csv(raw: str, label: str) -> List[int]:
    values: List[int] = []
    if not raw.strip():
        return values
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{label} must be a comma-separated list of integers: {raw}") from exc
    return values


def _parse_str_csv(raw: str) -> List[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_path_csv(raw: str) -> List[Path]:
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def _run_training_from_args(args: argparse.Namespace) -> int:

    eval_player_counts = _parse_int_csv(args.eval_player_counts, "--eval-player-counts")
    eval_baselines = _parse_str_csv(args.eval_baselines)
    eval_checkpoint_paths = _parse_path_csv(args.eval_checkpoint_paths)
    if (args.eval_episodes > 0 or args.arena_interval_epochs > 0) and not eval_player_counts:
        raise ValueError("--eval-player-counts must not be empty when evaluation is enabled.")

    os.environ["AZ_MP_MCTS_SIMS"] = str(args.num_simulations)
    os.environ["AZ_MP_C_PUCT"] = str(args.c_puct)
    os.environ["AZ_MP_TEMP_OPENING"] = str(args.temp_opening)
    os.environ["AZ_MP_TEMP_LATE"] = str(args.temp_late)
    os.environ["AZ_MP_TEMP_CUTOFF_MOVE"] = str(args.temp_cutoff_move)

    print_parameter_summary(
        [
            ("episodes", args.episodes),
            ("train_epochs", args.train_epochs),
            ("batch_size", args.batch_size),
            ("lr", args.lr),
            ("num_simulations", args.num_simulations),
            ("c_puct", args.c_puct),
            ("temp_opening", args.temp_opening),
            ("temp_late", args.temp_late),
            ("temp_cutoff_move", args.temp_cutoff_move),
            ("adjudicate_stale_moves_per_player", args.adjudicate_stale_moves),
            ("adjudicate_min_moves_per_player", args.adjudicate_min_moves),
            ("self_play_heuristic_fraction", args.self_play_heuristic_fraction),
            ("train_interval", args.train_interval),
            ("update_epochs", args.update_epochs),
            ("replay_cap", args.replay_cap),
            ("train_sample_size", args.train_sample_size),
            ("train_device", args.train_device or "auto"),
            ("background_train_epochs", args.background_train_epochs),
            ("background_train_until_self_play_done", args.background_train_until_self_play_done),
            ("background_train_max_epochs", args.background_train_max_epochs),
            ("self_play_workers", args.self_play_workers),
            ("self_play_device", args.self_play_device),
            ("worker_torch_threads", args.worker_torch_threads),
            ("chunk_heartbeat_sec", args.chunk_heartbeat_sec),
            ("inference_server", args.inference_server),
            ("inference_server_device", args.inference_server_device),
            ("inference_max_batch", args.inference_max_batch),
            ("inference_max_wait_ms", args.inference_max_wait_ms),
            ("player_count_min", args.player_count_min),
            ("player_count_max", args.player_count_max),
            ("fixed_player_count", args.fixed_player_count),
            ("max_moves_per_player", args.max_moves),
            ("model_path", args.model_path),
            ("init_model_path", args.init_model_path or "n/a"),
            ("log_path", "disabled" if args.no_log else args.log_path),
            ("eval_episodes", args.eval_episodes),
            ("eval_player_counts", ",".join(map(str, eval_player_counts)) or "n/a"),
            ("eval_max_moves_per_player", args.eval_max_moves or args.max_moves),
            ("eval_num_simulations", args.eval_num_simulations or args.num_simulations),
            ("eval_probe_path", args.eval_probe_path),
            ("eval_baselines", ",".join(eval_baselines) or "n/a"),
            ("eval_checkpoint_paths", ",".join(str(path) for path in eval_checkpoint_paths) or "n/a"),
            ("eval_checkpoint_pool_size", args.eval_checkpoint_pool_size),
            ("checkpoint_pool_dir", args.checkpoint_pool_dir),
            ("checkpoint_pool_size", args.checkpoint_pool_size),
            ("arena_interval_epochs", args.arena_interval_epochs),
            ("arena_eval_episodes", args.arena_eval_episodes),
            ("fresh", args.fresh),
            ("quiet", args.quiet),
        ]
    )

    print("[train] Starting multiplayer AlphaZero self-play training", flush=True)
    metrics = run_self_play_training(
        episodes=args.episodes,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        max_moves=args.max_moves,
        player_count_min=args.player_count_min,
        player_count_max=args.player_count_max,
        fixed_player_count=args.fixed_player_count,
        temp_opening=args.temp_opening,
        temp_late=args.temp_late,
        temperature_cutoff=args.temp_cutoff_move,
        adjudicate_stale_moves=args.adjudicate_stale_moves,
        adjudicate_min_moves=args.adjudicate_min_moves,
        heuristic_fraction=args.self_play_heuristic_fraction,
        model_path=Path(args.model_path),
        init_model_path=Path(args.init_model_path) if args.init_model_path else None,
        train_interval=args.train_interval,
        update_epochs=args.update_epochs,
        replay_cap=args.replay_cap,
        train_sample_size=args.train_sample_size,
        train_device=args.train_device or None,
        background_train_epochs=args.background_train_epochs,
        background_train_until_self_play_done=args.background_train_until_self_play_done,
        background_train_max_epochs=args.background_train_max_epochs,
        self_play_workers=args.self_play_workers,
        self_play_device=args.self_play_device,
        worker_torch_threads=args.worker_torch_threads,
        chunk_heartbeat_sec=args.chunk_heartbeat_sec,
        inference_server=args.inference_server,
        inference_server_device=args.inference_server_device,
        inference_max_batch=args.inference_max_batch,
        inference_max_wait_ms=args.inference_max_wait_ms,
        eval_episodes=args.eval_episodes,
        eval_player_counts=eval_player_counts,
        eval_max_moves=args.eval_max_moves or args.max_moves,
        eval_num_simulations=args.eval_num_simulations or args.num_simulations,
        eval_probe_path=Path(args.eval_probe_path),
        eval_baselines=eval_baselines,
        eval_checkpoint_paths=eval_checkpoint_paths,
        eval_checkpoint_pool_size=args.eval_checkpoint_pool_size,
        checkpoint_pool_dir=Path(args.checkpoint_pool_dir),
        checkpoint_pool_size=args.checkpoint_pool_size,
        arena_interval_epochs=args.arena_interval_epochs,
        arena_eval_episodes=args.arena_eval_episodes,
        fresh=args.fresh,
        verbose=not args.quiet,
    )
    _print_training_review(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
