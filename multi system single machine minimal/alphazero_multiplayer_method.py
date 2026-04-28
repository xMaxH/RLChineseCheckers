"""AlphaZero-style method for 3-6 player Chinese Checkers.

This module is a multiplayer companion to alphazero_method.py.
It keeps the same runtime interface for player.py but removes the
strict 2-player assumptions by using a root-perspective paranoid MCTS.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero_method import format_parameter_summary
from checkers_board import HexBoard
from checkers_pins import Pin


MAX_CELLS = 121
ACTION_SIZE = MAX_CELLS * MAX_CELLS
INPUT_CHANNELS = 24
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "alphazero_multi.pt"
MODEL_CACHE: Optional["MultiAlphaZeroAgent"] = None
COLOUR_ORDER = ["red", "lawn green", "yellow", "blue", "gray0", "purple"]


@dataclass(frozen=True)
class Action:
    """Move represented by logical pin id and target board index."""

    pin_id: int
    to_index: int


class MultiSimState:
    """Lightweight multiplayer simulator using existing board/pin logic."""

    def __init__(
        self,
        pins_by_colour: Mapping[str, Sequence[int]],
        turn_order: Sequence[str],
        current_turn_colour: str,
    ):
        if len(turn_order) < 2:
            raise ValueError("AlphaZero multiplayer mode needs at least 2 players.")
        if current_turn_colour not in turn_order:
            raise ValueError("Current turn colour must be part of the turn order.")

        self.board = HexBoard()
        self.turn_order = list(turn_order)
        self.current_turn_index = self.turn_order.index(current_turn_colour)

        self.pins_by_colour: Dict[str, List[Pin]] = {}
        for colour in self.turn_order:
            positions = list(pins_by_colour.get(colour, []))
            self.pins_by_colour[colour] = [
                Pin(self.board, axialindex=int(idx), id=i, color=colour)
                for i, idx in enumerate(positions)
            ]

    def clone(self) -> "MultiSimState":
        positions = {
            colour: [pin.axialindex for pin in pins]
            for colour, pins in self.pins_by_colour.items()
        }
        return MultiSimState(positions, self.turn_order, self.current_turn_colour())

    def current_turn_colour(self) -> str:
        return self.turn_order[self.current_turn_index]

    def legal_actions(self, colour: Optional[str] = None) -> List[Action]:
        chosen_colour = colour or self.current_turn_colour()
        actions: List[Action] = []
        for pin in self.pins_by_colour[chosen_colour]:
            for target in pin.getPossibleMoves():
                actions.append(Action(pin_id=pin.id, to_index=int(target)))
        return actions

    def apply_action(self, action: Action) -> None:
        colour = self.current_turn_colour()
        pin = self.pins_by_colour[colour][action.pin_id]
        old_idx = pin.axialindex

        self.board.cells[old_idx].occupied = False
        pin.axialindex = int(action.to_index)
        self.board.cells[action.to_index].occupied = True

        self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)

    def winner(self) -> Optional[str]:
        for colour in self.turn_order:
            opposite = self.board.colour_opposites[colour]
            pins = self.pins_by_colour[colour]
            if all(self.board.cells[p.axialindex].postype == opposite for p in pins):
                return colour
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or not self.legal_actions()

    def _hex_distance(self, a_idx: int, b_idx: int) -> int:
        a = self.board.cells[a_idx]
        b = self.board.cells[b_idx]
        dq = abs(a.q - b.q)
        dr = abs(a.r - b.r)
        ds = abs((-a.q - a.r) - (-b.q - b.r))
        return max(dq, dr, ds)

    def distance_to_goal(self, colour: str) -> float:
        opposite = self.board.colour_opposites[colour]
        target_idxs = self.board.axial_of_colour(opposite)
        total = 0.0
        for pin in self.pins_by_colour[colour]:
            if self.board.cells[pin.axialindex].postype == opposite:
                continue
            total += min(self._hex_distance(pin.axialindex, target) for target in target_idxs)
        return total

    def heuristic_value(self, root_colour: str) -> float:
        win = self.winner()
        if win == root_colour:
            return 1.0
        if win is not None:
            return -1.0

        root_dist = self.distance_to_goal(root_colour)
        opp_dists = [self.distance_to_goal(colour) for colour in self.turn_order if colour != root_colour]
        avg_opp_dist = sum(opp_dists) / max(1, len(opp_dists))
        raw = (avg_opp_dist - root_dist) / 40.0
        return float(max(-0.99, min(0.99, math.tanh(raw))))

    def encode(self, root_colour: str) -> torch.Tensor:
        """Encode as [24, MAX_CELLS] tensor for the policy-value net."""
        cur = self.current_turn_colour()
        n = len(self.board.cells)
        x = torch.zeros((INPUT_CHANNELS, MAX_CELLS), dtype=torch.float32)

        colour_to_idx = {colour: idx for idx, colour in enumerate(COLOUR_ORDER)}
        root_idx = colour_to_idx[root_colour]
        cur_idx = colour_to_idx[cur]

        for colour, idx in colour_to_idx.items():
            if colour in self.pins_by_colour:
                for pin in self.pins_by_colour[colour]:
                    x[idx, pin.axialindex] = 1.0

        x[6 + root_idx, :n] = 1.0
        x[12 + cur_idx, :n] = 1.0
        for colour, idx in colour_to_idx.items():
            if colour in self.board.colour_opposites:
                for cell_idx in self.board.axial_of_colour(colour):
                    x[18 + idx, cell_idx] = 1.0

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
    def __init__(self, state: MultiSimState, prior: float = 1.0):
        self.state = state
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Action, "MultiNode"] = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class TrainingExample:
    state_tensor: torch.Tensor
    policy: torch.Tensor
    value: float


class MultiAlphaZeroAgent:
    def __init__(self, model: Optional[MultiPolicyValueNet] = None, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device) if model is not None else None
        if self.model is not None:
            self.model.eval()

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

        if self.model is None:
            priors: Dict[Action, float] = {}
            total = 0.0
            for action in legal_actions:
                before = state.distance_to_goal(root_colour)
                tmp = state.clone()
                tmp.apply_action(action)
                after = tmp.distance_to_goal(root_colour)
                improvement = max(0.0, before - after)

                opponent_pressure = 0.0
                for colour in state.turn_order:
                    if colour == root_colour:
                        continue
                    opp_before = state.distance_to_goal(colour)
                    opp_after = tmp.distance_to_goal(colour)
                    opponent_pressure += max(0.0, opp_after - opp_before)

                into_goal = 1.0 if tmp.board.cells[tmp.pins_by_colour[root_colour][action.pin_id].axialindex].postype == tmp.board.colour_opposites[root_colour] else 0.0
                score = 1e-3 + improvement + 0.1 * opponent_pressure + 0.5 * into_goal
                priors[action] = score
                total += score

            if total > 0:
                priors = {a: p / total for a, p in priors.items()}
            else:
                priors = {a: 1.0 / len(legal_actions) for a in legal_actions}
            return priors, state.heuristic_value(root_colour)

        with torch.no_grad():
            x = state.encode(root_colour).flatten().unsqueeze(0).to(self.device)
            policy_logits, value = self.model(x)
            logits = policy_logits[0]
            probs: Dict[Action, float] = {}
            total = 0.0
            cur_colour = state.current_turn_colour()
            for action in legal_actions:
                from_idx = state.pins_by_colour[cur_colour][action.pin_id].axialindex
                action_id = int(from_idx) * MAX_CELLS + int(action.to_index)
                p = float(torch.exp(logits[action_id]).item())
                probs[action] = p
                total += p

            if total <= 0:
                probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
            else:
                probs = {a: p / total for a, p in probs.items()}
            return probs, float(value.item())

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
    ) -> Dict[Action, int]:
        root = MultiNode(root_state.clone(), prior=1.0)

        root_legal = root.state.legal_actions()
        root_priors, root_value = self._evaluate_policy_and_value(root.state, root_colour, root_legal)
        for action in root_legal:
            child_state = root.state.clone()
            child_state.apply_action(action)
            root.children[action] = MultiNode(child_state, prior=root_priors[action])

        noise = self._root_dirichlet(root_dirichlet_alpha, len(root_legal)) if root_legal else []
        for i, action in enumerate(root_legal):
            child = root.children[action]
            child.prior = (1 - root_dirichlet_frac) * child.prior + root_dirichlet_frac * noise[i]

        root.visit_count += 1
        root.value_sum += root_value

        for _ in range(num_simulations):
            node = root
            path = [node]

            while node.expanded() and not node.state.is_terminal():
                _, node = self._select_child(node, root_colour, c_puct)
                path.append(node)

            if node.state.is_terminal():
                winner = node.state.winner()
                if winner is None:
                    value = 0.0
                else:
                    value = 1.0 if winner == root_colour else -1.0
            else:
                legal = node.state.legal_actions()
                priors, value = self._evaluate_policy_and_value(node.state, root_colour, legal)
                for action in legal:
                    child_state = node.state.clone()
                    child_state.apply_action(action)
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
    root_state = MultiSimState(
        pins_by_colour={colour: list(map(int, pins.get(colour, []))) for colour in turn_order},
        turn_order=turn_order,
        current_turn_colour=current_turn,
    )

    legal_override = {int(k): [int(x) for x in v] for k, v in legal_moves.items() if v}
    if not legal_override:
        raise ValueError("No legal moves available for AlphaZero multiplayer mode.")

    agent = _load_agent()
    sims = int(os.getenv("AZ_MP_MCTS_SIMS", os.getenv("AZ_MCTS_SIMS", "96")))
    c_puct = float(os.getenv("AZ_MP_C_PUCT", os.getenv("AZ_C_PUCT", "1.5")))
    temp_opening = float(os.getenv("AZ_MP_TEMP_OPENING", os.getenv("AZ_TEMP_OPENING", "1.0")))
    temp_late = float(os.getenv("AZ_MP_TEMP_LATE", os.getenv("AZ_TEMP_LATE", "0.15")))
    cutoff = int(os.getenv("AZ_MP_TEMP_CUTOFF_MOVE", os.getenv("AZ_TEMP_CUTOFF_MOVE", "20")))
    move_count = int(state.get("move_count", 0))
    temperature = temp_opening if move_count < cutoff else temp_late

    visits = agent.run_mcts(
        root_state=root_state,
        root_colour=my_colour,
        num_simulations=sims,
        c_puct=c_puct,
    )

    chosen = _sample_action_from_visits(visits, temperature)
    delay = 0.0
    return chosen.pin_id, chosen.to_index, delay


def create_initial_multiplayer_state(colours: Sequence[str]) -> MultiSimState:
    board = HexBoard()
    positions = {
        colour: board.axial_of_colour(colour)[:10]
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


def build_policy_target(state: MultiSimState, action_visits: Mapping[Action, int]) -> torch.Tensor:
    target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    colour = state.current_turn_colour()
    total = sum(max(0, int(v)) for v in action_visits.values())
    if total <= 0:
        return target
    for action, count in action_visits.items():
        from_idx = state.pins_by_colour[colour][action.pin_id].axialindex
        action_id = int(from_idx) * MAX_CELLS + int(action.to_index)
        target[action_id] = float(count) / float(total)
    return target


def generate_self_play_game(
    agent: MultiAlphaZeroAgent,
    player_count: int = 6,
    num_simulations: int = 96,
    c_puct: float = 1.5,
    max_moves: int = 400,
    temperature_cutoff: int = 20,
    temp_opening: float = 1.0,
    temp_late: float = 0.15,
) -> List[TrainingExample]:
    if player_count < 2 or player_count > 6:
        raise ValueError("player_count must be between 2 and 6.")

    colours = random.sample(COLOUR_ORDER, k=player_count)
    state = create_initial_multiplayer_state(colours)
    raw_samples: List[Tuple[torch.Tensor, torch.Tensor, str]] = []

    for ply in range(max_moves):
        if state.is_terminal():
            break

        to_play = state.current_turn_colour()
        legal = state.legal_actions(to_play)
        if not legal:
            break

        visits = agent.run_mcts(
            root_state=state,
            root_colour=to_play,
            num_simulations=num_simulations,
            c_puct=c_puct,
        )

        policy_target = build_policy_target(state, visits)
        state_tensor = state.encode(to_play).flatten()
        raw_samples.append((state_tensor, policy_target, to_play))

        temperature = temp_opening if ply < temperature_cutoff else temp_late
        action = _sample_action_from_visits(visits, temperature)
        state.apply_action(action)

    winner = state.winner()
    samples: List[TrainingExample] = []
    for state_tensor, policy_target, to_play in raw_samples:
        if winner is None:
            value = 0.0
        else:
            value = 1.0 if winner == to_play else -1.0
        samples.append(
            TrainingExample(
                state_tensor=state_tensor,
                policy=policy_target,
                value=value,
            )
        )
    return samples


def train_step(
    model: MultiPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[TrainingExample],
    device: str,
) -> Dict[str, float]:
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


def run_self_play_training(
    episodes: int = 100,
    train_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_simulations: int = 96,
    c_puct: float = 1.5,
    max_moves: int = 500,
    player_count_min: int = 2,
    player_count_max: int = 6,
    fixed_player_count: Optional[int] = None,
    temp_opening: float = 1.0,
    temp_late: float = 0.15,
    temperature_cutoff: int = 20,
    model_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    started = time.time()
    model = MultiPolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    agent = MultiAlphaZeroAgent(model=model)

    replay: List[TrainingExample] = []
    for ep in range(1, episodes + 1):
        episode_player_count = _resolve_episode_player_count(
            player_count_min=player_count_min,
            player_count_max=player_count_max,
            fixed_player_count=fixed_player_count,
        )
        if verbose:
            print(
                f"[self-play] episode {ep}/{episodes} started "
                f"(players={episode_player_count})",
                flush=True,
            )
        episode_samples = generate_self_play_game(
            agent=agent,
            player_count=episode_player_count,
            num_simulations=num_simulations,
            c_puct=c_puct,
            max_moves=max_moves,
            temperature_cutoff=temperature_cutoff,
            temp_opening=temp_opening,
            temp_late=temp_late,
        )
        replay.extend(episode_samples)
        if verbose:
            print(
                f"[self-play] episode {ep}/{episodes} samples={len(episode_samples)} total_samples={len(replay)}",
                flush=True,
            )

    if not replay:
        raise RuntimeError("No self-play samples generated.")

    last_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    epoch_times: List[float] = []
    device = str(agent.device)
    for epoch in range(1, train_epochs + 1):
        epoch_started = time.time()
        random.shuffle(replay)
        epoch_loss = 0.0
        epoch_policy = 0.0
        epoch_value = 0.0
        batches = 0

        for i in range(0, len(replay), batch_size):
            batch = replay[i : i + batch_size]
            if not batch:
                continue
            last_metrics = train_step(model, optimizer, batch, device)
            epoch_loss += last_metrics["loss"]
            epoch_policy += last_metrics["policy_loss"]
            epoch_value += last_metrics["value_loss"]
            batches += 1

        epoch_elapsed = time.time() - epoch_started
        epoch_times.append(epoch_elapsed)

        if verbose and batches > 0:
            print(
                f"[train] epoch {epoch}/{train_epochs} loss={epoch_loss / batches:.4f} policy={epoch_policy / batches:.4f} value={epoch_value / batches:.4f} time={epoch_elapsed:.2f}s",
                flush=True,
            )

    save_path = Path(model_path or os.getenv("AZ_MP_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, save_path)

    elapsed = time.time() - started
    avg_epoch_time = (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0

    return {
        "samples": float(len(replay)),
        "train_seconds": float(elapsed),
        "avg_epoch_seconds": float(avg_epoch_time),
        "model_path": str(save_path),
        **last_metrics,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a multiplayer AlphaZero-style model via self-play.",
    )
    parser.add_argument("--episodes", type=int, default=500, help="Number of self-play games.")
    parser.add_argument("--train-epochs", type=int, default=8, help="Epochs over replay data.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for optimizer steps.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-simulations", type=int, default=96, help="MCTS simulations per move.") # high for training low forplaying
    parser.add_argument("--max-moves", type=int, default=500, help="Max plies per self-play game.")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument("--temp-opening", type=float, default=1.0, help="Temperature for early moves.")
    parser.add_argument("--temp-late", type=float, default=0.15, help="Temperature for late moves.")
    parser.add_argument("--temp-cutoff-move", type=int, default=20, help="Move number to switch temperatures.")
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
    print(f"Model saved to    : {metrics['model_path']}")
    print("=======================")


def main() -> int:
    args = _build_arg_parser().parse_args()

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
            ("player_count_min", args.player_count_min),
            ("player_count_max", args.player_count_max),
            ("fixed_player_count", args.fixed_player_count),
            ("max_moves", args.max_moves),
            ("model_path", args.model_path),
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
        model_path=Path(args.model_path),
        verbose=not args.quiet,
    )
    _print_training_review(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
