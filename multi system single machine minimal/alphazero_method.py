"""AlphaZero-style method for 2-player Chinese Checkers.

This module provides:
1. `choose_move_alphazero(...)` for runtime move selection in `player.py`
2. A local 2-player simulator from JSON state
3. MCTS (PUCT)
4. Optional policy-value network support with heuristic fallback
5. Self-play training utilities for offline training
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from checkers_board import HexBoard
from checkers_pins import Pin


MAX_CELLS = 121
ACTION_SIZE = MAX_CELLS * MAX_CELLS
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "alphazero_2p.pt"
MODEL_CACHE: Optional["AlphaZeroAgent"] = None


@dataclass(frozen=True)
class Action:
    """Move represented by logical pin id and target board index."""

    pin_id: int
    to_index: int


class SimState:
    """Lightweight 2-player simulator using existing board/pin logic."""

    def __init__(self, pins_by_colour: Mapping[str, Sequence[int]], turn_order: Sequence[str], current_turn_colour: str):
        if len(turn_order) != 2:
            raise ValueError("AlphaZero method supports exactly 2 players.")

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

    def clone(self) -> "SimState":
        positions = {
            colour: [pin.axialindex for pin in pins]
            for colour, pins in self.pins_by_colour.items()
        }
        return SimState(positions, self.turn_order, self.current_turn_colour())

    def current_turn_colour(self) -> str:
        return self.turn_order[self.current_turn_index]

    def opponent_colour(self, colour: str) -> str:
        return self.turn_order[1] if self.turn_order[0] == colour else self.turn_order[0]

    def legal_actions(self, colour: Optional[str] = None) -> List[Action]:
        c = colour or self.current_turn_colour()
        actions: List[Action] = []
        for pin in self.pins_by_colour[c]:
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

        self.current_turn_index = (self.current_turn_index + 1) % 2

    def winner(self) -> Optional[str]:
        for colour in self.turn_order:
            opposite = self.board.colour_opposites[colour]
            pins = self.pins_by_colour[colour]
            if all(self.board.cells[p.axialindex].postype == opposite for p in pins):
                return colour
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None

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
            total += min(self._hex_distance(pin.axialindex, t) for t in target_idxs)
        return total

    def heuristic_value(self, perspective_colour: str) -> float:
        win = self.winner()
        if win == perspective_colour:
            return 1.0
        if win is not None:
            return -1.0

        opp = self.opponent_colour(perspective_colour)
        own_dist = self.distance_to_goal(perspective_colour)
        opp_dist = self.distance_to_goal(opp)
        raw = (opp_dist - own_dist) / 40.0
        return float(max(-0.99, min(0.99, math.tanh(raw))))

    def encode(self, perspective_colour: str) -> torch.Tensor:
        """Encode board as [7, MAX_CELLS] tensor for policy-value net."""
        cur = self.current_turn_colour()
        opp = self.opponent_colour(perspective_colour)
        n = len(self.board.cells)

        x = torch.zeros((7, MAX_CELLS), dtype=torch.float32)

        for pin in self.pins_by_colour[perspective_colour]:
            x[0, pin.axialindex] = 1.0
        for pin in self.pins_by_colour[opp]:
            x[1, pin.axialindex] = 1.0

        own_goal = self.board.axial_of_colour(self.board.colour_opposites[perspective_colour])
        opp_goal = self.board.axial_of_colour(self.board.colour_opposites[opp])
        own_start = self.board.axial_of_colour(perspective_colour)
        opp_start = self.board.axial_of_colour(opp)

        x[2, own_goal] = 1.0
        x[3, opp_goal] = 1.0
        x[4, own_start] = 1.0
        x[5, opp_start] = 1.0
        x[6, :n] = 1.0 if cur == perspective_colour else 0.0
        return x


class PolicyValueNet(nn.Module):
    """Compact MLP policy-value network for fixed action space."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        input_dim = 7 * MAX_CELLS
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


def action_to_id(from_idx: int, to_idx: int) -> int:
    return int(from_idx) * MAX_CELLS + int(to_idx)


class Node:
    def __init__(self, state: SimState, prior: float = 1.0):
        self.state = state
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Action, "Node"] = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class AlphaZeroAgent:
    def __init__(self, model: Optional[PolicyValueNet] = None, device: Optional[str] = None):
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
        state: SimState,
        perspective: str,
        legal_actions: Sequence[Action],
    ) -> Tuple[Dict[Action, float], float]:
        if not legal_actions:
            return {}, state.heuristic_value(perspective)

        if self.model is None:
            priors: Dict[Action, float] = {}
            total = 0.0
            for action in legal_actions:
                before = state.distance_to_goal(perspective)
                tmp = state.clone()
                tmp.apply_action(action)
                after = tmp.distance_to_goal(perspective)
                improve = max(0.0, before - after)
                into_goal = 1.0 if tmp.board.cells[tmp.pins_by_colour[perspective][action.pin_id].axialindex].postype == tmp.board.colour_opposites[perspective] else 0.0
                score = 1e-3 + improve + 0.5 * into_goal
                priors[action] = score
                total += score
            priors = {a: p / total for a, p in priors.items()} if total > 0 else {a: 1.0 / len(legal_actions) for a in legal_actions}
            return priors, state.heuristic_value(perspective)

        with torch.no_grad():
            x = state.encode(perspective).flatten().unsqueeze(0).to(self.device)
            policy_logits, value = self.model(x)
            logits = policy_logits[0]
            probs: Dict[Action, float] = {}
            total = 0.0
            cur_colour = state.current_turn_colour()
            for action in legal_actions:
                from_idx = state.pins_by_colour[cur_colour][action.pin_id].axialindex
                aid = action_to_id(from_idx, action.to_index)
                p = float(torch.exp(logits[aid]).item())
                probs[action] = p
                total += p

            if total <= 0:
                probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
            else:
                probs = {a: p / total for a, p in probs.items()}
            return probs, float(value.item())

    def _select_child(self, node: Node, c_puct: float) -> Tuple[Action, Node]:
        best_score = -float("inf")
        best_action = None
        best_child = None
        parent_sqrt = math.sqrt(max(1, node.visit_count))

        for action, child in node.children.items():
            q = -child.value()
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
        root_state: SimState,
        perspective_colour: str,
        root_legal_override: Optional[Mapping[int, Sequence[int]]] = None,
        num_simulations: int = 96,
        c_puct: float = 1.5,
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_frac: float = 0.25,
    ) -> Dict[Action, int]:
        root = Node(root_state.clone(), prior=1.0)

        root_legal = self._legal_actions_for_node(root.state, root_legal_override)
        root_priors, root_value = self._evaluate_policy_and_value(root.state, perspective_colour, root_legal)
        for action in root_legal:
            child_state = root.state.clone()
            child_state.apply_action(action)
            root.children[action] = Node(child_state, prior=root_priors[action])

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
                _, node = self._select_child(node, c_puct)
                path.append(node)

            if node.state.is_terminal():
                winner = node.state.winner()
                if winner is None:
                    value = 0.0
                else:
                    to_play = node.state.current_turn_colour()
                    value = 1.0 if winner == to_play else -1.0
            else:
                legal = self._legal_actions_for_node(node.state, None)
                priors, value = self._evaluate_policy_and_value(node.state, perspective_colour, legal)
                for action in legal:
                    child_state = node.state.clone()
                    child_state.apply_action(action)
                    node.children[action] = Node(child_state, prior=priors[action])

            for p in reversed(path):
                p.visit_count += 1
                p.value_sum += value
                value = -value

        return {action: child.visit_count for action, child in root.children.items()}

    def _legal_actions_for_node(
        self,
        state: SimState,
        root_legal_override: Optional[Mapping[int, Sequence[int]]],
    ) -> List[Action]:
        if root_legal_override is None:
            return state.legal_actions()

        actions: List[Action] = []
        for pin_id, moves in root_legal_override.items():
            for target in moves:
                actions.append(Action(pin_id=int(pin_id), to_index=int(target)))
        return actions


def _load_agent() -> AlphaZeroAgent:
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE

    model_path = Path(os.getenv("AZ_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        MODEL_CACHE = AlphaZeroAgent(model=None)
        return MODEL_CACHE

    model = PolicyValueNet()
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise ValueError("Unsupported checkpoint format for AlphaZero model.")

    MODEL_CACHE = AlphaZeroAgent(model=model)
    return MODEL_CACHE


def _extract_two_player_turn_order(state: Mapping[str, Any]) -> List[str]:
    players = state.get("players", [])
    if len(players) != 2:
        raise ValueError("AlphaZero method currently supports exactly 2 players.")
    order = state.get("turn_order") or [p["colour"] for p in players]
    if len(order) != 2:
        raise ValueError("Invalid turn order for 2-player game.")
    return list(order)


def _sample_action_from_visits(
    visits: Mapping[Action, int],
    temperature: float,
) -> Action:
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


def choose_move_alphazero(
    legal_moves: Mapping[str, Sequence[int]],
    state: Mapping[str, Any],
    player_context: Dict[str, Any],
) -> Tuple[int, int, float]:
    """Choose a move with AlphaZero-style MCTS for 2-player games.

    Returns:
        (pin_id, to_index, delay_seconds)
    """
    my_colour = str(player_context.get("colour", ""))
    if not my_colour:
        raise ValueError("player_context must include colour for AlphaZero method.")

    turn_order = _extract_two_player_turn_order(state)
    current_turn = str(state.get("current_turn_colour", ""))
    if current_turn != my_colour:
        raise ValueError("AlphaZero called when it is not this player's turn.")

    pins = state.get("pins", {})
    if my_colour not in pins:
        raise ValueError("Current player colour not present in state pins.")

    root_state = SimState(
        pins_by_colour={colour: list(map(int, pins.get(colour, []))) for colour in turn_order},
        turn_order=turn_order,
        current_turn_colour=current_turn,
    )

    legal_override = {int(k): [int(x) for x in v] for k, v in legal_moves.items() if v}
    if not legal_override:
        raise ValueError("No legal moves available for AlphaZero method.")

    agent = _load_agent()
    sims = int(os.getenv("AZ_MCTS_SIMS", "96"))
    c_puct = float(os.getenv("AZ_C_PUCT", "1.5"))
    temp_opening = float(os.getenv("AZ_TEMP_OPENING", "1.0"))
    temp_late = float(os.getenv("AZ_TEMP_LATE", "0.15"))
    cutoff = int(os.getenv("AZ_TEMP_CUTOFF_MOVE", "20"))
    move_count = int(state.get("move_count", 0))
    temperature = temp_opening if move_count < cutoff else temp_late

    visits = agent.run_mcts(
        root_state=root_state,
        perspective_colour=my_colour,
        root_legal_override=legal_override,
        num_simulations=sims,
        c_puct=c_puct,
    )

    chosen = _sample_action_from_visits(visits, temperature)
    delay = 0.0 # random.uniform(0.05,0.12)
    return chosen.pin_id, chosen.to_index, delay


@dataclass
class TrainingExample:
    state_tensor: torch.Tensor
    policy: torch.Tensor
    value: float


def build_policy_target(
    state: SimState,
    action_visits: Mapping[Action, int],
) -> torch.Tensor:
    target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    colour = state.current_turn_colour()
    total = sum(max(0, int(v)) for v in action_visits.values())
    if total <= 0:
        return target
    for action, count in action_visits.items():
        from_idx = state.pins_by_colour[colour][action.pin_id].axialindex
        aid = action_to_id(from_idx, action.to_index)
        target[aid] = float(count) / float(total)
    return target


def train_step(
    model: PolicyValueNet,
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


def create_initial_two_player_state(
    first_colour: str = "yellow",
    second_colour: str = "purple",
) -> SimState:
    board = HexBoard()
    positions = {
        first_colour: board.axial_of_colour(first_colour)[:10],
        second_colour: board.axial_of_colour(second_colour)[:10],
    }
    return SimState(
        pins_by_colour=positions,
        turn_order=[first_colour, second_colour],
        current_turn_colour=first_colour,
    )


def generate_self_play_game(
    agent: AlphaZeroAgent,
    num_simulations: int = 96,
    c_puct: float = 1.5,
    max_moves: int = 400,
    temperature_cutoff: int = 20,
) -> List[TrainingExample]:
    state = create_initial_two_player_state()
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
            perspective_colour=to_play,
            root_legal_override=None,
            num_simulations=num_simulations,
            c_puct=c_puct,
        )

        policy_target = build_policy_target(state, visits)
        state_tensor = state.encode(to_play).flatten()
        raw_samples.append((state_tensor, policy_target, to_play))

        temperature = 1.0 if ply < temperature_cutoff else 0.15
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


def run_self_play_training(
    episodes: int = 100,
    train_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_simulations: int = 96,
    c_puct: float = 1.5,
    max_moves: int = 500,
    model_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    started = time.time()
    model = PolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    agent = AlphaZeroAgent(model=model)

    replay: List[TrainingExample] = []
    for ep in range(1, episodes + 1):
        if verbose:
            print(f"[self-play] episode {ep}/{episodes} started", flush=True)
        episode_samples = generate_self_play_game(
            agent=agent,
            num_simulations=num_simulations,
            c_puct=c_puct,
            max_moves=max_moves,
        )
        replay.extend(episode_samples)
        if verbose:
            print(
                f"[self-play] episode {ep}/{episodes} "
                f"samples={len(episode_samples)} total_samples={len(replay)}",
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
            batch = replay[i: i + batch_size]
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
                f"[train] epoch {epoch}/{train_epochs} "
                f"loss={epoch_loss / batches:.4f} "
                f"policy={epoch_policy / batches:.4f} "
                f"value={epoch_value / batches:.4f} "
                f"time={epoch_elapsed:.2f}s",
                flush=True,
            )

    save_path = Path(model_path or os.getenv("AZ_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
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
        description="Train a 2-player AlphaZero-style model via self-play."
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of self-play games (quick default: 5).")
    parser.add_argument("--train-epochs", type=int, default=10, help="Epochs over replay data (quick default: 2).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for optimizer steps.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-simulations", type=int, default=24, help="MCTS simulations per move (quick default: 24).")
    parser.add_argument("--max-moves", type=int, default=500, help="Max plies per self-play game.")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument("--temp-opening", type=float, default=1.0, help="Temperature for early moves.")
    parser.add_argument("--temp-late", type=float, default=0.15, help="Temperature for late moves.")
    parser.add_argument("--temp-cutoff-move", type=int, default=20, help="Move number to switch temperatures.")
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

    # Set env vars so gameplay uses same parameters
    os.environ["AZ_MCTS_SIMS"] = str(args.num_simulations)
    os.environ["AZ_C_PUCT"] = str(args.c_puct)
    os.environ["AZ_TEMP_OPENING"] = str(args.temp_opening)
    os.environ["AZ_TEMP_LATE"] = str(args.temp_late)
    os.environ["AZ_TEMP_CUTOFF_MOVE"] = str(args.temp_cutoff_move)

    print("[train] Starting AlphaZero self-play training", flush=True)
    print(
        "[train] config: "
        f"episodes={args.episodes}, "
        f"epochs={args.train_epochs}, "
        f"batch={args.batch_size}, "
        f"lr={args.lr}, "
        f"sims={args.num_simulations}, "
        f"c_puct={args.c_puct}, "
        f"temp_opening={args.temp_opening}, "
        f"temp_late={args.temp_late}, "
        f"temp_cutoff={args.temp_cutoff_move}, "
        f"max_moves={args.max_moves}, "
        f"model={args.model_path}",
        flush=True,
    )

    metrics = run_self_play_training(
        episodes=args.episodes,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        max_moves=args.max_moves,
        model_path=Path(args.model_path),
        verbose=not args.quiet,
    )
    _print_training_review(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
