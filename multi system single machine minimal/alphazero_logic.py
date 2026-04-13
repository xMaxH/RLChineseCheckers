"""AlphaZero-style move selection helpers.

This module keeps the decision logic separate from ``player.py`` so the game
loop stays small and readable.

Important: this is an inference-time shell for plugging in a trained model.
It does not train a network locally. You can inject your own model callback
through ``policy_value_fn``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

Move = Tuple[str, int]  # (pin_id, to_index)
PolicyDict = Dict[Move, float]


@dataclass
class AlphaZeroConfig:
    """Configuration knobs for AlphaZero-style move selection."""

    temperature: float = 1.0
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    simulations: int = 96
    seed: Optional[int] = None


@dataclass
class SearchResult:
    """Structured output for debugging and telemetry."""

    move: Move
    visit_distribution: PolicyDict
    value_estimate: float


class RootOnlyPUCT:
    """Small root-only PUCT implementation.

    Why root-only?
    The current client receives legal moves for just the current board and does
    not have a local transition simulator. Full tree search requires applying
    moves to generate deeper states. This class keeps the AlphaZero *selection*
    behavior while waiting for full game-model integration.
    """

    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def run(
        self,
        legal_moves: Dict[str, List[int]],
        priors: PolicyDict,
        root_value: float,
    ) -> SearchResult:
        moves = _flatten_legal_moves(legal_moves)
        if not moves:
            raise ValueError("No legal moves available for AlphaZero selector.")

        p = _normalize_priors(moves, priors)
        p = self._apply_root_noise(p)

        n: Dict[Move, int] = {m: 0 for m in moves}
        q: Dict[Move, float] = {m: 0.0 for m in moves}

        for _ in range(self.config.simulations):
            total_n = sum(n.values())
            best_move = None
            best_score = -float("inf")

            for m in moves:
                n_m = n[m]
                q_m = q[m]
                u_m = self.config.cpuct * p[m] * (math.sqrt(total_n + 1.0) / (1 + n_m))
                score = q_m + u_m
                if score > best_score:
                    best_score = score
                    best_move = m

            # Root-only value update: each simulation reuses the root value.
            chosen = best_move
            n[chosen] += 1
            q[chosen] += (root_value - q[chosen]) / n[chosen]

        visits = {m: float(n[m]) for m in moves}
        move = _sample_from_visits(visits, self.config.temperature, self.rng)
        visit_distribution = _normalize_distribution(visits)

        return SearchResult(
            move=move,
            visit_distribution=visit_distribution,
            value_estimate=root_value,
        )

    def _apply_root_noise(self, priors: PolicyDict) -> PolicyDict:
        moves = list(priors)
        if len(moves) <= 1:
            return priors

        alpha = self.config.dirichlet_alpha
        if alpha <= 0:
            return priors

        samples = [self.rng.gammavariate(alpha, 1.0) for _ in moves]
        total = sum(samples)
        if total <= 0:
            return priors

        noise = {m: s / total for m, s in zip(moves, samples)}
        eps = self.config.dirichlet_epsilon
        mixed = {m: (1 - eps) * priors[m] + eps * noise[m] for m in moves}
        return _normalize_distribution(mixed)


def choose_move_alphazero(
    state: dict,
    legal_moves: Dict[str, List[int]],
    colour: str,
    config: Optional[AlphaZeroConfig] = None,
    policy_value_fn: Optional[Callable[[dict, List[Move], str], Tuple[PolicyDict, float]]] = None,
) -> SearchResult:
    """Choose a move with AlphaZero-style policy/value-guided PUCT.

    Args:
        state: JSON game state returned by server.
        legal_moves: Mapping of ``pin_id -> [to_index...]`` from server.
        colour: Player colour used for perspective-sensitive callbacks.
        config: Optional search settings.
        policy_value_fn: Optional callback that returns ``(priors, value)`` for
            current state. If omitted, a lightweight fallback heuristic is used.

    Returns:
        SearchResult with chosen move and visit distribution.
    """

    cfg = config or AlphaZeroConfig()
    moves = _flatten_legal_moves(legal_moves)

    if policy_value_fn is None:
        priors, value = _fallback_policy_value(state, moves, colour)
    else:
        priors, value = policy_value_fn(state, moves, colour)

    selector = RootOnlyPUCT(cfg)
    return selector.run(legal_moves=legal_moves, priors=priors, root_value=value)


def play_turn_with_alphazero(
    rpc: Callable[[Dict[str, Any]], Dict[str, Any]],
    state: dict,
    game_id: str,
    player_id: str,
    colour: str,
    config: Optional[AlphaZeroConfig] = None,
) -> Dict[str, Any]:
    """Request legal moves, pick one with AlphaZero logic, and submit it.

    Returns a compact dict for caller-side logging:
        {
            "ok": bool,
            "pin_id": str | None,
            "to_index": int | None,
            "value_estimate": float | None,
            "error": str | None,
            "move_response": dict | None,
        }
    """

    legal_req = rpc({
        "op": "get_legal_moves",
        "game_id": game_id,
        "player_id": player_id,
    })

    if not legal_req.get("ok"):
        return {
            "ok": False,
            "pin_id": None,
            "to_index": None,
            "value_estimate": None,
            "error": f"get_legal_moves failed: {legal_req.get('error')}",
            "move_response": None,
        }

    legal_moves = legal_req.get("legal_moves", {})
    movable = [(pid, moves) for pid, moves in legal_moves.items() if moves]
    if not movable:
        return {
            "ok": False,
            "pin_id": None,
            "to_index": None,
            "value_estimate": None,
            "error": "No legal moves available.",
            "move_response": None,
        }

    az_result = choose_move_alphazero(
        state=state,
        legal_moves=legal_moves,
        colour=colour,
        config=config or AlphaZeroConfig(temperature=0.8, simulations=96),
    )
    pin_id, to_index = az_result.move

    move_response = rpc({
        "op": "move",
        "game_id": game_id,
        "player_id": player_id,
        "pin_id": pin_id,
        "to_index": to_index,
    })

    return {
        "ok": bool(move_response.get("ok")),
        "pin_id": pin_id,
        "to_index": to_index,
        "value_estimate": az_result.value_estimate,
        "error": move_response.get("error"),
        "move_response": move_response,
    }


def _flatten_legal_moves(legal_moves: Dict[str, List[int]]) -> List[Move]:
    moves: List[Move] = []
    for pin_id, destinations in legal_moves.items():
        for to_idx in destinations:
            moves.append((str(pin_id), int(to_idx)))
    return moves


def _fallback_policy_value(state: dict, moves: List[Move], colour: str) -> Tuple[PolicyDict, float]:
    """Simple heuristic priors/value until a trained network is connected."""
    if not moves:
        return {}, 0.0

    progress_hint = _progress_signal(state, colour)
    priors: PolicyDict = {}
    for move in moves:
        _, to_idx = move
        priors[move] = 1.0 + 0.1 * to_idx

    priors = _normalize_distribution(priors)
    value = max(-1.0, min(1.0, progress_hint))
    return priors, value


def _progress_signal(state: dict, colour: str) -> float:
    """Estimate progress from scores if available; otherwise return neutral."""
    players = state.get("players", [])
    me = next((p for p in players if p.get("colour") == colour), None)
    if not me:
        return 0.0

    score = me.get("score")
    if not score:
        return 0.0

    final_score = float(score.get("final_score", 0.0))
    return math.tanh(final_score / 50.0)


def _normalize_priors(moves: List[Move], priors: PolicyDict) -> PolicyDict:
    raw = {m: float(max(priors.get(m, 0.0), 0.0)) for m in moves}
    total = sum(raw.values())
    if total <= 0:
        uniform = 1.0 / len(moves)
        return {m: uniform for m in moves}
    return {m: raw[m] / total for m in moves}


def _normalize_distribution(dist: PolicyDict) -> PolicyDict:
    total = sum(max(v, 0.0) for v in dist.values())
    if total <= 0:
        n = len(dist)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in dist}
    return {k: max(v, 0.0) / total for k, v in dist.items()}


def _sample_from_visits(visits: PolicyDict, temperature: float, rng: random.Random) -> Move:
    moves = list(visits)
    if len(moves) == 1:
        return moves[0]

    if temperature <= 0:
        return max(moves, key=lambda m: visits[m])

    adjusted = {m: visits[m] ** (1.0 / temperature) for m in moves}
    probs = _normalize_distribution(adjusted)

    r = rng.random()
    cumulative = 0.0
    for m in moves:
        cumulative += probs[m]
        if r <= cumulative:
            return m
    return moves[-1]
