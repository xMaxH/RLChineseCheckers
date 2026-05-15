"""Tournament entry point loaded by player.py.

Public API (imported by player.py inside the PLAYING LOGIC markers):

    choose_move_alphazero(legal_moves, state, player_context) -> (pin_id, to_index, delay)

The function reconstructs the in-process simulator from the server's JSON
state and chooses a move. The default policy is the strongest mode observed
in local evaluations: the hand heuristic creates the safe tied-best move pool,
then the trained network ranks only inside that pool. Raw policy and MCTS are
kept as opt-in modes through ALPHAZERO_POLICY_MODE for diagnostics.
"""

from __future__ import annotations

import os
import copy
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

# Imported lazily so player.py's "fall back to random if unavailable" path works.
from az.config import (
    BOARD_CHANNELS, NUM_CELLS, MCTSConfig,
)
from az.net import AZNet
from az.encoder import decode_action, encode_action, encode_legal_mask, encode_state
from az.mcts import run_search
from az.sim import Sim
from az.heuristic import heuristic_choose_move, heuristic_move_pool
from az.selfplay import _finish_with_heuristic, _score_margin


# ---- Module-level state ----
_DEVICE: Optional[torch.device] = None
_NET: Optional[AZNet] = None
_NN_EVAL = None
_MCTS_CFG = MCTSConfig(n_sim=200, batch_leaves=32, dirichlet_alpha=0.0, dirichlet_eps=0.0,
                       virtual_loss=1.0, c_puct=1.5)


def _candidate_ckpts() -> List[str]:
    """Return prioritized list of checkpoint paths to try."""
    here = os.path.dirname(os.path.abspath(__file__))
    env_ckpt = os.getenv("ALPHAZERO_CKPT")
    paths = [
        env_ckpt,
        os.path.join(here, "runs", "best.pt"),
        os.path.join(here, "runs", "overnight_sweep_20260509_231426", "rt_bc_cap8", "final.pt"),
        os.path.join(here, "runs", "verify", "best.pt"),
        os.path.join(here, "runs", "full", "2p", "best.pt"),
    ]
    return [p for p in paths if p]


def _ensure_loaded() -> None:
    global _DEVICE, _NET, _NN_EVAL
    if _NET is not None:
        return
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _NET = AZNet().to(_DEVICE).eval()

    ckpt_path = None
    last_load_error = None
    for p in _candidate_ckpts():
        if os.path.exists(p):
            try:
                sd = torch.load(p, map_location=_DEVICE, weights_only=True)
                _NET.load_state_dict(sd)
                ckpt_path = p
                break
            except RuntimeError as e:
                last_load_error = e
                print(f"[alphazero_method] WARN: incompatible checkpoint skipped: {p}: {e}")
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No compatible checkpoint found. Looked in: {_candidate_ckpts()}. "
            f"Last load error: {last_load_error}"
        )

    # Pre-warm: run a dummy forward so cuDNN kernels are cached.
    with torch.no_grad():
        dummy_b = torch.zeros(1, BOARD_CHANNELS, NUM_CELLS, device=_DEVICE)
        dummy_g = torch.zeros(1, 8, device=_DEVICE)
        _NET(dummy_b, dummy_g)

    def _eval(boards: np.ndarray, globs: np.ndarray):
        with torch.no_grad():
            b = torch.from_numpy(boards).to(_DEVICE, non_blocking=True)
            g = torch.from_numpy(globs).to(_DEVICE, non_blocking=True)
            pol, val = _NET(b, g)
        return pol.cpu().numpy(), val.cpu().numpy()

    global _NN_EVAL
    _NN_EVAL = _eval
    print(f"[alphazero_method] loaded {ckpt_path} on {_DEVICE}")


def _build_sim_from_state(state: Dict[str, Any], my_colour: str) -> Sim:
    """Reconstruct an in-process Sim from the server's public state JSON."""
    pins_dict = state["pins"]            # {colour: [axialindex, ...]}
    turn_order = state.get("turn_order") or [my_colour]
    colours_present = list(pins_dict.keys())
    n_players = len(colours_present)

    # Build a Sim object then overwrite its state to match the server exactly.
    # Sim.__init__ takes num_players and seeds colour assignment randomly; we
    # then overwrite to match.
    sim = Sim(n_players, seed=0)

    # Reset board occupancy
    for cell in sim.board.cells:
        cell.occupied = False

    # Replace colours / pins / turn order
    sim.colours = colours_present
    sim.turn_order = turn_order

    # Re-init pins to match server
    from checkers_pins import Pin
    sim.pins_by_colour = {}
    for colour, idxs in pins_dict.items():
        pins = []
        for i, axidx in enumerate(idxs):
            p = Pin(sim.board, axidx, id=i, color=colour)
            pins.append(p)
        sim.pins_by_colour[colour] = pins

    # Set turn index to my colour
    if my_colour in turn_order:
        sim.current_turn_index = turn_order.index(my_colour)
    else:
        sim.current_turn_index = 0
    sim.move_count = state.get("move_count", 0)
    sim.player_status = {c: 'PLAYING' for c in colours_present}
    sim.game_status = 'PLAYING'
    sim.move_count_by_colour = {c: 0 for c in colours_present}
    return sim


def _normalize_legal_moves(legal_moves: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """JSON object keys arrive as strings; internal helpers use int pin IDs."""
    out: Dict[int, List[int]] = {}
    for pid, dests in legal_moves.items():
        out[int(pid)] = [int(to_idx) for to_idx in dests]
    return out


def _network_logits(sim: Sim, my_colour: str) -> np.ndarray:
    board, glob = encode_state(
        sim.pins_state(), my_colour, sim.turn_order, sim.move_count,
    )
    with torch.no_grad():
        b = torch.from_numpy(board[None]).to(_DEVICE, non_blocking=True)
        g = torch.from_numpy(glob[None]).to(_DEVICE, non_blocking=True)
        pol, _ = _NET(b, g)
    return pol[0].detach().cpu().numpy()


def _choose_raw_policy(
    sim: Sim,
    my_colour: str,
    legal_moves: Dict[int, List[int]],
) -> Tuple[int, int]:
    logits = _network_logits(sim, my_colour)
    mask = encode_legal_mask(legal_moves, my_colour)
    legal_actions = np.flatnonzero(mask)
    if len(legal_actions) == 0:
        return heuristic_choose_move(sim, my_colour, legal_moves)
    action = int(legal_actions[int(np.argmax(logits[legal_actions]))])
    return decode_action(action, my_colour)


def _rank_heuristic_pool_by_network(
    sim: Sim,
    my_colour: str,
    legal_moves: Dict[int, List[int]],
) -> List[Tuple[int, int]]:
    pool = heuristic_move_pool(sim, my_colour, legal_moves)
    if len(pool) == 1:
        _, _, pid, to_idx = pool[0]
        return [(pid, to_idx)]

    logits = _network_logits(sim, my_colour)
    ranked = sorted(
        ((float(logits[encode_action(pid, to_idx, my_colour)]), pid, to_idx)
         for _, _, pid, to_idx in pool),
        reverse=True,
    )
    return [(pid, to_idx) for _, pid, to_idx in ranked]


def _choose_heuristic_pool_rerank(
    sim: Sim,
    my_colour: str,
    legal_moves: Dict[int, List[int]],
) -> Tuple[int, int]:
    ranked = _rank_heuristic_pool_by_network(sim, my_colour, legal_moves)
    if not ranked:
        return heuristic_choose_move(sim, my_colour, legal_moves)
    return ranked[0]


def _rollout_seed(sim: Sim, colour: str, pid: int, to_idx: int, rollout_idx: int) -> int:
    return (
        (sim.move_count + 1) * 1_000_003
        + sum((i + 1) * p.axialindex for i, p in enumerate(sim.pins_by_colour[colour]))
        + pid * 9_176
        + to_idx * 37
        + rollout_idx * 1_009
    ) & 0xFFFFFFFF


def _choose_heuristic_pool_rollout(
    sim: Sim,
    my_colour: str,
    legal_moves: Dict[int, List[int]],
) -> Tuple[int, int]:
    ranked = _rank_heuristic_pool_by_network(sim, my_colour, legal_moves)
    if not ranked:
        return heuristic_choose_move(sim, my_colour, legal_moves)
    if len(ranked) == 1:
        return ranked[0]

    top_k = max(1, int(os.getenv("ALPHAZERO_ROLLOUT_TOPK", "3")))
    rollouts_per_move = max(1, int(os.getenv("ALPHAZERO_ROLLOUTS_PER_MOVE", "2")))
    max_moves = max(1, int(os.getenv("ALPHAZERO_ROLLOUT_MAX_MOVES", "500")))

    best_key = None
    best_move = ranked[0]
    for rank, (pid, to_idx) in enumerate(ranked[:top_k]):
        scores = []
        for rollout_idx in range(rollouts_per_move):
            s = copy.deepcopy(sim)
            s.apply_move(pid, to_idx)
            rollout_rng = random.Random(_rollout_seed(sim, my_colour, pid, to_idx, rollout_idx))
            _finish_with_heuristic(s, rollout_rng, max_moves=max_moves)
            scores.append(_score_margin(s, my_colour))
        # Keep network order as a stable tiebreaker after rollout score.
        key = (float(np.mean(scores)), -rank)
        if best_key is None or key > best_key:
            best_key = key
            best_move = (pid, to_idx)
    return best_move


def _choose_mcts(
    sim: Sim,
    my_colour: str,
    legal_moves: Dict[int, List[int]],
) -> Tuple[int, int]:
    visits, _, _ = run_search(sim, _NN_EVAL, _MCTS_CFG, add_dirichlet_at_root=False)

    if int(visits.sum()) == 0:
        return heuristic_choose_move(sim, my_colour, legal_moves)

    legal_action_set = {
        encode_action(pid, to_idx, my_colour)
        for pid, dests in legal_moves.items()
        for to_idx in dests
    }
    for action in np.argsort(-visits):
        action = int(action)
        if visits[action] <= 0:
            break
        if action in legal_action_set:
            return decode_action(action, my_colour)
    return heuristic_choose_move(sim, my_colour, legal_moves)


# Best-effort pre-load at import time so the first move avoids cold-start latency.
try:
    _ensure_loaded()
except Exception as _e:
    print(f"[alphazero_method] WARN: deferred load: {_e}")


def choose_move_alphazero(
    legal_moves: Dict[int, List[int]],
    state: Dict[str, Any],
    player_context: Dict[str, Any],
) -> Tuple[int, int, float]:
    """Choose a move. Returns (pin_id, to_index, delay_seconds)."""
    _ensure_loaded()

    my_colour = player_context["colour"]
    sim = _build_sim_from_state(state, my_colour)
    server_legal = _normalize_legal_moves(legal_moves)

    mode = os.getenv("ALPHAZERO_POLICY_MODE", "heuristic_rollout").strip().lower()
    if mode in ("heuristic_rollout", "heuristic-rollout", "rollout", "lookahead"):
        pid, to = _choose_heuristic_pool_rollout(sim, my_colour, server_legal)
    elif mode in ("heuristic_pool", "heuristic-pool", "hybrid", "rerank"):
        pid, to = _choose_heuristic_pool_rerank(sim, my_colour, server_legal)
    elif mode in ("raw", "raw_policy", "policy"):
        pid, to = _choose_raw_policy(sim, my_colour, server_legal)
    elif mode == "mcts":
        pid, to = _choose_mcts(sim, my_colour, server_legal)
    elif mode == "heuristic":
        pid, to = heuristic_choose_move(sim, my_colour, server_legal)
    else:
        print(f"[alphazero_method] WARN: unknown ALPHAZERO_POLICY_MODE={mode!r}; using heuristic_pool")
        pid, to = _choose_heuristic_pool_rerank(sim, my_colour, server_legal)

    if to not in server_legal.get(int(pid), []):
        pid, to = heuristic_choose_move(sim, my_colour, server_legal)
    return pid, to, 0.0
