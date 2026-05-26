"""Batched PUCT MCTS with virtual loss and MaxN backup for 2-6 player games.

Public entry point: `run_search(sim, nn_eval, cfg)` returns
    visit_counts: np.ndarray (1210,) — root-edge visit counts indexed by canonical action.
    root_value:   np.ndarray (6,)    — average leaf value at root (for diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import copy
import io
from contextlib import redirect_stdout

import numpy as np

from .config import MCTSConfig, NUM_ACTIONS, NUM_CELLS, MAX_PLAYERS
from .encoder import (
    encode_state, encode_legal_mask, decode_action, encode_action, slot_of,
)
from .heuristic import heuristic_move_pool
from .shaping import potentials
from .sim import Sim


# A snapshot of a sim that we can restore quickly without deep-copying HexBoard each time.
@dataclass
class SimSnapshot:
    pins_by_colour: Dict[str, List[int]]
    current_turn_index: int
    move_count: int
    move_count_by_colour: Dict[str, int]
    player_status: Dict[str, str]
    game_status: str
    winner: Optional[str]
    terminal_reason: Optional[str]


def snapshot_sim(sim: Sim) -> SimSnapshot:
    return SimSnapshot(
        pins_by_colour={c: [p.axialindex for p in pins] for c, pins in sim.pins_by_colour.items()},
        current_turn_index=sim.current_turn_index,
        move_count=sim.move_count,
        move_count_by_colour=dict(sim.move_count_by_colour),
        player_status=dict(sim.player_status),
        game_status=sim.game_status,
        winner=sim.winner,
        terminal_reason=sim.terminal_reason,
    )


def restore_sim(sim: Sim, snap: SimSnapshot) -> None:
    # Reset cell occupancy
    for cell in sim.board.cells:
        cell.occupied = False
    for c, pins in sim.pins_by_colour.items():
        positions = snap.pins_by_colour[c]
        for p, idx in zip(pins, positions):
            p.axialindex = idx
            sim.board.cells[idx].occupied = True
    sim.current_turn_index = snap.current_turn_index
    sim.move_count = snap.move_count
    sim.move_count_by_colour = dict(snap.move_count_by_colour)
    sim.player_status = dict(snap.player_status)
    sim.game_status = snap.game_status
    sim.winner = snap.winner
    sim.terminal_reason = snap.terminal_reason


def _advance_past_skips(sim: Sim) -> None:
    """If the current to-move has no legal moves, mark them DRAW and advance.
    Loops until a movable player or terminal."""
    while not sim.is_terminal:
        col = sim.current_colour()
        if any(len(p.getPossibleMoves()) > 0 for p in sim.pins_by_colour[col]):
            return
        sim.skip_no_moves()


def _terminal_v_dict(sim: Sim, cfg: MCTSConfig) -> Dict[str, float]:
    """Terminal value per colour. +tw/-tw win/loss; {} for max_moves.

    With shaping on the leaf carries T_c + Phi_c(terminal), matching the
    training target E[T + Phi_final] (Phi_final == Phi(terminal) at a terminal).
    """
    out: Dict[str, float] = {}
    if sim.terminal_reason in ('WIN', 'DRAW_CHAIN') and sim.winner is not None:
        tw = cfg.terminal_weight
        for c in sim.colours:
            out[c] = tw if c == sim.winner else -tw
        if cfg.shaping_enabled:
            phi = potentials(sim, cfg.shaping_scale, cfg.shaping_goal_weight)
            for c in sim.colours:
                out[c] += phi[c]
    return out


class MCTSNode:
    __slots__ = (
        'parent', 'parent_idx', 'to_move', 'turn_order', 'colours',
        'is_terminal', 'terminal_v_dict',
        'is_expanded',
        'legal_actions', 'priors', 'N', 'W', 'children',
    )

    def __init__(self, parent: Optional['MCTSNode'], parent_idx: int,
                 to_move: Optional[str], turn_order: List[str], colours: List[str]):
        self.parent = parent
        self.parent_idx = parent_idx
        self.to_move = to_move
        self.turn_order = turn_order
        self.colours = colours
        self.is_terminal = False
        self.terminal_v_dict: Dict[str, float] = {}
        self.is_expanded = False
        self.legal_actions: Optional[np.ndarray] = None
        self.priors: Optional[np.ndarray] = None
        self.N: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.children: Optional[List[Optional['MCTSNode']]] = None


def _puct_select(node: MCTSNode, c_puct: float) -> int:
    """Return index into node.legal_actions of the next action."""
    sqrt_total = float(np.sqrt(max(1, int(node.N.sum()))))
    # Q = W / N (treat N=0 as Q=0)
    Q = np.where(node.N > 0, node.W / np.maximum(1, node.N), 0.0)
    U = c_puct * node.priors * sqrt_total / (1.0 + node.N)
    return int(np.argmax(Q + U))


def _expand_node(
    node: MCTSNode,
    sim: Sim,
    policy_logits: np.ndarray,
    add_dirichlet_noise: bool,
    cfg: MCTSConfig,
) -> None:
    """Set up node's legal_actions, priors, N, W, children using NN policy."""
    legal_dict = sim.legal_moves(sim.current_colour())
    mask = encode_legal_mask(legal_dict, node.to_move)
    legal_indices = np.flatnonzero(mask)
    if (cfg.restrict_to_pool and legal_indices.size > 0
            and (not cfg.restrict_pool_root_only or node.parent is None)):
        # Restrict the action set to the heuristic's candidate-move pool so
        # multi-player search cannot wander into bad moves. PUCT + the RL value
        # net then search normally within that set. With restrict_pool_root_only
        # this applies only at the root (the move played stays bounded).
        pool = heuristic_move_pool(sim, sim.current_colour(), legal_dict)
        if pool:
            allowed = {encode_action(pid, to_idx, node.to_move)
                       for _, _, pid, to_idx in pool}
            restricted = np.array(
                [a for a in legal_indices if int(a) in allowed], dtype=np.int64)
            if restricted.size > 0:
                legal_indices = restricted
    n_legal = legal_indices.shape[0]
    if n_legal == 0:
        # Should not happen — _advance_past_skips has been called.
        node.is_terminal = True
        node.terminal_v_dict = _terminal_v_dict(sim, cfg)
        return

    # Softmax over legal logits
    logits = policy_logits[legal_indices].astype(np.float64)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()

    if add_dirichlet_noise and cfg.dirichlet_alpha > 0 and cfg.dirichlet_eps > 0:
        noise = np.random.dirichlet([cfg.dirichlet_alpha] * n_legal)
        p = (1 - cfg.dirichlet_eps) * p + cfg.dirichlet_eps * noise

    node.legal_actions = legal_indices.astype(np.int64)
    node.priors = p.astype(np.float32)
    node.N = np.zeros(n_legal, dtype=np.int64)
    node.W = np.zeros(n_legal, dtype=np.float32)
    node.children = [None] * n_legal
    node.is_expanded = True


def _select_leaf(
    root: MCTSNode,
    sim: Sim,
    snap: SimSnapshot,
    cfg: MCTSConfig,
) -> Tuple[List[Tuple[MCTSNode, int]], MCTSNode]:
    """Descend from root to an unexpanded (or terminal) leaf, applying virtual loss along the way.

    Returns (path, leaf). `path` is a list of (node, idx_taken). `leaf` is the final
    node — either unexpanded (caller should NN-evaluate) or terminal.
    sim is mutated to be at the leaf state.
    """
    restore_sim(sim, snap)
    # Root may need skip-advance if first player can't move at game start
    _advance_past_skips(sim)
    if sim.is_terminal and not root.is_terminal:
        # Edge case: starting position is already terminal.
        root.is_terminal = True
        root.terminal_v_dict = _terminal_v_dict(sim, cfg)
        return [], root

    node = root
    path: List[Tuple[MCTSNode, int]] = []
    while node.is_expanded and not node.is_terminal:
        idx = _puct_select(node, cfg.c_puct)
        # Apply virtual loss
        node.N[idx] += 1
        node.W[idx] -= cfg.virtual_loss
        path.append((node, idx))

        action = int(node.legal_actions[idx])
        pid, orig_to = decode_action(action, node.to_move)
        sim.apply_move(pid, orig_to)
        if not sim.is_terminal:
            _advance_past_skips(sim)

        child = node.children[idx]
        if child is None:
            # Create new child
            if sim.is_terminal:
                child = MCTSNode(
                    parent=node, parent_idx=idx,
                    to_move=None, turn_order=sim.turn_order, colours=sim.colours,
                )
                child.is_terminal = True
                child.terminal_v_dict = _terminal_v_dict(sim, cfg)
            else:
                child = MCTSNode(
                    parent=node, parent_idx=idx,
                    to_move=sim.current_colour(), turn_order=sim.turn_order, colours=sim.colours,
                )
            node.children[idx] = child
        node = child
    return path, node


def _backup(path: List[Tuple[MCTSNode, int]], v_dict: Dict[str, float], cfg: MCTSConfig) -> None:
    """Walk the path back, updating each edge's W. N was already incremented during selection (virtual)."""
    for node, idx in path:
        v_for_node_player = v_dict.get(node.to_move, 0.0)
        # Undo virtual loss (+vloss) and add real value
        node.W[idx] += cfg.virtual_loss + v_for_node_player


def _v_vec_to_dict(v_vec: np.ndarray, to_move: str, colours: List[str],
                   phi: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Convert a (6,) canonical-slot value vector to a {colour: value} dict.

    With shaping on, add Phi_c(leaf): the head learns V_head ≈ E[T+Phi_final]-Phi(s),
    so V_head(leaf)+Phi(leaf) recovers the unbiased E[T+Phi_final] for MaxN backup
    rather than a stall-biased quantity.
    """
    out: Dict[str, float] = {}
    for c in colours:
        s = slot_of(c, to_move)
        val = float(v_vec[s])
        if phi is not None:
            val += phi[c]
        out[c] = val
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

NNEval = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def run_search(
    sim_root: Sim,
    nn_eval: NNEval,
    cfg: MCTSConfig = MCTSConfig(),
    add_dirichlet_at_root: bool = False,
) -> Tuple[np.ndarray, np.ndarray, MCTSNode]:
    """Run `cfg.n_sim` PUCT simulations from the current sim state.

    `nn_eval(boards, globs) -> (policy_logits, values)` evaluates batches.
        boards: (B, BOARD_CHANNELS, 121), globs: (B, 8) -> policy_logits (B, 1210), values (B, 6).

    Returns (visit_counts (1210,), root_value (6,), root_node).
    """
    # Take a snapshot so we can keep restoring during search.
    work_sim = sim_root  # we use this as scratch — caller should not rely on its state after
    snap = snapshot_sim(work_sim)

    # Build root
    _advance_past_skips(work_sim)
    if work_sim.is_terminal:
        root = MCTSNode(parent=None, parent_idx=-1, to_move=None,
                        turn_order=work_sim.turn_order, colours=work_sim.colours)
        root.is_terminal = True
        root.terminal_v_dict = _terminal_v_dict(work_sim, cfg)
        return np.zeros(NUM_ACTIONS, dtype=np.int64), np.zeros(MAX_PLAYERS, dtype=np.float32), root

    root_to_move = work_sim.current_colour()
    root_turn_order = list(work_sim.turn_order)
    root_colours = list(work_sim.colours)
    # Refresh snap in case skip advanced state (it shouldn't unless first-mover has no moves at game start)
    snap = snapshot_sim(work_sim)

    root = MCTSNode(parent=None, parent_idx=-1, to_move=root_to_move,
                    turn_order=root_turn_order, colours=root_colours)

    # First expansion: NN-eval root
    board0, glob0 = encode_state(snap.pins_by_colour, root_to_move, root_turn_order, snap.move_count)
    pol0, val0 = nn_eval(board0[None], glob0[None])
    _expand_node(root, work_sim, pol0[0], add_dirichlet_at_root, cfg)
    root_first_value = val0[0].copy()
    if root.is_terminal:
        return np.zeros(NUM_ACTIONS, dtype=np.int64), np.zeros(MAX_PLAYERS, dtype=np.float32), root

    sims_done = 0
    while sims_done < cfg.n_sim:
        batch_paths: List[List[Tuple[MCTSNode, int]]] = []
        batch_leaves: List[MCTSNode] = []
        batch_boards: List[np.ndarray] = []
        batch_globs: List[np.ndarray] = []
        # Collect leaves
        target = min(cfg.batch_leaves, cfg.n_sim - sims_done)
        attempts = 0
        max_attempts = target * 4 + 8  # guard against cases where same path repeats
        while len(batch_leaves) < target and attempts < max_attempts:
            attempts += 1
            path, leaf = _select_leaf(root, work_sim, snap, cfg)
            if leaf.is_terminal:
                _backup(path, leaf.terminal_v_dict, cfg)
                sims_done += 1
                if sims_done >= cfg.n_sim:
                    break
                continue
            # Encode leaf state from sim (currently at leaf state)
            board, glob = encode_state(
                pins_by_colour={c: [p.axialindex for p in pins] for c, pins in work_sim.pins_by_colour.items()},
                to_move=leaf.to_move,
                turn_order=leaf.turn_order,
                move_count=work_sim.move_count,
            )
            batch_paths.append(path)
            batch_leaves.append(leaf)
            batch_boards.append(board)
            batch_globs.append(glob)
        # NN-eval batch
        if batch_leaves:
            B = len(batch_leaves)
            boards_arr = np.stack(batch_boards, axis=0)
            globs_arr = np.stack(batch_globs, axis=0)
            pol, vals = nn_eval(boards_arr, globs_arr)
            for i in range(B):
                # Need to restore sim to leaf state to call _expand_node (which inspects sim.legal_moves)
                # We know that during selection, sim ended at leaf i's state, but since we kept selecting
                # different leaves, sim is at the LAST leaf's state. Re-descend to leaf i.
                # Cheaper: just pass the legal_dict at expansion time. But we don't have it cached.
                # Simpler: re-descend.
                _redescend_to_leaf(root, batch_paths[i], batch_leaves[i], work_sim, snap)
                _expand_node(batch_leaves[i], work_sim, pol[i], add_dirichlet_at_root and (batch_leaves[i] is root), cfg)
                leaf_phi = (potentials(work_sim, cfg.shaping_scale, cfg.shaping_goal_weight)
                            if cfg.shaping_enabled else None)
                v_dict = _v_vec_to_dict(vals[i], batch_leaves[i].to_move,
                                        batch_leaves[i].colours, leaf_phi)
                _backup(batch_paths[i], v_dict, cfg)
                sims_done += 1
                if sims_done >= cfg.n_sim:
                    break

    # Collect root visit counts in (NUM_ACTIONS,) form
    visit_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    if root.is_expanded and root.legal_actions is not None:
        visit_counts[root.legal_actions] = root.N

    # Restore sim to root state so the caller can keep using it.
    restore_sim(work_sim, snap)
    return visit_counts, root_first_value.astype(np.float32), root


def _redescend_to_leaf(root: MCTSNode, path: List[Tuple[MCTSNode, int]], leaf: MCTSNode,
                      sim: Sim, snap: SimSnapshot) -> None:
    """Reset sim to root snap, then walk the path's actions to land at leaf state."""
    restore_sim(sim, snap)
    _advance_past_skips(sim)
    for node, idx in path:
        action = int(node.legal_actions[idx])
        pid, orig_to = decode_action(action, node.to_move)
        sim.apply_move(pid, orig_to)
        if not sim.is_terminal:
            _advance_past_skips(sim)
