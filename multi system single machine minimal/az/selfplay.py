"""Self-play game generation.

For each game:
  1. Pick player count from the curriculum stage.
  2. Assign each colour-slot a role: candidate / heuristic / snapshot.
  3. Run MCTS for candidate and snapshot moves, heuristic_choose_move for heuristic.
  4. Cap moves at max_moves; if exceeded, mark MAX_MOVES (game discarded by caller).
  5. Only candidate-owned moves emit replay samples.
  6. Value targets: +1 for the winner colour, -1 for everyone else (from canonical
     to-move's perspective at each candidate-owned position).
"""

from __future__ import annotations

import random
import copy
import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .config import (
    NUM_ACTIONS, MAX_PLAYERS, MCTSConfig, SelfPlayConfig,
    COLOUR_OPPOSITES,
)
from .sim import Sim
from .encoder import encode_state, decode_action, encode_action, value_target_to_canonical
from .mcts import run_search
from .heuristic import heuristic_choose_move, heuristic_move_pool
from .replay import Sample


# -----------------------------------------------------------------------------
@dataclass
class GameResult:
    samples: List[Sample]
    terminal_reason: str  # 'WIN', 'DRAW_CHAIN', 'MAX_MOVES'
    winner: Optional[str]
    move_count: int
    pins_in_goal_winner: int
    num_players: int
    candidate_colours: List[str]


def _assign_roles(
    colours: List[str],
    rng: random.Random,
    cfg: SelfPlayConfig,
    n_snapshots_available: int,
    bootstrap: bool = False,
) -> Dict[str, str]:
    """Return {colour: role} where role in {'candidate', 'heuristic', 'snapshot', 'random'}.

    Normal mode: candidate / heuristic / snapshot mix from cfg.
    Bootstrap mode: one candidate slot (which plays heuristic + records),
    rest are 'random' (so heuristic actually wins and produces samples).
    """
    if bootstrap:
        # All slots play heuristic; one is marked 'candidate' so we record its positions.
        out = {c: 'heuristic' for c in colours}
        idx = rng.randrange(len(colours))
        out[colours[idx]] = 'candidate'
        return out
    weights = [cfg.candidate_slot_frac, cfg.heuristic_slot_frac, cfg.snapshot_slot_frac]
    if n_snapshots_available == 0:
        weights = [weights[0] + weights[2], weights[1], 0.0]
    roles = ['candidate', 'heuristic', 'snapshot']
    out = {}
    chosen_roles = []
    for c in colours:
        r = rng.choices(roles, weights=weights, k=1)[0]
        chosen_roles.append(r)
        out[c] = r
    if 'candidate' not in chosen_roles:
        idx = rng.randrange(len(colours))
        out[colours[idx]] = 'candidate'
    return out


# -----------------------------------------------------------------------------
NNEval = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _heuristic_policy_target(
    sim: Sim,
    col: str,
    legal: Dict[int, List[int]],
    rng: random.Random,
) -> Tuple[int, int, np.ndarray]:
    """Choose a heuristic move and return a soft target over all tied moves."""
    pool = heuristic_move_pool(sim, col, legal)
    chosen = rng.choice(pool)
    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
    w = 1.0 / float(len(pool))
    for _, _, pid, to_idx in pool:
        pi[encode_action(pid, to_idx, col)] += w
    return chosen[2], chosen[3], pi


def _score_margin(sim: Sim, colour: str) -> float:
    """Score margin for `colour` using the tournament score proxy."""
    from .eval import _compute_player_score
    cand_score = _compute_player_score(
        sim, colour, sim.move_count_by_colour[colour],
    )
    opp_scores = [
        _compute_player_score(sim, c, sim.move_count_by_colour[c])
        for c in sim.colours if c != colour
    ]
    return cand_score["final_score"] - max(s["final_score"] for s in opp_scores)


def _finish_with_heuristic(sim: Sim, rng: random.Random, max_moves: int) -> None:
    """Roll a copied sim to terminal using the fast heuristic policy."""
    while not sim.is_terminal:
        col = sim.current_colour()
        legal = sim.legal_moves(col)
        if not any(legal.values()):
            sim.skip_no_moves()
            continue
        pid, to_idx = heuristic_choose_move(sim, col, legal, rng=rng)
        sim.apply_move(pid, to_idx)
        if sim.move_count >= max_moves and not sim.is_terminal:
            sim.force_max_moves()
            break


def _rollout_seed(
    sim: Sim,
    col: str,
    pid: int,
    to_idx: int,
    rollout_idx: int,
) -> int:
    """Stable per-state seed so rollout-teacher labels are reproducible.

    Python's built-in hash is process-randomized, so use a small explicit digest.
    This keeps the same state from receiving different policy labels when replay
    generation revisits it in a later chunk or worker.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(col.encode("utf-8"))
    h.update(str(sim.move_count).encode("ascii"))
    h.update(("|" + ",".join(sim.turn_order)).encode("utf-8"))
    for colour in sim.turn_order:
        h.update(("|" + colour + ":").encode("utf-8"))
        h.update(bytes(p.axialindex for p in sim.pins_by_colour[colour]))
    h.update(f"|{pid}|{to_idx}|{rollout_idx}".encode("ascii"))
    return int.from_bytes(h.digest(), "little")


def _heuristic_rollout_policy_target(
    sim: Sim,
    col: str,
    legal: Dict[int, List[int]],
    rng: random.Random,
    cfg: SelfPlayConfig,
    max_moves: int,
) -> Tuple[int, int, np.ndarray]:
    """Break local heuristic ties by scoring each tied move with rollouts.

    The old teacher target was uniform over all same-distance-gain moves. That
    fits the local heuristic, but deterministic network argmax can repeatedly
    choose a bad subset of those ties. A short rollout gives the policy a real
    signal for which tied moves preserve long-run game quality.
    """
    pool = sorted(
        heuristic_move_pool(sim, col, legal),
        key=lambda item: (-item[0], -item[1], item[2], item[3]),
    )
    cap = int(cfg.heuristic_rollout_pool_cap)
    if cap > 0 and len(pool) > cap:
        pool = pool[:cap]

    rollouts = max(1, int(cfg.heuristic_rollouts_per_move))
    scored: List[Tuple[float, int, int]] = []
    for _, _, pid, to_idx in pool:
        margins = []
        for rollout_idx in range(rollouts):
            s = copy.deepcopy(sim)
            s.apply_move(pid, to_idx)
            rollout_rng = random.Random(_rollout_seed(sim, col, pid, to_idx, rollout_idx))
            _finish_with_heuristic(s, rollout_rng, max_moves=max_moves)
            margins.append(_score_margin(s, col))
        scored.append((float(np.mean(margins)), pid, to_idx))

    best = max(scored, key=lambda x: x[0])[0]
    best_moves = [
        (pid, to_idx)
        for score, pid, to_idx in scored
        if np.isclose(score, best, rtol=0.0, atol=1e-6)
    ]
    chosen_pid, chosen_to = min(best_moves)

    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
    temp = float(cfg.heuristic_rollout_score_temperature)
    if temp > 0.0 and len(scored) > 1:
        scores = np.array([score for score, _, _ in scored], dtype=np.float64)
        logits = (scores - scores.max()) / temp
        probs = np.exp(np.clip(logits, -60.0, 0.0))
        probs /= probs.sum()
        for p, (_, pid, to_idx) in zip(probs, scored):
            pi[encode_action(pid, to_idx, col)] += float(p)
    else:
        w = 1.0 / float(len(best_moves))
        for pid, to_idx in best_moves:
            pi[encode_action(pid, to_idx, col)] += w
    return chosen_pid, chosen_to, pi


def _teacher_policy_target(
    sim: Sim,
    col: str,
    legal: Dict[int, List[int]],
    rng: random.Random,
    cfg: SelfPlayConfig,
    max_moves: int,
) -> Tuple[int, int, np.ndarray]:
    if cfg.heuristic_rollout_targets:
        return _heuristic_rollout_policy_target(sim, col, legal, rng, cfg, max_moves)
    return _heuristic_policy_target(sim, col, legal, rng)


def _policy_move(
    nn_eval: NNEval,
    sim: Sim,
    col: str,
    legal: Dict[int, List[int]],
    rng: Optional[random.Random] = None,
    temperature: float = 0.0,
):
    """Single forward pass over legal moves; argmax or softmax sample."""
    from .encoder import encode_legal_mask
    board, glob = encode_state(sim.pins_state(), col, sim.turn_order, sim.move_count)
    pol_logits, _ = nn_eval(board[None], glob[None])
    logits = pol_logits[0]
    mask = encode_legal_mask(legal, col)
    legal_idxs = np.flatnonzero(mask)
    if len(legal_idxs) == 0:
        return heuristic_choose_move(sim, col, legal, rng=None)
    sub = logits[legal_idxs].astype(np.float64)
    if temperature > 0.0:
        sub = sub / float(temperature)
        sub -= sub.max()
        probs = np.exp(sub)
        probs /= probs.sum()
        rng = rng or random.Random()
        chosen = rng.choices(list(legal_idxs), weights=list(probs), k=1)[0]
        return decode_action(int(chosen), col)
    sub -= sub.max()
    probs = np.exp(sub); probs /= probs.sum()
    chosen = legal_idxs[int(np.argmax(probs))]
    return decode_action(int(chosen), col)


def play_one_game(
    num_players: int,
    candidate_nn_eval: NNEval,
    snapshot_nn_evals: List[NNEval],   # list of NN evals for each snapshot in the pool
    mcts_cfg: MCTSConfig,
    selfplay_cfg: SelfPlayConfig,
    rng: random.Random,
    max_moves_override: Optional[int] = None,
    candidate_use_heuristic: bool = False,
    dagger: bool = False,
    ignore_moves_per_player: bool = False,
) -> GameResult:
    """Run one self-play game; return GameResult with candidate-owned samples."""
    sim = Sim(num_players, seed=rng.randrange(2**31))
    roles = _assign_roles(
        sim.colours, rng, selfplay_cfg, len(snapshot_nn_evals),
        bootstrap=candidate_use_heuristic,
    )
    candidate_colours = [c for c, r in roles.items() if r == 'candidate']
    snapshot_idx_for_colour = {c: rng.randrange(len(snapshot_nn_evals))
                                for c, r in roles.items() if r == 'snapshot'}

    if max_moves_override is not None:
        max_moves = max_moves_override
    elif selfplay_cfg.moves_per_player > 0 and not ignore_moves_per_player:
        max_moves = selfplay_cfg.moves_per_player * num_players
    else:
        max_moves = (selfplay_cfg.max_moves_2p if num_players == 2
                     else selfplay_cfg.max_moves_multi)

    # Per-position records (only candidate-owned). Each holds the encoded
    # board+glob, the visit-count distribution, and the canonical to-move at the time.
    pending_candidate_positions: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

    while not sim.is_terminal:
        col = sim.current_colour()
        legal = sim.legal_moves(col)
        if not any(legal.values()):
            sim.skip_no_moves()
            continue

        role = roles[col]

        if role == 'heuristic':
            pid, to_idx = heuristic_choose_move(sim, col, legal, rng=rng)
            sim.apply_move(pid, to_idx)
        elif role == 'random':
            movable = [(p, m) for p, m in legal.items() if m]
            pid, mvs = rng.choice(movable)
            to_idx = rng.choice(mvs)
            sim.apply_move(pid, to_idx)
        elif role == 'candidate' and dagger:
            # DAgger: candidate plays its own policy (argmax by default, or
            # sampled if configured) but records the teacher action as the pi
            # label. This closes the covariate-shift gap of pure BC.
            _, _, pi = _teacher_policy_target(
                sim, col, legal, rng, selfplay_cfg, max_moves,
            )
            board, glob = encode_state(sim.pins_state(), col, sim.turn_order, sim.move_count)
            pending_candidate_positions.append((board, glob, pi, col))
            pid, to_idx = _policy_move(
                candidate_nn_eval, sim, col, legal, rng=rng,
                temperature=selfplay_cfg.dagger_policy_temperature,
            )
            sim.apply_move(pid, to_idx)
        elif role == 'candidate' and candidate_use_heuristic:
            # Pure BC bootstrap: candidate plays and records the heuristic action from
            # heuristic-vs-heuristic states (on-policy for heuristic, off-policy for model).
            pid, to_idx, pi = _teacher_policy_target(
                sim, col, legal, rng, selfplay_cfg, max_moves,
            )
            board, glob = encode_state(
                sim.pins_state(), col, sim.turn_order, sim.move_count,
            )
            pending_candidate_positions.append((board, glob, pi, col))
            sim.apply_move(pid, to_idx)
        else:
            # candidate or snapshot — both run MCTS, just with different nets
            nn_eval = candidate_nn_eval if role == 'candidate' else snapshot_nn_evals[snapshot_idx_for_colour[col]]
            visits, _, _ = run_search(
                sim, nn_eval, mcts_cfg,
                add_dirichlet_at_root=(role == 'candidate'),
            )
            visits_sum = int(visits.sum())
            if visits_sum == 0:
                from .heuristic import heuristic_choose_move as _h
                pid, to_idx = _h(sim, col, legal, rng=rng)
            else:
                pi = visits.astype(np.float32) / float(visits_sum)
                if role == 'candidate':
                    board, glob = encode_state(
                        sim.pins_state(), col, sim.turn_order, sim.move_count,
                    )
                    pending_candidate_positions.append((board, glob, pi, col))
                action = int(np.argmax(visits))
                pid, to_idx = decode_action(action, col)
            sim.apply_move(pid, to_idx)

        if sim.move_count >= max_moves and not sim.is_terminal:
            sim.force_max_moves()
            break

    # Compute value targets from terminal outcomes
    samples: List[Sample] = []
    if sim.terminal_reason in ('WIN', 'DRAW_CHAIN'):
        # +1 winner, -1 everyone else
        outcomes = {c: (1.0 if c == sim.winner else -1.0) for c in sim.colours}
        for board, glob, pi, to_move in pending_candidate_positions:
            v_canon = value_target_to_canonical(outcomes, to_move)
            samples.append(Sample(board=board, glob=glob, pi=pi, v=v_canon))
    elif dagger and pending_candidate_positions:
        # DAgger: keep positions even from MAX_MOVES games — train policy head only.
        # NaN value target means the masked MSE in the training loop skips value loss.
        nan_v = value_target_to_canonical({}, sim.colours[0])  # all-NaN
        for board, glob, pi, to_move in pending_candidate_positions:
            samples.append(Sample(board=board, glob=glob, pi=pi, v=nan_v))

    pins_in_goal_winner = 0
    if sim.winner is not None:
        opp = COLOUR_OPPOSITES[sim.winner]
        pins_in_goal_winner = sum(
            1 for p in sim.pins_by_colour[sim.winner]
            if sim.board.cells[p.axialindex].postype == opp
        )

    return GameResult(
        samples=samples,
        terminal_reason=sim.terminal_reason or 'UNKNOWN',
        winner=sim.winner,
        move_count=sim.move_count,
        pins_in_goal_winner=pins_in_goal_winner,
        num_players=num_players,
        candidate_colours=candidate_colours,
    )


def pick_player_count(stage_player_counts, stage_weights, rng: random.Random) -> int:
    return rng.choices(stage_player_counts, weights=stage_weights, k=1)[0]
