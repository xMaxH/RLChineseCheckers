"""Evaluate candidate strength.

`eval_vs_heuristic`: candidate plays vs all-heuristic opponents at production
sim count. Returns win rate + score margin distribution (used for the
'eval_score_margin_unique' health check).
"""

from typing import Callable, Dict, List, Optional, Tuple
import random
import math
import numpy as np

from .config import MCTSConfig, COLOUR_OPPOSITES
from .sim import Sim
from .mcts import run_search
from .encoder import decode_action
from .heuristic import heuristic_choose_move


NNEval = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _axial_dist(a, b) -> int:
    dq = abs(a.q - b.q)
    dr = abs(a.r - b.r)
    ds = abs((-a.q - a.r) - (-b.q - b.r))
    return max(dq, dr, ds)


def _compute_player_score(sim: Sim, colour: str, move_count: int) -> Dict[str, float]:
    """Score components matching game.compute_scores (ignoring time)."""
    opp = COLOUR_OPPOSITES[colour]
    pins = sim.pins_by_colour[colour]
    target_idxs = sim.board.axial_of_colour(opp)
    target_cells = [sim.board.cells[i] for i in target_idxs]
    pins_in_goal = sum(1 for p in pins if sim.board.cells[p.axialindex].postype == opp)
    total_dist = 0
    for p in pins:
        if sim.board.cells[p.axialindex].postype != opp:
            best = min(_axial_dist(sim.board.cells[p.axialindex], t) for t in target_cells)
            total_dist += best
    pin_goal_score = pins_in_goal * 100.0
    distance_score = max(0.0, 200.0 - total_dist) if move_count > 0 else 0
    move_score = math.exp(-((move_count - 45) ** 2) / (2 * ((4 if move_count < 45 else 18) ** 2))) if move_count > 0 else 0
    win_bonus = 1000.0 if sim.player_status[colour] == 'WIN' else 0.0
    final_score = move_score + pin_goal_score + distance_score + win_bonus
    return {
        "final_score": final_score,
        "pins_in_goal": pins_in_goal,
        "total_distance": total_dist,
        "pin_goal_score": pin_goal_score,
        "distance_score": distance_score,
        "win_bonus": win_bonus,
        "move_score": move_score,
    }


def eval_vs_heuristic(
    candidate_nn_eval: NNEval,
    num_games: int,
    num_players: int,
    mcts_cfg: MCTSConfig,
    rng: random.Random,
    max_moves: int = 200,
) -> Dict[str, float]:
    """Play `num_games` candidate-vs-heuristic games. Candidate is one slot, others heuristic.

    Returns dict of metrics suitable for health checks.
    """
    wins = 0
    score_margins: List[float] = []
    candidate_finals: List[float] = []
    pins_in_goal_winner: List[int] = []
    finished = 0
    max_moves_count = 0

    for game_idx in range(num_games):
        sim = Sim(num_players, seed=rng.randrange(2**31))
        candidate_colour = sim.colours[game_idx % num_players]  # rotate so each colour plays candidate

        while not sim.is_terminal:
            col = sim.current_colour()
            legal = sim.legal_moves(col)
            if not any(legal.values()):
                sim.skip_no_moves()
                continue
            if col == candidate_colour:
                visits, _, _ = run_search(sim, candidate_nn_eval, mcts_cfg, add_dirichlet_at_root=False)
                if int(visits.sum()) == 0:
                    pid, to = heuristic_choose_move(sim, col, legal, rng=rng)
                else:
                    action = int(np.argmax(visits))
                    pid, to = decode_action(action, col)
            else:
                pid, to = heuristic_choose_move(sim, col, legal, rng=rng)
            sim.apply_move(pid, to)
            if sim.move_count >= max_moves:
                sim.force_max_moves()

        if sim.terminal_reason in ('WIN', 'DRAW_CHAIN'):
            finished += 1
            if sim.winner == candidate_colour:
                wins += 1

        # Score margin between candidate and best opponent
        cand_score = _compute_player_score(sim, candidate_colour, sim.move_count_by_colour[candidate_colour])
        opp_scores = [
            _compute_player_score(sim, c, sim.move_count_by_colour[c])
            for c in sim.colours if c != candidate_colour
        ]
        best_opp = max(s["final_score"] for s in opp_scores)
        score_margins.append(cand_score["final_score"] - best_opp)
        candidate_finals.append(cand_score["final_score"])
        if sim.terminal_reason == 'MAX_MOVES':
            max_moves_count += 1
        elif sim.winner is not None:
            opp = COLOUR_OPPOSITES[sim.winner]
            pig = sum(1 for p in sim.pins_by_colour[sim.winner]
                      if sim.board.cells[p.axialindex].postype == opp)
            pins_in_goal_winner.append(pig)

    return {
        "win_rate": wins / max(1, num_games),
        "wins": wins,
        "finished": finished,
        "max_moves_count": max_moves_count,
        "score_margin_mean": float(np.mean(score_margins)) if score_margins else 0.0,
        "score_margin_unique": int(len(set(round(m, 2) for m in score_margins))),
        "candidate_final_mean": float(np.mean(candidate_finals)) if candidate_finals else 0.0,
        "mean_pins_in_goal_winner": float(np.mean(pins_in_goal_winner)) if pins_in_goal_winner else 0.0,
    }
