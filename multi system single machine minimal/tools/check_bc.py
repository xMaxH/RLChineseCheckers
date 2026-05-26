"""Diagnostic: greedy-policy vs heuristic, compared against MCTS vs heuristic.

Run from the working directory:
    python tools/check_bc.py --ckpt runs/verify_no_early_eval/snapshots/snap_step1050.pt

Answers the question: is BC working (greedy ≈ heuristic) but MCTS breaking things,
or did BC not actually clone the heuristic at all?

Output columns:
  greedy  = argmax(policy | legal) — no MCTS, just the raw network head
  mcts    = MCTS with n_sim sims
  heur    = heuristic vs heuristic baseline

Also reports:
  exact_match_rate = fraction of moves where greedy picks the sampled teacher action
  pool_match_rate  = fraction of moves where greedy picks one of the teacher-best
                     actions. With --match-teacher rollout, this is the rollout
                     teacher's top-scored action set, not the broad soft target.
"""

import argparse
import copy
import math
import random
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from az.config import MCTSConfig, COLOUR_OPPOSITES, SelfPlayConfig
from az.net import AZNet
from az.sim import Sim
from az.encoder import encode_state, encode_legal_mask, decode_action, encode_action
from az.heuristic import heuristic_choose_move, heuristic_move_pool
from az.mcts import run_search
from az.eval import _compute_player_score
from az.selfplay import _finish_with_heuristic, _score_margin, _teacher_policy_target


def load_net(ckpt_path: str, device: str) -> AZNet:
    net = AZNet().to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()
    return net


def greedy_choose(net: AZNet, sim: Sim, colour: str, device: str):
    """One forward pass, argmax over legal moves. Returns (pid, to_idx)."""
    legal = sim.legal_moves(colour)
    return policy_choose(net, sim, colour, device, legal_scope="legal", legal=legal)


def policy_choose(
    net: AZNet,
    sim: Sim,
    colour: str,
    device: str,
    legal_scope: str = "legal",
    legal=None,
    rng: Optional[random.Random] = None,
    max_moves: int = 200,
    rollout_top_k: int = 3,
    rollouts_per_move: int = 2,
):
    """One forward pass, argmax over legal actions or the heuristic-best pool."""
    legal = legal if legal is not None else sim.legal_moves(colour)
    board, glob = encode_state(
        {c: [p.axialindex for p in pins] for c, pins in sim.pins_by_colour.items()},
        colour, sim.turn_order, sim.move_count,
    )
    board_t = torch.from_numpy(board[None]).to(device)
    glob_t  = torch.from_numpy(glob[None]).to(device)
    with torch.no_grad():
        pol_logits, _ = net(board_t, glob_t)
    logits = pol_logits[0].cpu().numpy()

    if legal_scope in ("heuristic-pool", "heuristic-rollout"):
        pool = heuristic_move_pool(sim, colour, legal)
        candidate_idxs = np.array(
            [encode_action(pid, to_idx, colour) for _, _, pid, to_idx in pool],
            dtype=np.int64,
        )
    else:
        mask = encode_legal_mask(legal, colour)
        candidate_idxs = np.flatnonzero(mask)

    if len(candidate_idxs) == 0:
        return heuristic_choose_move(sim, colour, legal)

    sub = logits[candidate_idxs]
    sub -= sub.max()
    probs = np.exp(sub)
    probs /= probs.sum()

    if legal_scope == "heuristic-rollout":
        order = np.argsort(-probs)
        ranked_actions = [int(candidate_idxs[i]) for i in order]
        best_key = None
        best_action = ranked_actions[0]
        for rank, action in enumerate(ranked_actions[:max(1, rollout_top_k)]):
            pid, to_idx = decode_action(action, colour)
            scores = []
            for rollout_idx in range(max(1, rollouts_per_move)):
                s = copy.deepcopy(sim)
                s.apply_move(pid, to_idx)
                seed = (
                    (sim.move_count + 1) * 1_000_003
                    + pid * 9_176
                    + to_idx * 37
                    + rollout_idx * 1_009
                ) & 0xFFFFFFFF
                _finish_with_heuristic(s, random.Random(seed), max_moves=max_moves)
                scores.append(_score_margin(s, colour))
            key = (float(np.mean(scores)), -rank)
            if best_key is None or key > best_key:
                best_key = key
                best_action = action
        return decode_action(best_action, colour)

    chosen_idx = candidate_idxs[int(np.argmax(probs))]
    return decode_action(chosen_idx, colour)


def mcts_choose(net: AZNet, sim: Sim, colour: str, device: str, cfg: MCTSConfig):
    def nn_eval(boards, globs):
        bt = torch.from_numpy(boards).to(device)
        gt = torch.from_numpy(globs).to(device)
        with torch.no_grad():
            pol, val = net(bt, gt)
        return pol.cpu().numpy(), val.cpu().numpy()

    visits, _, _ = run_search(sim, nn_eval, cfg, add_dirichlet_at_root=False)
    if int(visits.sum()) == 0:
        legal = sim.legal_moves(colour)
        return heuristic_choose_move(sim, colour, legal, rng=random.Random(42))
    return decode_action(int(np.argmax(visits)), colour)


def teacher_match_actions(
    sim: Sim,
    colour: str,
    legal,
    rng: random.Random,
    max_moves: int,
    teacher_cfg: Optional[SelfPlayConfig],
):
    """Return (sampled_action, teacher_best_actions) for match diagnostics."""
    if teacher_cfg is None:
        pool = heuristic_move_pool(sim, colour, legal)
        _, _, heur_pid, heur_to = rng.choice(pool)
        sampled_action = encode_action(heur_pid, heur_to, colour)
        best_actions = {
            encode_action(pid, to, colour)
            for _, _, pid, to in pool
        }
        return sampled_action, best_actions

    pid, to, pi = _teacher_policy_target(
        sim, colour, legal, rng, teacher_cfg, max_moves,
    )
    sampled_action = encode_action(pid, to, colour)
    if float(pi.max()) <= 0.0:
        return sampled_action, {sampled_action}
    best_actions = set(int(i) for i in np.flatnonzero(pi >= pi.max() - 1e-7))
    return sampled_action, best_actions


def play_game(
    mode: str,
    net,
    device: str,
    mcts_cfg: MCTSConfig,
    rng: random.Random,
    num_players: int = 2,
    max_moves: int = 200,
    track_match: bool = False,
    teacher_cfg: Optional[SelfPlayConfig] = None,
    greedy_scope: str = "legal",
    rollout_top_k: int = 3,
    rollouts_per_move: int = 2,
    verbose: bool = True,
):
    """Play one game. mode = 'greedy' | 'mcts' | 'heuristic'.
    Candidate plays one seat, heuristic plays the other.
    Returns (won, score_margin, exact_match_rate, pool_match_rate, terminal_reason).
    """
    sim = Sim(num_players, seed=rng.randrange(2**31))
    candidate_colour = sim.colours[0]

    match_total = match_agree = pool_agree = 0

    while not sim.is_terminal:
        col = sim.current_colour()
        legal = sim.legal_moves(col)
        if not any(legal.values()):
            sim.skip_no_moves()
            continue

        if col == candidate_colour and mode != 'heuristic':
            if track_match:
                heur_action, pool_actions = teacher_match_actions(
                    sim, col, legal, rng, max_moves, teacher_cfg,
                )

            if mode == 'greedy':
                pid, to = policy_choose(
                    net, sim, col, device, legal_scope=greedy_scope, legal=legal,
                    rng=rng, max_moves=max_moves,
                    rollout_top_k=rollout_top_k,
                    rollouts_per_move=rollouts_per_move,
                )
            else:
                pid, to = mcts_choose(net, sim, col, device, mcts_cfg)

            if track_match:
                cand_action = encode_action(pid, to, col)
                match_total += 1
                if cand_action == heur_action:
                    match_agree += 1
                if cand_action in pool_actions:
                    pool_agree += 1
        else:
            pid, to = heuristic_choose_move(sim, col, legal, rng=rng)

        sim.apply_move(pid, to)
        if sim.move_count >= max_moves:
            sim.force_max_moves()

    won = (sim.winner == candidate_colour)
    cand_score = _compute_player_score(sim, candidate_colour, sim.move_count_by_colour[candidate_colour])
    opp_scores = [_compute_player_score(sim, c, sim.move_count_by_colour[c])
                  for c in sim.colours if c != candidate_colour]
    margin = cand_score["final_score"] - max(s["final_score"] for s in opp_scores)
    match_rate = (match_agree / match_total) if match_total > 0 else None
    pool_rate = (pool_agree / match_total) if match_total > 0 else None
    return won, margin, match_rate, pool_rate, sim.terminal_reason


def run_eval(
    mode,
    net,
    device,
    mcts_cfg,
    n_games,
    seed,
    num_players=2,
    max_moves=200,
    track_match=False,
    teacher_cfg: Optional[SelfPlayConfig] = None,
    greedy_scope: str = "legal",
    rollout_top_k: int = 3,
    rollouts_per_move: int = 2,
    verbose: bool = True,
):
    rng = random.Random(seed)
    wins, margins, matches, pool_matches, reasons = [], [], [], [], []
    for i in range(n_games):
        won, margin, match_rate, pool_rate, reason = play_game(
            mode, net, device, mcts_cfg, rng, num_players=num_players,
            max_moves=max_moves,
            track_match=track_match, teacher_cfg=teacher_cfg,
            greedy_scope=greedy_scope,
            rollout_top_k=rollout_top_k,
            rollouts_per_move=rollouts_per_move,
        )
        wins.append(won)
        margins.append(margin)
        if match_rate is not None:
            matches.append(match_rate)
        if pool_rate is not None:
            pool_matches.append(pool_rate)
        reasons.append(reason)
        if reason == "WIN":
            status = "CAND_WIN" if won else "LOSS"
        else:
            status = reason
        if verbose:
            print(f"  [{mode}] game {i+1}/{n_games}  {status}  margin={margin:+.0f}"
                  + (f"  exact={match_rate:.2%}" if match_rate is not None else "")
                  + (f"  pool={pool_rate:.2%}" if pool_rate is not None else ""))

    win_rate = sum(wins) / n_games
    mean_margin = float(np.mean(margins))
    mean_match = float(np.mean(matches)) if matches else None
    mean_pool = float(np.mean(pool_matches)) if pool_matches else None
    max_moves_pct = reasons.count("MAX_MOVES") / n_games
    return win_rate, mean_margin, mean_match, mean_pool, max_moves_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--n-games", type=int, default=20)
    ap.add_argument("--num-players", type=int, default=2, choices=(2, 3, 4, 5, 6))
    ap.add_argument("--mcts-sims", type=int, default=200)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-mcts", action="store_true", help="Skip MCTS eval (faster)")
    ap.add_argument("--quiet", action="store_true", help="Only print summary tables, not every game.")
    ap.add_argument("--greedy-scope", choices=("legal", "heuristic-pool", "heuristic-rollout"), default="legal",
                    help="Action set for greedy policy eval. 'legal' is raw policy; "
                         "'heuristic-pool' lets the net only break ties among local heuristic-best moves; "
                         "'heuristic-rollout' adds small rollout lookahead over the net's top pool moves.")
    ap.add_argument("--policy-rollout-top-k", type=int, default=3,
                    help="For --greedy-scope heuristic-rollout, rollout this many NN-ranked pool moves.")
    ap.add_argument("--policy-rollouts-per-move", type=int, default=2,
                    help="For --greedy-scope heuristic-rollout, heuristic rollouts per candidate move.")
    ap.add_argument("--match-teacher", choices=("heuristic", "rollout"), default="heuristic",
                    help="Which teacher to compare greedy policy moves against.")
    ap.add_argument("--teacher-rollouts-per-move", type=int, default=1,
                    help="Rollouts per tied move when --match-teacher rollout is used.")
    ap.add_argument("--teacher-rollout-pool-cap", type=int, default=12,
                    help="Max tied moves to rollout when --match-teacher rollout is used; 0 means no cap.")
    ap.add_argument("--teacher-score-temperature", type=float, default=250.0,
                    help="Rollout teacher soft-target temperature for diagnostics.")
    args = ap.parse_args()

    print(f"\nLoading checkpoint: {args.ckpt}")
    net = load_net(args.ckpt, args.device)
    mcts_cfg = MCTSConfig(n_sim=args.mcts_sims)
    teacher_cfg = None
    if args.match_teacher == "rollout":
        teacher_cfg = SelfPlayConfig(
            heuristic_rollout_targets=True,
            heuristic_rollouts_per_move=args.teacher_rollouts_per_move,
            heuristic_rollout_pool_cap=args.teacher_rollout_pool_cap,
            heuristic_rollout_score_temperature=args.teacher_score_temperature,
        )

    print(
        f"\n=== GREEDY policy vs heuristic "
        f"({args.n_games} games, {args.num_players}p, scope={args.greedy_scope}, match={args.match_teacher}) ==="
    )
    g_wr, g_margin, g_match, g_pool, g_mm = run_eval(
        "greedy", net, args.device, mcts_cfg, args.n_games, args.seed,
        num_players=args.num_players, max_moves=args.max_moves,
        track_match=True, teacher_cfg=teacher_cfg,
        greedy_scope=args.greedy_scope,
        rollout_top_k=args.policy_rollout_top_k,
        rollouts_per_move=args.policy_rollouts_per_move,
        verbose=not args.quiet,
    )

    mcts_wr = mcts_margin = mcts_mm = None
    if not args.skip_mcts:
        print(f"\n=== MCTS ({args.mcts_sims} sims) vs heuristic ({args.n_games} games) ===")
        mcts_wr, mcts_margin, _, _, mcts_mm = run_eval(
            "mcts", net, args.device, mcts_cfg, args.n_games, args.seed,
            num_players=args.num_players, max_moves=args.max_moves,
            verbose=not args.quiet,
        )

    print(f"\n=== HEURISTIC vs heuristic baseline ({args.n_games} games, {args.num_players}p) ===")
    h_wr, h_margin, _, _, h_mm = run_eval(
        "heuristic", None, args.device, mcts_cfg, args.n_games, args.seed,
        num_players=args.num_players, max_moves=args.max_moves,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print(f"{'MODE':<12} {'WIN%':>6} {'MARGIN':>8} {'MAX_MOV%':>10} {'EXACT':>10} {'POOL':>10}")
    print("-" * 60)
    print(f"{'greedy':<12} {g_wr:>6.1%} {g_margin:>8.0f} {g_mm:>10.1%} {g_match:>10.1%} {g_pool:>10.1%}")
    if mcts_wr is not None:
        print(f"{'mcts':<12} {mcts_wr:>6.1%} {mcts_margin:>8.0f} {mcts_mm:>10.1%} {'n/a':>10} {'n/a':>10}")
    print(f"{'heuristic':<12} {h_wr:>6.1%} {h_margin:>8.0f} {h_mm:>10.1%} {'n/a':>10} {'n/a':>10}")
    print("=" * 60)

    print("\n--- INTERPRETATION ---")
    target_label = (
        "rollout teacher-best pool"
        if args.match_teacher == "rollout"
        else "tied heuristic-best pool"
    )
    if g_pool is not None and g_pool > 0.85:
        print(f"BC label fit: GOOD  (greedy is in the {target_label} {g_pool:.1%} of moves)")
        if g_wr < h_wr - 0.15 or g_margin < h_margin - 300:
            print("Game quality: POOR under greedy rollout — this is covariate shift, not a label-fit failure.")
            print("Next: run DAgger from this checkpoint so the policy sees its own off-teacher states.")
        elif mcts_wr is not None and mcts_wr < g_wr - 0.15:
            print("MCTS is the problem — BC is fine but search degrades the policy.")
            print("Try: lower c_puct, fewer sims during BC warmup, or longer DAgger before switching.")
        elif mcts_wr is None:
            print("Run again without --skip-mcts after greedy game quality is acceptable.")
    elif g_pool is not None and g_pool > 0.60:
        print(f"BC label fit: PARTIAL  (greedy is in the {target_label} {g_pool:.1%} of moves)")
        print("Policy has learned the direction but needs more DAgger/BC before search.")
    elif g_match is not None and g_match > 0.85:
        print(f"BC quality: GOOD  (greedy exactly matches sampled heuristic {g_match:.1%} of moves)")
        if mcts_wr is not None and mcts_wr < 0.2:
            print("MCTS is the problem — BC is fine but search breaks the policy.")
            print("Try: lower c_puct, fewer sims during BC warmup, or longer BC before switching.")
        elif mcts_wr is None:
            print("Run again without --skip-mcts to confirm whether MCTS is the problem.")
    else:
        pct = f"{g_pool:.1%}" if g_pool is not None else "unknown"
        print(f"BC label fit: POOR  (greedy is in the {target_label} {pct} of moves)")
        print("The NN hasn't cloned the heuristic. More BC steps / lower entropy bonus needed.")
        print("Check that encoder canonicalization is consistent between BC data and inference.")


if __name__ == "__main__":
    main()
