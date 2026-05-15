#!/usr/bin/env python3
"""Single-file launch wrapper for AlphaZero training.

Defaults run a 3-hour 2p verify. Override anything via CLI flags.

Examples:
    # Default 3h verify
    python train_run.py

    # Overnight 2p, bigger batches, longer
    python train_run.py --hours 12 --batch-size 4096 --sample-per-step 8192

    # 4p curriculum stage warm-started from a 2p ckpt
    python train_run.py --player-counts 2,3,4 --player-weights 0.4,0.3,0.3 \
        --hours 8 --seed-ckpt runs/verify_3h/snapshots/snap_step120.pt \
        --out runs/stage_2_4p

The first line of <out>/health.jsonl is a `run_meta` record with the
full effective config (so a single jsonl is enough to reproduce a run).
"""

import argparse
import sys
from typing import Tuple

import torch

from az.config import StageSpec, SelfPlayConfig, TrainConfig, MCTSConfig
from az.train import train_one_stage


def _parse_int_list(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _parse_float_list(s: str) -> Tuple[float, ...]:
    return tuple(float(x) for x in s.split(","))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Where + how long
    p.add_argument("--out", default="runs/verify_3h",
                   help="Output directory (health.jsonl + snapshots go here)")
    p.add_argument("--hours", type=float, default=3.0,
                   help="Wall-clock budget in hours")
    p.add_argument("--max-chunks", type=int, default=500,
                   help="Hard cap on chunks regardless of time")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-ckpt", default=None,
                   help="Optional warm-start checkpoint (.pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--name", default=None,
                   help="Stage name written to logs (defaults to dir basename)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Self-play worker processes (0 = single-process). "
                        "Pure BC bootstrap workers stay heuristic-only; NN self-play "
                        "workers hold a model/GPU context. With 6 cores leave 1-2 "
                        "for trainer/OS.")
    p.add_argument("--mcts-sims", type=int, default=200,
                   help="MCTS simulations per move (training+eval+inference). "
                        "Default 200; AlphaZero-paper used 800. Higher = stronger "
                        "but slower per game. 750-1000 typical for ambitious runs.")

    # Curriculum slice
    p.add_argument("--player-counts", type=_parse_int_list, default=(2,),
                   help="Comma-separated player counts to mix (e.g. 2,3,4)")
    p.add_argument("--player-weights", type=_parse_float_list, default=None,
                   help="Comma-separated weights for --player-counts; defaults to uniform")
    p.add_argument("--pass-winrate", type=float, default=0.99,
                   help="Stop early if eval win-rate ≥ this (0.99 = effectively never)")

    # Self-play
    p.add_argument("--games-per-chunk", type=int, default=10)
    p.add_argument("--bootstrap-chunks", type=int, default=8,
                   help="First N chunks use heuristic-clone bootstrap (cold start)")
    p.add_argument("--candidate-frac", type=float, default=0.5)
    p.add_argument("--heuristic-frac", type=float, default=0.25)
    p.add_argument("--snapshot-frac", type=float, default=0.25)
    p.add_argument("--snapshot-pool-size", type=int, default=10)
    p.add_argument("--snapshot-every-train-steps", type=int, default=5)
    p.add_argument("--dagger-in-bootstrap", action="store_true",
                   help="During bootstrap, let the candidate play its greedy policy "
                        "while recording heuristic labels. Default is pure BC: "
                        "candidate plays heuristic, which gives clean winning trajectories.")
    p.add_argument("--dagger-policy-temperature", type=float, default=0.0,
                   help="If >0, DAgger candidate samples legal policy moves with this softmax temperature instead of argmax.")
    p.add_argument("--heuristic-rollout-targets", action="store_true",
                   help="Use short heuristic rollouts to break local-gain teacher ties before writing policy labels.")
    p.add_argument("--heuristic-rollouts-per-move", type=int, default=1,
                   help="Rollouts per candidate teacher move when --heuristic-rollout-targets is enabled.")
    p.add_argument("--heuristic-rollout-pool-cap", type=int, default=12,
                   help="Max tied teacher moves to rollout per labelled position; 0 means no cap.")
    p.add_argument("--heuristic-rollout-score-temperature", type=float, default=250.0,
                   help="Softmax temperature for rollout-score policy targets. "
                        "Use 0 for the old tied-best one-hot target; 200-400 keeps "
                        "near-best moves alive while still preferring winning rollouts.")
    p.add_argument("--max-moves-2p", type=int, default=200,
                   help="Max total moves in a 2p game before MAX_MOVES (game discarded)")
    p.add_argument("--max-moves-multi", type=int, default=300,
                   help="Max total moves in a 3p+ game before MAX_MOVES")
    p.add_argument("--moves-per-player", type=int, default=0,
                   help="If >0, max_moves = N * num_players (overrides --max-moves-2p/--max-moves-multi)")

    # Training
    p.add_argument("--replay-capacity", type=int, default=200_000)
    p.add_argument("--min-samples-to-train", type=int, default=2000)
    p.add_argument("--sample-per-step", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--entropy-bonus", type=float, default=0.01)
    p.add_argument("--min-train-steps", type=int, default=4,
                   help="Always run at least N gradient steps per chunk (regardless of new samples). "
                        "Bump to 20+ to push BC convergence faster during cold start.")
    p.add_argument("--eval-every-steps", "--eval-every", dest="eval_every_steps", type=int, default=40)
    p.add_argument("--eval-games", type=int, default=40)

    args = p.parse_args()

    if args.player_weights is None:
        n = len(args.player_counts)
        weights = tuple(1.0 / n for _ in range(n))
    else:
        if len(args.player_weights) != len(args.player_counts):
            sys.exit(f"--player-weights ({len(args.player_weights)}) must match --player-counts ({len(args.player_counts)})")
        weights = args.player_weights

    sp = SelfPlayConfig(
        games_per_chunk=args.games_per_chunk,
        max_moves_2p=args.max_moves_2p,
        max_moves_multi=args.max_moves_multi,
        moves_per_player=args.moves_per_player,
        candidate_slot_frac=args.candidate_frac,
        heuristic_slot_frac=args.heuristic_frac,
        snapshot_slot_frac=args.snapshot_frac,
        snapshot_pool_size=args.snapshot_pool_size,
        snapshot_every_train_steps=args.snapshot_every_train_steps,
        dagger_in_bootstrap=args.dagger_in_bootstrap,
        dagger_policy_temperature=args.dagger_policy_temperature,
        heuristic_rollout_targets=args.heuristic_rollout_targets,
        heuristic_rollouts_per_move=args.heuristic_rollouts_per_move,
        heuristic_rollout_pool_cap=args.heuristic_rollout_pool_cap,
        heuristic_rollout_score_temperature=args.heuristic_rollout_score_temperature,
    )
    tr = TrainConfig(
        replay_capacity=args.replay_capacity,
        min_samples_to_train=args.min_samples_to_train,
        sample_per_step=args.sample_per_step,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        entropy_bonus=args.entropy_bonus,
        eval_every_steps=args.eval_every_steps,
        eval_games=args.eval_games,
        min_train_steps=args.min_train_steps,
    )
    stage_name = args.name or args.out.rstrip("/").split("/")[-1]
    stage = StageSpec(
        name=stage_name,
        player_counts=tuple(args.player_counts),
        player_count_weights=tuple(weights),
        pass_winrate=args.pass_winrate,
        max_wallclock_hours=args.hours,
        selfplay=sp,
        train=tr,
    )

    device = torch.device(args.device)
    print(f"[launch] stage={stage_name} out={args.out} hours={args.hours} "
          f"player_counts={args.player_counts} weights={weights} "
          f"bootstrap_chunks={args.bootstrap_chunks} device={device}")
    train_one_stage(
        stage, args.out,
        seed_ckpt=args.seed_ckpt,
        device=device,
        rng_seed=args.seed,
        bootstrap_chunks=args.bootstrap_chunks,
        max_chunks=args.max_chunks,
        num_workers=args.num_workers,
        mcts_sims=args.mcts_sims,
    )


if __name__ == "__main__":
    main()
