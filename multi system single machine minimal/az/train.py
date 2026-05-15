"""Alternating self-play + training loop for one curriculum stage."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import os
import random
import subprocess
import sys
import threading
import time
import json
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    NetConfig, MCTSConfig, SelfPlayConfig, TrainConfig, StageSpec,
    NUM_ACTIONS, MAX_PLAYERS,
    verify_stage,
)
from .net import AZNet
from .inference_server import make_nn_eval, load_model
from .replay import ReplayBuffer
from .selfplay import play_one_game, pick_player_count
from .eval import eval_vs_heuristic
from .health import HealthMonitor


# -----------------------------------------------------------------------------
def az_loss(
    policy_logits: torch.Tensor,        # (B, 1210)
    value_pred: torch.Tensor,           # (B, 6)
    pi_target: torch.Tensor,            # (B, 1210)
    v_target: torch.Tensor,             # (B, 6) — NaN at absent slots
    entropy_bonus: float = 0.01,
):
    # Policy: cross-entropy over the *target distribution* (not one-hot — pi is visit dist).
    log_probs = F.log_softmax(policy_logits, dim=-1)
    pol_loss = -(pi_target * log_probs).sum(dim=-1).mean()

    # Value: masked MSE per-slot.
    mask = ~torch.isnan(v_target)
    v_target_clean = torch.where(mask, v_target, torch.zeros_like(v_target))
    se = (value_pred - v_target_clean) ** 2
    se = se * mask.float()
    val_loss = se.sum() / mask.float().sum().clamp(min=1.0)

    # Entropy bonus on policy (encourages diverse policy)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    ent_term = -entropy_bonus * entropy

    total = pol_loss + val_loss + ent_term
    return total, pol_loss.detach(), val_loss.detach(), entropy.detach()


# -----------------------------------------------------------------------------
def train_one_stage(
    stage: StageSpec,
    out_dir: str,
    seed_ckpt: Optional[str] = None,
    device: torch.device = torch.device("cuda"),
    rng_seed: int = 0,
    bootstrap_chunks: int = 2,    # first N chunks use heuristic-clone bootstrap
    max_chunks: int = 200,
    num_workers: int = 0,         # 0 = single-process; >0 = mp pool for self-play
    mcts_sims: int = 200,         # MCTS simulations per move
):
    """Train through one curriculum stage. Returns final checkpoint path."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)
    health = HealthMonitor(log_path=os.path.join(out_dir, "health.jsonl"))

    # ---- Plain-text progress / heartbeat log (separate from health.jsonl) ----
    progress_path = os.path.join(out_dir, "progress.log")
    progress_f = open(progress_path, "a", buffering=1)  # line-buffered

    def plog(msg: str):
        line = f"[{_dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        progress_f.write(line + "\n")

    # Shared state that the heartbeat thread reads from.
    hb_state = {
        "started_perf": time.perf_counter(),
        "chunk_idx": 0,
        "bootstrap": False,
        "game_idx": 0,
        "games_total": 0,
        "train_step": 0,
        "replay_size": 0,
        "last_pol_loss": None,
        "last_val_loss": None,
        "last_chunk_wins": None,
        "phase": "init",   # 'selfplay' / 'train' / 'eval'
    }
    hb_stop = threading.Event()

    def heartbeat_loop():
        period = 120.0
        while not hb_stop.wait(period):
            elapsed_h = (time.perf_counter() - hb_state["started_perf"]) / 3600.0
            pol = hb_state["last_pol_loss"]
            val = hb_state["last_val_loss"]
            pol_s = f"{pol:.2f}" if pol is not None else "—"
            val_s = f"{val:.2f}" if val is not None else "—"
            mode = "BS" if hb_state["bootstrap"] else "MCTS"
            plog(
                f"[HB] elapsed={elapsed_h:.2f}h chunk={hb_state['chunk_idx']} mode={mode} "
                f"phase={hb_state['phase']} game={hb_state['game_idx']}/{hb_state['games_total']} "
                f"step={hb_state['train_step']} replay={hb_state['replay_size']} "
                f"last_pol={pol_s} last_val={val_s} last_wins={hb_state['last_chunk_wins']}"
            )

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    # ---- Run-config record (one line, first in health.jsonl) ----
    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj
    run_meta = {
        "run_meta": True,
        "started_iso": _dt.datetime.now().isoformat(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "device": str(device),
        "seed_ckpt": seed_ckpt,
        "rng_seed": rng_seed,
        "bootstrap_chunks": bootstrap_chunks,
        "max_chunks": max_chunks,
        "num_workers": num_workers,
        "mcts_sims": mcts_sims,
        "stage_config": _to_dict(stage),
    }
    with open(os.path.join(out_dir, "health.jsonl"), "a") as f:
        f.write(json.dumps(run_meta, default=str) + "\n")

    rng = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    net = AZNet().to(device)
    if seed_ckpt is not None and os.path.exists(seed_ckpt):
        net.load_state_dict(torch.load(seed_ckpt, map_location=device, weights_only=True))
        print(f"[train] warm-started from {seed_ckpt}")
    else:
        print(f"[train] starting from random init")

    optim = torch.optim.AdamW(
        net.parameters(),
        lr=stage.train.lr,
        weight_decay=stage.train.weight_decay,
    )

    candidate_nn_eval = make_nn_eval(net, device)
    snapshot_nets: List[AZNet] = []
    snapshot_evals = []

    replay = ReplayBuffer(stage.train.replay_capacity)

    mcts_cfg = MCTSConfig(n_sim=mcts_sims)
    sp_cfg = stage.selfplay
    train_cfg = stage.train

    # ---- Print run config to progress.log (human-readable mirror of run_meta) ----
    plog("=" * 72)
    plog(f"RUN CONFIG  stage={stage.name}  out={out_dir}")
    plog(f"  argv: {' '.join(sys.argv)}")
    plog(f"  device={device}  seed={rng_seed}  seed_ckpt={seed_ckpt}")
    plog(f"  bootstrap_chunks={bootstrap_chunks}  max_chunks={max_chunks}  "
         f"num_workers={num_workers}  mcts_sims={mcts_sims}")
    plog(f"  player_counts={list(stage.player_counts)}  "
         f"weights={list(stage.player_count_weights)}  hours={stage.max_wallclock_hours}")
    plog(f"  selfplay: games_per_chunk={sp_cfg.games_per_chunk} "
         f"candidate={sp_cfg.candidate_slot_frac} heuristic={sp_cfg.heuristic_slot_frac} "
         f"snapshot={sp_cfg.snapshot_slot_frac} "
         f"dagger_in_bootstrap={sp_cfg.dagger_in_bootstrap} "
         f"dagger_temp={sp_cfg.dagger_policy_temperature} "
         f"rollout_targets={sp_cfg.heuristic_rollout_targets} "
         f"rollouts_per_move={sp_cfg.heuristic_rollouts_per_move} "
         f"rollout_pool_cap={sp_cfg.heuristic_rollout_pool_cap} "
         f"rollout_score_temp={sp_cfg.heuristic_rollout_score_temperature} "
         f"snap_pool={sp_cfg.snapshot_pool_size} snap_every={sp_cfg.snapshot_every_train_steps} "
         f"max_moves_2p={sp_cfg.max_moves_2p} max_moves_multi={sp_cfg.max_moves_multi} "
         f"moves_per_player={sp_cfg.moves_per_player}")
    plog(f"  train: replay={train_cfg.replay_capacity} "
         f"min_to_train={train_cfg.min_samples_to_train} "
         f"sample_per_step={train_cfg.sample_per_step} "
         f"batch={train_cfg.batch_size} lr={train_cfg.lr} "
         f"weight_decay={train_cfg.weight_decay} entropy_bonus={train_cfg.entropy_bonus} "
         f"min_train_steps={train_cfg.min_train_steps} "
         f"eval_every={train_cfg.eval_every_steps} eval_games={train_cfg.eval_games}")
    plog("=" * 72)

    # ---- Multi-process worker pool (optional) ----
    pool = None
    snapshot_paths_on_disk: List[str] = []
    weights_tmp_path = os.path.join(out_dir, "_current_weights.pt")
    if num_workers > 0:
        from .mp_selfplay import WorkerPool
        plog(f"Spawning {num_workers} self-play worker processes ...")
        pool = WorkerPool(
            num_workers=num_workers, sp_cfg=sp_cfg, mcts_cfg=mcts_cfg,
            device=str(device), seed=rng_seed,
        )
        plog(f"All {num_workers} workers ready.")

    chunk_idx = 0
    train_step = 0
    wallclock_start = time.perf_counter()
    deadline = wallclock_start + stage.max_wallclock_hours * 3600.0

    print(f"[stage {stage.name}] start. budget={stage.max_wallclock_hours}h")

    has_trained_at_least_once = False
    consecutive_empty_mcts_chunks = 0
    last_eval_step = 0
    stage_result_path: Optional[str] = None
    while chunk_idx < max_chunks:
        if time.perf_counter() > deadline:
            print(f"[stage {stage.name}] time budget reached at chunk {chunk_idx}")
            break
        if health.should_stop():
            print(f"[stage {stage.name}] health killed: {health.kill_reason}")
            break

        chunk_t0 = time.perf_counter()
        # Bootstrap if any of:
        #   (a) we asked for it via bootstrap_chunks
        #   (b) we still haven't trained once and replay is too small to train
        #   (c) the candidate's MCTS-driven games stopped producing samples
        #       (NN not yet strong enough — fall back to BC to feed it more)
        bootstrap = (
            chunk_idx < bootstrap_chunks
            or (not has_trained_at_least_once and len(replay) < train_cfg.min_samples_to_train)
            or consecutive_empty_mcts_chunks >= 2
        )
        bootstrap_dagger = bootstrap and sp_cfg.dagger_in_bootstrap
        chunk_metrics = {
            "chunk": chunk_idx,
            "stage": stage.name,
            "bootstrap": bootstrap,
            "wallclock_h": (time.perf_counter() - wallclock_start) / 3600.0,
            "iso_time": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        hb_state["chunk_idx"] = chunk_idx
        hb_state["bootstrap"] = bootstrap
        hb_state["phase"] = "selfplay"
        hb_state["games_total"] = sp_cfg.games_per_chunk
        hb_state["game_idx"] = 0
        mode_name = "DAGGER" if bootstrap_dagger else ("BS" if bootstrap else "MCTS")
        plog(f"chunk {chunk_idx} START mode={mode_name} "
             f"replay={len(replay)} train_step={train_step}")

        # ---- Self-play ----
        kept_samples = 0
        discarded_samples = 0
        wins = 0
        max_moves = 0
        chunk_pins_in_goal = []
        chunk_terminal_reasons = {"WIN": 0, "DRAW_CHAIN": 0, "MAX_MOVES": 0, "UNKNOWN": 0}

        # Bootstrap should use the normal stage cap even if a later RL stage asks
        # for a high moves_per_player budget. Long random-policy bootstrap games
        # flood replay with unfinished positions and slow down useful training.
        max_moves_override = None

        if pool is not None:
            # Pure BC bootstrap is heuristic-only. Do not load CUDA models in
            # workers until DAgger/MCTS actually needs neural-network eval.
            worker_needs_model = (not bootstrap) or bootstrap_dagger
            if worker_needs_model:
                torch.save(net.state_dict(), weights_tmp_path)
                pool.reload_main(weights_tmp_path)
                pool.reload_snapshots([] if bootstrap else snapshot_paths_on_disk)
            else:
                pool.clear_models()

            def _picker():
                return pick_player_count(stage.player_counts, stage.player_count_weights, rng)

            def _on_done(idx: int, total: int, r: Dict):
                hb_state["game_idx"] = idx
                hb_state["replay_size"] = len(replay) + 0  # will be updated after aggregation
                shown_kept = (
                    len(r["samples"])
                    if r["terminal_reason"] in ("WIN", "DRAW_CHAIN") or (bootstrap_dagger and r["samples"])
                    else 0
                )
                plog(f"  chunk {chunk_idx} game {idx}/{total} "
                     f"n={r['num_players']}p term={r['terminal_reason']} "
                     f"winner={r['winner']} moves={r['move_count']} "
                     f"kept={shown_kept} "
                     f"worker={r['worker_id']} ({r['duration_s']:.1f}s)")

            results = pool.play_chunk(
                n_games=sp_cfg.games_per_chunk,
                n_players_picker=_picker,
                bootstrap=bootstrap,
                max_moves_override=max_moves_override,
                progress_cb=_on_done,
                dagger=bootstrap_dagger,
                ignore_moves_per_player=bootstrap,
            )
            for r in results:
                term = r["terminal_reason"]
                chunk_terminal_reasons[term] = chunk_terminal_reasons.get(term, 0) + 1
                if term in ("WIN", "DRAW_CHAIN"):
                    wins += 1
                    added = replay.add_game(r["samples"])
                    kept_samples += added
                    chunk_pins_in_goal.append(r["pins_in_goal_winner"])
                elif bootstrap_dagger and r["samples"]:
                    # DAgger: keep policy-only samples even from MAX_MOVES games
                    added = replay.add_game(r["samples"])
                    kept_samples += added
                else:
                    discarded_samples += len(r["samples"])
                if term == "MAX_MOVES":
                    max_moves += 1
            hb_state["replay_size"] = len(replay)
        else:
            for g in range(sp_cfg.games_per_chunk):
                n_players = pick_player_count(stage.player_counts, stage.player_count_weights, rng)
                game_t0 = time.perf_counter()
                result = play_one_game(
                    num_players=n_players,
                    candidate_nn_eval=candidate_nn_eval,
                    snapshot_nn_evals=snapshot_evals,
                    mcts_cfg=mcts_cfg,
                    selfplay_cfg=sp_cfg,
                    rng=rng,
                    candidate_use_heuristic=bootstrap,
                    dagger=bootstrap_dagger,
                    max_moves_override=max_moves_override,
                    ignore_moves_per_player=bootstrap,
                )
                chunk_terminal_reasons[result.terminal_reason] = chunk_terminal_reasons.get(result.terminal_reason, 0) + 1
                if result.terminal_reason in ('WIN', 'DRAW_CHAIN'):
                    wins += 1
                    added = replay.add_game(result.samples)
                    kept_samples += added
                    chunk_pins_in_goal.append(result.pins_in_goal_winner)
                elif bootstrap_dagger and result.samples:
                    added = replay.add_game(result.samples)
                    kept_samples += added
                else:
                    discarded_samples += len(result.samples)
                if result.terminal_reason == 'MAX_MOVES':
                    max_moves += 1
                game_dt = time.perf_counter() - game_t0
                hb_state["game_idx"] = g + 1
                hb_state["replay_size"] = len(replay)
                shown_kept = (
                    len(result.samples)
                    if result.terminal_reason in ("WIN", "DRAW_CHAIN") or (bootstrap_dagger and result.samples)
                    else 0
                )
                plog(f"  chunk {chunk_idx} game {g+1}/{sp_cfg.games_per_chunk} "
                     f"n={n_players}p term={result.terminal_reason} "
                     f"winner={result.winner} moves={result.move_count} "
                     f"kept={shown_kept} "
                     f"({game_dt:.1f}s)")

        chunk_metrics.update({
            "chunk_wins": wins,
            "chunk_max_moves": max_moves,
            "chunk_kept_samples": kept_samples,
            "chunk_discarded_samples": discarded_samples,
            "replay_size": len(replay),
            "terminal_reasons": chunk_terminal_reasons,
            "mean_pins_in_goal_winner": float(np.mean(chunk_pins_in_goal)) if chunk_pins_in_goal else 0.0,
            "selfplay_seconds": time.perf_counter() - chunk_t0,
        })

        # Track empty MCTS chunks for the bootstrap-fallback heuristic.
        if (not bootstrap) and kept_samples == 0:
            consecutive_empty_mcts_chunks += 1
        else:
            consecutive_empty_mcts_chunks = 0

        # ---- Train ----
        if len(replay) >= train_cfg.min_samples_to_train:
            train_t0 = time.perf_counter()
            net.train()
            hb_state["phase"] = "train"
            # Run at least train_cfg.min_train_steps per chunk; otherwise scale with new samples.
            n_steps = max(train_cfg.min_train_steps, kept_samples // train_cfg.sample_per_step)
            plog(f"  chunk {chunk_idx} TRAIN n_steps={n_steps} replay={len(replay)}")
            for _ in range(n_steps):
                batch = replay.sample(train_cfg.batch_size, np_rng)
                boards = torch.from_numpy(batch["boards"]).to(device, non_blocking=True)
                globs = torch.from_numpy(batch["globs"]).to(device, non_blocking=True)
                pis = torch.from_numpy(batch["pis"]).to(device, non_blocking=True)
                vs = torch.from_numpy(batch["vs"]).to(device, non_blocking=True)
                pol_logits, val = net(boards, globs)
                loss, pol_l, val_l, ent = az_loss(pol_logits, val, pis, vs, entropy_bonus=train_cfg.entropy_bonus)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optim.step()
                train_step += 1
                has_trained_at_least_once = True
                hb_state["train_step"] = train_step
                hb_state["last_pol_loss"] = float(pol_l.item())
                hb_state["last_val_loss"] = float(val_l.item())
            chunk_metrics.update({
                "train_steps_in_chunk": n_steps,
                "policy_loss": float(pol_l.item()),
                "value_loss": float(val_l.item()),
                "policy_entropy": float(ent.item()),
                "train_seconds": time.perf_counter() - train_t0,
                "train_step_total": train_step,
            })
            net.eval()
            plog(f"  chunk {chunk_idx} TRAIN done step={train_step} "
                 f"pol={chunk_metrics['policy_loss']:.3f} val={chunk_metrics['value_loss']:.3f}")

            # ---- Push snapshot ----
            if train_step % sp_cfg.snapshot_every_train_steps == 0:
                # The main process only needs live snapshot modules in
                # single-process MCTS mode. Worker-pool runs reload snapshots
                # from disk, and bootstrap does not consult snapshots.
                if pool is None and not bootstrap:
                    snap = AZNet().to(device)
                    snap.load_state_dict(net.state_dict())
                    snap.eval()
                    snapshot_nets.append(snap)
                    snapshot_evals.append(make_nn_eval(snap, device))
                    # FIFO trim
                    trimmed_snapshot = False
                    while len(snapshot_nets) > sp_cfg.snapshot_pool_size:
                        snapshot_nets.pop(0)
                        snapshot_evals.pop(0)
                        trimmed_snapshot = True
                    if trimmed_snapshot and device.type == "cuda":
                        torch.cuda.empty_cache()
                # Save snapshot to disk
                snap_path = os.path.join(out_dir, "snapshots", f"snap_step{train_step}.pt")
                torch.save(net.state_dict(), snap_path)
                chunk_metrics["snapshot_path"] = snap_path
                # Maintain the disk-snapshot list for worker reload
                snapshot_paths_on_disk.append(snap_path)
                while len(snapshot_paths_on_disk) > sp_cfg.snapshot_pool_size:
                    snapshot_paths_on_disk.pop(0)

            # ---- Eval ----
            if (not bootstrap) and (train_step - last_eval_step >= train_cfg.eval_every_steps):
                hb_state["phase"] = "eval"
                plog(f"  chunk {chunk_idx} EVAL n={train_cfg.eval_games} games...")
                eval_t0 = time.perf_counter()
                eval_metrics = eval_vs_heuristic(
                    candidate_nn_eval,
                    num_games=train_cfg.eval_games,
                    num_players=stage.player_counts[0],   # primary player count for eval
                    mcts_cfg=mcts_cfg,
                    rng=rng,
                )
                chunk_metrics["eval"] = eval_metrics
                health.record_eval(eval_metrics)
                last_eval_step = train_step
                plog(f"  chunk {chunk_idx} EVAL done win_rate={eval_metrics['win_rate']:.2f} "
                     f"margin={eval_metrics['score_margin_mean']:.0f} "
                     f"unique={eval_metrics['score_margin_unique']} "
                     f"({time.perf_counter() - eval_t0:.1f}s)")
                # Save best
                best_path = os.path.join(out_dir, "best.pt")
                if eval_metrics["win_rate"] >= stage.pass_winrate:
                    torch.save(net.state_dict(), best_path)
                    print(f"[stage {stage.name}] PASS at step {train_step}, win_rate={eval_metrics['win_rate']:.2f}")
                    chunk_metrics["chunk_seconds"] = time.perf_counter() - chunk_t0
                    health.record_chunk(chunk_metrics)
                    stage_result_path = best_path
                    break
        else:
            chunk_metrics["train_skipped"] = f"replay {len(replay)} < min {train_cfg.min_samples_to_train}"

        chunk_metrics["chunk_seconds"] = time.perf_counter() - chunk_t0
        health.record_chunk(chunk_metrics)
        hb_state["last_chunk_wins"] = wins
        hb_state["replay_size"] = len(replay)
        plog(f"chunk {chunk_idx} END wins={wins}/{sp_cfg.games_per_chunk} "
             f"kept={kept_samples} replay={len(replay)} "
             f"({chunk_metrics['chunk_seconds']:.1f}s)")
        print(f"[chunk {chunk_idx}] " + json.dumps({k: v for k, v in chunk_metrics.items() if k != "terminal_reasons"}, default=str))
        chunk_idx += 1

    # End of stage — save final
    final_path = os.path.join(out_dir, "final.pt")
    torch.save(net.state_dict(), final_path)
    best_path = os.path.join(out_dir, "best.pt")
    if stage_result_path is None:
        # If no eval/pass selected a true best checkpoint, make best.pt an
        # alias of the final model. This prevents stale best.pt files when an
        # output directory is reused for pure BC/DAgger runs with eval disabled.
        torch.save(net.state_dict(), best_path)
    elif not os.path.exists(best_path):
        torch.save(net.state_dict(), best_path)
    hb_stop.set()
    progress_f.close()
    if pool is not None:
        pool.shutdown()
    return stage_result_path or best_path


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="verify")
    p.add_argument("--out", type=str, default="runs/verify")
    p.add_argument("--seed-ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap-chunks", type=int, default=2)
    p.add_argument("--max-chunks", type=int, default=200)
    args = p.parse_args()

    if args.stage == "verify":
        stage = verify_stage()
    else:
        from . import config as cfg_mod
        stage = getattr(cfg_mod, args.stage + "_stage_2p", None) or getattr(cfg_mod, "stage_" + args.stage)
        if callable(stage):
            stage = stage()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_one_stage(
        stage, args.out, seed_ckpt=args.seed_ckpt, device=device,
        rng_seed=args.seed, bootstrap_chunks=args.bootstrap_chunks,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
