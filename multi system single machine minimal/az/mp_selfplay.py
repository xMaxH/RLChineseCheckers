"""Multi-process self-play worker pool.

Architecture:
  - main process: owns trainer (optimizer + replay buffer + heartbeat)
  - N persistent worker processes: each lazily holds its own AZNet on GPU/CPU
    only when neural-network self-play is requested.
  - per neural-network chunk: main saves current weights to a tmp file, tells
    workers to reload, then dispatches `games_per_chunk` play tasks across workers.
  - workers return GameResult-like dicts (samples are numpy arrays, picklable).

Why a pool rather than torch.multiprocessing shared tensors:
  - simpler — no synchronization races between training updates and worker reads
  - per-chunk weight reload is ~50ms (12MB ckpt) vs hours of MCTS savings
  - pure heuristic/BC bootstrap workers avoid CUDA model allocation entirely
  - each neural-network worker keeps its own CUDA context (~500MB-1GB on V100;
    4 workers fits comfortably)

Snapshot pool: workers maintain their own copy of the snapshot list. Main
broadcasts a 'load_snapshots' task only when the list changes.
"""

from __future__ import annotations

import dataclasses
import atexit
import multiprocessing as _mp
import os
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Use spawn (required for CUDA processes) — initialised lazily in get_context.
_CTX = None


def _ctx():
    global _CTX
    if _CTX is None:
        _CTX = _mp.get_context("spawn")
    return _CTX


# -----------------------------------------------------------------------------
def _worker_main(worker_id: int, task_q, result_q, init_kwargs: Dict[str, Any]):
    """Long-running worker loop. One process per worker."""
    # Re-import inside the worker (spawn context).
    import torch
    from az.net import AZNet
    from az.selfplay import play_one_game
    from az.config import MCTSConfig, SelfPlayConfig
    from az.inference_server import make_nn_eval

    device = torch.device(init_kwargs["device"])
    net: Optional[AZNet] = None
    nn_eval = None
    snap_nets: List[AZNet] = []
    snap_evals = []

    sp_cfg = SelfPlayConfig(**init_kwargs["sp_cfg_kwargs"])
    mcts_cfg = MCTSConfig(**init_kwargs.get("mcts_cfg_kwargs", {}))
    rng = random.Random(init_kwargs["seed"] * 100003 + worker_id * 7919 + 1)

    def _ensure_main_net():
        nonlocal net, nn_eval
        if net is None:
            net = AZNet().to(device).eval()
            nn_eval = make_nn_eval(net, device)
            with torch.no_grad():
                from az.config import BOARD_CHANNELS, NUM_CELLS
                net(torch.zeros(1, BOARD_CHANNELS, NUM_CELLS, device=device),
                    torch.zeros(1, 8, device=device))
        return net, nn_eval

    def _missing_eval(*_args, **_kwargs):
        raise RuntimeError("worker NN eval requested before model was loaded")

    result_q.put({"worker_id": worker_id, "op": "ready"})

    while True:
        task = task_q.get()
        if task is None:
            break
        try:
            op = task["op"]
            if op == "load_main":
                net, nn_eval = _ensure_main_net()
                sd = torch.load(task["path"], map_location=device, weights_only=True)
                net.load_state_dict(sd)
                net.eval()
                result_q.put({"worker_id": worker_id, "op": "load_main_ok"})
            elif op == "load_snapshots":
                snap_nets.clear()
                snap_evals.clear()
                for p in task["paths"]:
                    sn = AZNet().to(device).eval()
                    sn.load_state_dict(torch.load(p, map_location=device, weights_only=True))
                    snap_nets.append(sn)
                    snap_evals.append(make_nn_eval(sn, device))
                result_q.put({"worker_id": worker_id, "op": "load_snapshots_ok",
                              "n": len(snap_evals)})
            elif op == "clear_models":
                snap_nets.clear()
                snap_evals.clear()
                net = None
                nn_eval = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                result_q.put({"worker_id": worker_id, "op": "clear_models_ok"})
            elif op == "play":
                t0 = time.perf_counter()
                result = play_one_game(
                    num_players=task["n_players"],
                    candidate_nn_eval=nn_eval if nn_eval is not None else _missing_eval,
                    snapshot_nn_evals=snap_evals,
                    mcts_cfg=mcts_cfg,
                    selfplay_cfg=sp_cfg,
                    rng=rng,
                    candidate_use_heuristic=task["bootstrap"],
                    dagger=task.get("dagger", False),
                    max_moves_override=task.get("max_moves_override"),
                    ignore_moves_per_player=task.get("ignore_moves_per_player", False),
                )
                result_q.put({
                    "worker_id": worker_id,
                    "op": "play_done",
                    "task_id": task["task_id"],
                    "samples": result.samples,
                    "terminal_reason": result.terminal_reason,
                    "winner": result.winner,
                    "move_count": result.move_count,
                    "pins_in_goal_winner": result.pins_in_goal_winner,
                    "num_players": result.num_players,
                    "duration_s": time.perf_counter() - t0,
                })
            else:
                result_q.put({"worker_id": worker_id, "op": "error",
                              "error": f"unknown op {op}"})
        except Exception as e:
            result_q.put({
                "worker_id": worker_id, "op": "error",
                "error": str(e), "tb": traceback.format_exc(),
                "task_id": task.get("task_id"),
            })


# -----------------------------------------------------------------------------
class WorkerPool:
    def __init__(
        self,
        num_workers: int,
        sp_cfg,            # SelfPlayConfig instance
        mcts_cfg,          # MCTSConfig instance
        device: str,
        seed: int,
    ):
        ctx = _ctx()
        self.num_workers = num_workers
        self.task_q = ctx.Queue()
        self.result_q = ctx.Queue()
        sp_cfg_kwargs = dataclasses.asdict(sp_cfg)
        mcts_cfg_kwargs = dataclasses.asdict(mcts_cfg)
        init_kwargs = {
            "device": device,
            "sp_cfg_kwargs": sp_cfg_kwargs,
            "mcts_cfg_kwargs": mcts_cfg_kwargs,
            "seed": seed,
        }
        self.workers: List[Any] = []
        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_main, args=(i, self.task_q, self.result_q, init_kwargs),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
        # Wait for all workers to signal ready
        ready_count = 0
        while ready_count < num_workers:
            r = self.result_q.get()
            if r.get("op") == "ready":
                ready_count += 1
            elif r.get("op") == "error":
                raise RuntimeError(f"Worker init failed: {r}")
        self._snapshot_paths: List[str] = []
        self._models_loaded = False
        self._closed = False
        atexit.register(self.shutdown)

    def reload_main(self, ckpt_path: str) -> None:
        for _ in self.workers:
            self.task_q.put({"op": "load_main", "path": ckpt_path})
        ack = 0
        while ack < self.num_workers:
            r = self.result_q.get()
            if r.get("op") == "load_main_ok":
                ack += 1
            elif r.get("op") == "error":
                raise RuntimeError(f"Worker reload failed: {r}")
        self._models_loaded = True

    def reload_snapshots(self, paths: List[str]) -> None:
        if paths == self._snapshot_paths:
            return
        for _ in self.workers:
            self.task_q.put({"op": "load_snapshots", "paths": paths})
        ack = 0
        while ack < self.num_workers:
            r = self.result_q.get()
            if r.get("op") == "load_snapshots_ok":
                ack += 1
            elif r.get("op") == "error":
                raise RuntimeError(f"Worker snapshot load failed: {r}")
        self._snapshot_paths = list(paths)
        if paths:
            self._models_loaded = True

    def clear_models(self) -> None:
        if not self._models_loaded and not self._snapshot_paths:
            return
        for _ in self.workers:
            self.task_q.put({"op": "clear_models"})
        ack = 0
        while ack < self.num_workers:
            r = self.result_q.get()
            if r.get("op") == "clear_models_ok":
                ack += 1
            elif r.get("op") == "error":
                raise RuntimeError(f"Worker model clear failed: {r}")
        self._snapshot_paths = []
        self._models_loaded = False

    def play_chunk(
        self,
        n_games: int,
        n_players_picker,    # callable -> int
        bootstrap: bool,
        max_moves_override: Optional[int],
        progress_cb=None,
        dagger: bool = False,
        ignore_moves_per_player: bool = False,
    ) -> List[Dict[str, Any]]:
        """Dispatch n_games play tasks across workers; collect results in completion order.

        progress_cb(idx, total, result) is called after each completed game.
        """
        for i in range(n_games):
            self.task_q.put({
                "op": "play",
                "task_id": i,
                "n_players": n_players_picker(),
                "bootstrap": bootstrap,
                "dagger": dagger,
                "max_moves_override": max_moves_override,
                "ignore_moves_per_player": ignore_moves_per_player,
            })
        results: List[Dict[str, Any]] = []
        completed = 0
        while completed < n_games:
            r = self.result_q.get()
            if r.get("op") == "play_done":
                results.append(r)
                completed += 1
                if progress_cb is not None:
                    progress_cb(completed, n_games, r)
            elif r.get("op") == "error":
                raise RuntimeError(f"Worker play failed: {r}")
        return results

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        for _ in self.workers:
            try:
                self.task_q.put(None)
            except Exception:
                pass
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        for q in (self.task_q, self.result_q):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
