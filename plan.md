# AlphaZero-Style Chinese Checkers Tournament Agent — Plan

## Context

The prior PPO + ad-hoc-AlphaZero attempts collapsed into pin-shuffling (one pin races, others sit) and stall-for-margin behaviour. The training tree has been wiped; only base game files remain. We have ~14 days until the tournament (2026-05-22), a V100, 6 cores and 96GB RAM, and a hard **2s/turn** inference budget at tournament time. The user wants an iterative AlphaZero rebuild: validate the pipeline on a short run, then commit to a multi-day run, stepping curriculum 2p → 3p → 4p → 5p → 6p with a real pass criterion at every stage.

The plan below incorporates every "must respect" lesson from [CLAUDE.md](RLChineseCheckers/multi%20system%20single%20machine%20minimal/../CLAUDE.md): no score-margin adjudication, drop max_moves games, never pure self-play, identical sim count across train/eval/tournament, large batches, full curriculum.

## Working directory

`/home/coder/IKT460/RLChineseCheckers/multi system single machine minimal/` (always quote — has spaces).

## High-level pipeline

```
[selfplay workers] --states--> [GPU inference server] --(pi,v)--> [workers]
        |                                                            |
        v                                                            v
   raw games                                                   games (filtered)
        |                                                            |
        +-----------------> [replay buffer (FIFO)] <-----------------+
                                     |
                                     v
                            [trainer (alternating)]
                                     |
                          checkpoints + snapshots
                                     |
                                     v
                              [eval @ N_sim=200]
                                     |
                                     v
                             runs/best.pt → player.py
```

Alternation, **not** background training: generate self-play chunk → run K train steps → eval/snapshot → repeat. Background training was a known collapse mode last time.

## File layout (new files only)

```
multi system single machine minimal/
├── az/
│   ├── __init__.py
│   ├── config.py           # per-stage hyperparameters (verify / overnight / multiday)
│   ├── encoder.py          # board↔tensor; canonicalization rotations (6 LUTs)
│   ├── sim.py              # in-process simulator; reuses HexBoard + Pin
│   ├── heuristic.py        # greedy pin-racer baseline
│   ├── net.py              # ResNet trunk + MaxN value head
│   ├── mcts.py             # batched PUCT MCTS w/ virtual loss + MaxN backup
│   ├── inference_server.py # single GPU process; batches NN evals across workers
│   ├── selfplay.py         # 1 worker process; rolls games and emits samples
│   ├── replay.py           # FIFO buffer; max_moves filter at insertion
│   ├── train.py            # AdamW loop, alternating with selfplay
│   ├── eval.py             # candidate vs heuristic / snapshot at N_sim=200
│   ├── health.py           # writes health.jsonl; raises kill-flag on bad signals
│   └── curriculum.py       # drives 2p→3p→4p→5p→6p with warm-start
├── alphazero_method.py     # imported by player.py; loads runs/best.pt
├── runs/
│   └── best.pt             # tournament artifact (symlink to current best ckpt)
└── tests/
    └── test_sim_parity.py  # 50 seeded games via TCP vs in-process; assert equality
```

`player.py` is **not modified** — its existing PLAYING LOGIC block already imports and calls `choose_move_alphazero`. We only supply `alphazero_method.py` and the .pt artifact.

## State / action / value representation

**Board tensor**: `(C=14, N=121)` float32 + global vec `(g=8)`.
- 6 channels: pin presence per colour-slot (canonicalized so to-move = slot 0, opposite = slot 1).
- 6 channels: zone postype one-hot.
- 1 channel: empty flag.
- 1 channel: broadcast scalar `pins_remaining_to_score`.
- Global: `[N_players_one_hot(5), move_count_norm, my_pins_in_goal_norm, my_turn_idx_norm]`.

**Canonicalization**: precompute six 121→121 axial-rotation LUTs in `encoder.py` so the to-move player is always in slot 0. Unit-test `decode(encode(state)) == state` for 1000 random configs across all 6 to-move colours.

**Action space**: flat `1210 = 10 pins × 121 destinations`, action = `pin_id * 121 + to_index`. Illegal actions are masked with `-1e9` before softmax. MCTS only expands legal children.

**Value head**: 6-vector `tanh` output in colour-slot space (not absolute colour). Loss masks absent players to zero. MCTS reads `v[to_move_slot]`.

## Network architecture

ResNet over the cell axis (1-D — hex adjacency is irregular; padding a 2-D conv wastes compute).

- Stem: `Conv1d(14→192, k=1)` + learned 192×121 positional embedding.
- Trunk: 8 residual blocks `Conv1d(192,192,k=1) → ReLU → Conv1d → +residual → ReLU`. Every other block adds a learned 121×121 mixing matrix to propagate info across cells.
- Policy head: `Conv1d(192→16, k=1)` → flatten → `Linear(16·121 → 1210)`.
- Value head: global mean pool → concat global vec → `MLP(200 → 128 → 6) → tanh`.

~2.4M params. V100 batched forward (B=32, 200 sims): ~150-300ms. End-to-end move under 500ms — well within 2s.

## MCTS

Standard PUCT.

- `c_puct = 1.5`
- Root Dirichlet noise `α=0.3, ε=0.25` — **only during self-play**, disabled at eval and tournament.
- Virtual loss `vloss = 1.0` — batch ~32 leaves per NN call.
- **`N_sim = 200` everywhere** (training, eval, tournament). This is the lesson-4 hard rule. Mismatched sim counts hid policy collapse last run.
- Multi-player MaxN backup: leaf returns `v[6]`; on the way back, each node updates W with the value component for *that node's owner*.

Tournament inference is fixed `N_sim=200`, not adaptive.

## Self-play

**In-process simulator (`az/sim.py`)** — required. Going through TCP would cost 12k+ RPCs/game and dominate. Reuse `HexBoard` and `Pin` directly (read-only imports allowed). Must mirror `game.py`:
- WIN: all 10 pins on cells with `postype == colour_opposites[colour]` ([game.py:188](RLChineseCheckers/multi%20system%20single%20machine%20minimal/game.py#L188))
- DRAW: no legal moves any pin ([game.py:192](RLChineseCheckers/multi%20system%20single%20machine%20minimal/game.py#L192))
- N-1 DRAWs ⇒ remaining player WINs ([game.py:477-487](RLChineseCheckers/multi%20system%20single%20machine%20minimal/game.py#L477-L487))
- Turn order: `COLOUR_ORDER = ['red','lawn green','yellow','blue','gray0','purple']` rotated to first joiner

Synthetic `max_moves = 200` cap (300 for 4-6p). **All such games are dropped at insertion to the replay buffer** — no value targets from non-WIN terminations (lesson 1, 2).

**Workers**: 4 self-play processes. Inference server (1 process, owns the V100) batches NN requests across workers. Trainer (1 process) reads from replay buffer. Total 6 processes ↔ 6 cores.

**Opponent slot mix per game** (lesson 3):
- 50% candidate (only candidate moves emit training samples)
- 25% heuristic pin-racer
- 25% past-snapshot from rolling pool of 10 (push every 5 train steps)
- At least one candidate slot per game.

**Heuristic** (`az/heuristic.py`): for each legal move, score = decrease in sum of axial distances of own pins to nearest opposite-zone cell. Pick max gain; deterministic tiebreak by `(pin_id, to_index)`.

## Replay + training

| Stage | Replay | Min before train | Sample/step | Batch | LR |
|---|---|---|---|---|---|
| verify | 200k | 50k | 8192 | 4096 | 1e-4 |
| overnight | 1.0M | 200k | 8192 | 4096 | 1e-4 |
| multiday | 4.0M | 800k | 65536 | 32768 | 3e-5 cosine |

**Loss**: `L = CE(policy) + 1.0 * MSE(value, masked) ` ; AdamW weight_decay=1e-4. Policy CE only over legal actions. Value MSE masked to present players. Add small per-policy entropy bonus (coef 0.01) as belt-and-suspenders against single-pin shuffling.

**Targets**:
- π_target = visit-count distribution at root
- v_target[p] = +1 if player p WON the game, −1 if LOST, never 0 (max_moves games dropped entirely)

**Cycle**: generate ~200 games → ingest filtered samples → run K = chunk_samples / sample_per_step train steps → eval if step % eval_every == 0 → snapshot if step % snapshot_every == 0.

## Curriculum

Each pass criterion is at production sim count (200):

1. **2p only** — pass: ≥80% win rate vs heuristic over 100 games, mean `pins_in_goal[winner] = 10`.
2. **2p / 3p mix (50/50)** — pass: ≥65% win rate vs 2 heuristics in 3p slot.
3. **2p / 3p / 4p mix (40/30/30)** — pass: ≥50% rank-1 in 4p (2 heuristics + 1 snapshot).
4. **+5p mix** — pass: ≥40% rank-1 vs 4 heuristics.
5. **+6p mix** — pass: ≥35% rank-1 vs 5 heuristics.

Warm-start each next stage from `runs/<prev>/best.pt`. Reset AdamW state but hold prior LR for 1k steps to avoid shock.

If a stage fails to pass within budget (verify: 2h, later: 12h), kick to multi-day config and stay at that stage.

## Health checks / kill criteria

Every chunk → `health.jsonl`:
- `terminal_reason_counts {WIN, max_moves, draw_chain}`
- `mean_pins_in_goal[winner]` (target = 10.0)
- `chunk_samples_kept`, `chunk_samples_discarded`, `replay_size`, `replay_growth`
- `policy_loss_ema`, `value_loss_ema`, `policy_entropy`
- `eval_win_rate_vs_heuristic`, `eval_score_margin_mean`, `eval_score_margin_unique`
- `mean_search_depth`, `mean_unique_actions_per_episode`

**Auto-kill**:
- 0 WIN-terminated games for 10 consecutive chunks
- Replay didn't grow at all for 5 chunks
- `eval_score_margin_unique ≤ 3` across 2 evals (lesson 4 — bitwise-identical = collapse)
- `eval_score_margin_mean` trending more negative for 3 consecutive evals
- Value loss diverging > 2× rolling min

## Tournament artifact

`alphazero_method.py`:
1. At import: load `runs/best.pt` to `cuda` if available else `cpu`, `eval()`. Pre-warm with one dummy forward (avoids 200-300ms first-call latency).
2. `choose_move_alphazero(legal_moves, state, player_context)`:
   - Reconstruct board from `state["pins"]` and `current_turn_colour`.
   - Run MCTS, `N_sim=200`, **no Dirichlet noise**, deterministic argmax over visit counts.
   - Return `(pid, to_index, delay=0.05)`. No artificial slowdown — fast moves protect time_score.
3. If `runs/best.pt` is missing, raise — `player.py` already falls back to random.

End-to-end latency budget per turn: get_legal_moves RPC (~5ms) + MCTS 200 sims (~300ms) + move RPC (~5ms) + sleep 50ms ≈ 400ms. Margin = 1600ms.

Run with: `PLAYER_METHOD=alphazero python player.py`.

## Critical files to create / modify

Create:
- [az/sim.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/az/sim.py) — perf-critical, must mirror game.py rules exactly
- [az/encoder.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/az/encoder.py) — canonicalization LUTs
- [az/mcts.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/az/mcts.py) — batched MaxN PUCT
- [az/net.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/az/net.py)
- [az/inference_server.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/az/inference_server.py)
- [az/selfplay.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/az/selfplay.py)
- [az/train.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/az/train.py)
- [alphazero_method.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/alphazero_method.py) — tournament entry point

Reuse (read-only):
- [checkers_board.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/checkers_board.py) — `HexBoard`, `axial_of_colour`, adjacency
- [checkers_pins.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/checkers_pins.py) — `Pin.getPossibleMoves()`
- [game.py](RLChineseCheckers/multi%20system%20single%20machine%20minimal/game.py) — reference for legality / scoring (do not modify)

Do **not** modify: `game.py`, `checkers_board.py`, `checkers_pins.py`, `player.py` outside the PLAYING LOGIC markers.

## Verification

Order matters — each step gates the next.

1. **Sim parity test** (`tests/test_sim_parity.py`): drive 50 random-vs-random games via TCP and 50 via in-process sim with the same seeds; assert move-by-move trajectory equality. Catches divergent legality before training pollutes the replay.
2. **Encoder roundtrip test**: 1000 random board configs across all 6 to-move colours; assert `decode(encode(state)) == state`.
3. **End-to-end smoke test (verify config)**: 2-hour budget, 2p-only stage. Check `health.jsonl` shows `terminal_reason=WIN ≥ 60%`, `eval_score_margin_unique > 30`, value_loss declining over 5 chunks.
4. **Tournament dry run**: run `python game.py` (server) and `PLAYER_METHOD=alphazero python player.py` (×N) locally; verify per-move latency < 500ms in the game log, and that the agent actually reaches goal cells (not pin-shuffling).
5. **Heuristic eval gate**: candidate beats heuristic at every curriculum stage's pass criterion before promotion.
6. **Visual sanity check**: run `game_visualizer.py` on a sample game from each stage; confirm pins move toward the correct opposite zone, and all 10 pins move (not just one).

Once verify run is green, commit to overnight, then multi-day.

## Open risks (with mitigations)

- **In-process sim divergence from `game.py`** (especially multi-hop chains, draw chain logic) → parity test step 1 above.
- **MaxN bias from heuristic/snapshot opponents** → only train on candidate-owned positions; use real terminal outcomes (not bootstrapped MaxN values) as v_target.
- **Sample sparsity in 4-6p**: most games may hit max_moves and be dropped → raise cap to 300, rely heavily on warm-start, and accept slow progress at higher player counts.
- **Cold-start CUDA latency** at tournament first move → explicit pre-warm at module import + a second pre-warm before JOIN.
- **Mode collapse to single-pin shuffling** → 25% heuristic injection (lesson 3) + entropy bonus + visualizer spot-checks.
- **GPU contention with inference batching** → 5ms hard timeout on batch fill; profile after verify run; tune batch size.
