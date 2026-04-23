# Training guide

How to train, evaluate, and deploy the DQN agent. Read
`architecture.md` first for the conceptual picture.

## Prerequisites

Install dependencies (the base `requirements.txt` already pins
`torch`, `numpy`, `matplotlib`, etc.):

```bash
pip install -r requirements.txt
```

All commands below are run from the repository root.

## Quick smoke test (~10 min)

Verify that training launches and the trainer actually makes progress:

```bash
python -m rl.scripts.train_dqn --config rl/configs/dqn_quicktest.json --minutes 10
```

Checkpoints and metrics end up under `rl/checkpoints/quicktest/`.

## Fast bootstrap (imitation pretraining)

If pure DQN warmup is unstable on your machine, pretrain the policy by
imitation of `GreedyAgent` first, then run RL fine-tuning:

```bash
python -m rl.scripts.pretrain_from_greedy --episodes 4000 --epochs 12 \
    --out rl/checkpoints/dqn_best.pt
```

This already gives a usable model (in our validation run: 14% solo wins,
all under 600 moves), and is a good initialization for subsequent RL.

## Full training (~45 min budget)

```bash
python -m rl.scripts.train_dqn --config rl/configs/dqn_default.json --minutes 45
```

Expected timeline (CPU-ish; GPU is ~2x faster):

| Phase                  | Approx. time | What to watch                            |
| ---------------------- | ------------ | ---------------------------------------- |
| greedy warmup          |   ~30 s      | replay buffer fills to ~3000 transitions |
| epsilon=1 -> 0.5       |  5-8 min     | `return_sum` climbs, `loss` still high   |
| epsilon=0.5 -> 0.1     | 10-15 min    | first episodes with `pins_home >= 4-5`   |
| epsilon stable (0.05)  | remainder    | first wins, rolling win-rate rises       |

Training exits early when the rolling-100-episode win-rate crosses
`exit_win_rate`. You can stop any time; `rl/checkpoints/dqn_latest.pt`
is written after every episode.

### CLI options

```bash
python -m rl.scripts.train_dqn \
    --config rl/configs/dqn_default.json \
    --minutes 45                 \  # hard wall-clock cap
    --skip-vs-random             \  # run only stage 1
    --skip-solo                  \  # run only stage 2 (needs a checkpoint)
    --resume rl/checkpoints/dqn_latest.pt
```

## Evaluating a checkpoint

```bash
# Solo race: does the agent clear all 10 pins under the step cap?
python -m rl.scripts.eval_agent --checkpoint rl/checkpoints/dqn_best.pt \
    --mode solo_race --episodes 50 --max-steps 600

# 2-player vs random: tournament-shaped evaluation.
python -m rl.scripts.eval_agent --checkpoint rl/checkpoints/dqn_best.pt \
    --mode vs_random --episodes 50 --max-steps 600

# Sanity baseline: random agent in the same env.
python -m rl.scripts.eval_agent --random-baseline --episodes 50 --mode solo_race
```

Sample output:

```
=== RESULTS ===
episodes: 50
win rate: 14.0% (14/100)
avg moves (when won): 152.4 +/- 80.7 (min 61, max 396)
avg pins_home: 7.13 / 10
avg return: 9.09
```

Milestone 3 target: at least one win with `agent_moves < 600`.

## Interpreting metrics

Every episode is logged to `rl/checkpoints/metrics/episodes.csv`:

```
episode, stage, steps, agent_moves, return_sum, pins_home,
outcome, epsilon, loss
```

`MetricsLogger.plot()` produces `summary.png` in the same folder when
training finishes. Useful columns:

- `return_sum`: total reward for the episode. Should trend upward.
- `pins_home`: proxy for "progress". Should cross 5, then 8, then 10.
- `outcome`: `win`, `truncated`, `draw_no_moves`, `opponent_win:<c>`.
- `loss`: Huber-loss of the last training batch. High + rising = LR too
   high or target net too stale; high + flat = value targets are
   large (rescale rewards); low + flat with no progress = stuck.

## Deploying the trained agent

The integration path is already in place. After training, the
checkpoint at `rl/checkpoints/dqn_best.pt` is picked up automatically
by `dqn_method.py`. To use it:

```bash
# Self-play (agent vs agent)
python "multi system single machine minimal/run_game.py" --players 2 --method dqn

# Mixed match: agent vs random
python "multi system single machine minimal/run_game.py" \
    --players 2 --player-methods dqn,random

# Override the checkpoint with an environment variable
$env:DQN_CHECKPOINT="rl/checkpoints/my_experiment.pt"
python "multi system single machine minimal/run_game.py" --players 2 --method dqn
```

## Troubleshooting

**"Falling back to random" on startup**
`dqn_method.py` couldn't load the checkpoint. Confirm the file exists
and was produced by the current version of `DQNAgent` (obs-dim mismatch
will raise on `load_state_dict`). Rerun training.

**Agent chooses an illegal move**
Should not happen — the mask is applied both at train and inference
time. If it does, it indicates a mismatch between the
`legal_moves` shape at training vs. inference. Both should be
`{pin_id: [to_idx, ...]}`; the server returns keys as JSON ints.

**Training loss explodes (>100)**
Reduce `lr`, increase `target_update_interval`, or scale down reward
weights. Stage 1's `all_home_bonus=100` can cause spikes; the default
`0.995` gamma keeps propagation stable but only if reward scale is
reasonable.

**Agent plateaus with 1-3 pins in goal**
This is usually because it has learned to shove pins near the goal
entrance and then stalls. Increase `distance_shaping` modestly (0.1 ->
0.2), give more training time, or extend `max_steps_per_episode`.

## Reproducibility

All configs include a `seed`. Torch CUDA results are still not fully
deterministic; set `PYTHONHASHSEED` and disable CUDA determinism if you
need bit-for-bit repeats.
