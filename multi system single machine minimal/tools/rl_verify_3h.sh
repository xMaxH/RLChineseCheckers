#!/usr/bin/env bash
set -euo pipefail

# Warm-started AlphaZero-style RL verification run.
#
# This is intentionally different from the heuristic bootstrap runs:
# - starts from checkpointOldHeuristic/best.pt
# - uses --bootstrap-chunks 0
# - uses --no-auto-bootstrap so chunk 0 is MCTS self-play
# - does not pass --heuristic-rollout-targets, so policy targets are MCTS visits
# - uses candidate/snapshot opponents, not heuristic teacher opponents

RUN="${1:-runs/rl_verify_3h_$(date +%Y%m%d_%H%M%S)}"
SEED_CKPT="${SEED_CKPT:-checkpointOldHeuristic/best.pt}"
DEVICE="${DEVICE:-cuda}"
HOURS="${HOURS:-3}"
MCTS_SIMS="${MCTS_SIMS:-150}"
NUM_WORKERS="${NUM_WORKERS:-6}"

if [[ ! -f "$SEED_CKPT" ]]; then
  echo "Missing warm-start checkpoint: $SEED_CKPT" >&2
  exit 2
fi

mkdir -p "$RUN"

echo "[rl-verify] run=$RUN"
echo "[rl-verify] seed_ckpt=$SEED_CKPT"
echo "[rl-verify] device=$DEVICE hours=$HOURS mcts_sims=$MCTS_SIMS workers=$NUM_WORKERS"

python3 train_run.py \
  --out "$RUN" \
  --name rl_verify_from_old_heuristic \
  --hours "$HOURS" \
  --max-chunks 1000 \
  --seed 20260515 \
  --seed-ckpt "$SEED_CKPT" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS" \
  --mcts-sims "$MCTS_SIMS" \
  --player-counts 2,3,4,5,6 \
  --player-weights 0.08,0.12,0.18,0.27,0.35 \
  --games-per-chunk 20 \
  --bootstrap-chunks 0 \
  --no-auto-bootstrap \
  --candidate-frac 0.75 \
  --heuristic-frac 0.0 \
  --snapshot-frac 0.25 \
  --max-moves-2p 500 \
  --max-moves-multi 500 \
  --replay-capacity 300000 \
  --min-samples-to-train 512 \
  --batch-size 1024 \
  --sample-per-step 512 \
  --min-train-steps 8 \
  --lr 3e-6 \
  --entropy-bonus 0.001 \
  --snapshot-pool-size 12 \
  --snapshot-every-train-steps 200 \
  --eval-every 200 \
  --eval-games 8

echo "[rl-verify] checking that the run actually entered RL/MCTS mode"
rg '"bootstrap": false' "$RUN/health.jsonl" >/dev/null
rg 'mode=MCTS' "$RUN/progress.log" >/dev/null

VERIFY="$RUN/final_verify"
mkdir -p "$VERIFY"

echo "[rl-verify] final quick sweep vs warm-start checkpoint"
python3 tools/eval_checkpoint_sweep.py \
  --ckpt "$RUN/final.pt" \
  --baseline-ckpt "$SEED_CKPT" \
  --device "$DEVICE" \
  --games "2:30,3:20,4:20,5:12,6:12" \
  --max-moves 500 \
  --policy-rollout-top-k 3 \
  --policy-rollouts-per-move 2 \
  --out "$VERIFY/final_vs_old_heuristic.json" \
  --summary-jsonl "$VERIFY/summary.jsonl" \
  2>&1 | tee "$VERIFY/final_vs_old_heuristic.log"

echo "[rl-verify] done. Use this checkpoint with:"
echo "  ALPHAZERO_CKPT=$RUN/final.pt PLAYER_METHOD=alphazero python player.py"
