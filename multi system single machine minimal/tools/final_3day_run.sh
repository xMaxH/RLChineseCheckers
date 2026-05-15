#!/usr/bin/env bash
set -euo pipefail

RUN="${1:-runs/final_3day_$(date +%Y%m%d_%H%M%S)}"
SEED_CKPT="${SEED_CKPT:-runs/final_reranker_pilot_12h_20260510_195708/final.pt}"
BASELINE_CKPT="${BASELINE_CKPT:-$SEED_CKPT}"
DEVICE="${DEVICE:-cuda}"
AUTO_MONITOR="${AUTO_MONITOR:-0}"
MONITOR_INTERVAL_MIN="${MONITOR_INTERVAL_MIN:-360}"

mkdir -p "$RUN"

echo "[driver] run=$RUN"
echo "[driver] started=$(date -Is)"
echo "[driver] seed_ckpt=$SEED_CKPT"
echo "[driver] baseline_ckpt=$BASELINE_CKPT"
echo "[driver] device=$DEVICE"
echo "[driver] monitor command:"
echo "  tools/monitor_long_run.sh \"$RUN\" \"$BASELINE_CKPT\" \"$MONITOR_INTERVAL_MIN\""

monitor_pid=""
if [[ "$AUTO_MONITOR" == "1" ]]; then
  echo "[driver] starting background monitor"
  tools/monitor_long_run.sh "$RUN" "$BASELINE_CKPT" "$MONITOR_INTERVAL_MIN" \
    > "$RUN/live_monitor.console.log" 2>&1 &
  monitor_pid="$!"
  echo "[driver] monitor_pid=$monitor_pid"
fi

python3 train_run.py \
  --out "$RUN" \
  --name final_3day_rollout_teacher \
  --hours 72 \
  --max-chunks 20000 \
  --seed 20260511 \
  --seed-ckpt "$SEED_CKPT" \
  --device "$DEVICE" \
  --num-workers 6 \
  --mcts-sims 150 \
  --player-counts 2,3,4,5,6 \
  --player-weights 0.08,0.12,0.18,0.27,0.35 \
  --games-per-chunk 20 \
  --bootstrap-chunks 999999 \
  --heuristic-rollout-targets \
  --heuristic-rollouts-per-move 2 \
  --heuristic-rollout-pool-cap 10 \
  --heuristic-rollout-score-temperature 0 \
  --max-moves-2p 500 \
  --max-moves-multi 500 \
  --replay-capacity 1000000 \
  --min-samples-to-train 8000 \
  --batch-size 4096 \
  --sample-per-step 8192 \
  --min-train-steps 40 \
  --lr 1e-5 \
  --entropy-bonus 0.0 \
  --snapshot-pool-size 20 \
  --snapshot-every-train-steps 1000 \
  --eval-every 999999

CKPT="$RUN/final.pt"
VERIFY="$RUN/final_verify"
mkdir -p "$VERIFY"

echo "[driver] training_finished=$(date -Is)"
echo "[driver] final quick sweep vs baseline"
python3 tools/eval_checkpoint_sweep.py \
  --ckpt "$CKPT" \
  --baseline-ckpt "$BASELINE_CKPT" \
  --device "$DEVICE" \
  --games "2:80,3:60,4:60,5:40,6:40" \
  --max-moves 500 \
  --policy-rollout-top-k 3 \
  --policy-rollouts-per-move 2 \
  --out "$VERIFY/final_vs_baseline.json" \
  --summary-jsonl "$VERIFY/summary.jsonl" \
  2>&1 | tee "$VERIFY/final_vs_baseline.log"

if [[ -n "$monitor_pid" ]]; then
  echo "[driver] stopping background monitor pid=$monitor_pid"
  kill "$monitor_pid" 2>/dev/null || true
fi

echo "[driver] finished=$(date -Is)"
