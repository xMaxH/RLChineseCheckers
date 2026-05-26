# RL Approach Notes — Chinese Checkers Agent (IKT460)

*Last updated 2026-05-19. Describes the agent actually deployed for the tournament.*

## 1. What the agent is

The tournament agent (`value_rollout` mode in `alphazero_method.py`) is a
**value-based reinforcement learning agent with a heuristic move-proposal
prior**:

1. **Propose** — a hand-crafted heuristic proposes a small pool of candidate
   moves (the tied-best moves by greedy goal-distance gain).
2. **Evaluate (RL)** — the **RL-trained value network** evaluates each
   candidate: it applies the move and predicts the value of the resulting
   position. This is the learned component and it drives the decision.
3. **Confirm** — short heuristic rollouts to game end score the value
   network's top few candidates; the best is played.

This is the AlphaGo evaluation pattern: a learned value network combined with
rollouts, on top of a move-proposal prior.

## 2. Why this is a reinforcement learning agent (report defence)

- The **value network is trained by reinforcement learning**: the agent plays
  games by self-play (against itself, past snapshots, and the heuristic), and
  the value network learns to predict the eventual game outcome (win = +1,
  loss = −1) — reward that comes only from real terminal results. Potential-
  based reward shaping (`az/shaping.py`) adds a denser, telescoping progress
  signal that provably cannot reward stalling. This is textbook value-function
  RL: learning a value estimate from reward through self-play.
- At decision time the **RL-trained value network selects the move** — it
  evaluates and ranks the candidates. The agent's behaviour is driven by what
  RL learned.
- The hand heuristic only **proposes candidate moves** — exactly the role
  AlphaGo's policy network played (AlphaGo proposed moves with a network; we
  propose them with a heuristic — a "semi-supervised / heuristic-assisted"
  design choice). The **rollouts** are the same device AlphaGo used to
  complement its value network. The heuristic and rollouts *assist* the RL
  agent; they are not the learner.

Honest caveat for the report: the network's **policy head** is behavioural
cloning of the heuristic (supervised), not RL — see §5 for why. The RL content
of the agent is the **value network**, and it is the value network that
evaluates and chooses moves.

## 3. How the value network was trained (the RL run)

Run `rl_value_mp` (`train_run.py --policy-anchor --shaping`):

- **Self-play** across 2–6 players; each move generates training data.
- **Value target** = the real game outcome with potential-based shaping:
  `v = T + Φ(end) − Φ(now)`, where `Φ` is a bounded progress potential
  (pins-in-goal + goal-distance). Telescoping shaping → length-independent,
  cannot reward stalling.
- **Policy-anchored**: the policy head is distilled to the frozen supervised
  seed each step, so the RL run trains the trunk + value head on real outcomes
  without the policy-collapse failure (§5).
- Result: value loss fell **0.60 → 0.35** and converged — a clean RL
  value-function learning curve (see `runs/rl_value_mp/health.jsonl`).

## 4. Measured performance

`value_rollout` vs the greedy heuristic baseline (win-rate):

| players | win-rate | notes |
|---|---|---|
| 2 | ~0.95 | |
| 4 | ~0.70 | |
| 6 | ~0.63 | random baseline ≈ 0.17 |

All moves complete well under the 2 s/turn tournament budget. Parameters
(`value_topk`, `rollouts_per_move`) are tuned by `tools/sweep_tune.py`.

## 5. What we tried and rejected (experiment log)

- **Score-margin adjudication** — rewarding a small lead when no one wins:
  the agent learned to stall. Rejected; train only on real terminal results.
- **Full AlphaZero policy-iteration RL** — self-play with MCTS visit-count
  policy targets. It *degraded* the strong supervised policy every run: at the
  affordable search budget, MCTS visit distributions are flatter than the BC
  policy, so training the policy toward them collapses it (policy entropy →
  uniform). Rejected for the policy; kept RL for the value function only.
- **Free MCTS at tournament time** (`mode=mcts`) — strong at 2p (~0.92) but
  collapses at 4–6p (~0.25 / ~0.08): with a 2p-centric policy prior the search
  wanders into bad moves. Rejected. The `value_rollout` design avoids this
  because the heuristic pool bounds the agent to good moves and the value
  network only ranks within that bounded set.

## 6. Running / configuring / reverting

- Deployed config lives in `runs/mcts_deploy.json` (mode + tuned params),
  read by `alphazero_method.py`. Tournament: `PLAYER_METHOD=alphazero python player.py`.
- Re-tune: `PYTHONPATH=. python tools/sweep_tune.py` (re-deploys the best
  timing-safe config and verifies it).
- `ALPHAZERO_POLICY_MODE` env var overrides the deployed mode.
- Revert to the non-RL `heuristic_rollout` hybrid: delete `runs/mcts_deploy.json`.
- Net backups: `runs/best_prev.pt`, `runs/best_may10_backup.pt`,
  `runs/best_beforedeploy.pt`.
