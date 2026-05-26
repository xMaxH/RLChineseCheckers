#!/usr/bin/env python3
"""PPO policy-gradient RL for the Chinese Checkers agent.

Genuine policy learning (AlphaGo's RL-policy stage, done with PPO):
  * the policy network plays self-play games, SAMPLING its own moves;
  * reward = real win/loss (+0.8 / -0.8) plus potential-based progress shaping;
  * the policy is improved by PPO (clipped policy gradient) with KL early-stop;
  * the value head is the critic (advantage = shaped return - V).

Different algorithm from the AlphaZero visit-count training that collapsed the
policy: PPO directly optimises win-rate, it does not chase MCTS visit counts.

Stability: low lr, PPO clip, **KL early-stopping** (caps policy movement per
iteration), and a value-critic warmup so advantages start calibrated.

Starts from runs/best.pt (BC policy + RL value head). Opponents: the greedy
heuristic. Eval-gated: only saves runs/ppo/best.pt when the raw-policy win-rate
vs the heuristic improves. Does NOT touch runs/best.pt (the deployed
value_rollout fallback stays intact).

    PYTHONPATH=. python tools/ppo_train.py --hours 30
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from az.sim import Sim
from az.encoder import encode_state, encode_legal_mask, decode_action
from az.heuristic import heuristic_choose_move
from az.shaping import potential
from az.inference_server import load_model

SHP_SCALE = 0.15
SHP_GW = 0.5
TERMINAL_W = 0.8


def masked_logp_entropy(logits, mask):
    """Log-softmax over legal moves only; returns (log_probs, entropy) per row."""
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg))
    logp = F.log_softmax(masked, dim=-1)
    p = logp.exp()
    ent = -(torch.where(mask, p * logp, torch.zeros_like(p))).sum(-1)
    return logp, ent


@torch.no_grad()
def play_game(net, device, rng, n_players, max_moves, greedy=False):
    """One game: candidate (one colour) vs heuristics. Returns (transitions,
    won, used). transitions carry the shaped Monte-Carlo return."""
    sim = Sim(n_players, seed=rng.randrange(2 ** 31))
    cand = sim.colours[rng.randrange(n_players)]
    trans = []
    while not sim.is_terminal:
        col = sim.current_colour()
        legal = sim.legal_moves(col)
        if not any(legal.values()):
            sim.skip_no_moves()
            continue
        if col == cand:
            board, glob = encode_state(sim.pins_state(), col, sim.turn_order, sim.move_count)
            bt = torch.from_numpy(board[None]).to(device)
            gt = torch.from_numpy(glob[None]).to(device)
            logits, value = net(bt, gt)
            mask_np = encode_legal_mask(legal, col)
            mask = torch.from_numpy(mask_np[None]).to(device)
            logp, _ = masked_logp_entropy(logits, mask)
            probs = logp[0].exp().cpu().numpy()
            legal_idx = np.flatnonzero(mask_np)
            p_legal = probs[legal_idx]
            p_legal = p_legal / p_legal.sum()
            if greedy:
                action = int(legal_idx[int(np.argmax(p_legal))])
            else:
                action = int(rng.choices(list(legal_idx), weights=list(p_legal), k=1)[0])
            if not greedy:
                trans.append(dict(
                    board=board, glob=glob, action=action,
                    logp=float(logp[0, action].cpu()),
                    value=float(value[0, 0].cpu()),     # to_move==cand -> slot 0
                    mask=mask_np,
                    phi=potential(sim, cand, SHP_SCALE, SHP_GW),
                ))
            pid, to = decode_action(action, col)
            sim.apply_move(pid, to)
        else:
            pid, to = heuristic_choose_move(sim, col, legal, rng=rng)
            sim.apply_move(pid, to)
        if sim.move_count >= max_moves and not sim.is_terminal:
            sim.force_max_moves()
    won = (sim.winner == cand)
    if sim.terminal_reason in ('WIN', 'DRAW_CHAIN'):
        T = TERMINAL_W if won else -TERMINAL_W
        phi_end = potential(sim, cand, SHP_SCALE, SHP_GW)
        for tr in trans:
            tr['ret'] = T + phi_end - tr['phi']
        return trans, won, True
    return [], won, False   # MAX_MOVES -> discarded


def collect(net, device, rng, n_games, n_players, max_moves):
    net.eval()
    trans, wins, used = [], 0, 0
    for _ in range(n_games):
        tr, won, ok = play_game(net, device, rng, n_players, max_moves)
        if ok:
            used += 1
            wins += int(won)
            trans.extend(tr)
    return trans, (wins / used if used else 0.0), used


def ppo_update(net, optim, trans, device, clip_eps, c_value, c_entropy,
               epochs, minibatch, target_kl, policy_weight):
    """One PPO update over the collected batch. Stops early once the policy has
    moved `target_kl` from the collection policy (the key stability guard).
    policy_weight=0 -> critic-only update (value warmup)."""
    boards = torch.from_numpy(np.stack([t['board'] for t in trans])).to(device)
    globs = torch.from_numpy(np.stack([t['glob'] for t in trans])).to(device)
    masks = torch.from_numpy(np.stack([t['mask'] for t in trans])).to(device)
    actions = torch.tensor([t['action'] for t in trans], device=device, dtype=torch.long)
    logp_old = torch.tensor([t['logp'] for t in trans], device=device)
    rets = torch.tensor([t['ret'] for t in trans], device=device, dtype=torch.float32)
    adv = rets - torch.tensor([t['value'] for t in trans], device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    N = len(trans)
    s = dict(pg=0.0, v=0.0, ent=0.0, kl=0.0, n=0)
    stop = False
    for _ in range(epochs):
        if stop:
            break
        perm = torch.randperm(N, device=device)
        for i in range(0, N, minibatch):
            idx = perm[i:i + minibatch]
            logits, value = net(boards[idx], globs[idx])
            logp, ent = masked_logp_entropy(logits, masks[idx])
            logp_a = logp.gather(1, actions[idx, None]).squeeze(1)
            ratio = (logp_a - logp_old[idx]).exp()
            a = adv[idx]
            pg = -torch.min(ratio * a,
                            torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a).mean()
            v_loss = F.mse_loss(value[:, 0], rets[idx])
            ent_m = ent.mean()
            loss = policy_weight * (pg - c_entropy * ent_m) + c_value * v_loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
            approx_kl = float((logp_old[idx] - logp_a.detach()).mean())
            s['pg'] += float(pg.detach())
            s['v'] += float(v_loss.detach())
            s['ent'] += float(ent_m.detach())
            s['kl'] += approx_kl
            s['n'] += 1
            if policy_weight > 0.0 and approx_kl > target_kl:
                stop = True
                break
    n = max(1, s['n'])
    return dict(pg=s['pg'] / n, v=s['v'] / n, ent=s['ent'] / n,
                kl=s['kl'] / n, steps=s['n'], stopped=int(stop))


def evaluate(net, device, rng, n_games, n_players, max_moves):
    """Raw-policy (greedy) win-rate vs the heuristic — the genuine RL agent."""
    net.eval()
    wins = 0
    for _ in range(n_games):
        _, won, _ = play_game(net, device, rng, n_players, max_moves, greedy=True)
        wins += int(won)
    return wins / max(1, n_games)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-ckpt", default="runs/best.pt")
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--hours", type=float, default=30.0)
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--games-per-iter", type=int, default=256)
    p.add_argument("--max-moves", type=int, default=220)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--c-value", type=float, default=0.5)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--value-warmup", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=3)
    p.add_argument("--eval-games", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    net = load_model(a.seed_ckpt, dev)
    net.eval()   # keep eval mode: BatchNorm frozen -> PPO ratios stay consistent
    optim = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = random.Random(a.seed)
    log = open(os.path.join(a.out, "ppo_progress.log"), "a", buffering=1)

    def w(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        log.write(line + "\n")

    w(f"PPO start: seed={a.seed_ckpt} players={a.players} lr={a.lr} clip={a.clip} "
      f"target_kl={a.target_kl} entropy={a.entropy} warmup={a.value_warmup} "
      f"games/iter={a.games_per_iter}")
    base_wr = evaluate(net, dev, random.Random(777), a.eval_games, a.players, a.max_moves)
    w(f"baseline raw-policy win-rate vs heuristic ({a.players}p): {base_wr:.3f}")
    best_wr = base_wr
    torch.save(net.state_dict(), os.path.join(a.out, "best.pt"))

    deadline = time.time() + a.hours * 3600.0
    it = 0
    while time.time() < deadline:
        it += 1
        t0 = time.time()
        trans, sp_wr, used = collect(net, dev, rng, a.games_per_iter,
                                     a.players, a.max_moves)
        if not trans:
            w(f"iter {it}: 0 usable games (all max_moves) — skipping")
            continue
        pw = 0.0 if it <= a.value_warmup else 1.0
        st = ppo_update(net, optim, trans, dev, a.clip, a.c_value, a.entropy,
                        a.epochs, a.minibatch, a.target_kl, pw)
        phase = "warmup" if pw == 0.0 else "ppo"
        msg = (f"iter {it} [{phase}]: selfplay_wr={sp_wr:.3f} used={used} "
               f"samples={len(trans)} | pg={st['pg']:.3f} v={st['v']:.3f} "
               f"ent={st['ent']:.2f} kl={st['kl']:.4f} steps={st['steps']}"
               f"{' KLSTOP' if st['stopped'] else ''} | {time.time()-t0:.0f}s")
        if it % a.eval_every == 0:
            wr = evaluate(net, dev, random.Random(777), a.eval_games,
                          a.players, a.max_moves)
            torch.save(net.state_dict(), os.path.join(a.out, "latest.pt"))
            tag = ""
            if wr > best_wr:
                best_wr = wr
                torch.save(net.state_dict(), os.path.join(a.out, "best.pt"))
                tag = "  <-- NEW BEST (saved)"
            msg += f" || EVAL raw-policy wr={wr:.3f} (best {best_wr:.3f}){tag}"
        w(msg)
    w(f"PPO done. best raw-policy win-rate vs heuristic: {best_wr:.3f} "
      f"(baseline {base_wr:.3f})")
    print("PPO DONE", flush=True)


if __name__ == "__main__":
    main()
