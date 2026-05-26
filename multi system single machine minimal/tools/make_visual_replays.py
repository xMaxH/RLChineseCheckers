#!/usr/bin/env python3
"""Create static HTML replays for inspecting a trained policy.

The generated viewer is intentionally dependency-free. It plays the model as
one colour against heuristic opponents and marks suspicious patterns such as
full-state repeats and immediate piece reversals.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from az.config import BOARD_CHANNELS, NUM_CELLS
from az.encoder import encode_action, encode_state
from az.heuristic import heuristic_choose_move, heuristic_move_pool
from az.net import AZNet
from az.selfplay import _finish_with_heuristic, _score_margin
from az.sim import Sim
from checkers_board import HexBoard


Move = Tuple[int, int]


def parse_counts(text: str) -> List[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("empty --player-counts")
    for n in out:
        if n < 2 or n > 6:
            raise ValueError("--player-counts must be between 2 and 6")
    return out


def load_net(ckpt: Path, device: torch.device) -> AZNet:
    net = AZNet().to(device).eval()
    state = torch.load(ckpt, map_location=device, weights_only=True)
    net.load_state_dict(state)
    with torch.no_grad():
        net(
            torch.zeros(1, BOARD_CHANNELS, NUM_CELLS, device=device),
            torch.zeros(1, 8, device=device),
        )
    return net


def network_logits(net: AZNet, device: torch.device, sim: Sim, colour: str) -> np.ndarray:
    board, glob = encode_state(sim.pins_state(), colour, sim.turn_order, sim.move_count)
    with torch.no_grad():
        b = torch.from_numpy(board[None]).to(device)
        g = torch.from_numpy(glob[None]).to(device)
        policy, _ = net(b, g)
    return policy[0].detach().cpu().numpy()


def rank_heuristic_pool(
    net: AZNet,
    device: torch.device,
    sim: Sim,
    colour: str,
    legal: Dict[int, List[int]],
) -> List[Tuple[float, int, int]]:
    pool = heuristic_move_pool(sim, colour, legal)
    logits = network_logits(net, device, sim, colour)
    ranked = sorted(
        (
            (float(logits[encode_action(pid, to_idx, colour)]), pid, to_idx)
            for _, _, pid, to_idx in pool
        ),
        reverse=True,
    )
    return ranked


def rollout_seed(sim: Sim, colour: str, pid: int, to_idx: int, rollout_idx: int) -> int:
    return (
        (sim.move_count + 1) * 1_000_003
        + sum((i + 1) * p.axialindex for i, p in enumerate(sim.pins_by_colour[colour]))
        + pid * 9_176
        + to_idx * 37
        + rollout_idx * 1_009
    ) & 0xFFFFFFFF


def choose_model_move(
    net: AZNet,
    device: torch.device,
    sim: Sim,
    colour: str,
    legal: Dict[int, List[int]],
    top_k: int,
    rollouts_per_move: int,
    rollout_max_moves: int,
) -> Tuple[int, int, Dict]:
    ranked = rank_heuristic_pool(net, device, sim, colour, legal)
    if not ranked:
        pid, to_idx = heuristic_choose_move(sim, colour, legal)
        return pid, to_idx, {"fallback": "heuristic", "pool_size": 0}
    if len(ranked) == 1:
        logit, pid, to_idx = ranked[0]
        return pid, to_idx, {
            "pool_size": 1,
            "top": [{"pid": pid, "to": to_idx, "logit": logit, "mean_margin": None}],
        }

    best_key = None
    best_move = ranked[0][1], ranked[0][2]
    scored = []
    for rank, (logit, pid, to_idx) in enumerate(ranked[: max(1, top_k)]):
        margins = []
        for rollout_idx in range(max(1, rollouts_per_move)):
            s = copy.deepcopy(sim)
            s.apply_move(pid, to_idx)
            rng = random.Random(rollout_seed(sim, colour, pid, to_idx, rollout_idx))
            _finish_with_heuristic(s, rng, max_moves=rollout_max_moves)
            margins.append(float(_score_margin(s, colour)))
        mean_margin = float(np.mean(margins))
        scored.append(
            {
                "pid": pid,
                "to": to_idx,
                "logit": logit,
                "mean_margin": mean_margin,
                "margins": margins,
            }
        )
        key = (mean_margin, -rank)
        if best_key is None or key > best_key:
            best_key = key
            best_move = (pid, to_idx)

    return best_move[0], best_move[1], {
        "pool_size": len(ranked),
        "top": scored,
        "selected_mean_margin": best_key[0] if best_key else None,
    }


def state_signature(sim: Sim) -> str:
    parts = [str(sim.current_turn_index)]
    for colour in sim.turn_order:
        parts.append(colour)
        parts.extend(str(p.axialindex) for p in sim.pins_by_colour[colour])
    return "|".join(parts)


def board_cells() -> List[Dict]:
    board = HexBoard()
    return [
        {
            "i": i,
            "q": cell.q,
            "r": cell.r,
            "x": round(cell.x, 3),
            "y": round(cell.y, 3),
            "postype": cell.postype,
        }
        for i, cell in enumerate(board.cells)
    ]


def capture_frame(sim: Sim, move_id: int, move: Dict | None) -> Dict:
    return {
        "move_id": move_id,
        "turn": sim.current_colour(),
        "terminal": sim.is_terminal,
        "pins": {
            colour: [p.axialindex for p in sim.pins_by_colour[colour]]
            for colour in sim.turn_order
        },
        "move": move,
    }


def simulate_game(
    net: AZNet,
    device: torch.device,
    num_players: int,
    seed: int,
    top_k: int,
    rollouts_per_move: int,
    max_moves: int,
) -> Dict:
    sim = Sim(num_players, seed=seed)
    candidate = sim.colours[0]
    rng = random.Random(seed + 1009)

    seen_states = {state_signature(sim): 1}
    last_piece_move: Dict[Tuple[str, int], Tuple[int, int]] = {}
    visited_by_piece = {
        (colour, p.id): {p.axialindex}
        for colour in sim.turn_order
        for p in sim.pins_by_colour[colour]
    }

    frames = [capture_frame(sim, 0, None)]
    moves = []
    diagnostics = {
        "illegal_moves": 0,
        "full_state_repeats": 0,
        "candidate_immediate_reversals": 0,
        "opponent_immediate_reversals": 0,
        "candidate_piece_revisits": 0,
        "opponent_piece_revisits": 0,
    }

    while not sim.is_terminal:
        colour = sim.current_colour()
        legal = sim.legal_moves(colour)
        if not any(legal.values()):
            sim.skip_no_moves()
            frames.append(
                capture_frame(
                    sim,
                    sim.move_count,
                    {"colour": colour, "role": "skip", "flags": ["no_legal_moves"]},
                )
            )
            continue

        role = "model" if colour == candidate else "heuristic"
        choose_diag = {}
        if role == "model":
            pid, to_idx, choose_diag = choose_model_move(
                net, device, sim, colour, legal, top_k, rollouts_per_move, max_moves
            )
        else:
            pid, to_idx = heuristic_choose_move(sim, colour, legal, rng=rng)

        from_idx = sim.pins_by_colour[colour][pid].axialindex
        flags = []
        if to_idx not in legal.get(pid, []):
            diagnostics["illegal_moves"] += 1
            flags.append("illegal")

        previous = last_piece_move.get((colour, pid))
        if previous == (to_idx, from_idx):
            if role == "model":
                diagnostics["candidate_immediate_reversals"] += 1
            else:
                diagnostics["opponent_immediate_reversals"] += 1
            flags.append("immediate_reverse")

        if to_idx in visited_by_piece[(colour, pid)]:
            if role == "model":
                diagnostics["candidate_piece_revisits"] += 1
            else:
                diagnostics["opponent_piece_revisits"] += 1
            flags.append("piece_revisit")

        sim.apply_move(pid, to_idx)

        sig = state_signature(sim)
        if sig in seen_states:
            diagnostics["full_state_repeats"] += 1
            flags.append("full_state_repeat")
        seen_states[sig] = seen_states.get(sig, 0) + 1

        visited_by_piece[(colour, pid)].add(to_idx)
        last_piece_move[(colour, pid)] = (from_idx, to_idx)

        move = {
            "move": sim.move_count,
            "colour": colour,
            "role": role,
            "pid": pid,
            "from": from_idx,
            "to": to_idx,
            "flags": flags,
            "choice": choose_diag,
        }
        try:
            move["candidate_margin_after"] = float(_score_margin(sim, candidate))
        except Exception:
            move["candidate_margin_after"] = None
        moves.append(move)
        frames.append(capture_frame(sim, sim.move_count, move))

        if sim.move_count >= max_moves and not sim.is_terminal:
            sim.force_max_moves()
            break

    terminal_margin = float(_score_margin(sim, candidate))
    return {
        "id": f"{num_players}p_seed{seed}",
        "num_players": num_players,
        "seed": seed,
        "candidate": candidate,
        "turn_order": sim.turn_order,
        "winner": sim.winner,
        "terminal_reason": sim.terminal_reason,
        "move_count": sim.move_count,
        "candidate_margin": terminal_margin,
        "candidate_win": sim.winner == candidate,
        "diagnostics": diagnostics,
        "moves": moves,
        "frames": frames,
    }


def html_page(data: Dict) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy Replay Inspector</title>
<style>
body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #18202a; background: #f4f6f8; }}
header {{ padding: 16px 24px; background: #ffffff; border-bottom: 1px solid #d8dee6; }}
main {{ display: grid; grid-template-columns: minmax(360px, 1fr) 420px; gap: 16px; padding: 16px; }}
button, select, input {{ font: inherit; }}
button {{ border: 1px solid #aab4c1; background: #fff; border-radius: 6px; padding: 6px 10px; cursor: pointer; }}
select {{ border: 1px solid #aab4c1; border-radius: 6px; padding: 6px 10px; background: #fff; }}
.panel {{ background: #fff; border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 12px; border-bottom: 1px solid #d8dee6; }}
.board-wrap {{ padding: 12px; }}
svg {{ width: 100%; height: min(72vh, 760px); background: #fbfcfd; border: 1px solid #d8dee6; border-radius: 6px; }}
.side {{ display: grid; gap: 16px; align-content: start; }}
.section {{ padding: 12px; }}
h1 {{ font-size: 20px; margin: 0 0 4px; }}
h2 {{ font-size: 15px; margin: 0 0 10px; }}
p {{ margin: 0; color: #536071; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid #edf0f3; text-align: left; vertical-align: top; }}
tr.active {{ background: #fff7d6; }}
.bad {{ color: #ad2727; font-weight: 700; }}
.ok {{ color: #207a3c; font-weight: 700; }}
.muted {{ color: #6b7685; }}
.flag {{ display: inline-block; margin: 1px 3px 1px 0; padding: 1px 5px; border-radius: 999px; background: #ffe0e0; color: #8f1d1d; font-size: 12px; }}
.meta {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 10px; font-size: 13px; }}
.players {{ display: grid; gap: 8px; }}
.player-row {{ display: grid; grid-template-columns: 18px 95px 1fr; gap: 8px; align-items: center; font-size: 13px; }}
.swatch {{ width: 16px; height: 16px; border-radius: 50%; border: 2px solid #18202a; box-sizing: border-box; }}
.role {{ color: #536071; }}
.role.me {{ color: #0f5fc9; font-weight: 700; }}
@media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} svg {{ height: 68vh; }} }}
</style>
</head>
<body>
<header>
  <h1>Policy Replay Inspector</h1>
  <p>Model plays one colour against heuristic opponents. Use the slider to step through the game.</p>
</header>
<main>
  <section class="panel">
    <div class="toolbar">
      <select id="gameSelect"></select>
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <input id="stepSlider" type="range" min="0" max="0" value="0">
      <span id="stepLabel"></span>
    </div>
    <div class="board-wrap"><svg id="board"></svg></div>
  </section>
  <aside class="side">
    <section class="panel"><div class="section">
      <h2>Game</h2>
      <div id="gameMeta" class="meta"></div>
    </div></section>
    <section class="panel"><div class="section">
      <h2>Players</h2>
      <div id="playersLegend" class="players"></div>
    </div></section>
    <section class="panel"><div class="section">
      <h2>Diagnostics</h2>
      <div id="diagnostics"></div>
    </div></section>
    <section class="panel"><div class="section">
      <h2>Current Move</h2>
      <div id="moveDetails"></div>
    </div></section>
    <section class="panel"><div class="section">
      <h2>Move Window</h2>
      <div id="moveTable"></div>
    </div></section>
  </aside>
</main>
<script>
const DATA = {payload};
const cells = DATA.cells;
let gameIndex = 0;
let step = 0;
const colourFill = {{
  "red": "#d13b3b",
  "blue": "#2676cc",
  "yellow": "#e6bd27",
  "lawn green": "#55a84f",
  "purple": "#8b55c7",
  "gray0": "#333944",
  "board": "#f3f6f8"
}};
const textFill = {{"yellow": "#20242a", "lawn green": "#102715"}};

function cellByIndex(i) {{ return cells[i]; }}
function hexPoints(x, y, r) {{
  const pts = [];
  for (let k = 0; k < 6; k++) {{
    const a = Math.PI / 180 * (60 * k - 30);
    pts.push(`${{(x + r * Math.cos(a)).toFixed(2)}},${{(y + r * Math.sin(a)).toFixed(2)}}`);
  }}
  return pts.join(" ");
}}
function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
}}
function fmt(v) {{ return v === null || v === undefined ? "n/a" : String(v); }}
function flagsHtml(flags) {{
  if (!flags || !flags.length) return '<span class="muted">none</span>';
  return flags.map(f => `<span class="flag">${{escapeHtml(f)}}</span>`).join("");
}}
function currentGame() {{ return DATA.games[gameIndex]; }}

function setupSelect() {{
  const select = document.getElementById("gameSelect");
  select.innerHTML = "";
  DATA.games.forEach((g, i) => {{
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${{g.num_players}}p seed ${{g.seed}} (${{
      g.candidate_win ? "win" : "not win"
    }}, margin ${{g.candidate_margin.toFixed(0)}})`;
    select.appendChild(opt);
  }});
  select.addEventListener("change", () => {{
    gameIndex = Number(select.value);
    step = 0;
    render();
  }});
}}

function renderBoard(g, frame) {{
  const svg = document.getElementById("board");
  const xs = cells.map(c => c.x), ys = cells.map(c => c.y);
  const minX = Math.min(...xs) - 48, maxX = Math.max(...xs) + 48;
  const minY = Math.min(...ys) - 48, maxY = Math.max(...ys) + 48;
  svg.setAttribute("viewBox", `${{minX}} ${{minY}} ${{maxX - minX}} ${{maxY - minY}}`);
  svg.innerHTML = "";

  const last = frame.move;
  cells.forEach(c => {{
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("points", hexPoints(c.x, c.y, 17));
    poly.setAttribute("fill", colourFill[c.postype] || "#e7ebef");
    poly.setAttribute("fill-opacity", c.postype === "board" ? "1" : "0.22");
    poly.setAttribute("stroke", "#c4ccd6");
    poly.setAttribute("stroke-width", "1");
    svg.appendChild(poly);
  }});

  if (last && Number.isInteger(last.from) && Number.isInteger(last.to)) {{
    const a = cellByIndex(last.from), b = cellByIndex(last.to);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
    line.setAttribute("stroke", "#111827");
    line.setAttribute("stroke-width", "5");
    line.setAttribute("stroke-opacity", "0.38");
    line.setAttribute("stroke-linecap", "round");
    svg.appendChild(line);
  }}

  Object.entries(frame.pins).forEach(([colour, positions]) => {{
    positions.forEach((idx, pid) => {{
      const c = cellByIndex(idx);
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", c.x);
      circle.setAttribute("cy", c.y);
      circle.setAttribute("r", colour === g.candidate ? "12.5" : "10.5");
      circle.setAttribute("fill", colourFill[colour] || "#777");
      circle.setAttribute("stroke", colour === g.candidate ? "#111827" : "#ffffff");
      circle.setAttribute("stroke-width", colour === g.candidate ? "3" : "2");
      group.appendChild(circle);
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", c.x);
      label.setAttribute("y", c.y + 4);
      label.setAttribute("font-size", "10");
      label.setAttribute("font-weight", "700");
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("fill", textFill[colour] || "#ffffff");
      label.textContent = String(pid);
      group.appendChild(label);
      svg.appendChild(group);
    }});
  }});
}}

function renderMeta(g) {{
  const rows = [
    ["players", g.num_players],
    ["candidate", g.candidate],
    ["turn order", g.turn_order.join(", ")],
    ["winner", g.winner || "none"],
    ["terminal", g.terminal_reason],
    ["moves", g.move_count],
    ["candidate margin", g.candidate_margin.toFixed(1)],
    ["checkpoint", DATA.ckpt]
  ];
  document.getElementById("gameMeta").innerHTML = rows.map(([k,v]) =>
    `<div class="muted">${{escapeHtml(k)}}</div><div>${{escapeHtml(v)}}</div>`
  ).join("");
}}

function ckptName() {{
  const parts = String(DATA.ckpt || "").split(/[\\\\/]/);
  return parts[parts.length - 1] || "checkpoint";
}}

function renderPlayers(g) {{
  const name = ckptName();
  document.getElementById("playersLegend").innerHTML = g.turn_order.map(colour => {{
    const isMe = colour === g.candidate;
    const role = isMe ? `ME / model (${{name}})` : "heuristic greedy";
    return `<div class="player-row">
      <span class="swatch" style="background:${{colourFill[colour] || "#777"}}"></span>
      <strong>${{escapeHtml(colour)}}</strong>
      <span class="role ${{isMe ? "me" : ""}}">${{escapeHtml(role)}}</span>
    </div>`;
  }}).join("");
}}

function renderDiagnostics(g) {{
  const d = g.diagnostics;
  const rows = Object.entries(d).map(([k, v]) => {{
    const bad = v > 0 && (k.includes("illegal") || k.includes("full_state") || k.includes("candidate_immediate"));
    return `<tr><td>${{escapeHtml(k)}}</td><td class="${{bad ? "bad" : "ok"}}">${{v}}</td></tr>`;
  }}).join("");
  document.getElementById("diagnostics").innerHTML = `<table>${{rows}}</table>`;
}}

function renderMoveDetails(frame) {{
  const m = frame.move;
  if (!m) {{
    document.getElementById("moveDetails").innerHTML = '<p class="muted">Initial board.</p>';
    return;
  }}
  let choice = "";
  if (m.choice && m.choice.top) {{
    const topRows = m.choice.top.map(x =>
      `<tr><td>${{x.pid}} -> ${{x.to}}</td><td>${{Number(x.logit).toFixed(2)}}</td><td>${{x.mean_margin === null ? "n/a" : Number(x.mean_margin).toFixed(1)}}</td></tr>`
    ).join("");
    choice = `<h2>Model Candidates</h2><table><tr><th>move</th><th>logit</th><th>rollout</th></tr>${{topRows}}</table>`;
  }}
  document.getElementById("moveDetails").innerHTML = `
    <table>
      <tr><td>move</td><td>${{m.move}}</td></tr>
      <tr><td>role</td><td>${{escapeHtml(m.role)}}</td></tr>
      <tr><td>colour</td><td>${{escapeHtml(m.colour)}}</td></tr>
      <tr><td>pin</td><td>${{m.pid}}</td></tr>
      <tr><td>from -> to</td><td>${{m.from}} -> ${{m.to}}</td></tr>
      <tr><td>flags</td><td>${{flagsHtml(m.flags)}}</td></tr>
      <tr><td>candidate margin</td><td>${{fmt(m.candidate_margin_after)}}</td></tr>
    </table>
    ${{choice}}`;
}}

function renderMoveTable(g) {{
  const start = Math.max(0, step - 8);
  const end = Math.min(g.moves.length, step + 8);
  const rows = g.moves.slice(start, end).map(m => `
    <tr class="${{m.move === step ? "active" : ""}}">
      <td>${{m.move}}</td><td>${{escapeHtml(m.role)}}</td><td>${{escapeHtml(m.colour)}}</td>
      <td>${{m.pid}}</td><td>${{m.from}} -> ${{m.to}}</td><td>${{flagsHtml(m.flags)}}</td>
    </tr>`).join("");
  document.getElementById("moveTable").innerHTML =
    `<table><tr><th>#</th><th>role</th><th>colour</th><th>pin</th><th>move</th><th>flags</th></tr>${{rows}}</table>`;
}}

function render() {{
  const g = currentGame();
  const slider = document.getElementById("stepSlider");
  slider.max = String(g.frames.length - 1);
  slider.value = String(step);
  document.getElementById("stepLabel").textContent = `${{step}} / ${{g.frames.length - 1}}`;
  const frame = g.frames[step];
  renderBoard(g, frame);
  renderMeta(g);
  renderPlayers(g);
  renderDiagnostics(g);
  renderMoveDetails(frame);
  renderMoveTable(g);
}}

document.getElementById("prevBtn").addEventListener("click", () => {{
  step = Math.max(0, step - 1); render();
}});
document.getElementById("nextBtn").addEventListener("click", () => {{
  step = Math.min(currentGame().frames.length - 1, step + 1); render();
}});
document.getElementById("stepSlider").addEventListener("input", e => {{
  step = Number(e.target.value); render();
}});
document.addEventListener("keydown", e => {{
  if (e.key === "ArrowLeft") {{ step = Math.max(0, step - 1); render(); }}
  if (e.key === "ArrowRight") {{ step = Math.min(currentGame().frames.length - 1, step + 1); render(); }}
}});
setupSelect();
render();
</script>
</body>
</html>
"""


def build_summary(games: Iterable[Dict]) -> List[Dict]:
    out = []
    for g in games:
        out.append(
            {
                "id": g["id"],
                "num_players": g["num_players"],
                "seed": g["seed"],
                "candidate": g["candidate"],
                "winner": g["winner"],
                "terminal_reason": g["terminal_reason"],
                "move_count": g["move_count"],
                "candidate_margin": g["candidate_margin"],
                "candidate_win": g["candidate_win"],
                "diagnostics": g["diagnostics"],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--player-counts", default="2,5,6")
    ap.add_argument("--games-per-count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--policy-rollout-top-k", type=int, default=3)
    ap.add_argument("--policy-rollouts-per-move", type=int, default=2)
    ap.add_argument("--max-moves", type=int, default=500)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    counts = parse_counts(args.player_counts)

    print(f"[visual] loading {ckpt} on {device}")
    net = load_net(ckpt, device)

    games = []
    for n in counts:
        for i in range(args.games_per_count):
            seed = args.seed + n * 100 + i
            print(f"[visual] simulating {n}p seed={seed}")
            games.append(
                simulate_game(
                    net,
                    device,
                    n,
                    seed,
                    args.policy_rollout_top_k,
                    args.policy_rollouts_per_move,
                    args.max_moves,
                )
            )

    data = {
        "ckpt": str(ckpt),
        "cells": board_cells(),
        "games": games,
        "settings": {
            "player_counts": counts,
            "games_per_count": args.games_per_count,
            "seed": args.seed,
            "policy_rollout_top_k": args.policy_rollout_top_k,
            "policy_rollouts_per_move": args.policy_rollouts_per_move,
            "max_moves": args.max_moves,
        },
    }
    (out / "summary.json").write_text(json.dumps(build_summary(games), indent=2), encoding="utf-8")
    (out / "games.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (out / "index.html").write_text(html_page(data), encoding="utf-8")

    print(f"[visual] wrote {out / 'index.html'}")
    for row in build_summary(games):
        print(
            "[visual] "
            f"{row['id']} winner={row['winner']} reason={row['terminal_reason']} "
            f"moves={row['move_count']} margin={row['candidate_margin']:.1f} "
            f"diag={row['diagnostics']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
