#!/usr/bin/env python3
"""Local live match viewer for model-vs-heuristic games."""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from az.heuristic import heuristic_choose_move
from az.selfplay import _score_margin
from az.sim import Sim
from make_visual_replays import board_cells, choose_model_move, load_net

VALID_ROLES = {"model", "heuristic", "old"}


def _abs_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p


def _state_signature(sim: Sim) -> str:
    parts = [str(sim.current_turn_index)]
    for colour in sim.turn_order:
        parts.append(colour)
        parts.extend(str(p.axialindex) for p in sim.pins_by_colour[colour])
    return "|".join(parts)


def _capture_frame(sim: Sim, move_id: int, move: Dict | None) -> Dict:
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


class LiveEngine:
    def __init__(
        self,
        ckpt: str,
        device: str,
        num_players: int,
        seed: int,
        top_k: int,
        rollouts_per_move: int,
        max_moves: int,
        old_ckpt: str,
    ):
        self.lock = threading.Lock()
        self.device = torch.device(device)
        self.net = None
        self.old_net = None
        self.ckpt = ""
        self.old_ckpt = ""
        self.cells = board_cells()
        self.stats = {
            "games": 0,
            "candidate_wins": 0,
            "losses": 0,
            "max_moves": 0,
            "margins": [],
        }
        self.game_serial = 0
        self._load_checkpoint(ckpt)
        self._set_old_checkpoint(old_ckpt)
        self.new_game(
            num_players=num_players,
            seed=seed,
            top_k=top_k,
            rollouts_per_move=rollouts_per_move,
            max_moves=max_moves,
            reset_stats=False,
        )

    def _load_checkpoint(self, ckpt: str) -> None:
        p = _abs_path(ckpt)
        if str(p) == self.ckpt and self.net is not None:
            return
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        self.net = load_net(p, self.device)
        self.ckpt = str(p)

    def _set_old_checkpoint(self, old_ckpt: str) -> None:
        p = _abs_path(old_ckpt)
        if str(p) != self.old_ckpt:
            self.old_ckpt = str(p)
            self.old_net = None

    def _load_old_checkpoint(self):
        if self.old_net is not None:
            return self.old_net
        if not self.old_ckpt:
            raise FileNotFoundError("old checkpoint path is empty")
        p = Path(self.old_ckpt)
        if not p.exists():
            raise FileNotFoundError(f"old checkpoint not found: {p}")
        self.old_net = load_net(p, self.device)
        return self.old_net

    def _default_roles_unlocked(self) -> Dict[str, str]:
        return {
            colour: ("model" if i == 0 else "heuristic")
            for i, colour in enumerate(self.sim.turn_order)
        }

    def _normalise_roles_unlocked(self, roles: Dict | None) -> Dict[str, str]:
        if not roles:
            return self._default_roles_unlocked()
        out = {}
        for colour in self.sim.turn_order:
            role = str(roles.get(colour, "heuristic")).strip().lower()
            out[colour] = role if role in VALID_ROLES else "heuristic"
        if not any(role in ("model", "old") for role in out.values()):
            out[self.sim.turn_order[0]] = "model"
        return out

    def _sync_candidate_unlocked(self) -> None:
        for wanted in ("model", "old", "heuristic"):
            for colour in self.sim.turn_order:
                if self.roles.get(colour) == wanted:
                    self.candidate = colour
                    return
        self.candidate = self.sim.turn_order[0]

    def set_roles(self, roles: Dict | None = None, old_ckpt: str | None = None) -> Dict:
        with self.lock:
            if old_ckpt is not None:
                self._set_old_checkpoint(old_ckpt)
            self.roles = self._normalise_roles_unlocked(roles)
            if "old" in self.roles.values():
                self._load_old_checkpoint()
            self._sync_candidate_unlocked()
            self.settings["old_ckpt"] = self.old_ckpt
            self.settings["roles"] = dict(self.roles)
            return self.snapshot_unlocked()

    def new_game(
        self,
        num_players: int | None = None,
        seed: int | None = None,
        top_k: int | None = None,
        rollouts_per_move: int | None = None,
        max_moves: int | None = None,
        ckpt: str | None = None,
        old_ckpt: str | None = None,
        roles: Dict | None = None,
        reset_stats: bool = False,
    ) -> Dict:
        with self.lock:
            if ckpt:
                self._load_checkpoint(ckpt)
            if old_ckpt is not None:
                self._set_old_checkpoint(old_ckpt)
            if reset_stats:
                self.stats = {
                    "games": 0,
                    "candidate_wins": 0,
                    "losses": 0,
                    "max_moves": 0,
                    "margins": [],
                }
            if hasattr(self, "settings"):
                base = self.settings
            else:
                base = {
                    "num_players": 2,
                    "seed": 7,
                    "top_k": 3,
                    "rollouts_per_move": 2,
                    "max_moves": 500,
                    "old_ckpt": self.old_ckpt,
                    "roles": {},
                }
            self.settings = {
                "ckpt": self.ckpt,
                "old_ckpt": self.old_ckpt,
                "device": str(self.device),
                "num_players": int(num_players if num_players is not None else base["num_players"]),
                "seed": int(seed if seed is not None else base["seed"] + 1),
                "top_k": int(top_k if top_k is not None else base["top_k"]),
                "rollouts_per_move": int(
                    rollouts_per_move if rollouts_per_move is not None else base["rollouts_per_move"]
                ),
                "max_moves": int(max_moves if max_moves is not None else base["max_moves"]),
            }
            self.game_serial += 1
            self.sim = Sim(self.settings["num_players"], seed=self.settings["seed"])
            requested_roles = roles if roles is not None else base.get("roles")
            self.roles = self._normalise_roles_unlocked(requested_roles)
            if "old" in self.roles.values():
                self._load_old_checkpoint()
            self._sync_candidate_unlocked()
            self.settings["roles"] = dict(self.roles)
            self.rng = random.Random(self.settings["seed"] + 1009)
            self.frames = [_capture_frame(self.sim, 0, None)]
            self.moves = []
            self.result_recorded = False
            self.seen_states = {_state_signature(self.sim): 1}
            self.last_piece_move: Dict[Tuple[str, int], Tuple[int, int]] = {}
            self.visited_by_piece = {
                (colour, p.id): {p.axialindex}
                for colour in self.sim.turn_order
                for p in self.sim.pins_by_colour[colour]
            }
            self.diagnostics = {
                "illegal_moves": 0,
                "full_state_repeats": 0,
                "candidate_immediate_reversals": 0,
                "opponent_immediate_reversals": 0,
                "candidate_piece_revisits": 0,
                "opponent_piece_revisits": 0,
            }
            return self.snapshot_unlocked()

    def _record_result_unlocked(self) -> None:
        if self.result_recorded:
            return
        if not self.sim.is_terminal:
            return
        margin = float(_score_margin(self.sim, self.candidate))
        self.stats["games"] += 1
        self.stats["margins"].append(margin)
        if self.sim.terminal_reason == "MAX_MOVES":
            self.stats["max_moves"] += 1
        elif self.sim.winner == self.candidate:
            self.stats["candidate_wins"] += 1
        else:
            self.stats["losses"] += 1
        self.result_recorded = True

    def advance(self, steps: int = 1) -> Dict:
        with self.lock:
            for _ in range(max(1, int(steps))):
                if self.sim.is_terminal:
                    self._record_result_unlocked()
                    break
                self._advance_one_unlocked()
                if self.sim.is_terminal:
                    self._record_result_unlocked()
                    break
            return self.snapshot_unlocked()

    def _advance_one_unlocked(self) -> None:
        colour = self.sim.current_colour()
        legal = self.sim.legal_moves(colour)
        if not any(legal.values()):
            self.sim.skip_no_moves()
            move = {"move": self.sim.move_count, "colour": colour, "role": "skip", "flags": ["no_legal_moves"]}
            self.frames.append(_capture_frame(self.sim, self.sim.move_count, move))
            return

        role = "model" if colour == self.candidate else "heuristic"
        choice = {}
        role = self.roles.get(colour, "heuristic")
        if role == "model":
            pid, to_idx, choice = choose_model_move(
                self.net,
                self.device,
                self.sim,
                colour,
                legal,
                self.settings["top_k"],
                self.settings["rollouts_per_move"],
                self.settings["max_moves"],
            )
        elif role == "old":
            pid, to_idx, choice = choose_model_move(
                self._load_old_checkpoint(),
                self.device,
                self.sim,
                colour,
                legal,
                self.settings["top_k"],
                self.settings["rollouts_per_move"],
                self.settings["max_moves"],
            )
        else:
            pid, to_idx = heuristic_choose_move(self.sim, colour, legal, rng=self.rng)

        from_idx = self.sim.pins_by_colour[colour][pid].axialindex
        flags = []
        if to_idx not in legal.get(pid, []):
            self.diagnostics["illegal_moves"] += 1
            flags.append("illegal")

        previous = self.last_piece_move.get((colour, pid))
        if previous == (to_idx, from_idx):
            if colour == self.candidate:
                self.diagnostics["candidate_immediate_reversals"] += 1
            else:
                self.diagnostics["opponent_immediate_reversals"] += 1
            flags.append("immediate_reverse")

        if to_idx in self.visited_by_piece[(colour, pid)]:
            if colour == self.candidate:
                self.diagnostics["candidate_piece_revisits"] += 1
            else:
                self.diagnostics["opponent_piece_revisits"] += 1
            flags.append("piece_revisit")

        self.sim.apply_move(pid, to_idx)
        if self.sim.move_count >= self.settings["max_moves"] and not self.sim.is_terminal:
            self.sim.force_max_moves()

        sig = _state_signature(self.sim)
        if sig in self.seen_states:
            self.diagnostics["full_state_repeats"] += 1
            flags.append("full_state_repeat")
        self.seen_states[sig] = self.seen_states.get(sig, 0) + 1
        self.visited_by_piece[(colour, pid)].add(to_idx)
        self.last_piece_move[(colour, pid)] = (from_idx, to_idx)

        move = {
            "move": self.sim.move_count,
            "colour": colour,
            "role": role,
            "pid": pid,
            "from": from_idx,
            "to": to_idx,
            "flags": flags,
            "choice": choice,
            "candidate_margin_after": float(_score_margin(self.sim, self.candidate)),
        }
        self.moves.append(move)
        self.frames.append(_capture_frame(self.sim, self.sim.move_count, move))

    def snapshot(self) -> Dict:
        with self.lock:
            return self.snapshot_unlocked()

    def snapshot_unlocked(self) -> Dict:
        margins = self.stats["margins"]
        avg_margin = sum(margins) / len(margins) if margins else 0.0
        games = max(1, self.stats["games"])
        stats = dict(self.stats)
        stats["avg_margin"] = avg_margin
        stats["win_rate"] = self.stats["candidate_wins"] / games
        game = {
            "serial": self.game_serial,
            "candidate": self.candidate,
            "turn_order": self.sim.turn_order,
            "winner": self.sim.winner,
            "terminal_reason": self.sim.terminal_reason,
            "move_count": self.sim.move_count,
            "candidate_margin": float(_score_margin(self.sim, self.candidate)),
            "candidate_win": self.sim.winner == self.candidate,
            "diagnostics": dict(self.diagnostics),
        }
        return {
            "settings": dict(self.settings),
            "stats": stats,
            "game": game,
            "cells": self.cells,
            "frames": self.frames,
            "moves": self.moves,
        }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Chinese Checkers Match Viewer</title>
<style>
body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #17202a; background: #f5f7fa; }
header { padding: 14px 18px; background: #fff; border-bottom: 1px solid #d8dee6; display: grid; gap: 10px; }
main { display: grid; grid-template-columns: minmax(420px, 1fr) 430px; gap: 14px; padding: 14px; }
label { font-size: 12px; color: #536071; display: grid; gap: 4px; }
input, select, button { font: inherit; }
input, select { border: 1px solid #abb5c2; border-radius: 6px; padding: 6px 8px; background: #fff; min-width: 0; }
button { border: 1px solid #9da8b6; border-radius: 6px; padding: 7px 10px; background: #fff; cursor: pointer; }
button.primary { background: #1f6feb; color: white; border-color: #1f6feb; }
.grid { display: grid; grid-template-columns: 2fr repeat(6, minmax(72px, 110px)); gap: 8px; align-items: end; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.panel { background: #fff; border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; }
.toolbar { padding: 10px 12px; border-bottom: 1px solid #d8dee6; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.board-wrap { padding: 12px; }
svg { width: 100%; height: min(72vh, 760px); background: #fbfcfd; border: 1px solid #d8dee6; border-radius: 6px; }
.side { display: grid; gap: 14px; align-content: start; }
.section { padding: 12px; }
h1 { margin: 0; font-size: 19px; }
h2 { margin: 0 0 8px; font-size: 15px; }
p { margin: 0; color: #536071; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 8px; border-bottom: 1px solid #edf0f3; text-align: left; vertical-align: top; }
tr.active { background: #fff7d6; }
.meta { display: grid; grid-template-columns: 130px 1fr; gap: 6px 10px; font-size: 13px; }
.players { display: grid; gap: 8px; }
.player-row { display: grid; grid-template-columns: 18px 86px 1fr; gap: 8px; align-items: center; font-size: 13px; }
.swatch { width: 16px; height: 16px; border-radius: 50%; border: 2px solid #17202a; box-sizing: border-box; }
.role { color: #536071; }
.role.me { color: #0f5fc9; font-weight: 700; }
.role.old { color: #9a5200; font-weight: 700; }
.role-select { width: 100%; }
.players-actions { display: flex; gap: 8px; margin-top: 10px; }
.muted { color: #6b7685; }
.ok { color: #207a3c; font-weight: 700; }
.bad { color: #b42318; font-weight: 700; }
.flag { display: inline-block; margin: 1px 3px 1px 0; padding: 1px 5px; border-radius: 999px; background: #ffe0e0; color: #8f1d1d; font-size: 12px; }
#status { font-size: 13px; color: #536071; }
@media (max-width: 980px) { main { grid-template-columns: 1fr; } .grid { grid-template-columns: 1fr 1fr; } svg { height: 68vh; } }
</style>
</head>
<body>
<header>
  <div>
    <h1>Live Match Viewer</h1>
    <p>Configure each color as current model, heuristic greedy, or old checkpoint.</p>
  </div>
  <div class="grid">
    <label>Checkpoint <input id="ckptInput" type="text"></label>
    <label>Old Checkpoint <input id="oldCkptInput" type="text"></label>
    <label>Players <select id="playersInput"><option>2</option><option>3</option><option>4</option><option>5</option><option selected>6</option></select></label>
    <label>Seed <input id="seedInput" type="number" value="7"></label>
    <label>Max Moves <input id="maxMovesInput" type="number" value="500"></label>
    <label>Top K <input id="topKInput" type="number" value="3"></label>
    <label>Rollouts <input id="rolloutsInput" type="number" value="2"></label>
    <button class="primary" id="newBtn">New Game</button>
  </div>
  <div class="controls">
    <button id="playBtn">Play</button>
    <button id="backBtn">Back</button>
    <button id="forwardBtn">Forward</button>
    <label style="display:flex; gap:8px; align-items:center;">Speed <input id="speedInput" type="range" min="0.5" max="12" step="0.5" value="3"><span id="speedLabel">3x</span></label>
    <label style="display:flex; gap:8px; align-items:center;"><input id="autoNextInput" type="checkbox"> Auto next game</label>
    <button id="resetStatsBtn">Reset Stats</button>
    <span id="status"></span>
  </div>
</header>
<main>
  <section class="panel">
    <div class="toolbar">
      <input id="stepSlider" type="range" min="0" max="0" value="0" style="flex:1;">
      <span id="stepLabel"></span>
    </div>
    <div class="board-wrap"><svg id="board"></svg></div>
  </section>
  <aside class="side">
    <section class="panel"><div class="section"><h2>Game</h2><div id="gameMeta" class="meta"></div></div></section>
    <section class="panel"><div class="section"><h2>Players</h2><div id="playersLegend" class="players"></div><div class="players-actions"><button id="applyRolesBtn">Apply Roles</button></div></div></section>
    <section class="panel"><div class="section"><h2>Running Stats</h2><div id="statsMeta" class="meta"></div></div></section>
    <section class="panel"><div class="section"><h2>Diagnostics</h2><div id="diagnostics"></div></div></section>
    <section class="panel"><div class="section"><h2>Current Move</h2><div id="moveDetails"></div></div></section>
    <section class="panel"><div class="section"><h2>Move Window</h2><div id="moveTable"></div></div></section>
  </aside>
</main>
<script>
let DATA = null;
let frameIndex = 0;
let playing = false;
let timer = null;
const colourFill = {"red":"#d13b3b","blue":"#2676cc","yellow":"#e6bd27","lawn green":"#55a84f","purple":"#8b55c7","gray0":"#333944","board":"#f3f6f8"};
const textFill = {"yellow":"#20242a","lawn green":"#102715"};

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c])); }
function cell(i) { return DATA.cells[i]; }
function flagsHtml(flags) {
  if (!flags || !flags.length) return '<span class="muted">none</span>';
  return flags.map(f => `<span class="flag">${escapeHtml(f)}</span>`).join("");
}
function hexPoints(x, y, r) {
  const pts = [];
  for (let k = 0; k < 6; k++) {
    const a = Math.PI / 180 * (60 * k - 30);
    pts.push(`${(x + r * Math.cos(a)).toFixed(2)},${(y + r * Math.sin(a)).toFixed(2)}`);
  }
  return pts.join(" ");
}
async function postJson(path, body) {
  const res = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body || {})});
  if (!res.ok) throw new Error(await res.text());
  DATA = await res.json();
  return DATA;
}
async function fetchState() {
  const res = await fetch("/api/state");
  DATA = await res.json();
  hydrateInputs();
  frameIndex = Math.min(frameIndex, DATA.frames.length - 1);
  render();
}
function hydrateInputs() {
  document.getElementById("ckptInput").value = DATA.settings.ckpt;
  document.getElementById("oldCkptInput").value = DATA.settings.old_ckpt || "";
  document.getElementById("playersInput").value = String(DATA.settings.num_players);
  document.getElementById("seedInput").value = String(DATA.settings.seed);
  document.getElementById("maxMovesInput").value = String(DATA.settings.max_moves);
  document.getElementById("topKInput").value = String(DATA.settings.top_k);
  document.getElementById("rolloutsInput").value = String(DATA.settings.rollouts_per_move);
}
function currentRoles() {
  const roles = {};
  document.querySelectorAll(".role-select").forEach(sel => {
    roles[sel.dataset.colour] = sel.value;
  });
  return roles;
}
function payload(resetStats=false, nextSeed=false) {
  const seedBox = document.getElementById("seedInput");
  const seed = Number(seedBox.value) + (nextSeed ? 1 : 0);
  seedBox.value = String(seed);
  return {
    ckpt: document.getElementById("ckptInput").value,
    old_ckpt: document.getElementById("oldCkptInput").value,
    num_players: Number(document.getElementById("playersInput").value),
    seed,
    max_moves: Number(document.getElementById("maxMovesInput").value),
    top_k: Number(document.getElementById("topKInput").value),
    rollouts_per_move: Number(document.getElementById("rolloutsInput").value),
    reset_stats: resetStats,
    roles: currentRoles()
  };
}
function updateStatus(text) { document.getElementById("status").textContent = text || ""; }
async function applyRoles() {
  updateStatus("applying roles...");
  DATA = await postJson("/api/roles", {
    old_ckpt: document.getElementById("oldCkptInput").value,
    roles: currentRoles()
  });
  hydrateInputs();
  render();
  updateStatus("");
}

async function newGame(resetStats=false, nextSeed=false) {
  updateStatus("starting game...");
  DATA = await postJson("/api/new", payload(resetStats, nextSeed));
  frameIndex = 0;
  hydrateInputs();
  render();
  updateStatus("");
}
async function ensureForward() {
  if (frameIndex < DATA.frames.length - 1) {
    frameIndex += 1;
    render();
    return;
  }
  if (DATA.game.terminal_reason) {
    if (document.getElementById("autoNextInput").checked) {
      await newGame(false, true);
    }
    return;
  }
  updateStatus("thinking...");
  DATA = await postJson("/api/advance", {steps: 1});
  frameIndex = DATA.frames.length - 1;
  render();
  updateStatus("");
}
function speedMs() {
  const v = Number(document.getElementById("speedInput").value);
  return Math.max(80, 1000 / v);
}
function schedule() {
  clearTimeout(timer);
  if (!playing) return;
  timer = setTimeout(async () => {
    try { await ensureForward(); } catch (e) { updateStatus(String(e)); playing = false; }
    schedule();
  }, speedMs());
}
function setPlaying(v) {
  playing = v;
  document.getElementById("playBtn").textContent = playing ? "Pause" : "Play";
  schedule();
}
function renderBoard(frame) {
  const svg = document.getElementById("board");
  const xs = DATA.cells.map(c => c.x), ys = DATA.cells.map(c => c.y);
  const minX = Math.min(...xs) - 48, maxX = Math.max(...xs) + 48;
  const minY = Math.min(...ys) - 48, maxY = Math.max(...ys) + 48;
  svg.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);
  svg.innerHTML = "";
  DATA.cells.forEach(c => {
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("points", hexPoints(c.x, c.y, 17));
    poly.setAttribute("fill", colourFill[c.postype] || "#e7ebef");
    poly.setAttribute("fill-opacity", c.postype === "board" ? "1" : "0.22");
    poly.setAttribute("stroke", "#c4ccd6");
    svg.appendChild(poly);
  });
  const last = frame.move;
  if (last && Number.isInteger(last.from) && Number.isInteger(last.to)) {
    const a = cell(last.from), b = cell(last.to);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
    line.setAttribute("stroke", "#111827");
    line.setAttribute("stroke-width", "5");
    line.setAttribute("stroke-opacity", "0.38");
    line.setAttribute("stroke-linecap", "round");
    svg.appendChild(line);
  }
  Object.entries(frame.pins).forEach(([colour, positions]) => {
    const role = roleOf(colour);
    positions.forEach((idx, pid) => {
      const c = cell(idx);
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", c.x); circle.setAttribute("cy", c.y);
      circle.setAttribute("r", role === "model" ? "12.8" : role === "old" ? "12" : "10.5");
      circle.setAttribute("fill", colourFill[colour] || "#777");
      circle.setAttribute("stroke", role === "model" ? "#111827" : role === "old" ? "#c26a00" : "#ffffff");
      circle.setAttribute("stroke-width", role === "heuristic" ? "2" : "3");
      svg.appendChild(circle);
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", c.x); label.setAttribute("y", c.y + 4);
      label.setAttribute("font-size", "10"); label.setAttribute("font-weight", "700");
      label.setAttribute("text-anchor", "middle"); label.setAttribute("fill", textFill[colour] || "#ffffff");
      label.textContent = String(pid);
      svg.appendChild(label);
    });
  });
}
function renderMeta() {
  const g = DATA.game, s = DATA.settings;
  const rows = [
    ["players", s.num_players], ["tracked color", g.candidate], ["turn order", g.turn_order.join(", ")],
    ["winner", g.winner || "none"], ["terminal", g.terminal_reason || "playing"],
    ["moves", g.move_count], ["candidate margin", g.candidate_margin.toFixed(1)], ["game seed", s.seed]
  ];
  document.getElementById("gameMeta").innerHTML = rows.map(([k,v]) => `<div class="muted">${escapeHtml(k)}</div><div>${escapeHtml(v)}</div>`).join("");
  const st = DATA.stats;
  const rows2 = [
    ["games", st.games], ["win rate", (100 * st.win_rate).toFixed(1) + "%"], ["wins", st.candidate_wins],
    ["losses", st.losses], ["max moves", st.max_moves], ["avg margin", st.avg_margin.toFixed(1)]
  ];
  document.getElementById("statsMeta").innerHTML = rows2.map(([k,v]) => `<div class="muted">${escapeHtml(k)}</div><div>${escapeHtml(v)}</div>`).join("");
}
function ckptName() {
  const parts = String(DATA.settings.ckpt || "").split(/[\\/]/);
  return parts[parts.length - 1] || "checkpoint";
}
function oldCkptName() {
  const parts = String(DATA.settings.old_ckpt || "").split(/[\\/]/);
  return parts[parts.length - 1] || "old checkpoint";
}
function roleOf(colour) {
  return (DATA.settings.roles && DATA.settings.roles[colour]) || "heuristic";
}
function roleText(role) {
  if (role === "model") return `ME / current checkpoint (${ckptName()})`;
  if (role === "old") return `old checkpoint (${oldCkptName()})`;
  return "heuristic greedy";
}
function renderPlayers() {
  document.getElementById("playersLegend").innerHTML = DATA.game.turn_order.map(colour => {
    const role = roleOf(colour);
    return `<div class="player-row">
      <span class="swatch" style="background:${colourFill[colour] || "#777"}"></span>
      <strong>${escapeHtml(colour)}</strong>
      <select class="role-select role ${role === "model" ? "me" : role === "old" ? "old" : ""}" data-colour="${escapeHtml(colour)}">
        <option value="model" ${role === "model" ? "selected" : ""}>ME / current</option>
        <option value="heuristic" ${role === "heuristic" ? "selected" : ""}>heuristic greedy</option>
        <option value="old" ${role === "old" ? "selected" : ""}>old checkpoint</option>
      </select>
      <div class="role ${role === "model" ? "me" : role === "old" ? "old" : ""}" style="grid-column: 3;">${escapeHtml(roleText(role))}</div>
    </div>`;
  }).join("");
}
function renderDiagnostics() {
  const d = DATA.game.diagnostics;
  const rows = Object.entries(d).map(([k,v]) => {
    const bad = v > 0 && (k.includes("illegal") || k.includes("full_state") || k.includes("candidate_immediate"));
    return `<tr><td>${escapeHtml(k)}</td><td class="${bad ? "bad" : "ok"}">${v}</td></tr>`;
  }).join("");
  document.getElementById("diagnostics").innerHTML = `<table>${rows}</table>`;
}
function renderMoveDetails(frame) {
  const m = frame.move;
  if (!m) {
    document.getElementById("moveDetails").innerHTML = '<p class="muted">Initial board.</p>';
    return;
  }
  let choice = "";
  if (m.choice && m.choice.top) {
    const rows = m.choice.top.map(x => `<tr><td>${x.pid} -> ${x.to}</td><td>${Number(x.logit).toFixed(2)}</td><td>${x.mean_margin === null ? "n/a" : Number(x.mean_margin).toFixed(1)}</td></tr>`).join("");
    choice = `<h2>Model Candidates</h2><table><tr><th>move</th><th>logit</th><th>rollout</th></tr>${rows}</table>`;
  }
  document.getElementById("moveDetails").innerHTML = `
    <table>
      <tr><td>move</td><td>${m.move}</td></tr><tr><td>role</td><td>${escapeHtml(m.role)}</td></tr>
      <tr><td>colour</td><td>${escapeHtml(m.colour)}</td></tr><tr><td>pin</td><td>${m.pid}</td></tr>
      <tr><td>from -> to</td><td>${m.from} -> ${m.to}</td></tr><tr><td>flags</td><td>${flagsHtml(m.flags)}</td></tr>
      <tr><td>candidate margin</td><td>${Number(m.candidate_margin_after).toFixed(1)}</td></tr>
    </table>${choice}`;
}
function renderMoveTable() {
  const start = Math.max(0, frameIndex - 8);
  const end = Math.min(DATA.moves.length, frameIndex + 8);
  const rows = DATA.moves.slice(start, end).map(m => `
    <tr class="${m.move === frameIndex ? "active" : ""}">
      <td>${m.move}</td><td>${escapeHtml(m.role)}</td><td>${escapeHtml(m.colour)}</td>
      <td>${m.pid}</td><td>${m.from} -> ${m.to}</td><td>${flagsHtml(m.flags)}</td>
    </tr>`).join("");
  document.getElementById("moveTable").innerHTML = `<table><tr><th>#</th><th>role</th><th>colour</th><th>pin</th><th>move</th><th>flags</th></tr>${rows}</table>`;
}
function render() {
  if (!DATA) return;
  frameIndex = Math.max(0, Math.min(frameIndex, DATA.frames.length - 1));
  const slider = document.getElementById("stepSlider");
  slider.max = String(DATA.frames.length - 1);
  slider.value = String(frameIndex);
  document.getElementById("stepLabel").textContent = `${frameIndex} / ${DATA.frames.length - 1}`;
  const frame = DATA.frames[frameIndex];
  renderBoard(frame);
  renderMeta();
  renderPlayers();
  renderDiagnostics();
  renderMoveDetails(frame);
  renderMoveTable();
}
document.getElementById("newBtn").addEventListener("click", () => newGame(false, false).catch(e => updateStatus(String(e))));
document.getElementById("resetStatsBtn").addEventListener("click", () => newGame(true, false).catch(e => updateStatus(String(e))));
document.getElementById("applyRolesBtn").addEventListener("click", () => applyRoles().catch(e => updateStatus(String(e))));
document.getElementById("playBtn").addEventListener("click", () => setPlaying(!playing));
document.getElementById("backBtn").addEventListener("click", () => { frameIndex = Math.max(0, frameIndex - 1); render(); });
document.getElementById("forwardBtn").addEventListener("click", () => ensureForward().catch(e => updateStatus(String(e))));
document.getElementById("stepSlider").addEventListener("input", e => { frameIndex = Number(e.target.value); render(); });
document.getElementById("speedInput").addEventListener("input", e => { document.getElementById("speedLabel").textContent = e.target.value + "x"; schedule(); });
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { frameIndex = Math.max(0, frameIndex - 1); render(); }
  if (e.key === "ArrowRight") { ensureForward().catch(err => updateStatus(String(err))); }
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
});
fetchState().catch(e => updateStatus(String(e)));
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    engine: LiveEngine

    def log_message(self, fmt, *args):
        sys.stdout.write("[viewer] " + fmt % args + "\n")

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> Dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps(self.engine.snapshot()).encode("utf-8"))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:
            self._send(500, str(exc).encode("utf-8"), "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._json_body()
            if path == "/api/new":
                state = self.engine.new_game(
                    ckpt=body.get("ckpt"),
                    old_ckpt=body.get("old_ckpt"),
                    roles=body.get("roles"),
                    num_players=int(body.get("num_players", 2)),
                    seed=int(body.get("seed", 7)),
                    max_moves=int(body.get("max_moves", 500)),
                    top_k=int(body.get("top_k", 3)),
                    rollouts_per_move=int(body.get("rollouts_per_move", 2)),
                    reset_stats=bool(body.get("reset_stats", False)),
                )
            elif path == "/api/roles":
                state = self.engine.set_roles(
                    roles=body.get("roles"),
                    old_ckpt=body.get("old_ckpt"),
                )
            elif path == "/api/advance":
                state = self.engine.advance(int(body.get("steps", 1)))
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, json.dumps(state).encode("utf-8"))
        except Exception as exc:
            self._send(500, str(exc).encode("utf-8"), "text/plain")


def pick_port(host: str, start_port: int) -> int:
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port found from {start_port}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/best.pt")
    ap.add_argument("--old-ckpt", default="runs/best.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--num-players", type=int, default=6, choices=(2, 3, 4, 5, 6))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-moves", type=int, default=500)
    ap.add_argument("--policy-rollout-top-k", type=int, default=3)
    ap.add_argument("--policy-rollouts-per-move", type=int, default=2)
    args = ap.parse_args()

    port = pick_port(args.host, args.port)
    Handler.engine = LiveEngine(
        ckpt=args.ckpt,
        device=args.device,
        num_players=args.num_players,
        seed=args.seed,
        top_k=args.policy_rollout_top_k,
        rollouts_per_move=args.policy_rollouts_per_move,
        max_moves=args.max_moves,
        old_ckpt=args.old_ckpt,
    )
    server = ThreadingHTTPServer((args.host, port), Handler)
    print(f"[viewer] serving http://{args.host}:{port}")
    print(f"[viewer] checkpoint={Handler.engine.ckpt}")
    print(f"[viewer] old_checkpoint={Handler.engine.old_ckpt}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
