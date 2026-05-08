#!/usr/bin/env python3
"""
Game Visualizer — reads game logs and renders board states as images.
Usage: python game_visualizer.py <game_log_file>
Outputs: renders/<game_id>/*.png and renders/<game_id>/viewer.html
"""

import argparse
import html
import http.server
import json
import re
import socketserver
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

from checkers_board import HexBoard
from checkers_pins import Pin


class GameVisualizer:
    def __init__(self, log_file: str, max_moves: Optional[int] = None, stride: int = 1):
        self.log_file = log_file
        self.game_id = Path(log_file).stem.replace('game_', '')
        script_dir = Path(__file__).resolve().parent
        self.output_dir = script_dir / "renders" / self.game_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_moves = max_moves
        self.stride = max(1, int(stride))
        
        # Parse log to build move sequence
        self.moves = []
        self.players = {}
        self.colors = {}
        self.turn_order = []
        self.scores_by_move = {}
        self.parse_log()
        
    def parse_log(self):
        """Extract move sequence and player info from log."""
        with open(self.log_file, 'r') as f:
            for line in f:
                # Player joins: [timestamp] PLAYER JOINED: Name as color
                m = re.search(r'PLAYER JOINED:\s+(.+?)\s+as\s+(.+?)\s*$', line.strip())
                if m:
                    name, color = m.groups()
                    color = color.strip()
                    self.players[color] = name
                    self.colors[name] = color

                m = re.search(r'GAME START.*turn order\s+\[(.*?)\]', line)
                if m:
                    self.turn_order = [
                        part.strip().strip("'\"")
                        for part in m.group(1).split(',')
                        if part.strip()
                    ]
                
                # Move: [timestamp] MOVE N: Name (color) from->to [time]
                m = re.search(
                    r'MOVE\s+(\d+):\s+(.+?)\s+\((.+?)\)\s+(\d+)->(\d+)\s+\[([0-9.]+)ms\]',
                    line,
                )
                if m:
                    move_num, name, color, from_idx, to_idx, move_ms = m.groups()
                    self.moves.append({
                        'num': int(move_num),
                        'player': name.strip(),
                        'color': color.strip(),
                        'from': int(from_idx),
                        'to': int(to_idx),
                        'move_ms': float(move_ms),
                    })

                m = re.search(
                    r'SCORE\s+(.+?)\s+\((.+?)\):\s+Final=([0-9.]+),\s+'
                    r'Time=([0-9.]+),\s+Moves\((\d+)\)=([0-9.]+),\s+'
                    r'Pins\((\d+)\)=([0-9.]+),\s+Dist=([0-9.]+)',
                    line,
                )
                if m and self.moves:
                    name, color, final, time_score, moves, move_score, pins, pin_score, dist = m.groups()
                    self.scores_by_move.setdefault(self.moves[-1]['num'], []).append({
                        'player': name.strip(),
                        'color': color.strip(),
                        'final': float(final),
                        'time': float(time_score),
                        'moves': int(moves),
                        'move_score': float(move_score),
                        'pins': int(pins),
                        'pin_score': float(pin_score),
                        'dist': float(dist),
                    })
    
    def render_board(self, board: HexBoard, pins: Dict[str, List[Pin]], move_num: int = 0, title: str = "", highlight: Tuple[str, int] = None):
        """Render hexagonal board with pins to a matplotlib figure.

        highlight: optional (colour, axial_index) to draw a ring around the last moved pin.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Color mapping for board zones
        zone_colors = {
            'red': '#ffcccc',
            'lawn green': '#ccffcc',
            'blue': '#ccccff',
            'yellow': '#ffffcc',
            'purple': '#ffccff',
            'gray0': '#cccccc',
            'board': '#f0f0f0'
        }
        
        # Draw board cells
        for cell in board.cells:
            x, y = cell.x, cell.y
            hex_patch = RegularPolygon(
                (x, y), 6, radius=board.hole_radius / 1.1,
                facecolor=zone_colors.get(cell.postype, '#ffffff'),
                edgecolor='black', linewidth=1
            )
            ax.add_patch(hex_patch)
        
        # Draw pins
        pin_colors = {
            'red': '#cc0000',
            'lawn green': '#00cc00',
            'blue': '#0000cc',
            'yellow': '#cccc00',
            'purple': '#cc00cc',
            'gray0': '#666666'
        }
        
        for color, pin_list in pins.items():
            for pin in pin_list:
                x, y = pin.position
                circle = plt.Circle((x, y), board.hole_radius * 0.5, 
                                   color=pin_colors.get(color, '#999999'),
                                   ec='black', linewidth=2, zorder=10)
                ax.add_patch(circle)

        # Highlight last moved pin with a ring
        if highlight:
            h_colour, h_idx = highlight
            if 0 <= h_idx < len(board.cells):
                hx, hy = board.cartesian[h_idx]
                ring = plt.Circle((hx, hy), board.hole_radius * 0.65,
                                  fill=False, lw=3, ec='#ff8800', zorder=15)
                ax.add_patch(ring)
        
        # Set axis limits and aspect
        xs = [cell.x for cell in board.cells]
        ys = [cell.y for cell in board.cells]
        margin = 50
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title
        if not title:
            title = f"Initial Board State"
        if move_num > 0:
            title = f"Move {move_num}"
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig
    
    def _should_render_move(self, move_num: int) -> bool:
        if self.max_moves is not None and move_num > self.max_moves:
            return False
        return move_num == 1 or move_num % self.stride == 0

    def visualize(self):
        """Render game progression as sequence of images."""
        board = HexBoard()
        pins_by_colour = {}
        
        # Initialize pins for all colours present in this game log.
        for color in self.players.keys():
            idxs = board.axial_of_colour(color)[:10]
            pins_by_colour[color] = [
                Pin(board, idxs[i], id=i, color=color)
                for i in range(len(idxs))
            ]
        
        # Render initial state
        fig = self.render_board(board, pins_by_colour, move_num=0, title="Initial Board State")
        initial_image = "move_000_initial.png"
        fig.savefig(str(self.output_dir / initial_image), dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Rendered initial state")

        frames = [{
            "move": 0,
            "label": "Initial board",
            "image": initial_image,
            "player": "",
            "color": "",
            "from": "",
            "to": "",
            "move_ms": "",
            "scores": [],
        }]
        
        rendered_moves = 0
        skipped_moves = 0
        applied_moves = 0

        # Apply each move and render
        for move in self.moves:
            color = move['color']
            from_idx = move['from']
            to_idx = move['to']

            if self.max_moves is not None and move['num'] > self.max_moves:
                break
            
            # Find the pin at from_idx
            if color not in pins_by_colour:
                skipped_moves += 1
                continue
            
            pin_found = None
            for pin in pins_by_colour[color]:
                if pin.axialindex == from_idx:
                    pin_found = pin
                    break
            
            if pin_found:
                # Apply move
                pin_found.placePin(to_idx)
                applied_moves += 1

                if not self._should_render_move(move['num']):
                    continue
                
                # Render
                fig = self.render_board(
                    board, pins_by_colour, 
                    move_num=move['num'],
                    title=f"Move {move['num']}: {move['player']} ({color}) {from_idx}→{to_idx}",
                    highlight=(color, to_idx)
                )
                image_name = f"move_{move['num']:03d}_{self._safe_filename(move['player'])}.png"
                fig.savefig(str(self.output_dir / image_name), dpi=100, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Rendered move {move['num']}: {move['player']} ({color}) {from_idx}→{to_idx}")
                frames.append({
                    "move": move['num'],
                    "label": f"Move {move['num']}: {move['player']} ({color}) {from_idx}->{to_idx}",
                    "image": image_name,
                    "player": move['player'],
                    "color": color,
                    "from": from_idx,
                    "to": to_idx,
                    "move_ms": move.get('move_ms', ''),
                    "scores": self.scores_by_move.get(move['num'], []),
                })
                rendered_moves += 1
            else:
                skipped_moves += 1

        viewer_path = self.write_viewer(frames)
        print(f"\n✅ Game visualization saved to {self.output_dir}/")
        print(f"Viewer: {viewer_path}")
        print(
            f"Parsed moves: {len(self.moves)} | Applied: {applied_moves} | "
            f"Rendered: {rendered_moves} | Skipped: {skipped_moves}"
        )

    @staticmethod
    def _safe_filename(value: str) -> str:
        safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', value.strip().lower())
        return safe or "player"

    def write_viewer(self, frames: List[Dict]) -> Path:
        """Write an interactive HTML viewer for the rendered frames."""
        viewer_path = self.output_dir / "viewer.html"
        frames_json = json.dumps(frames)
        player_rows = "\n".join(
            f"<li><span class=\"swatch {self._css_class(color)}\"></span>"
            f"<strong>{html.escape(name)}</strong> <span>{html.escape(color)}</span></li>"
            for color, name in self.players.items()
        )
        turn_order = " -> ".join(html.escape(color) for color in self.turn_order) or "unknown"
        source_name = html.escape(Path(self.log_file).name)

        viewer_path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chinese Checkers Viewer - {html.escape(self.game_id)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --ink: #1d2428;
      --muted: #66737a;
      --line: #d8e0e4;
      --accent: #1f7a8c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 3;
    }}
    h1 {{ margin: 0; font-size: 18px; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 360px;
      min-height: calc(100vh - 61px);
    }}
    .stage {{
      padding: 18px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
      min-width: 0;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 10px;
    }}
    button {{
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      padding: 7px 10px;
      cursor: pointer;
      font: inherit;
    }}
    button:hover {{ border-color: var(--accent); }}
    input[type="range"] {{ flex: 1 1 220px; min-width: 160px; }}
    .counter {{ min-width: 90px; color: var(--muted); text-align: right; }}
    .board {{
      background: var(--panel);
      border: 1px solid var(--line);
      display: grid;
      place-items: center;
      overflow: hidden;
      min-height: 0;
    }}
    .board img {{
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 170px);
      object-fit: contain;
      display: block;
    }}
    .caption {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 12px;
      display: grid;
      gap: 4px;
    }}
    .caption strong {{ font-size: 16px; }}
    aside {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-width: 0;
    }}
    .side-section {{ padding: 14px; border-bottom: 1px solid var(--line); }}
    .side-section h2 {{ margin: 0 0 8px; font-size: 14px; }}
    ul {{ margin: 0; padding: 0; list-style: none; }}
    .players li {{ display: flex; align-items: center; gap: 8px; padding: 4px 0; }}
    .players span:last-child {{ margin-left: auto; color: var(--muted); }}
    .swatch {{ width: 12px; height: 12px; border: 1px solid rgba(0,0,0,.25); display: inline-block; }}
    .red {{ background: #cc0000; }}
    .lawn-green {{ background: #00cc00; }}
    .blue {{ background: #0000cc; }}
    .yellow {{ background: #cccc00; }}
    .purple {{ background: #cc00cc; }}
    .gray0 {{ background: #666666; }}
    .moves {{
      overflow: auto;
      padding: 8px;
    }}
    .move {{
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      background: transparent;
      padding: 8px;
      display: grid;
      gap: 2px;
    }}
    .move.active {{
      background: #e8f3f5;
      border-color: #9ccbd3;
    }}
    .move small {{ color: var(--muted); }}
    .scores {{
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 6px 10px;
      margin-top: 8px;
      font-size: 13px;
    }}
    .scores div {{ border-bottom: 1px solid #edf1f3; padding-bottom: 3px; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
      .board img {{ max-height: 68vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Chinese Checkers Viewer</h1>
      <div class="meta">{source_name} · {len(frames)} rendered frames · turn order: {turn_order}</div>
    </div>
  </header>
  <main>
    <section class="stage">
      <div class="toolbar">
        <button id="first">First</button>
        <button id="prev">Prev</button>
        <button id="play">Play</button>
        <button id="next">Next</button>
        <button id="last">Last</button>
        <input id="slider" type="range" min="0" max="{max(0, len(frames) - 1)}" value="0">
        <span class="counter" id="counter"></span>
      </div>
      <div class="board"><img id="board" src="{frames[0]['image'] if frames else ''}" alt="Board state"></div>
      <div class="caption">
        <strong id="title"></strong>
        <span class="meta" id="detail"></span>
        <div class="scores" id="scores"></div>
      </div>
    </section>
    <aside>
      <section class="side-section">
        <h2>Players</h2>
        <ul class="players">{player_rows}</ul>
      </section>
      <section class="side-section meta">
        Use Left/Right to step moves. Space toggles playback.
      </section>
      <section class="moves" id="moves"></section>
    </aside>
  </main>
  <script>
    const frames = {frames_json};
    let index = 0;
    let timer = null;
    const board = document.getElementById('board');
    const title = document.getElementById('title');
    const detail = document.getElementById('detail');
    const scores = document.getElementById('scores');
    const slider = document.getElementById('slider');
    const counter = document.getElementById('counter');
    const moveList = document.getElementById('moves');
    const playButton = document.getElementById('play');

    function scoreTable(frame) {{
      if (!frame.scores || frame.scores.length === 0) return '';
      const header = '<div><strong>Player</strong></div><div><strong>Score</strong></div><div><strong>Pins</strong></div><div><strong>Dist</strong></div>';
      const rows = frame.scores.map(s =>
        `<div>${{s.player}} (${{s.color}})</div><div>${{s.final.toFixed(1)}}</div><div>${{s.pins}}</div><div>${{s.dist.toFixed(1)}}</div>`
      ).join('');
      return header + rows;
    }}

    function renderMoveList() {{
      moveList.innerHTML = frames.map((frame, i) => `
        <button class="move" data-index="${{i}}">
          <strong>${{frame.move === 0 ? 'Initial board' : 'Move ' + frame.move + ': ' + frame.player}}</strong>
          <small>${{frame.move === 0 ? '' : frame.color + ' ' + frame.from + ' -> ' + frame.to}}</small>
        </button>
      `).join('');
      moveList.addEventListener('click', (event) => {{
        const item = event.target.closest('[data-index]');
        if (item) show(Number(item.dataset.index));
      }});
    }}

    function show(nextIndex) {{
      index = Math.max(0, Math.min(frames.length - 1, nextIndex));
      const frame = frames[index];
      board.src = frame.image;
      title.textContent = frame.label;
      detail.textContent = frame.move === 0
        ? 'Initial position'
        : `${{frame.color}} moved ${{frame.from}} -> ${{frame.to}} in ${{frame.move_ms}} ms`;
      scores.innerHTML = scoreTable(frame);
      slider.value = String(index);
      counter.textContent = `${{index + 1}} / ${{frames.length}}`;
      document.querySelectorAll('.move').forEach((el, i) => {{
        el.classList.toggle('active', i === index);
        if (i === index) el.scrollIntoView({{ block: 'nearest' }});
      }});
    }}

    function togglePlay() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
        playButton.textContent = 'Play';
        return;
      }}
      playButton.textContent = 'Pause';
      timer = setInterval(() => {{
        if (index >= frames.length - 1) {{
          togglePlay();
        }} else {{
          show(index + 1);
        }}
      }}, 650);
    }}

    document.getElementById('first').onclick = () => show(0);
    document.getElementById('prev').onclick = () => show(index - 1);
    document.getElementById('play').onclick = togglePlay;
    document.getElementById('next').onclick = () => show(index + 1);
    document.getElementById('last').onclick = () => show(frames.length - 1);
    slider.oninput = () => show(Number(slider.value));
    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowLeft') show(index - 1);
      if (event.key === 'ArrowRight') show(index + 1);
      if (event.key === ' ') {{
        event.preventDefault();
        togglePlay();
      }}
    }});

    renderMoveList();
    show(0);
  </script>
</body>
</html>
""", encoding="utf-8")
        return viewer_path

    @staticmethod
    def _css_class(color: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]+', '-', color.strip().lower())


def find_game_dir() -> Optional[Path]:
    script_dir = Path(__file__).resolve().parent
    candidate_dirs = [
        script_dir / "games",
        script_dir.parent / "games",
        script_dir.parent.parent / "games",
        Path("games"),
        Path("../games"),
        Path("../../games"),
    ]
    for path in candidate_dirs:
        if path.exists() and path.is_dir():
            return path
    return None


def choose_log_interactively(game_dir: Path) -> str:
    log_files = sorted(game_dir.glob("game_*.log"), key=lambda path: path.stat().st_mtime)
    if not log_files:
        print("Error: no game logs found in games/")
        sys.exit(1)

    print("Available game logs:")
    for idx, path in enumerate(reversed(log_files), start=1):
        print(f"  {idx}. {path.name}")

    choice = input("Select game number (Enter = latest): ").strip()
    if choice:
        try:
            choice_idx = int(choice)
            ordered_logs = list(reversed(log_files))
            if not (1 <= choice_idx <= len(ordered_logs)):
                raise ValueError
            return str(ordered_logs[choice_idx - 1])
        except ValueError:
            print("Error: invalid selection.")
            sys.exit(1)
    return str(log_files[-1])


class LiveGameSource:
    def __init__(self, log_file: Optional[str], follow_latest: bool):
        self.log_file = Path(log_file).resolve() if log_file else None
        self.follow_latest = follow_latest or self.log_file is None
        self.game_dir = find_game_dir()
        self.board = HexBoard()
        self.board_payload = self._build_board_payload()

    def _build_board_payload(self) -> Dict:
        xs = [cell.x for cell in self.board.cells]
        ys = [cell.y for cell in self.board.cells]
        pad = 42
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cells = []
        for idx, cell in enumerate(self.board.cells):
            cells.append({
                "index": idx,
                "q": cell.q,
                "r": cell.r,
                "x": round(cell.x - min_x + pad, 2),
                # SVG's Y axis points down, so invert to match the rendered PNG orientation.
                "y": round(max_y - cell.y + pad, 2),
                "zone": cell.postype,
            })
        return {
            "width": round(max_x - min_x + pad * 2, 2),
            "height": round(max_y - min_y + pad * 2, 2),
            "hole_radius": self.board.hole_radius,
            "cells": cells,
        }

    def _select_log(self) -> Optional[Path]:
        if self.follow_latest:
            if not self.game_dir:
                return None
            logs = sorted(self.game_dir.glob("game_*.log"), key=lambda path: path.stat().st_mtime)
            return logs[-1].resolve() if logs else None
        return self.log_file

    def state(self) -> Dict:
        log_path = self._select_log()
        base_state = {
            "ok": True,
            "board": self.board_payload,
            "log_file": str(log_path) if log_path else "",
            "game_id": log_path.stem.replace("game_", "") if log_path else "",
            "updated_at": time.strftime("%H:%M:%S"),
            "players": [],
            "turn_order": [],
            "status": "waiting for log",
            "move_count": 0,
            "initial_pins": {},
            "pins": {},
            "last_move": None,
            "moves": [],
            "recent_moves": [],
            "scores_by_move": {},
            "scores": [],
            "warnings": [],
        }
        if not log_path or not log_path.exists():
            base_state["ok"] = False
            base_state["warnings"].append("No game log found yet.")
            return base_state

        try:
            return self._parse_log(log_path, base_state)
        except Exception as exc:
            base_state["ok"] = False
            base_state["warnings"].append(f"Could not parse log: {exc}")
            return base_state

    def _parse_log(self, log_path: Path, state: Dict) -> Dict:
        players_by_color: Dict[str, str] = {}
        moves: List[Dict] = []
        scores_by_move: Dict[int, List[Dict]] = {}
        turn_order: List[str] = []
        status = "created"

        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.search(r'PLAYER JOINED:\s+(.+?)\s+as\s+(.+?)\s*$', line.strip())
                if m:
                    name, color = m.groups()
                    players_by_color[color.strip()] = name.strip()
                    continue

                m = re.search(r'GAME START.*turn order\s+\[(.*?)\]', line)
                if m:
                    turn_order = [
                        part.strip().strip("'\"")
                        for part in m.group(1).split(',')
                        if part.strip()
                    ]
                    status = "playing"
                    continue

                if re.search(r'\bFINISHED\b|\bGAME FINISHED\b|\bWins\b|\bwins\b', line):
                    status = "finished"

                m = re.search(
                    r'MOVE\s+(\d+):\s+(.+?)\s+\((.+?)\)\s+(\d+)->(\d+)\s+\[([0-9.]+)ms\]',
                    line,
                )
                if m:
                    move_num, name, color, from_idx, to_idx, move_ms = m.groups()
                    moves.append({
                        "num": int(move_num),
                        "player": name.strip(),
                        "color": color.strip(),
                        "from": int(from_idx),
                        "to": int(to_idx),
                        "move_ms": float(move_ms),
                    })
                    continue

                m = re.search(
                    r'SCORE\s+(.+?)\s+\((.+?)\):\s+Final=([0-9.]+),\s+'
                    r'Time=([0-9.]+),\s+Moves\((\d+)\)=([0-9.]+),\s+'
                    r'Pins\((\d+)\)=([0-9.]+),\s+Dist=([0-9.]+)',
                    line,
                )
                if m and moves:
                    name, color, final, time_score, player_moves, move_score, pins, pin_score, dist = m.groups()
                    scores_by_move.setdefault(moves[-1]["num"], []).append({
                        "player": name.strip(),
                        "color": color.strip(),
                        "final": float(final),
                        "time": float(time_score),
                        "moves": int(player_moves),
                        "move_score": float(move_score),
                        "pins": int(pins),
                        "pin_score": float(pin_score),
                        "dist": float(dist),
                    })

        pins = {
            color: self.board.axial_of_colour(color)[:10]
            for color in players_by_color
        }
        initial_pins = {
            color: list(positions)
            for color, positions in pins.items()
        }
        warnings = []
        for move in moves:
            color = move["color"]
            positions = pins.get(color)
            if positions is None:
                warnings.append(f"Move {move['num']} references unknown color {color}.")
                continue
            try:
                pin_idx = positions.index(move["from"])
            except ValueError:
                warnings.append(
                    f"Move {move['num']} could not find {color} pin at {move['from']}."
                )
                continue
            positions[pin_idx] = move["to"]

        log_mtime = log_path.stat().st_mtime

        if moves and status != "finished":
            status = "playing"
        elif players_by_color and status == "created":
            status = "waiting to start"

        if status == "playing":
            log_age_sec = max(0.0, time.time() - log_mtime)
            if log_age_sec > 30.0:
                status = "stale"
                warnings.append(
                    f"Log has not updated for {log_age_sec:.0f}s; the game process may have stopped."
                )

        state.update({
            "players": [
                {"color": color, "name": name}
                for color, name in players_by_color.items()
            ],
            "turn_order": turn_order,
            "status": status,
            "move_count": moves[-1]["num"] if moves else 0,
            "initial_pins": initial_pins,
            "pins": pins,
            "last_move": moves[-1] if moves else None,
            "moves": moves,
            "recent_moves": moves[-80:],
            "scores_by_move": scores_by_move,
            "scores": scores_by_move.get(moves[-1]["num"], []) if moves else [],
            "warnings": warnings[-8:],
            "log_mtime": log_mtime,
        })
        return state


def live_viewer_html(poll_ms: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Chinese Checkers</title>
  <style>
    :root {{
      --bg: #eef2f4;
      --panel: #ffffff;
      --ink: #182126;
      --muted: #66747c;
      --line: #d4dde2;
      --accent: #1f7a8c;
      --warn: #b45500;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 10px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0; font-size: 18px; }}
    .subtle {{ color: var(--muted); font-size: 13px; }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      padding: 6px 9px;
      background: #f8fafb;
      font-size: 13px;
    }}
    .dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #8aa1aa;
    }}
    .dot.live {{ background: #23a55a; box-shadow: 0 0 0 4px rgba(35,165,90,.14); }}
    main {{
      height: calc(100vh - 58px);
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 380px;
      min-height: 0;
    }}
    .board-wrap {{
      padding: 16px;
      min-width: 0;
      min-height: 0;
      display: grid;
    }}
    .board-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      min-height: 0;
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    aside {{
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto auto 1fr;
      background: var(--panel);
      border-left: 1px solid var(--line);
    }}
    section {{
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
    }}
    h2 {{ margin: 0 0 8px; font-size: 14px; }}
    .players, .scores, .moves, .warnings {{ margin: 0; padding: 0; list-style: none; }}
    .players li, .scores li {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 4px 0;
      font-size: 13px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border: 1px solid rgba(0,0,0,.3);
      display: inline-block;
    }}
    .moves {{
      overflow: auto;
      padding: 8px;
    }}
    .move {{
      display: grid;
      grid-template-columns: 52px 1fr;
      gap: 8px;
      align-items: center;
      padding: 8px;
      border-bottom: 1px solid #edf1f3;
      font-size: 13px;
    }}
    .move.latest {{
      background: #e8f3f5;
      border: 1px solid #9ccbd3;
    }}
    .move-num {{ color: var(--muted); }}
    .cell {{
      stroke: rgba(24,33,38,.58);
      stroke-width: 1.2;
    }}
    .cell.board {{ fill: #f4f6f7; }}
    .cell.red {{ fill: #ffd3d3; }}
    .cell.lawn-green {{ fill: #d9f6d2; }}
    .cell.blue {{ fill: #d8ddff; }}
    .cell.yellow {{ fill: #fff4bb; }}
    .cell.purple {{ fill: #f1d7ff; }}
    .cell.gray0 {{ fill: #d7d7d7; }}
    .pin {{
      stroke: rgba(0,0,0,.75);
      stroke-width: 3;
      transition: cx .28s ease, cy .28s ease;
    }}
    .pin-label {{
      font-size: 10px;
      text-anchor: middle;
      dominant-baseline: central;
      fill: #fff;
      font-weight: 700;
      pointer-events: none;
      transition: x .28s ease, y .28s ease;
    }}
    .index-label {{
      font-size: 8px;
      text-anchor: middle;
      dominant-baseline: central;
      fill: rgba(24,33,38,.48);
      display: none;
      pointer-events: none;
    }}
    body.show-indices .index-label {{ display: block; }}
    .last-line {{
      stroke: #ff8a00;
      stroke-width: 4;
      stroke-linecap: round;
      opacity: .78;
    }}
    .last-ring {{
      fill: none;
      stroke: #ff8a00;
      stroke-width: 5;
      opacity: .9;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      margin-top: 10px;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 6px 9px;
      cursor: pointer;
      font: inherit;
    }}
    button:hover {{ border-color: var(--accent); }}
    input[type="range"] {{ width: 130px; }}
    .warnings li {{ color: var(--warn); font-size: 13px; padding: 3px 0; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; height: auto; }}
      .board-wrap {{ height: 72vh; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Live Chinese Checkers</h1>
      <div class="subtle" id="logName">Waiting for game log...</div>
    </div>
    <div class="status"><span class="dot" id="liveDot"></span><span id="statusText">Connecting</span></div>
  </header>
  <main>
    <div class="board-wrap">
      <div class="board-panel">
        <svg id="board" viewBox="0 0 100 100" role="img" aria-label="Live Chinese Checkers board"></svg>
      </div>
    </div>
    <aside>
      <section>
        <h2>Game</h2>
        <div class="subtle" id="gameMeta">No moves yet</div>
        <div class="subtle" id="replayMeta">Replay move 0 / 0</div>
        <div class="controls">
          <button id="playPause">Pause</button>
          <button id="stepMove">Step</button>
          <button id="catchUp">Catch up</button>
          <label>speed <input type="range" id="speed" min="250" max="2200" step="50" value="850"></label>
          <label><input type="checkbox" id="showIndices"> show board indexes</label>
        </div>
      </section>
      <section>
        <h2>Players</h2>
        <ul class="players" id="players"></ul>
      </section>
      <section>
        <h2>Scores</h2>
        <ul class="scores" id="scores"></ul>
        <ul class="warnings" id="warnings"></ul>
      </section>
      <ol class="moves" id="moves"></ol>
    </aside>
  </main>
  <script>
    const POLL_MS = {int(poll_ms)};
    const SVG_NS = 'http://www.w3.org/2000/svg';
    const zoneClass = (color) => String(color || 'board').replace(/[^a-zA-Z0-9_-]+/g, '-').toLowerCase();
    const pinColor = {{
      red: '#cc0000',
      'lawn green': '#00a83a',
      blue: '#1a3fd2',
      yellow: '#d1a800',
      purple: '#b22acc',
      gray0: '#676f73'
    }};
    let cellsByIndex = new Map();
    let boardReady = false;
    let pinNodes = new Map();
    let labelNodes = new Map();
    let lastMoveEls = [];
    let currentGameId = null;
    let initialPins = {{}};
    let displayPins = {{}};
    let allMoves = [];
    let scoresByMove = {{}};
    let appliedIndex = 0;
    let lastDisplayedMove = null;
    let isPlaying = true;
    let moveDelayMs = 850;
    let playbackTimer = null;

    function createSvg(tag, attrs = {{}}) {{
      const el = document.createElementNS(SVG_NS, tag);
      for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
      return el;
    }}

    function hexPoints(cx, cy, r) {{
      const pts = [];
      for (let i = 0; i < 6; i++) {{
        const angle = Math.PI / 180 * (60 * i - 30);
        pts.push(`${{(cx + r * Math.cos(angle)).toFixed(2)}},${{(cy + r * Math.sin(angle)).toFixed(2)}}`);
      }}
      return pts.join(' ');
    }}

    function buildBoard(board) {{
      if (boardReady) return;
      const svg = document.getElementById('board');
      svg.setAttribute('viewBox', `0 0 ${{board.width}} ${{board.height}}`);
      svg.innerHTML = '';
      const cellLayer = createSvg('g', {{ id: 'cells' }});
      const indexLayer = createSvg('g', {{ id: 'indices' }});
      const traceLayer = createSvg('g', {{ id: 'trace' }});
      const pinLayer = createSvg('g', {{ id: 'pins' }});
      cellsByIndex = new Map();
      for (const cell of board.cells) {{
        cellsByIndex.set(cell.index, cell);
        cellLayer.appendChild(createSvg('polygon', {{
          class: `cell ${{zoneClass(cell.zone)}}`,
          points: hexPoints(cell.x, cell.y, board.hole_radius / 1.08)
        }}));
        const label = createSvg('text', {{
          class: 'index-label',
          x: cell.x,
          y: cell.y
        }});
        label.textContent = cell.index;
        indexLayer.appendChild(label);
      }}
      svg.appendChild(cellLayer);
      svg.appendChild(indexLayer);
      svg.appendChild(traceLayer);
      svg.appendChild(pinLayer);
      boardReady = true;
    }}

    function setText(id, value) {{
      document.getElementById(id).textContent = value;
    }}

    function renderPlayers(players) {{
      const el = document.getElementById('players');
      el.innerHTML = players.map(p => `
        <li><span class="swatch" style="background:${{pinColor[p.color] || '#999'}}"></span>
        <strong>${{p.name}}</strong><span>${{p.color}}</span></li>
      `).join('');
    }}

    function renderScores(scores) {{
      const el = document.getElementById('scores');
      if (!scores || !scores.length) {{
        el.innerHTML = '<li><span></span><span class="subtle">No scores yet</span><span></span></li>';
        return;
      }}
      el.innerHTML = scores.map(s => `
        <li><span class="swatch" style="background:${{pinColor[s.color] || '#999'}}"></span>
        <strong>${{s.player}}</strong><span>${{s.final.toFixed(1)}} · pins ${{s.pins}} · dist ${{s.dist.toFixed(1)}}</span></li>
      `).join('');
    }}

    function renderMoves(moves, activeMoveNum = null) {{
      const el = document.getElementById('moves');
      const activeIdx = activeMoveNum == null ? appliedIndex - 1 : moves.findIndex(m => m.num === activeMoveNum);
      const center = Math.max(0, activeIdx);
      const start = Math.max(0, center - 36);
      const windowMoves = moves.slice(start, start + 80);
      el.innerHTML = windowMoves.map((m) => `
        <li class="move ${{m.num === activeMoveNum ? 'latest' : ''}}">
          <span class="move-num">#${{m.num}}</span>
          <span><strong>${{m.player}}</strong> (${{m.color}}) ${{m.from}} -> ${{m.to}}</span>
        </li>
      `).join('');
      const active = el.querySelector('.move.latest');
      if (active) active.scrollIntoView({{ block: 'nearest' }});
    }}

    function clearLastMove() {{
      for (const el of lastMoveEls) el.remove();
      lastMoveEls = [];
    }}

    function renderLastMove(move) {{
      clearLastMove();
      if (!move) return;
      const from = cellsByIndex.get(move.from);
      const to = cellsByIndex.get(move.to);
      if (!from || !to) return;
      const traceLayer = document.getElementById('trace');
      const line = createSvg('line', {{
        class: 'last-line',
        x1: from.x,
        y1: from.y,
        x2: to.x,
        y2: to.y
      }});
      const ring = createSvg('circle', {{
        class: 'last-ring',
        cx: to.x,
        cy: to.y,
        r: 21
      }});
      traceLayer.appendChild(line);
      traceLayer.appendChild(ring);
      lastMoveEls = [line, ring];
    }}

    function renderPins(pins) {{
      const pinLayer = document.getElementById('pins');
      const liveKeys = new Set();
      for (const [color, positions] of Object.entries(pins || {{}})) {{
        positions.forEach((cellIndex, pinIndex) => {{
          const key = `${{color}}:${{pinIndex}}`;
          const cell = cellsByIndex.get(cellIndex);
          if (!cell) return;
          liveKeys.add(key);
          let pin = pinNodes.get(key);
          let label = labelNodes.get(key);
          if (!pin) {{
            pin = createSvg('circle', {{
              class: 'pin',
              r: 13,
              fill: pinColor[color] || '#999'
            }});
            label = createSvg('text', {{ class: 'pin-label' }});
            label.textContent = String(pinIndex + 1);
            pinNodes.set(key, pin);
            labelNodes.set(key, label);
            pinLayer.appendChild(pin);
            pinLayer.appendChild(label);
          }}
          pin.setAttribute('cx', cell.x);
          pin.setAttribute('cy', cell.y);
          label.setAttribute('x', cell.x);
          label.setAttribute('y', cell.y);
        }});
      }}
      for (const [key, node] of pinNodes.entries()) {{
        if (!liveKeys.has(key)) {{
          node.remove();
          pinNodes.delete(key);
          const label = labelNodes.get(key);
          if (label) label.remove();
          labelNodes.delete(key);
        }}
      }}
    }}

    function clonePins(pins) {{
      const cloned = {{}};
      for (const [color, positions] of Object.entries(pins || {{}})) {{
        cloned[color] = positions.slice();
      }}
      return cloned;
    }}

    function scoresForMove(moveNum) {{
      if (moveNum == null) return [];
      return scoresByMove[String(moveNum)] || scoresByMove[moveNum] || [];
    }}

    function resetReplay(state) {{
      currentGameId = state.game_id || null;
      initialPins = clonePins(state.initial_pins || {{}});
      displayPins = clonePins(initialPins);
      allMoves = state.moves || [];
      scoresByMove = state.scores_by_move || {{}};
      appliedIndex = 0;
      lastDisplayedMove = null;
      clearLastMove();
      renderPins(displayPins);
      renderScores([]);
      renderMoves(allMoves, null);
      updateReplayMeta(state);
    }}

    function updateReplaySource(state) {{
      allMoves = state.moves || [];
      scoresByMove = state.scores_by_move || {{}};
      const nextInitial = state.initial_pins || {{}};
      const existingColors = Object.keys(initialPins).sort().join('|');
      const nextColors = Object.keys(nextInitial).sort().join('|');
      if (appliedIndex === 0 && (Object.keys(displayPins).length === 0 || existingColors !== nextColors)) {{
        initialPins = clonePins(nextInitial);
        displayPins = clonePins(initialPins);
        renderPins(displayPins);
      }}
      updateReplayMeta(state);
    }}

    function updateReplayMeta(state) {{
      const shown = lastDisplayedMove ? lastDisplayedMove.num : 0;
      const logged = state.move_count || 0;
      const backlog = Math.max(0, allMoves.length - appliedIndex);
      setText('replayMeta', `Showing move ${{shown}} / logged ${{logged}} · queued ${{backlog}}`);
    }}

    function applyNextMove() {{
      if (appliedIndex >= allMoves.length) return false;
      const move = allMoves[appliedIndex];
      if (!displayPins[move.color]) displayPins[move.color] = [];
      const pinIndex = displayPins[move.color].indexOf(move.from);
      if (pinIndex >= 0) {{
        displayPins[move.color][pinIndex] = move.to;
      }}
      appliedIndex += 1;
      lastDisplayedMove = move;
      renderPins(displayPins);
      renderLastMove(move);
      renderScores(scoresForMove(move.num));
      renderMoves(allMoves, move.num);
      return true;
    }}

    function catchUpReplay() {{
      while (appliedIndex < allMoves.length) applyNextMove();
    }}

    function startPlayback() {{
      if (playbackTimer) clearInterval(playbackTimer);
      playbackTimer = setInterval(() => {{
        if (isPlaying) applyNextMove();
      }}, moveDelayMs);
    }}

    function renderWarnings(warnings) {{
      const el = document.getElementById('warnings');
      el.innerHTML = (warnings || []).map(w => `<li>${{w}}</li>`).join('');
    }}

    async function poll() {{
      try {{
        const response = await fetch('/state.json?ts=' + Date.now(), {{ cache: 'no-store' }});
        const state = await response.json();
        buildBoard(state.board);
        const dot = document.getElementById('liveDot');
        dot.classList.toggle('live', state.ok && state.status === 'playing');
        setText('statusText', `${{state.status}} · move ${{state.move_count}} · ${{state.updated_at}}`);
        setText('logName', state.log_file || 'Waiting for game log...');
        const turnOrder = state.turn_order && state.turn_order.length ? ` · turn order: ${{state.turn_order.join(' -> ')}}` : '';
        setText('gameMeta', `${{state.game_id || 'no game'}} · ${{state.players.length}} players${{turnOrder}}`);
        renderPlayers(state.players || []);
        if (state.game_id !== currentGameId) {{
          resetReplay(state);
        }} else {{
          updateReplaySource(state);
        }}
        renderWarnings(state.warnings || []);
      }} catch (error) {{
        document.getElementById('liveDot').classList.remove('live');
        setText('statusText', 'Disconnected');
      }} finally {{
        setTimeout(poll, POLL_MS);
      }}
    }}

    document.getElementById('showIndices').addEventListener('change', (event) => {{
      document.body.classList.toggle('show-indices', event.target.checked);
    }});
    document.getElementById('playPause').addEventListener('click', () => {{
      isPlaying = !isPlaying;
      document.getElementById('playPause').textContent = isPlaying ? 'Pause' : 'Play';
    }});
    document.getElementById('stepMove').addEventListener('click', () => {{
      isPlaying = false;
      document.getElementById('playPause').textContent = 'Play';
      applyNextMove();
    }});
    document.getElementById('catchUp').addEventListener('click', () => {{
      catchUpReplay();
    }});
    document.getElementById('speed').addEventListener('input', (event) => {{
      moveDelayMs = Number(event.target.value);
      startPlayback();
    }});

    startPlayback();
    poll();
  </script>
</body>
</html>
"""


class ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_live_viewer(log_file: Optional[str], latest: bool, host: str, port: int, poll_ms: int) -> None:
    source = LiveGameSource(log_file=log_file, follow_latest=latest)
    viewer_html = live_viewer_html(poll_ms=poll_ms)
    script_dir = Path(__file__).resolve().parent
    live_path = script_dir / "renders" / "live_viewer.html"
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text(viewer_html, encoding="utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state.json"):
                payload = json.dumps(source.state()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            payload = viewer_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            return

    selected_port = int(port)
    server = None
    for candidate_port in range(int(port), int(port) + 20):
        try:
            server = ReusableThreadingHTTPServer((host, candidate_port), Handler)
            selected_port = candidate_port
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError(f"Could not bind live viewer to {host}:{port}-{port + 19}")

    url = f"http://{host}:{selected_port}/"
    print(f"Live viewer HTML file: {live_path}", flush=True)
    print(f"Live viewer URL      : {url}", flush=True)
    print("Press Ctrl+C to stop the live viewer.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live viewer.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Chinese Checkers game log and HTML move viewer.")
    parser.add_argument("log_file", nargs="?", help="Path to a game_*.log file. Omit to choose from games/.")
    parser.add_argument("--latest", action="store_true", help="Use the newest game log without prompting.")
    parser.add_argument("--max-moves", type=int, default=None, help="Only apply/render moves up to this move number.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth move after applying all moves. Default: 1.")
    parser.add_argument("--live", action="store_true", help="Start a live browser viewer that follows a game log.")
    parser.add_argument("--host", default="127.0.0.1", help="Live viewer host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Live viewer port. Default: 8765")
    parser.add_argument("--poll-ms", type=int, default=600, help="Browser polling interval for live mode. Default: 600")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.live:
        start_live_viewer(
            log_file=args.log_file,
            latest=args.latest or not args.log_file,
            host=args.host,
            port=args.port,
            poll_ms=args.poll_ms,
        )
        sys.exit(0)

    if args.log_file:
        log_file = args.log_file
    else:
        game_dir = find_game_dir()
        if not game_dir:
            print("Error: games/ folder not found. Run a game first.")
            sys.exit(1)
        log_files = sorted(game_dir.glob("game_*.log"), key=lambda path: path.stat().st_mtime)
        if not log_files:
            print("Error: no game logs found in games/")
            sys.exit(1)
        if args.latest:
            log_file = str(log_files[-1])
        else:
            log_file = choose_log_interactively(game_dir)
        print(f"Using log: {log_file}\n")
    
    if not Path(log_file).exists():
        print(f"Error: log file not found: {log_file}")
        sys.exit(1)
    
    visualizer = GameVisualizer(log_file, max_moves=args.max_moves, stride=args.stride)
    visualizer.visualize()
