#!/usr/bin/env python3
"""
Game Visualizer — reads game logs and renders board states as images.
Usage: python game_visualizer.py <game_log_file>
Outputs: renders/<game_id>/*.png
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import RegularPolygon
import math

from checkers_board import HexBoard
from checkers_pins import Pin


class GameVisualizer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.game_id = Path(log_file).stem.replace('game_', '')
        script_dir = Path(__file__).resolve().parent
        self.output_dir = script_dir / "renders" / self.game_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse log to build move sequence
        self.moves = []
        self.players = {}
        self.colors = {}
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
                
                # Move: [timestamp] MOVE N: Name (color) from->to [time]
                m = re.search(r'MOVE\s+(\d+):\s+(.+?)\s+\((.+?)\)\s+(\d+)->(\d+)', line)
                if m:
                    move_num, name, color, from_idx, to_idx = m.groups()
                    self.moves.append({
                        'num': int(move_num),
                        'player': name.strip(),
                        'color': color.strip(),
                        'from': int(from_idx),
                        'to': int(to_idx),
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
        fig.savefig(str(self.output_dir / "move_0_initial.png"), dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Rendered initial state")
        
        rendered_moves = 0
        skipped_moves = 0

        # Apply each move and render
        for move in self.moves:
            color = move['color']
            from_idx = move['from']
            to_idx = move['to']
            
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
                
                # Render
                fig = self.render_board(
                    board, pins_by_colour, 
                    move_num=move['num'],
                    title=f"Move {move['num']}: {move['player']} ({color}) {from_idx}→{to_idx}",
                    highlight=(color, to_idx)
                )
                fig.savefig(str(self.output_dir / f"move_{move['num']:03d}_{move['player'].lower()}.png"), 
                           dpi=100, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Rendered move {move['num']}: {move['player']} ({color}) {from_idx}→{to_idx}")
                rendered_moves += 1
            else:
                skipped_moves += 1

        print(f"\n✅ Game visualization saved to {self.output_dir}/")
        print(f"Parsed moves: {len(self.moves)} | Rendered: {rendered_moves} | Skipped: {skipped_moves}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find latest game log near this script first, then fallback to cwd-relative paths.
        script_dir = Path(__file__).resolve().parent
        game_dir = None
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
                game_dir = path
                break
        
        if not game_dir:
            print("Error: games/ folder not found. Run a game first.")
            sys.exit(1)
        
        log_files = sorted(game_dir.glob("game_*.log"))
        if not log_files:
            print("Error: no game logs found in games/")
            sys.exit(1)
        
        log_file = str(log_files[-1])
        print(f"Using latest log: {log_file}\n")
    else:
        log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"Error: log file not found: {log_file}")
        sys.exit(1)
    
    visualizer = GameVisualizer(log_file)
    visualizer.visualize()
