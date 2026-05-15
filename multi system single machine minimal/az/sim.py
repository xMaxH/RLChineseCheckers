"""In-process Chinese Checkers simulator that mirrors game.py exactly.

Reuses HexBoard and Pin (read-only imports). The legality, WIN, DRAW, and
draw-chain rules match game.py:
  - WIN: all 10 pins of a colour on cells with postype == colour_opposites[colour]
  - DRAW: no legal moves for any pin of that colour
  - If N-1 players have status DRAW, the remaining player gets WIN
  - Turn order: COLOUR_ORDER rotated to start at first joiner

Differences from the server we deliberately drop (not relevant in headless
self-play):
  - Per-turn timeouts (TURN_TIMEOUT_SEC). We impose a synthetic move-count
    cap externally; max_moves games are flagged and discarded by the trainer.
  - Network / logging.
"""

import random
from typing import Dict, List, Optional, Tuple
from contextlib import redirect_stdout
import io

from checkers_board import HexBoard
from checkers_pins import Pin

from .config import COLOUR_ORDER, COLOUR_OPPOSITES


PRIMARIES_BASE = ['red', 'lawn green', 'yellow']
COMPLEMENT_OF = {'red': 'blue', 'lawn green': 'gray0', 'yellow': 'purple'}


def _assign_colours(num_players: int, rng: random.Random) -> List[str]:
    """Mirror game.Game.assign_colour over `num_players` joins."""
    primaries = PRIMARIES_BASE.copy()
    rng.shuffle(primaries)
    colours: List[str] = []
    for i in range(num_players):
        primary_idx = i // 2
        if i % 2 == 0:  # odd-numbered joiner (1st, 3rd, 5th)
            colours.append(primaries[primary_idx])
        else:           # even-numbered joiner — complement of latest primary
            colours.append(COMPLEMENT_OF[primaries[primary_idx]])
    return colours


def _compute_turn_order(colours: List[str]) -> List[str]:
    first = colours[0]
    idx = COLOUR_ORDER.index(first)
    rotated = COLOUR_ORDER[idx:] + COLOUR_ORDER[:idx]
    return [c for c in rotated if c in colours]


class Sim:
    def __init__(self, num_players: int, seed: Optional[int] = None):
        assert 2 <= num_players <= 6
        self.rng = random.Random(seed)
        self.colours: List[str] = _assign_colours(num_players, self.rng)
        self.turn_order: List[str] = _compute_turn_order(self.colours)
        self.num_players = num_players

        # Pin's __init__ writes to stdout via placePin in some paths; suppress.
        with redirect_stdout(io.StringIO()):
            self.board = HexBoard()
            self.pins_by_colour: Dict[str, List[Pin]] = {}
            for c in self.colours:
                idxs = self.board.axial_of_colour(c)[:10]
                self.pins_by_colour[c] = [
                    Pin(self.board, idxs[i], id=i, color=c) for i in range(10)
                ]

        self.move_count = 0
        self.current_turn_index = 0
        self.player_status: Dict[str, str] = {c: 'PLAYING' for c in self.colours}
        self.game_status = 'PLAYING'
        self.winner: Optional[str] = None
        self.terminal_reason: Optional[str] = None
        # Track per-colour move counts (parity with game.py's pl.move_count)
        self.move_count_by_colour: Dict[str, int] = {c: 0 for c in self.colours}
        # History (canonical encoder samples are taken externally)
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.game_status == 'FINISHED'

    def current_colour(self) -> Optional[str]:
        if self.is_terminal:
            return None
        return self.turn_order[self.current_turn_index]

    def legal_moves(self, colour: str) -> Dict[int, List[int]]:
        return {p.id: p.getPossibleMoves() for p in self.pins_by_colour[colour]}

    def pins_state(self) -> Dict[str, List[int]]:
        return {c: [p.axialindex for p in pins] for c, pins in self.pins_by_colour.items()}

    # ------------------------------------------------------------------
    def _check_colour_status(self, colour: str) -> str:
        opp = COLOUR_OPPOSITES[colour]
        pins = self.pins_by_colour[colour]
        if all(self.board.cells[p.axialindex].postype == opp for p in pins):
            return 'WIN'
        if all(len(p.getPossibleMoves()) == 0 for p in pins):
            return 'DRAW'
        return 'PLAYING'

    def _maybe_finalize_after_status_update(self, colour: str) -> bool:
        """Mirror game.py's WIN / DRAW-chain finalization logic. Return True if game ended."""
        st = self.player_status[colour]
        if st == 'WIN':
            self.game_status = 'FINISHED'
            self.winner = colour
            self.terminal_reason = 'WIN'
            return True
        if st == 'DRAW':
            draws = [c for c in self.colours if self._check_colour_status(c) == 'DRAW']
            if len(draws) == len(self.colours) - 1:
                winner = next(c for c in self.colours if c not in draws)
                self.player_status[winner] = 'WIN'
                self.game_status = 'FINISHED'
                self.winner = winner
                self.terminal_reason = 'DRAW_CHAIN'
                return True
        return False

    # ------------------------------------------------------------------
    def apply_move(self, pin_id: int, to_idx: int) -> None:
        """Apply a move for the current to-move colour. Caller must validate legality."""
        if self.is_terminal:
            raise RuntimeError("apply_move on terminal sim")
        colour = self.current_colour()
        pins = self.pins_by_colour[colour]
        if not (0 <= pin_id < len(pins)):
            raise ValueError(f"bad pin_id {pin_id}")
        pin = pins[pin_id]
        if to_idx not in pin.getPossibleMoves():
            raise ValueError(f"illegal move {colour} pin {pin_id} -> {to_idx}")

        # Direct mutation, skipping Pin.placePin's print noise.
        from_idx = pin.axialindex
        self.board.cells[from_idx].occupied = False
        pin.axialindex = to_idx
        self.board.cells[to_idx].occupied = True

        self.move_count += 1
        self.move_count_by_colour[colour] += 1
        self.history.append({"colour": colour, "pin": pin_id, "from": from_idx, "to": to_idx})

        # Update player's status (matches game.py: only the moving player's status is recomputed)
        self.player_status[colour] = self._check_colour_status(colour)
        if self._maybe_finalize_after_status_update(colour):
            return

        # Advance turn
        self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)

    def skip_no_moves(self) -> None:
        """Current to-move has no legal moves at all; mark DRAW and advance.

        game.py would handle this via per-turn timeout; in headless sim we
        just record DRAW and move on.
        """
        if self.is_terminal:
            return
        colour = self.current_colour()
        # Sanity: confirm no legal moves
        any_moves = any(len(p.getPossibleMoves()) > 0 for p in self.pins_by_colour[colour])
        if any_moves:
            raise RuntimeError(f"skip_no_moves called but {colour} has legal moves")
        self.player_status[colour] = 'DRAW'
        if self._maybe_finalize_after_status_update(colour):
            return
        self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)

    def force_max_moves(self) -> None:
        """External orchestrator decided the move cap is reached. No winner."""
        if self.is_terminal:
            return
        self.game_status = 'FINISHED'
        self.terminal_reason = 'MAX_MOVES'
        self.winner = None

    # ------------------------------------------------------------------
    def outcomes_by_colour(self) -> Dict[str, float]:
        """+1 winner, -1 everyone else. Empty if non-WIN terminal (max_moves)."""
        if self.terminal_reason != 'WIN' and self.terminal_reason != 'DRAW_CHAIN':
            return {}
        out = {c: -1.0 for c in self.colours}
        if self.winner is not None:
            out[self.winner] = 1.0
        return out
