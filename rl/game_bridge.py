"""Bridge module that exposes the existing game engine to the RL layer.

The base game engine lives in a folder whose name contains spaces
(``multi system single machine minimal``). This module adds that folder
to ``sys.path`` so we can import ``checkers_board`` and ``checkers_pins``
directly, and re-exports the symbols we rely on.

Importing ``rl`` automatically triggers this module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "multi system single machine minimal"

if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

# These imports must come after the sys.path tweak above.
from checkers_board import HexBoard, BoardPosition  # noqa: E402
from checkers_pins import Pin  # noqa: E402

# Colour constants mirror game.py but kept local to avoid importing the
# server module (which has side effects).
COLOUR_ORDER = ["red", "lawn green", "yellow", "blue", "gray0", "purple"]
PRIMARY_COLOURS = ["red", "lawn green", "yellow"]
COMPLEMENT = {"red": "blue", "lawn green": "gray0", "yellow": "purple"}
OPPOSITES = {
    "red": "blue",
    "blue": "red",
    "lawn green": "gray0",
    "gray0": "lawn green",
    "yellow": "purple",
    "purple": "yellow",
}

PINS_PER_PLAYER = 10


def new_board() -> HexBoard:
    """Factory helper."""
    return HexBoard()


def num_cells() -> int:
    return len(new_board().cells)


__all__ = [
    "REPO_ROOT",
    "GAME_DIR",
    "HexBoard",
    "BoardPosition",
    "Pin",
    "COLOUR_ORDER",
    "PRIMARY_COLOURS",
    "COMPLEMENT",
    "OPPOSITES",
    "PINS_PER_PLAYER",
    "new_board",
    "num_cells",
]
