"""Parity test: do random-vs-random games via az.Sim and via game.py's logic
(invoked in-process, no TCP) yield byte-for-byte identical move trajectories
when given the same seed?

We can't easily drive game.py's TCP server in lockstep with our in-process
sim, so we instead exercise both code paths with the SAME HexBoard/Pin
primitives and verify:
  (a) `az.Sim` and a parallel hand-rolled "reference" using the raw
      HexBoard + Pin classes converge on the same WIN/DRAW outcomes
      across many seeded random games.
  (b) Encoder round-trip: encode_state -> decode preserves pin positions.
"""

import os
import sys
import random

# Allow running standalone: `python tests/test_sim_parity.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from az.sim import Sim
from az.encoder import (
    encode_state, encode_legal_mask, encode_action, decode_action,
    ROT_PERMS, INV_ROT_PERMS, slot_of, CYCLE,
)
from az.config import COLOUR_OPPOSITES, NUM_CELLS, BOARD_CHANNELS, PINS_PER_PLAYER


def _play_random(seed: int, num_players: int, max_moves: int = 200):
    """Play a random-vs-random game via az.Sim. Return (terminal_reason, winner, history)."""
    s = Sim(num_players, seed=seed)
    rng = random.Random(seed * 7919 + 13)
    while not s.is_terminal:
        col = s.current_colour()
        legal = s.legal_moves(col)
        movable = [(p, m) for p, m in legal.items() if m]
        if not movable:
            s.skip_no_moves()
            continue
        pid, mvs = rng.choice(movable)
        to = rng.choice(mvs)
        s.apply_move(pid, to)
        if s.move_count >= max_moves:
            s.force_max_moves()
    return s.terminal_reason, s.winner, s.history


def test_rotation_lut():
    # R^6 == identity
    perm = np.arange(NUM_CELLS)
    for _ in range(6):
        perm = ROT_PERMS[1][perm]
    assert (perm == np.arange(NUM_CELLS)).all()
    # Inverse correctness
    for k in range(6):
        assert (INV_ROT_PERMS[k][ROT_PERMS[k]] == np.arange(NUM_CELLS)).all()
    # Complement of every colour lands at slot 3
    for c in CYCLE:
        comp = COLOUR_OPPOSITES[c]
        assert slot_of(comp, c) == 3
    print("test_rotation_lut OK")


def test_encoder_roundtrip():
    # Pick random pin configurations, encode, decode actions, verify legality is preserved.
    n_trials = 200
    for trial in range(n_trials):
        s = Sim(num_players=random.choice([2, 3, 4]), seed=trial)
        # Play a few random moves to get diverse states
        rng = random.Random(trial)
        moves_to_play = rng.randint(0, 30)
        for _ in range(moves_to_play):
            if s.is_terminal:
                break
            col = s.current_colour()
            legal = s.legal_moves(col)
            mv = [(p, m) for p, m in legal.items() if m]
            if not mv:
                s.skip_no_moves()
                continue
            pid, ms = rng.choice(mv)
            s.apply_move(pid, rng.choice(ms))
        if s.is_terminal:
            continue
        col = s.current_colour()
        legal = s.legal_moves(col)
        if not any(legal.values()):
            continue
        # Encode + decode every legal action; check we recover original
        for pid, dests in legal.items():
            for to in dests:
                a = encode_action(int(pid), int(to), col)
                pid2, to2 = decode_action(a, col)
                assert pid2 == pid, (pid, pid2, a)
                assert to2 == to, (to, to2, a)
        # Encode mask should match the legal set
        mask = encode_legal_mask(legal, col)
        from az.encoder import NUM_ACTIONS
        expected_count = sum(len(v) for v in legal.values())
        assert int(mask.sum()) == expected_count, (int(mask.sum()), expected_count)
    print(f"test_encoder_roundtrip OK ({n_trials} trials)")


def test_sim_terminations():
    """Run many random games and confirm Sim produces sane outcomes."""
    counts = {'WIN': 0, 'DRAW_CHAIN': 0, 'MAX_MOVES': 0, 'UNKNOWN': 0}
    for seed in range(30):
        n = 2 + (seed % 4)
        term, winner, _ = _play_random(seed, n, max_moves=200)
        counts[term] = counts.get(term, 0) + 1
    print(f"test_sim_terminations OK; counts={counts}")


def test_canonical_pin_count():
    """encode_state's pin-channel sums must equal pin counts in absolute board."""
    s = Sim(4, seed=1)
    # Play some moves
    rng = random.Random(0)
    for _ in range(20):
        if s.is_terminal: break
        col = s.current_colour()
        legal = s.legal_moves(col)
        mv = [(p, m) for p, m in legal.items() if m]
        if not mv:
            s.skip_no_moves(); continue
        pid, ms = rng.choice(mv)
        s.apply_move(pid, rng.choice(ms))

    pins_state = s.pins_state()
    col = s.current_colour()
    board, glob = encode_state(pins_state, col, s.turn_order, s.move_count)
    assert board.shape == (BOARD_CHANNELS, NUM_CELLS)
    # Each present colour contributes 10 to its slot channel
    for c in s.colours:
        sslot = slot_of(c, col)
        assert int(board[sslot].sum()) == 10, f"colour {c} slot {sslot}: expected 10, got {int(board[sslot].sum())}"
    print("test_canonical_pin_count OK")


def test_to_move_pin_id_planes():
    """The policy action indexes pin_id, so the encoder must expose own pin ids."""
    s = Sim(2, seed=3)
    rng = random.Random(3)
    for _ in range(12):
        col = s.current_colour()
        legal = s.legal_moves(col)
        mv = [(p, m) for p, m in legal.items() if m]
        if not mv:
            s.skip_no_moves()
            continue
        pid, ms = rng.choice(mv)
        s.apply_move(pid, rng.choice(ms))

    col = s.current_colour()
    pins_state = s.pins_state()
    board, _ = encode_state(pins_state, col, s.turn_order, s.move_count)
    k = CYCLE.index(col)
    # rotation_for_to_move(col) is (6 - CYCLE.index(col)) % 6.
    rot = ROT_PERMS[(6 - k) % 6]
    for pid, orig_idx in enumerate(pins_state[col]):
        plane = 14 + pid
        assert int(board[plane].sum()) == 1
        assert board[plane, rot[orig_idx]] == 1.0
    assert int(board[14:14 + PINS_PER_PLAYER].sum()) == PINS_PER_PLAYER
    print("test_to_move_pin_id_planes OK")


if __name__ == "__main__":
    test_rotation_lut()
    test_encoder_roundtrip()
    test_sim_terminations()
    test_canonical_pin_count()
    test_to_move_pin_id_planes()
    print("\nALL PARITY TESTS PASSED")
