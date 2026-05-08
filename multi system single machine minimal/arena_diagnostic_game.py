#!/usr/bin/env python3
"""Run a visualizable arena diagnostic game.

This bypasses the socket launcher so one colour can use a candidate model while
the other colours use a probe checkpoint or the built-in heuristic policy.  It
writes the same MOVE/SCORE log format that game_visualizer.py already reads.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from alphazero_multiplayer_method import (
    MultiAlphaZeroAgent,
    MultiPolicyValueNet,
    _blocker_winner,
    _evaluation_colours,
    _load_model_weights_if_available,
    _max_plies_for_player_count,
    _sample_action_from_visits,
    _score_margin,
    _should_blocker_adjudicate,
    _terminal_reason,
    create_initial_multiplayer_state,
)


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{ts()}] {message}\n")


def load_agent(model_path: Optional[Path], device: str) -> MultiAlphaZeroAgent:
    if model_path is None:
        return MultiAlphaZeroAgent(model=None, device=device)
    model = MultiPolicyValueNet()
    _load_model_weights_if_available(model, model_path, verbose=True)
    return MultiAlphaZeroAgent(model=model, device=device)


def score_components(state, colour: str) -> Dict[str, float]:
    moves = int(state.move_counts_by_colour.get(colour, 0))
    pins = int(state.pins_in_goal(colour))
    distance = float(state.distance_to_goal(colour))
    distance_score = max(0.0, 200.0 - distance) if moves > 0 else 0.0
    pin_score = float(pins * 100)
    final = float(state.score_proxy(colour))
    move_score = max(0.0, final - pin_score - distance_score)
    return {
        "final": final,
        "time": 0.0,
        "moves": float(moves),
        "move_score": move_score,
        "pins": float(pins),
        "pin_score": pin_score,
        "dist": distance_score,
    }


def write_scores(path: Path, names_by_colour: Dict[str, str], state) -> None:
    for colour in state.turn_order:
        parts = score_components(state, colour)
        write_log(
            path,
            f"SCORE {names_by_colour[colour]} ({colour}): "
            f"Final={parts['final']:.1f}, "
            f"Time={parts['time']:.1f}, "
            f"Moves({int(parts['moves'])})={parts['move_score']:.1f}, "
            f"Pins({int(parts['pins'])})={parts['pin_score']:.1f}, "
            f"Dist={parts['dist']:.1f}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one candidate-vs-probe/heuristic game and log it for game_visualizer.py.",
    )
    parser.add_argument("--players", type=int, choices=(2, 3, 4, 5, 6), default=2)
    parser.add_argument("--candidate-model", required=True, help="Candidate .pt checkpoint.")
    parser.add_argument("--candidate-colour", default="", help="Colour controlled by candidate. Default: first colour.")
    parser.add_argument("--opponent", choices=("heuristic", "probe"), default="heuristic")
    parser.add_argument("--probe-model", default="checkpoints/alphazero_multi_probe.pt")
    parser.add_argument("--device", default="cpu", help="Torch device for model inference. Default: cpu.")
    parser.add_argument("--num-simulations", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-moves", type=int, default=250, help="Max moves per player.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means deterministic max-visits moves.")
    parser.add_argument(
        "--move-delay-sec",
        type=float,
        default=0.0,
        help="Sleep after each logged move so game_visualizer.py --live can follow the game.",
    )
    parser.add_argument("--adjudicate-stale-moves", type=int, default=160)
    parser.add_argument("--adjudicate-min-moves", type=int, default=80)
    parser.add_argument("--output-dir", default="games")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    colours = _evaluation_colours(args.players)
    candidate_colour = args.candidate_colour.strip() or colours[0]
    if candidate_colour not in colours:
        raise ValueError(f"--candidate-colour must be one of: {', '.join(colours)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    game_id = f"arena_diag_{uuid.uuid4()}"
    log_path = output_dir / f"game_{game_id}.log"
    print(f"Diagnostic log: {log_path}", flush=True)
    print("Live viewer: python3 game_visualizer.py --live --latest --host 0.0.0.0 --port 8765", flush=True)
    print("Open http://127.0.0.1:8765 locally, or the forwarded 8765 URL in the IDE.", flush=True)

    candidate_agent = load_agent(Path(args.candidate_model), args.device)
    if args.opponent == "probe":
        opponent_agent = load_agent(Path(args.probe_model), args.device)
        opponent_label = f"probe:{Path(args.probe_model).stem}"
    else:
        opponent_agent = load_agent(None, args.device)
        opponent_label = "heuristic"

    names_by_colour = {
        colour: (
            f"candidate_{Path(args.candidate_model).stem}"
            if colour == candidate_colour
            else f"{opponent_label}_{colour.replace(' ', '_')}"
        )
        for colour in colours
    }

    state = create_initial_multiplayer_state(colours)
    max_plies = _max_plies_for_player_count(args.players, args.max_moves)
    stale_plies = args.players * max(0, int(args.adjudicate_stale_moves))
    min_plies = args.players * max(0, int(args.adjudicate_min_moves))
    blocker_since_ply: Optional[int] = None
    blocker_colour: Optional[str] = None
    adjudicated = False
    adjudicated_winner: Optional[str] = None

    write_log(log_path, "GAME CREATED")
    for colour in colours:
        write_log(log_path, f"PLAYER JOINED: {names_by_colour[colour]} as {colour}")
        write_log(log_path, f"PLAYER START: {names_by_colour[colour]} ({colour})")
    write_log(log_path, f"GAME START - turn order {colours}")
    write_log(
        log_path,
        f"ARENA DIAGNOSTIC: candidate={candidate_colour} "
        f"opponent={opponent_label} simulations={args.num_simulations}",
    )

    try:
        for ply in range(max_plies):
            if state.is_terminal():
                break
            colour = state.current_turn_colour()
            legal = state.legal_actions(colour)
            if not legal:
                break
            agent = candidate_agent if colour == candidate_colour else opponent_agent
            started = time.time()
            visits = agent.run_mcts(
                root_state=state,
                root_colour=colour,
                num_simulations=int(args.num_simulations),
                c_puct=float(args.c_puct),
                root_dirichlet_frac=0.0,
            )
            action = _sample_action_from_visits(visits, float(args.temperature))
            from_idx = state.pin_position(colour, action.pin_id)
            state.apply_action(action)
            move_ms = (time.time() - started) * 1000.0

            move_text = (
                f"MOVE {ply + 1}: {names_by_colour[colour]} ({colour}) "
                f"{from_idx}->{action.to_index} [{move_ms:.2f}ms]"
            )
            write_log(log_path, move_text)
            print(move_text, flush=True)
            write_scores(log_path, names_by_colour, state)
            if args.move_delay_sec > 0:
                time.sleep(float(args.move_delay_sec))

            current_ply = ply + 1
            if state.winner() is not None:
                break
            candidate_blocker = _blocker_winner(state)
            if candidate_blocker is None:
                blocker_since_ply = None
                blocker_colour = None
            else:
                if blocker_colour != candidate_blocker:
                    blocker_colour = candidate_blocker
                    blocker_since_ply = current_ply
                if _should_blocker_adjudicate(
                    state=state,
                    ply=current_ply,
                    blocker_since_ply=blocker_since_ply,
                    stale_plies=stale_plies,
                    min_plies=min_plies,
                ):
                    adjudicated = True
                    adjudicated_winner = candidate_blocker
                    break
    except KeyboardInterrupt:
        write_log(log_path, f"FINISHED: interrupted after {state.move_count} moves")
        write_scores(log_path, names_by_colour, state)
        print(f"\nInterrupted. Partial diagnostic log: {log_path}", flush=True)
        return 130

    reason = _terminal_reason(state, state.move_count, max_plies, adjudicated=adjudicated)
    scores = {colour: float(state.score_proxy(colour)) for colour in state.turn_order}
    winner = state.winner() or adjudicated_winner
    if winner is not None:
        suffix = " (blocker-adjudicated)" if adjudicated else ""
        write_log(log_path, f"FINISHED: {names_by_colour[winner]} ({winner}) wins after {state.move_count} moves{suffix}")
    else:
        write_log(
            log_path,
            f"FINISHED: terminal_reason={reason} moves={state.move_count} "
            f"score_margin={_score_margin(scores):.1f}",
        )
    write_scores(log_path, names_by_colour, state)

    print(f"Diagnostic log: {log_path}")
    print(f"Terminal reason: {reason}")
    print("Scores: " + ", ".join(f"{colour}={score:.1f}" for colour, score in scores.items()))
    print(f"Visualize with: python3 game_visualizer.py '{log_path}' --stride 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
