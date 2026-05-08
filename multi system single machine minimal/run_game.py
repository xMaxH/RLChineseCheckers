#!/usr/bin/env python3
"""Launch the local multi-player Chinese Checkers game from one command."""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from alphazero_method import DEFAULT_MODEL_PATH as DEFAULT_2P_MODEL_PATH
from alphazero_method import format_parameter_summary
from alphazero_multiplayer_method import DEFAULT_MODEL_PATH as DEFAULT_MULTI_MODEL_PATH


DEFAULT_CREATE_DELAY_SEC = 1.0
DEFAULT_PLAYER_STAGGER_SEC = 0.5
DEFAULT_START_DELAY_SEC = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start server + auto-join players for local testing."
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Number of players to launch (2-6). Default: 2",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable network debug output in each player process.",
    )
    parser.add_argument(
        "--create-delay",
        type=float,
        default=DEFAULT_CREATE_DELAY_SEC,
        help="Seconds to wait before sending Create to the server. Default: 1.0",
    )
    parser.add_argument(
        "--player-stagger",
        type=float,
        default=DEFAULT_PLAYER_STAGGER_SEC,
        help="Seconds between launching player processes. Default: 0.5",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="random",
        help="Playing method for all players (random or alphazero). Default: random",
    )
    parser.add_argument(
        "--player-methods",
        type=str,
        default="",
        help="Comma-separated playing methods per player, e.g. random,alphazero,random",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=("auto", "2p", "multiplayer"),
        default="auto",
        help="AlphaZero implementation to use. auto resolves from --players. Default: auto",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="2-player AlphaZero model path. Sets AZ_MODEL_PATH for this run.",
    )
    parser.add_argument(
        "--multiplayer-model-path",
        type=str,
        default="",
        help="Multiplayer AlphaZero model path. Sets AZ_MP_MODEL_PATH for this run.",
    )
    parser.add_argument(
        "--search-sims",
        type=int,
        default=None,
        help="MCTS simulations per move for both AlphaZero modes.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=DEFAULT_START_DELAY_SEC,
        help="Seconds to wait after launching players before sending START. Default: 2.0",
    )
    return parser.parse_args()


def resolve_player_methods(players: int, default_method: str, per_player_methods: str) -> list[str]:
    if per_player_methods.strip():
        methods = [m.strip().lower() for m in per_player_methods.split(",") if m.strip()]
        if len(methods) != players:
            raise ValueError(
                f"--player-methods must provide exactly {players} entries "
                f"(got {len(methods)})."
            )
    else:
        methods = [default_method.strip().lower()] * players

    valid = {"random", "alphazero"}
    invalid = [m for m in methods if m not in valid]
    if invalid:
        raise ValueError(
            "Invalid method(s): "
            + ", ".join(invalid)
            + ". Valid options are: random, alphazero."
        )
    return methods


def stream_output(prefix: str, pipe):
    for line in iter(pipe.readline, ""):
        print(f"[{prefix}] {line}", end="")
    pipe.close()


def terminate_all(processes: list[subprocess.Popen]):
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()

    deadline = time.time() + 3
    for proc in processes:
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)

    for proc in processes:
        if proc.poll() is None:
            proc.kill()


def main() -> int:
    args = parse_args()

    if args.players < 2 or args.players > 6:
        print("Error: --players must be between 2 and 6.")
        return 2

    try:
        player_methods = resolve_player_methods(
            players=args.players,
            default_method=args.method,
            per_player_methods=args.player_methods,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 2

    alpha_enabled = any(method == "alphazero" for method in player_methods)
    alpha_mode = args.alpha_mode
    if alpha_mode == "auto":
        alpha_mode = "2p" if args.players <= 2 else "multiplayer"
    model_path = args.model_path or os.getenv("AZ_MODEL_PATH", str(DEFAULT_2P_MODEL_PATH))
    multiplayer_model_path = args.multiplayer_model_path or os.getenv(
        "AZ_MP_MODEL_PATH",
        str(DEFAULT_MULTI_MODEL_PATH),
    )
    search_sims = str(args.search_sims) if args.search_sims is not None else os.getenv("AZ_MCTS_SIMS", "96")

    print(
        format_parameter_summary(
            "Game Run Parameters",
            [
                ("players", args.players),
                ("methods", ",".join(player_methods)),
                ("alpha_enabled", alpha_enabled),
                ("alpha_mode", alpha_mode if alpha_enabled else "n/a"),
                ("create_delay", args.create_delay),
                ("player_stagger", args.player_stagger),
                ("start_delay", args.start_delay),
                ("debug", args.debug),
            ],
        ),
        flush=True,
    )

    if alpha_enabled:
        print(
            format_parameter_summary(
                "AlphaZero Game Defaults",
                [
                    ("player_mode", alpha_mode),
                    ("num_simulations", search_sims),
                    ("c_puct", os.getenv("AZ_C_PUCT", "1.5")),
                    ("temp_opening", os.getenv("AZ_TEMP_OPENING", "1.0")),
                    ("temp_late", os.getenv("AZ_TEMP_LATE", "0.15")),
                    ("temp_cutoff_move", os.getenv("AZ_TEMP_CUTOFF_MOVE", "20")),
                    ("model_path", model_path),
                    ("multiplayer_model_path", multiplayer_model_path),
                ],
            ),
            flush=True,
        )

    base_dir = Path(__file__).resolve().parent
    server_script = base_dir / "game.py"
    player_script = base_dir / "player.py"

    if not server_script.exists() or not player_script.exists():
        print("Error: Expected game.py and player.py in the same folder as this launcher.")
        return 2

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if alpha_enabled:
        env["AZ_PLAYER_MODE"] = alpha_mode
        env["AZ_MODEL_PATH"] = model_path
        env["AZ_MP_MODEL_PATH"] = multiplayer_model_path
        if args.search_sims is not None:
            env["AZ_MCTS_SIMS"] = str(args.search_sims)
            env["AZ_MP_MCTS_SIMS"] = str(args.search_sims)
    if args.debug:
        env["DEBUG_NET"] = "1"
    else:
        env.pop("DEBUG_NET", None)

    processes: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []

    def on_signal(signum, _frame):
        print(f"\n[launcher] Caught signal {signum}, shutting down all processes...")
        terminate_all(processes)
        sys.exit(130)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        server = subprocess.Popen(
            [sys.executable, str(server_script)],
            cwd=str(base_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        processes.append(server)
        t = threading.Thread(target=stream_output, args=("server", server.stdout), daemon=True)
        t.start()
        threads.append(t)

        time.sleep(args.create_delay)
        if server.poll() is not None:
            print("[launcher] Server exited before game creation.")
            return server.returncode or 1

        server.stdin.write("create\n")
        server.stdin.flush()
        print("[launcher] Sent 'create' to server.")

        for i in range(1, args.players + 1):
            player_env = env.copy()
            method = player_methods[i - 1]
            player_env["PLAYER_METHOD"] = method

            player = subprocess.Popen(
                [sys.executable, str(player_script)],
                cwd=str(base_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=player_env,
            )
            processes.append(player)

            pt = threading.Thread(
                target=stream_output,
                args=(f"player{i}", player.stdout),
                daemon=True,
            )
            pt.start()
            threads.append(pt)

            # Answer "Enter name" now. START is sent after all requested players join.
            player.stdin.write(f"Player{i}\n")
            player.stdin.flush()
            print(f"[launcher] Started player {i} with method '{method}'.")
            time.sleep(args.player_stagger)

        if args.start_delay > 0:
            print(f"[launcher] Waiting {args.start_delay:.1f}s before START.")
            time.sleep(args.start_delay)

        for player in processes[1:]:
            if player.poll() is None and player.stdin:
                player.stdin.write("\n")
                player.stdin.flush()
        print("[launcher] Sent START to all players.")

        print("[launcher] Game running. Press Ctrl+C to stop all processes.")

        exit_codes = []
        for proc in processes[1:]:
            exit_codes.append(proc.wait())

        if server.poll() is None:
            terminate_all([server])

        return 0 if all(code == 0 for code in exit_codes) else 1

    except KeyboardInterrupt:
        print("\n[launcher] Interrupted, terminating...")
        terminate_all(processes)
        return 130
    finally:
        terminate_all(processes)


if __name__ == "__main__":
    raise SystemExit(main())
