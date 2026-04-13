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


DEFAULT_CREATE_DELAY_SEC = 1.0
DEFAULT_PLAYER_STAGGER_SEC = 0.5


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
    return parser.parse_args()


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

    base_dir = Path(__file__).resolve().parent
    server_script = base_dir / "game.py"
    player_script = base_dir / "player.py"

    if not server_script.exists() or not player_script.exists():
        print("Error: Expected game.py and player.py in the same folder as this launcher.")
        return 2

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
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
            player = subprocess.Popen(
                [sys.executable, str(player_script)],
                cwd=str(base_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            processes.append(player)

            pt = threading.Thread(
                target=stream_output,
                args=(f"player{i}", player.stdout),
                daemon=True,
            )
            pt.start()
            threads.append(pt)

            # First line answers "Enter name"; second newline pre-answers "Press ENTER to send START...".
            player.stdin.write(f"Player{i}\n\n")
            player.stdin.flush()
            print(f"[launcher] Started player {i}.")
            time.sleep(args.player_stagger)

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
