# ============================================
# player.py — prints final scores
# ============================================

import os
import json
import random
import socket
import time
from typing import Dict, Any

from checkers_board import HexBoard
from checkers_pins import Pin

HOST = "127.0.0.1"
PORT = 50555

DEBUG_NET = os.getenv("DEBUG_NET", "0") not in ("0", "", "false", "False")

def debug(*args):
    if DEBUG_NET:
        print("[NET]", *args)

def rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10.0)
    try:
        s.connect((HOST, PORT))
    except Exception as e:
        return {"ok": False, "error": f"connect-failed: {e}"}

    s.sendall(json.dumps(payload).encode("utf-8"))
    data = s.recv(1_000_000)
    s.close()

    if not data:
        return {"ok": False, "error": "no-response"}

    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}

def build_board_from_state(state):
    board = HexBoard()
    for cell in board.cells:
        cell.occupied = False
    pins_map = {}
    all_pins = []

    for colour, indices in state["pins"].items():
        pins = []
        for i, idx in enumerate(indices):
            idx = int(idx)
            board.cells[idx].occupied = True
            p = Pin(board, idx, id=i, color=colour)
            pins.append(p)
            all_pins.append(p)
        pins_map[colour] = pins

    return board, pins_map, all_pins

def render_ascii(board: HexBoard, pins):
    board.print_ascii(pins=pins)

def main():
    print("==== Player ====")
    name = input("Enter name: ").strip()
    if not name:
        return

    # JOIN
    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        print("JOIN ERROR:", r.get("error"))
        return

    game_id = r["game_id"]
    player_id = r["player_id"]
    colour = r["colour"]

    print(f"Joined game {game_id} as {colour}")

    # WAIT for READY_TO_START
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") in ("READY_TO_START", "PLAYING"):
            break
        print("Waiting for players...")
        time.sleep(0.5)

    # SEND START
    input("Press ENTER to send Start...")
    rpc({"op": "start", "game_id": game_id, "player_id": player_id})
    print("Sent START")

    # WAIT for PLAYING
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    print("=== GAME STARTED ===\n")
    last_move_seen = 0

    # MAIN LOOP
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("Error:", st.get("error"))
            return

        state = st["state"]

        notice = state.get("turn_timeout_notice")
        if notice:
            print("⚠ TIMEOUT:", notice)

        if state["status"] == "FINISHED":
            print("\n=== GAME FINISHED ===")
            print("FINAL SCORES:")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    print(
                        f"{pl['name']} ({pl['colour']}): "
                        f"{sc['final_score']:.1f}  "
                        f"[time={sc['time_score']:.1f}, "
                        f"moves({sc['moves']})={sc['move_score']:.1f}, "
                        f"pins={sc['pin_goal_score']:.1f}, "
                        f"dist={sc['distance_score']:.1f}]"
                    )
            print("======================")
            break

        board, pins_map, all_pins = build_board_from_state(state)

        if state["move_count"] > last_move_seen:
            lm = state.get("last_move") or {}
            if lm:
                print(f"MOVE: {lm['by']} ({lm['colour']}) "
                      f"{lm['from']}→{lm['to']} [{lm['move_ms']:.1f}ms]")
            last_move_seen = state["move_count"]

        render_ascii(board, all_pins)
        print()

        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            render_ascii(board, all_pins)
            print()
            my_pins = pins_map.get(colour, [])
            movable = [(i,p) for i,p in enumerate(my_pins) if p.getPossibleMoves()]
            if not movable:
                time.sleep(0.5)
                continue

            pid, pin = random.choice(movable)
            moves = pin.getPossibleMoves()
            to_idx = random.choice(moves)

            sleepdelay = random.randint(1, 12)
            print("Randomized delay ", sleepdelay)
            time.sleep(sleepdelay)

            mv = rpc({
                "op": "move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": pid,
                "to_index": to_idx
            })

            if not mv.get("ok"):
                print("Move rejected:", mv.get("error"))
            else:
                render_ascii(board, all_pins)
                print()
                if mv.get("status") == "WIN":
                    print("YOU WIN!")
                    print(mv.get("msg"))
                    continue
                if mv.get("status") == "DRAW":
                    print("DRAW")
                    print(mv.get("msg"))
                    continue
        else:
            time.sleep(0.5)


if __name__ == "__main__":
    main()