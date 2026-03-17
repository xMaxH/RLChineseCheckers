# =============================================================
# player.py — simplified client that relies ONLY on JSON state
# =============================================================

import os
import json
import random
import socket
import time
from typing import Dict, Any

HOST = "127.0.0.1"
PORT = 50555
DEBUG_NET = os.getenv("DEBUG_NET", "0") not in ("0", "", "false", "False")


def debug(*args):
    if DEBUG_NET:
        print("[NET]", *args)

def render_ascii_from_json(state):
    """
    Render the hex board in ASCII using only the JSON state.
    No HexBoard, no Pin classes required.

    The board is represented as 121 cells arranged in 11 rows:
    Row length pattern:  9 10 11 12 13 14 13 12 11 10 9
    This matches the Chinese Checkers star-shaped center hex.

    Each cell in the JSON is rendered with:
        '.' for empty
        first letter of colour for occupied
    """

    pins = state.get("pins", {})

    # Build an occupancy map: index -> symbol
    occ = {}
    for colour, idx_list in pins.items():
        symbol = colour[0].upper()   # R,L,Y,B,G,P
        for idx in idx_list:
            occ[int(idx)] = symbol

    # Known row sizes for a 121‑cell Chinese Checkers-like board
    row_lengths = [9,10,11,12,13,14,13,12,11,10,9]

    # Build rows according to lengths
    rows = []
    cur = 0
    for length in row_lengths:
        row_cells = []
        for _ in range(length):
            idx = cur
            cur += 1
            cell = occ.get(idx, ".")
            row_cells.append(cell)
        rows.append(row_cells)

    # Center the board visually for ASCII output
    widest = max(len(r) for r in rows)

    print("\n=== ASCII BOARD ===")
    for r in rows:
        pad = " " * ((widest - len(r)) * 2)
        print(pad + " ".join(r))
    print("===================\n")

def rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send JSON to server and receive JSON reply."""
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


# =============================================================
# Simple renderer for the server's JSON board (optional)
# =============================================================
def render_json_board(state):
    """
    Rudimentary visualization using only JSON information.
    Does NOT require HexBoard or Pin.
    """
    pins = state.get("pins", {})
    print("=== BOARD STATE ===")
    for colour, indices in pins.items():
        print(f"{colour}: {indices}")
    print("===================")


# =============================================================
# Main client loop
# =============================================================
def main():
    print("==== Player ====")
    name = input("Enter name: ").strip()
    if not name:
        return

    # JOIN GAME
    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        print("JOIN ERROR:", r.get("error"))
        return

    game_id = r["game_id"]
    player_id = r["player_id"]
    colour = r["colour"]

    print(f"Joined game {game_id} as {colour}")

    # Wait until game ready
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") in ("READY_TO_START", "PLAYING"):
            break
        print("Waiting for players...")
        time.sleep(0.5)

    input("Press ENTER to send START...")
    rpc({"op": "start", "game_id": game_id, "player_id": player_id})
    print("Sent START")

    # Wait until PLAYING
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    print("=== GAME STARTED ===\n")

    last_move_seen = 0

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("Error:", st.get("error"))
            return

        state = st["state"]

        # Timeout messages
        if state.get("turn_timeout_notice"):
            print("⚠ TIMEOUT:", state["turn_timeout_notice"])

        # Finished?
        if state["status"] == "FINISHED":
            print("\n=== GAME FINISHED ===")
            print("FINAL SCORES:")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    print(
                        f"{pl['name']} ({pl['colour']}): "
                        f"{sc['final_score']:.1f} "
                        f"[time={sc['time_score']:.1f}, "
                        f"moves({sc['moves']})={sc['move_score']:.1f}, "
                        f"pins={sc['pin_goal_score']:.1f}, "
                        f"dist={sc['distance_score']:.1f}]"
                    )
            print("======================")
            break

        # Render board from JSON-only
        

        # Show last move
        if state["move_count"] > last_move_seen:
            mv = state.get("last_move")
            if mv:
                print(
                    f"MOVE: {mv['by']} ({mv['colour']}) "
                    f"{mv['from']}→{mv['to']}  [{mv['move_ms']:.1f}ms]"
                )
            last_move_seen = state["move_count"]

        # If it's our turn, choose a random move
        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            # Request legal moves for each pin from server
            # Since player.py no longer knows how to compute moves, we add a request:
            legal_req = rpc({
                "op": "get_legal_moves",
                "game_id": game_id,
                "player_id": player_id
            })

            if not legal_req.get("ok"):
                print("Error requesting legal moves:", legal_req.get("error"))
                time.sleep(0.5)
                continue

            legal_moves = legal_req.get("legal_moves", {})

            # legal_moves example structure:
            # { pin_id: [to_index1, to_index2, ...], ... }

            movable = [(pid, moves) for pid, moves in legal_moves.items() if moves]
            if not movable:
                print("No legal moves available.")
                time.sleep(0.5)
                continue

            pid, moves = random.choice(movable)
            to_index = random.choice(moves)

            delay = random.randint(1, 12)
            print("Randomized delay:", delay)
            time.sleep(delay)

            mv = rpc({
                "op": "move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": pid,
                "to_index": to_index
            })
            render_json_board(state)
            if not mv.get("ok"):
                print("Move rejected:", mv.get("error"))
            else:
                if mv.get("status") == "WIN":
                    print("YOU WIN!")
                    print(mv.get("msg"))
                elif mv.get("status") == "DRAW":
                    print("DRAW")
                    print(mv.get("msg"))

        time.sleep(0.5)


if __name__ == "__main__":
    main()