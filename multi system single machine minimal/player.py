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
    timeoutnotice_move = -1
    loop_sleep_s = float(os.getenv("PLAYER_LOOP_SLEEP_S", "0.05"))
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
        time.sleep(loop_sleep_s)

    input("Press ENTER to send START...")
    rpc({"op": "start", "game_id": game_id, "player_id": player_id})
    print("Sent START")

    # Wait until PLAYING
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(loop_sleep_s)

    print("=== GAME STARTED ===\n")

    # Tkinter GUI (realtime visualization). Vi pumper events manuelt via root.update()
    # i main-loop'en for å slippe å kjøre mainloop() blokkende.
    from checkers_board import HexBoard
    from checkers_pins import Pin
    from checkers_gui import BoardGUI

    board = HexBoard()

    def build_pins_from_state(state: Dict[str, Any]):
        # Reset occupancy and recreate pins from server state.
        for cell in board.cells:
            cell.occupied = False

        pins = []
        pins_by_colour = state.get("pins", {})
        for colour_name, indices in pins_by_colour.items():
            for i, axialindex in enumerate(indices):
                pins.append(Pin(board, int(axialindex), id=i, color=colour_name))
        return pins

    st0 = rpc({"op": "get_state", "game_id": game_id})
    if not st0.get("ok"):
        print("Error getting initial state:", st0.get("error"))
        return

    gui = BoardGUI(board, build_pins_from_state(st0["state"]))

    last_move_seen = 0
    game_finished = False

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("Error:", st.get("error"))
            return

        state = st["state"]

        # Timeout messages
        if state.get("turn_timeout_notice") and timeoutnotice_move< state.get("move_count"):
            print("⚠ TIMEOUT:", state["turn_timeout_notice"])
            timeoutnotice_move =  state.get("move_count")


        # Finished?
        if state["status"] == "FINISHED":
            if not game_finished:
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

                # Vis siste bretttilstand og hold vinduet åpent.
                try:
                    gui.refresh(build_pins_from_state(state))
                    gui.root.update_idletasks()
                    gui.root.update()
                except Exception:
                    pass

                game_finished = True

            # Keep the visualization open until the server process stops.
            try:
                gui.root.update_idletasks()
                gui.root.update()
            except Exception:
                break

            time.sleep(loop_sleep_s)
            continue

        # Render board from JSON-only
        

        # Show last move
        if state["move_count"] > last_move_seen:
            mv = state.get("last_move")
            if mv:
                print(
                    f"MOVE: {mv['by']} ({mv['colour']}) "
                    f"{mv['from']}→{mv['to']}  [{mv['move_ms']:.1f}ms]"
                )
            gui.refresh(build_pins_from_state(state))
            last_move_seen = state["move_count"]

        
        # If it's our turn, choose a random move
        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            print("\nMy turn")
            '''------------PLAYING LOGIC-----------'''
            # Request legal moves for each pin from server
            legal_req = rpc({
                "op": "get_legal_moves",
                "game_id": game_id,
                "player_id": player_id
            })

            if not legal_req.get("ok"):
                print("Error requesting legal moves:", legal_req.get("error"))
                time.sleep(loop_sleep_s)
                continue

            legal_moves = legal_req.get("legal_moves", {})

            # legal_moves example structure:
            # { pin_id: [to_index1, to_index2, ...], ... }

            movable = [(pid, moves) for pid, moves in legal_moves.items() if moves]
            if not movable:
                print("No legal moves available.")
                time.sleep(loop_sleep_s)
                continue

            pid, moves = random.choice(movable)
            to_index = random.choice(moves)

            max_delay_s = float(os.getenv("PLAYER_MAX_DELAY_S", "0"))
            if max_delay_s > 0:
                time.sleep(random.uniform(0, max_delay_s))
            '''-----------------PLAYING LOGIC----------------'''

            mv = rpc({
                "op": "move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": pid,
                "to_index": to_index
            })
            if not mv.get("ok"):
                print("Move rejected:", mv.get("error"))
            else:
                if mv.get("status") == "WIN":
                    print("YOU WIN!")
                    print(mv.get("msg"))
                elif mv.get("status") == "DRAW":
                    print("DRAW")
                    print(mv.get("msg"))

        # Pump Tk events so the window updates during play.
        try:
            gui.root.update_idletasks()
            gui.root.update()
        except Exception:
            # If the window was closed, just stop looping.
            break

        time.sleep(loop_sleep_s)


if __name__ == "__main__":
    main()