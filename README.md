# RLChineseCheckers
For IKT 460 (RL) base Chinese Checkers


# Rules
- each player starts with their colored pieces on one of the six points or corners of the star and attempts to race them all home into the opposite corner
- Players take turns moving a single piece,
  - either by moving one step in any direction to an adjacent empty space,
  - or by jumping in one or any number of available consecutive hops over other single pieces.
- A player may not combine hopping with a single-step move :: a move consists of one or the other.
- ~~A pin maynot go into a 'coloured triangle' unless it is either its source or its destination~~ A pin may go into any coloured region, provided it is a valid move
- There is no capturing

## Multi system

game.py :: server side (run on terminal 1)

player.py :: client side (run on terminal 2/3/4... Each terminal is a player.)

In game.py ::
  Options are Create/Status/Quit
  Typing Create + Enter - creates new game
  Typing Status + Enter - shows status of games in current session

In player.py ::
  Enter player name
  Press Enter when you get message 'Press enter to send Start'

  **You'll need to replace the code inside 'PLAYING LOGIC' to your own logic.**


- You can see the scoring inside game.py
- game log is stored in folder 'games'
- get_state as player gives you access to game board, moves made by all players, timings, game status etc.
- Change TURN_TIMEOUT_SEC and GAME_TIME_LIMIT_SEC accordingly for training. *Please let me know your average time taken per move and average total time to finish a game. Once I recieve responses from atleast 5 different teams, I will base the final values of these parameters accordingly.*


## single system
Designed to be played on a single system (no separate teams joining from separate systems)

Run : python checkers_main.py

You'll be prompted to type 'assign' to assign a random colour (without the quotes)
```
Type 'exit' to exit. Type 'assign' and press enter for first player:
```
Again for the next player
```
Type 'assign' and press enter for second player:
```

You'll see the updated ascii board on the commandline screen everytime there is a change.

Next prompt:
```
Type 'assign' and press enter for more players, else type 'start game':
```

Once you enter 'start_game', you'll see
  - Tkinter version in a separate window
  - *colour* 's current positions : format (pin_id, pin_axialindex) *list of (id, position) of that colour*
  - If 'help_mode = True' under checkers_main.py, you'll be prompted:
    - ```
      Helpmode:Which pin will you move?:
      ```
    - Enter a pin number, e.g: 5
    - ```
      Possible moves: *list of possible axial positions that the given pin can go to*
      ```
    - ```
      Need more help? Yes/No
      ```
    - If 'Yes'
      - Helpmode continues
    - If 'No', 
      - ```
        Enter *colour* 's Move: (pin_number, dest_axial):
        ```
      - enter the move **with** '(' ,')', eg: (2,66)
      - If it was a valid move, see update on Ascii/Tkinter, and control moves to the next player.
      - Else, current player's turn continues
     
  # ~~Current Limitations/Expected Changes~~ Updates
  - ~~There is no logic implemented yet for checking if a player has won - Expect changes~~ Game terminates when 1 player wins. Game also terminates if all players excpet 1 Draw (have no possible valid moves).
  - ~~Expect changes in Starting a Game/Assigning colours.~~
  - ~~There is no logic implemented yet for storing moves made - Expect changes~~ Game moves are stored in game-specific log text file in folder games.
  - ~~There is no logic implemented yet to indicate 'Pass' on a turn - Expect changes~~ No explicit 'Pass', but timeout implemented for a turn
  - ~~There is no logic implemented yet to timeout a players turn - Expect changes~~ Timeout for total gameplay implemented
  - ~~There is no logic implemented yet for scoring - Expect changes (will be based on time taken, total number of moves, number of pins successfully moved to opposite colour)~~ Scoring ::
    - time score : max(0.0, 100.0 - time_taken_sec) // time_taken_sec is total time taken for all moves made by player, less total time, more time score
    - move score : math.exp(-((move_count - 45) ** 2) / (2 * ((4 if move_count < 45 else 18) ** 2))) // move_count is total number of moves made by player.assymetrical Gaussian. max score if move_count around 45, penalized both for too few movements and too many movements.
    - pin score : pins_in_goal * 100.0
    - distance score :  max(0.0, 200.0 - total_dist) // total_dist is sum of min(distance(pin_not_in_goal, target_colour))
    - total score : time score + move score + pin score + distance score
  - ~~There is no logic implemented yet for maximum allowable time per game - Expect changes~~ See above
