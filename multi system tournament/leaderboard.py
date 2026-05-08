#Read log files from folder 'games' and create a leaderboard based on the scores of the games. The leaderboard should be sorted by score and include the game ID, player names, and scores. The top 10 on the leaderboard should be printed to the console in a readable format. The entire leaderboard should be written to a tab-separated file. 
#This is an example log file format:
'''
[2026-05-04 09:13:24] GAME CREATED
[2026-05-04 09:13:40] PLAYER JOINED: pl_B as yellow
[2026-05-04 09:13:55] PLAYER JOINED: PL_C as purple
[2026-05-04 09:13:57] PLAYER START: PL_C (purple)
[2026-05-04 09:13:59] PLAYER START: pl_B (yellow)
[2026-05-04 09:13:59] GAME START — turn order ['yellow', 'purple']
[2026-05-04 09:14:03] MOVE 1: pl_B (yellow) 36->37 [0.38ms]
[2026-05-04 09:14:03] SCORE pl_B (yellow): Final=187.0, Time=96.0, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:03] SCORE PL_C (purple): Final=0.0, Time=0.0, Moves(0)=0.0, Pins(0)=0.0, Dist=0.0
[2026-05-04 09:14:05] MOVE 2: PL_C (purple) 84->73 [0.14ms]
[2026-05-04 09:14:05] SCORE pl_B (yellow): Final=187.0, Time=96.0, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:05] SCORE PL_C (purple): Final=188.8, Time=97.8, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:16] SCORE pl_B (yellow): Final=187.0, Time=96.0, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:16] SCORE PL_C (purple): Final=188.8, Time=97.8, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:16] TURN TIMEOUT: Player with colour yellow exceeded 10s at move 2. Turn skipped.
[2026-05-04 09:14:26] SCORE pl_B (yellow): Final=187.0, Time=96.0, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:26] SCORE PL_C (purple): Final=188.8, Time=97.8, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:26] TURN TIMEOUT: Player with colour purple exceeded 10s at move 2. Turn skipped.
[2026-05-04 09:14:31] MOVE 3: pl_B (yellow) 46->47 [0.21ms]
[2026-05-04 09:14:31] SCORE pl_B (yellow): Final=182.6, Time=90.6, Moves(2)=0.0, Pins(0)=0.0, Dist=92.0
[2026-05-04 09:14:31] SCORE PL_C (purple): Final=188.8, Time=97.8, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:41] SCORE pl_B (yellow): Final=182.6, Time=90.6, Moves(2)=0.0, Pins(0)=0.0, Dist=92.0
[2026-05-04 09:14:41] SCORE PL_C (purple): Final=188.8, Time=97.8, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:41] TURN TIMEOUT: Player with colour purple exceeded 10s at move 3. Turn skipped.
[2026-05-04 09:14:44] MOVE 4: pl_B (yellow) 11->57 [0.24ms]
[2026-05-04 09:14:44] SCORE pl_B (yellow): Final=183.6, Time=87.6, Moves(3)=0.0, Pins(0)=0.0, Dist=96.0
[2026-05-04 09:14:44] SCORE PL_C (purple): Final=188.8, Time=97.8, Moves(1)=0.0, Pins(0)=0.0, Dist=91.0
[2026-05-04 09:14:46] MOVE 5: PL_C (purple) 73->84 [0.18ms]
[2026-05-04 09:14:46] SCORE pl_B (yellow): Final=183.6, Time=87.6, Moves(3)=0.0, Pins(0)=0.0, Dist=96.0
[2026-05-04 09:14:46] SCORE PL_C (purple): Final=185.7, Time=95.7, Moves(2)=0.0, Pins(0)=0.0, Dist=90.0
[2026-05-04 09:14:51] MOVE 6: pl_B (yellow) 13->38 [0.16ms]
[2026-05-04 09:14:51] SCORE pl_B (yellow): Final=181.1, Time=83.1, Moves(4)=0.0, Pins(0)=0.0, Dist=98.0
[2026-05-04 09:14:51] SCORE PL_C (purple): Final=185.7, Time=95.7, Moves(2)=0.0, Pins(0)=0.0, Dist=90.0
[2026-05-04 09:14:56] MOVE 7: PL_C (purple) 85->64 [0.15ms]
[2026-05-04 09:14:56] SCORE pl_B (yellow): Final=181.1, Time=83.1, Moves(4)=0.0, Pins(0)=0.0, Dist=98.0
[2026-05-04 09:14:56] SCORE PL_C (purple): Final=182.6, Time=90.6, Moves(3)=0.0, Pins(0)=0.0, Dist=92.0
[2026-05-04 09:14:59] SCORE pl_B (yellow): Final=181.1, Time=83.1, Moves(4)=0.0, Pins(0)=0.0, Dist=98.0
[2026-05-04 09:14:59] SCORE PL_C (purple): Final=182.6, Time=90.6, Moves(3)=0.0, Pins(0)=0.0, Dist=92.0
[2026-05-04 09:14:59] GAME TIME LIMIT REACHED.
[2026-05-04 09:15:03] SCORE pl_B (yellow): Final=181.1, Time=83.1, Moves(4)=0.0, Pins(0)=0.0, Dist=98.0
[2026-05-04 09:15:03] SCORE PL_C (purple): Final=182.6, Time=90.6, Moves(3)=0.0, Pins(0)=0.0, Dist=92.0
[2026-05-04 09:15:03] GAME TIME LIMIT REACHED.
[2026-05-04 09:15:03] SCORE pl_B (yellow): Final=181.1, Time=83.1, Moves(4)=0.0, Pins(0)=0.0, Dist=98.0
[2026-05-04 09:15:03] SCORE PL_C (purple): Final=182.6, Time=90.6, Moves(3)=0.0, Pins(0)=0.0, Dist=92.0
[2026-05-04 09:15:03] GAME TIME LIMIT REACHED.
'''
#While create_leaderboard.py is running, it should periodically check for new log files and update the leaderboard accordingly. The script should run indefinitely until manually stopped. 
#If player already exists in the leaderboard, update their score to sum of existing score and new score. Otherwise, add new entry to leaderboard.
#also have columns for games played, average score per game, total time taken, total moves, total pins in goal, total distance, average time per move, average moves per game, average distance per game, and win bonus (if player won any games). Update these columns accordingly when processing each log file.    
#After processing all log files, calculate average score per game for each player and sort leaderboard by score in descending order. Write leaderboard to tab-separated file and print top 10 to console.
#reorder columns to have Player Name, Score, Games Played, Average Score, Total Time, Average Time, Total Moves, Average Moves, Total Pins in Goal, Average Pins in Goal, Total Distance, Average Distance, Win Bonus
#sort by Average Score in descending order, then by Games Played in ascending order (to break ties), then by Average Pins in Goal in descending order (to break further ties), then by Average Distance in descending order (to break further ties)

import os
import time
import pandas as pd

def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def extract_game_info(lines):
    game_id = None
    players = {}
    scores = {}

    for line in lines:
        line = line.strip()
        if "GAME CREATED" in line:
            game_id = line.split(']')[1].strip()
        elif "PLAYER JOINED" in line:
            parts = line.split('PLAYER JOINED:')[1].strip().split(' as ')
            player_name = parts[0].strip()
            player_color = parts[1].strip()
            players[player_color] = player_name
        elif "SCORE" in line:
            parts = line.split('SCORE')[1].strip().split(':')
            player_info = parts[0].strip()
            score_info = parts[1].strip()
            player_color = player_info.split(' ')[-1].strip('()')
            score_parts = score_info.split(', ') 
            final_score = float(score_parts[0].split('=')[1])
            time_score = float(score_parts[1].split('=')[1])
            move_score = float(score_parts[2].split('=')[1])
            pin_goal_score = float(score_parts[3].split('=')[1])
            distance_score = float(score_parts[4].split('=')[1])
            win_bonus = float(score_parts[5].split('=')[1]) if len(score_parts) > 5 else 0.0
            scores[player_color] = {
                "final_score": final_score,
                "time_score": time_score,
                "move_score": move_score,
                "pin_goal_score": pin_goal_score,
                "distance_score": distance_score,
                "win_bonus": win_bonus
            }
    return game_id, players, scores

def update_leaderboard(log_folder, leaderboard_file):
    leaderboard = []
    processed_files = set()

    while True:
        log_files = [f for f in os.listdir(log_folder) if f.endswith('.log')]
        new_files = [f for f in log_files if f not in processed_files]

        for file_name in new_files:
            file_path = os.path.join(log_folder, file_name)
            game_id, players, scores = extract_game_info(parse_log_file(file_path))
            for color, player_name in players.items():
                score = scores[color]
                existing_entry = next((entry for entry in leaderboard if entry['Player Name'] == player_name), None)
                if existing_entry:
                    existing_entry['Score'] += score.get('final_score', 0)
                    existing_entry['Games Played'] += 1
                    existing_entry['Total Time'] += score.get('time_score', 0)
                    existing_entry['Total Moves'] += score.get('move_score', 0)
                    existing_entry['Total Pins in Goal'] += score.get('pin_goal_score', 0)
                    existing_entry['Total Distance'] += score.get('distance_score', 0)
                else:
                    leaderboard.append({'Player Name': player_name, 'Score': score.get('final_score', 0), 'Total Time': score.get('time_score', 0), 'Total Moves': score.get('move_score', 0), 'Total Pins in Goal': score.get('pin_goal_score', 0), 'Total Distance': score.get('distance_score', 0), 'Games Played': 1})
            processed_files.add(file_name)
        for entry in leaderboard:
            entry['Average Score'] = entry['Score'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
            entry['Average Time'] = entry['Total Time'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
            entry['Average Moves'] = entry['Total Moves'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
            entry['Average Pins in Goal'] = entry['Total Pins in Goal'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
            entry['Average Distance'] = entry['Total Distance'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
        leaderboard.sort(key=lambda x: (-x['Average Score'], x['Games Played'], -x['Average Pins in Goal'], -x['Average Distance']))
        df = pd.DataFrame(leaderboard, columns=['Player Name', 'Score', 'Games Played', 'Average Score', 'Total Time', 'Average Time', 'Total Moves', 'Average Moves', 'Total Pins in Goal', 'Average Pins in Goal', 'Total Distance', 'Average Distance', 'Win Bonus'])
        df.to_csv(leaderboard_file, sep='\t', index=False) 

        print(df.head(10))
        time.sleep(10)  # Check for new log files every 10 seconds

if __name__ == "__main__":
    log_folder = 'games'
    leaderboard_file = 'leaderboard.tsv'
    update_leaderboard(log_folder, leaderboard_file)
    