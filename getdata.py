import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

class GetPlayerStats:
    """
    Extracts match-by-match batter statistics (batting position, runs, balls, dismissals, 
    and performance vs opposition, venue, and bowlers) from Cricsheet JSON files.
    Saves output into CSVs.
    """

    def __init__(self, player_name: str, data_dir: str = "data"):
        """
        Initialises the class with the player's name and data directory.

        Args:
            player_name (str): Name of the player whose stats will be extracted.
            data_dir (str): Directory where JSON files are stored.
        """
        self.player_name = player_name
        self.data_folder = Path(data_dir)
        self.output_folder = Path(f"{player_name} Stats")
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.processed_ids = self.load_processed_match_ids()

        # Initialize lists to hold row data for each type of CSV output
        self.bowler_rows = []
        self.pos_rows = []
        self.oppo_rows = []
        self.venue_rows = []
        self.innings_num_rows = []
    
    def load_processed_match_ids(self):
        """
        Loads match IDs already processed (based on existing Batting Position file).
        Prevents reprocessing of matches and enables incremental updates.

        Returns:
            set: Set of already-processed match IDs as strings.
        """
        path = Path(f"{self.player_name} Stats/{self.player_name} Batting Position matchwise.csv")

        if path.exists():
            df = pd.read_csv(path)
            return set(df["match_id"].astype(str).values)
        
        return set()

    def update_bowler_stats(self, stats, delivery):
        """
        Updates a single bowler's stats (runs conceded, balls bowled, dismissals caused) for a delivery against our player.

        Args:
            stats (dict): Current dictionary of the bowler's stats.
            delivery (dict): Delivery JSON data from Cricsheet.
        """
        runs = delivery.get("runs", {}).get("batter", 0)
        wickets = delivery.get("wickets")
        extras = delivery.get("extras", {})

        stats["runs"] += runs

        if "wides" not in extras:
            stats["balls"] += 1
            
        if wickets:
            for wicket in wickets:
                if wicket.get("player_out") == self.player_name:
                    stats["dismissals"] += 1

    def process_innings(self, inning_data, match_id, match_date, player_team, oppo_team, oppo_bowlers, venue, innings_num):
        """
        Processes one innings of a match and updates internal data structures with match-level stats.

        Args:
            inning_data (dict): JSON data for a single innings.
            match_id (str): Unique id of the match.
            match_date (datetime): Date of the match.
            player_team (str): Player's team in the match.
            oppo_team (str): Opposition team.
            oppo_bowlers (set): Set of opposition bowlers.
            venue (str): Venue where the match was played.
            innings_num (str): Innings number which the player batted in.
        """
        overs = inning_data.get("overs", [])
        batting_order = []
        batting_pos = None
        dismissed = False
        runs_scored = 0
        balls_faced = 0
        bowler_stats = defaultdict(lambda: {"runs": 0, "balls": 0, "dismissals": 0})

        for over in overs:
            for delivery in over.get("deliveries", []):
                batter = delivery.get("batter")
                if batter not in batting_order:
                    batting_order.append(batter)

                # Skip deliveries where the player wasn't batting
                if batter != self.player_name:
                    continue

                bowler = delivery.get("bowler")
                if bowler in oppo_bowlers:
                    self.update_bowler_stats(bowler_stats[bowler], delivery)

                runs_scored += delivery.get("runs", {}).get("batter", 0)
                if "wides" not in delivery.get("extras", {}):
                    balls_faced += 1

                wickets = delivery.get("wickets")
                if wickets:
                    for wicket in wickets:
                        if wicket.get("player_out") == self.player_name:
                            dismissed = True

        # Determine batting position (1-based index)
        if self.player_name in batting_order:
            batting_pos = batting_order.index(self.player_name) + 1
                    
        # Record bowler-specific performance
        for bowler, stats in bowler_stats.items():
            if stats["balls"] > 0:
                self.bowler_rows.append({
                    "match_id": match_id,
                    "match_date": match_date,
                    "bowler": bowler,
                    "runs": stats["runs"],
                    "balls": stats["balls"],
                    "dismissals": stats["dismissals"]
                })

        # Record batting position
        if batting_pos is not None:
            self.pos_rows.append({
                "match_id": match_id,
                "match_date": match_date,
                "batting_pos": batting_pos,
                "runs": runs_scored,
                "balls": balls_faced,
                "dismissals": 1 if dismissed else 0
            })

        # Record performance vs opposition
        self.oppo_rows.append({
            "match_id": match_id,
            "match_date": match_date,
            "opposition": oppo_team,
            "runs": runs_scored,
            "balls": balls_faced,
            "dismissals": 1 if dismissed else 0
        })

        # Record performance at venue
        self.venue_rows.append({
            "match_id": match_id,
            "match_date": match_date,
            "venue": venue,
            "runs": runs_scored,
            "balls": balls_faced,
            "dismissals": 1 if dismissed else 0
        })

        # Record performance by innings number (1st or 2nd)
        self.innings_num_rows.append({
            "match_id": match_id,
            "match_date": match_date,
            "innings_num": innings_num,
            "runs": runs_scored,
            "balls": balls_faced,
            "dismissals": 1 if dismissed else 0
        })

    def save_matchwise_to_csv(self, rows, label):
        """
        Saves new matchwise stats to a CSV file, appending to any existing file for the player.

        Args:
            rows (list): List of dictionaries containing match-level data.
            label (str): Descriptor used in filename (e.g., "Bowler matchwise").
        """
        df_new = pd.DataFrame(rows)
        out_path = self.output_folder / f"{self.player_name} {label}.csv"

        # If file exists, append new data
        if out_path.exists():
            df_old = pd.read_csv(out_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new

        if "match_date" in df_combined.columns:
            df_combined["match_date"] = pd.to_datetime(df_combined["match_date"])
            df_combined = df_combined.sort_values("match_date")

        df_combined = df_combined.reset_index(drop=True)
        df_combined.to_csv(out_path, index=False)

    def process_files(self):
        """
        Loops through all JSON match files, extracts stats for the target player, 
        and writes match-level data to output CSVs.
        """
        for file_path in sorted(self.data_folder.rglob("*.json")):
            match_id = file_path.stem

            # Skip files already processed
            if match_id in self.processed_ids:
                continue

            match_data = json.load(file_path.open())
            info = match_data.get("info", {})

            match_date = pd.to_datetime(info.get("dates", [""])[0])
            teams = info.get("teams", ["TeamA", "TeamB"])
            players = info.get("players", {})
            innings = match_data.get("innings", [])
            venue = info.get("venue", "Unknown")

            # Identify player's team
            player_team = None
            for team, team_players in players.items():
                if self.player_name in team_players:
                    player_team = team
                    break
            if not player_team:
                continue  # Player did not play in this match

            oppo_team = [t for t in teams if t != player_team][0]
            oppo_bowlers = set(players.get(oppo_team, []))

            # Process innings played by player's team
            for i, innings in enumerate(innings):
                if innings.get("team") != player_team:
                    continue
                    
                innings_num = "1st" if i == 0 else "2nd"
                self.process_innings(
                    inning_data=innings,
                    match_id=match_id,
                    match_date=match_date,
                    player_team=player_team,
                    oppo_team=oppo_team,
                    oppo_bowlers=oppo_bowlers,
                    venue=venue,
                    innings_num=innings_num                    
                )

        # Save final datasets
        self.save_matchwise_to_csv(self.bowler_rows, "Bowler matchwise")
        self.save_matchwise_to_csv(self.pos_rows, "Batting Position matchwise")
        self.save_matchwise_to_csv(self.oppo_rows, "Opposition matchwise")
        self.save_matchwise_to_csv(self.venue_rows, "Venue matchwise")
        self.save_matchwise_to_csv(self.innings_num_rows, "Innings Num matchwise")


if __name__ == "__main__":   
    # Example usage: extract and save matchwise stats for Virat Kohli
    player_name = "V Kohli" 
    processer = GetPlayerStats(player_name=player_name)
    processer.process_files()
