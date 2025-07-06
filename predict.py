import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

class DataLoader:
    """
    DataLoader class handles loading player data from CSV files.
    """

    def __init__(self, player_name: str, base_path: Path = None):
        """
        Initialises DataLoader with player name and base path.
        If no base path is provided, it defaults to "{player_name} Stats".

        Args:
            player_name (str): Name of the player (used to load data files).
            base_path (Path, optional): Path to the folder containing player data (defaults to None).
        """
        self.player_name = player_name
        self.base_path = base_path or Path(f"{player_name} Stats")
        
    def load_data(self):
        """
        Loads data for various match statistics for the player: bowler, batting position, opposition,
        venue, and innings number.

        Returns:
            tuple: DataFrames containing data for bowler, batting position, opposition, venue, and innings.
        """
        df_bowler = pd.read_csv(self.base_path / f"{self.player_name} Bowler matchwise.csv", parse_dates=["match_date"])
        df_pos = pd.read_csv(self.base_path / f"{self.player_name} Batting Position matchwise.csv", parse_dates=["match_date"])
        df_oppo = pd.read_csv(self.base_path / f"{self.player_name} Opposition matchwise.csv", parse_dates=["match_date"])
        df_venue = pd.read_csv(self.base_path / f"{self.player_name} Venue matchwise.csv", parse_dates=["match_date"])
        df_innings = pd.read_csv(self.base_path / f"{self.player_name} Innings Num matchwise.csv", parse_dates=["match_date"])

        return df_bowler, df_pos, df_oppo, df_venue, df_innings

class FeatureEngineer:
    """
    FeatureEngineer class creates features for predictive modeling using player data.
    """

    def __init__(self, player_name: str, run_target: int):
        """
        Initialises the FeatureEngineer object with player name and target run value.

        Args:
            player_name (str): Name of the player.
            run_target (int): The target number of runs for the classification task.
        """
        self.player_name = player_name
        self.run_target = run_target
    
    def create_features(self, df_bowler, df_pos, df_oppo, df_venue, df_innings):
        """
        Creates features for predictive modeling, merging different datasets and calculating statistics.

        Args:
            df_bowler (DataFrame): Bowler data.
            df_pos (DataFrame): Batting position data.
            df_oppo (DataFrame): Opposition team data.
            df_venue (DataFrame): Venue data.
            df_innings (DataFrame): Innings number data.

        Returns:
            DataFrame: A DataFrame containing features and actual runs for each match.
        """
        df = df_pos.merge(df_oppo[["match_id", "opposition"]], on="match_id")
        df = df.merge(df_venue[["match_id", "venue"]], on="match_id")
        df = df.merge(df_innings[["match_id", "innings_num"]], on="match_id")
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.sort_values("match_date").reset_index(drop=True)

        # Prepare feature rows
        feature_rows = []
        for i, row in df.iterrows():
            match_id = row["match_id"]
            match_date = row["match_date"]
            opposition = row["opposition"]
            venue = row["venue"]
            innings_num = row["innings_num"]
            batting_pos = row["batting_pos"]
            actual_runs = row["runs"]

            df_prior_matches = df[df["match_date"] < match_date]
            if df_prior_matches.empty:
                continue

            all_time_avg = df_prior_matches["runs"].mean()
            df_last_3 = df_prior_matches.tail(3)
            avg_last_3 = df_last_3["runs"].mean() if len(df_last_3) >= 1 else np.nan

            df_bowlers_prior = df_bowler[df_bowler["match_date"] < match_date]
            match_bowlers = df_bowler[df_bowler["match_id"] == match_id]["bowler"].unique()

            bowler_avgs = []
            bowler_srs = []

            # Calculate bowler statistics
            for bowler in match_bowlers:
                bowler_stats = df_bowlers_prior[df_bowlers_prior["bowler"] == bowler]
                runs = bowler_stats["runs"].sum()
                balls = bowler_stats["balls"].sum()
                outs = bowler_stats["dismissals"].sum()

                if balls > 0:
                    sr = (runs / balls) * 100
                    bowler_srs.append(sr)
                if outs > 0:
                    avg = runs / outs
                    bowler_avgs.append(avg)

            career_runs = df_bowlers_prior["runs"].sum()
            career_balls = df_bowlers_prior["balls"].sum()
            career_outs = df_bowlers_prior["dismissals"].sum()

            career_avg = (career_runs / career_outs) if career_outs > 0 else np.nan
            career_sr = (career_runs / career_balls) * 100 if career_balls > 0 else np.nan

            avg_vs_bowlers = np.mean(bowler_avgs) if bowler_avgs else career_avg
            sr_vs_bowlers = np.mean(bowler_srs) if bowler_srs else career_sr

            if np.isnan(avg_vs_bowlers) or np.isnan(sr_vs_bowlers):
                continue

            feature_rows.append({
                "match_id": match_id,
                "match_date": match_date,
                "avg_vs_bowlers": avg_vs_bowlers,
                "sr_vs_bowlers": sr_vs_bowlers,
                "batting_pos": batting_pos,
                "opposition": opposition,
                "venue": venue,
                "innings_num": innings_num,
                "all_time_avg": all_time_avg,
                "avg_last_3": avg_last_3,
                "actual_runs": actual_runs
            })

        # Create feature DataFrame
        df_features = pd.DataFrame(feature_rows)
        df_features["runs_over_target"] = (df_features["actual_runs"] > self.run_target).astype(int)
        df_features = df_features.sort_values("match_date").reset_index(drop=True)

        return df_features
    

class Predictor:
    """
    Predictor handles preprocessing, model evaluation, and result interpretation for predicting cricket performance.
    """

    def __init__(self, run_target: int):
        """
        Initialises the Predictor object with run target and other necessary configurations for model evaluation.

        Args:
            run_target (int): The target number of runs for classification tasks (greater or lesser).
        """
        self.run_target = run_target
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.cv = TimeSeriesSplit(n_splits=5)

    def preprocess(self, df_features):
        """
        Preprocesses features by encoding categorical variables and splitting into training/test sets.

        Args:
            df_features (DataFrame): The feature DataFrame.

        Returns:
            tuple: (x_train, x_test, df_train, df_test) prepared for training/testing.
        """
        # Split into train/test BEFORE encoding
        split_idx = int(len(df_features) * 0.8)
        df_train = df_features.iloc[:split_idx].copy()
        df_test = df_features.iloc[split_idx:].copy()

        # One-hot encoding after split
        categorical_cols = ["opposition", "venue", "innings_num"]
        self.encoder.fit(df_train[categorical_cols])

        def encode(df_part):
            encoded = self.encoder.transform(df_part[categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(categorical_cols))
            x = pd.concat([df_part[["avg_vs_bowlers", "sr_vs_bowlers", "batting_pos", "all_time_avg", "avg_last_3"]].reset_index(drop=True),
                        encoded_df.reset_index(drop=True)], axis=1)
            return x

        x_train = encode(df_train)
        x_test = encode(df_test)
        
        train_mean_avg3 = x_train["avg_last_3"].mean()

        # Fill NaN values in avg_last_3 with the mean of training set
        x_train["avg_last_3"] = x_train["avg_last_3"].fillna(train_mean_avg3)
        x_test["avg_last_3"] = x_test["avg_last_3"].fillna(train_mean_avg3)
        
        return x_train, x_test, df_train, df_test

    def evaluate_models(self, x_train, y_train, model_dict, scoring):
        """
        Evaluates the given models using cross-validation and the specified scoring metric.

        Args:
            x_train (DataFrame): Training features.
            y_train (Series): Training labels.
            model_dict (dict): Dictionary of models to evaluate.
            scoring (str): Scoring metric for evaluation.

        Returns:
            dict: Results of model evaluations.
        """
        print(f"\n--- Cross-Validated {scoring.upper()} ---")
        best_model_name, best_score = None, -np.inf
        for name, model in model_dict.items():
            scores = cross_val_score(model, x_train, y_train, scoring=scoring, cv=self.cv)
            mean_score = np.mean(scores) if scoring != 'neg_root_mean_squared_error' else -np.mean(scores)
            print(f"{name}: CV {scoring} = {mean_score:.2f}")
            if mean_score > best_score:
                best_score, best_model_name = mean_score, name
        return best_model_name
        

    def print_model_summary(self, probs, actuals, clf_name, threshold=0.5):
        """
        Prints a detailed summary of model predictions, including performance metrics and PnL.

        Args:
            probs (list): List of predicted probabilities.
            actuals (list): List of actual outcomes (0 or 1).
            clf_name (str): The name of the classifier.
            threshold (float): The probability threshold to classify predictions as 'Above' or 'Below'. Default is 0.5.
        """

        # Classify predictions based on the threshold and store them as 'Above' or 'Below'
        decisions = ["Above" if p > threshold else "Below" for p in probs]
 
        # Count predictions
        above_preds = decisions.count("Above")
        below_preds = decisions.count("Below")
        total_preds = len(decisions)

        # Calculate correct and incorrect predictions
        correct_preds = sum((d == "Above" and a == 1) or (d == "Below" and a == 0) for d, a in zip(decisions, actuals))
        incorrect_preds = total_preds - correct_preds
        
        # Calculate profit/loss (PnL)
        pnl = correct_preds - incorrect_preds # pnl if we bet an arbitrary unit 1 on above target runs vs below target runs

        # Count correct 'Above' and 'Below' predictions
        correct_above_preds = sum(1 for d, a in zip(decisions, actuals) if d == "Above" and a == 1)
        correct_below_preds = sum(1 for d, a in zip(decisions, actuals) if d == "Below" and a == 0)

        # Print summary
        print(f"\n--- {clf_name} Summary ---")
        print(f"Total Predictions: {total_preds}")
        print(f"Above {self.run_target} Predictions: {above_preds}")
        print(f"Below {self.run_target} Predictions: {below_preds}")
        print(f"Correct Predictions: {correct_preds}")
        print(f"Incorrect Predictions: {incorrect_preds}")
        print(f"Total Profit/Loss: {pnl}")
        print(f"Average PnL per Trade: {pnl/total_preds:.2f}")
        print(f"Correct Above {self.run_target} Predictions: {correct_above_preds}")
        print(f"Correct Below {self.run_target} Predictions: {correct_below_preds}")
        print(f"Accuracy: {(correct_preds/total_preds):.2f}")



if __name__ == "__main__":
    # Set player name for analysis and run target for classification
    player_name = "V Kohli"
    run_target = 30

    # Load data and prepare features
    data_loader = DataLoader(player_name)
    df_bowler, df_pos, df_oppo, df_venue, df_innings = data_loader.load_data()

    engineer = FeatureEngineer(player_name, run_target)
    df_features = engineer.create_features(df_bowler, df_pos, df_oppo, df_venue, df_innings)

    predictor = Predictor(run_target)
    x_train, x_test, df_train, df_test = predictor.preprocess(df_features)

    # Prepare target variables for regression and classification models
    y_train_reg = df_train["actual_runs"]
    y_test_reg = df_test["actual_runs"]
    y_train_clf = df_train["runs_over_target"]
    y_test_clf = df_test["runs_over_target"]

    # Dictionary of regression and classification models
    regressors = {
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=1, learning_rate=0.05, random_state=42, subsample=0.6, colsample_bytree=0.8, n_jobs=-1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression()
    }

    classifiers = {
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Logistic Regression (L2)": LogisticRegression(max_iter=10000, penalty='l2', random_state=42)
    }

    # Train and evaluate the best regression model for predicting runs
    print(f"\nSCORE PREDICTOR FOR {player_name.upper()}")
    best_reg_name= predictor.evaluate_models(x_train, y_train_reg, regressors, scoring="neg_root_mean_squared_error")

    # Train and evaluate the best classifier for predicting if player will score above/below target
    print(f"\n{player_name.upper()} SCORE GREATER/LOWER THAN {run_target}")
    best_clf_name = predictor.evaluate_models(x_train, y_train_clf, classifiers, scoring="accuracy")

    # Select XGBoost and the best classifier model for further evaluation
    trading_clf_names = {"XGBoost"}
    trading_clf_names.add(best_clf_name)

    # Output predictions summary for the chosen models
    print("\nPREDICTIONS SUMMARY")
    for clf_name in trading_clf_names:
        clf = classifiers[clf_name]
        clf.fit(x_train, y_train_clf)
        preds = clf.predict_proba(x_test)[:, 1]
        predictor.print_model_summary(preds, y_test_clf, clf_name)


