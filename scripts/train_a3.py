
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# A3 PHASE 4 — HISTORICAL MODEL BACKTEST
# Train: 2024–2025
# Holdout test: 2026
# No sportsbook lines used.

DATA = Path("data/processed")
OUT = Path("data/model_results")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = [2024, 2025]
TEST_SEASON = 2026

# Observed play roles, not official roster positions.
TARGETS = {
    "passer": [
        "pass_yards",
        "completions",
        "pass_attempts",
    ],
    "rusher": [
        "rush_yards",
        "carries",
    ],
    "receiver": [
        "rec_yards",
        "receptions",
        "targets",
    ],
}

HISTORY_METRICS = [
    "pass_yards",
    "completions",
    "pass_attempts",
    "rush_yards",
    "carries",
    "rec_yards",
    "receptions",
    "targets",
]

DEFENSE_METRICS = [
    "prior5_epa_allowed",
    "prior5_success_allowed",
    "prior5_yards_per_play_allowed",
    "prior5_explosive_rate_allowed",
]

def load_data():
    path = DATA / "player_matchup_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the A3 data workflow."
        )

    df = pd.read_csv(path, low_memory=False)

    required = [
        "season", "week", "game_id",
        "player_id", "team", "opponent",
        "roles", "prior_games",
    ]

    for metrics in TARGETS.values():
        required.extend(metrics)

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    keys = [
        "season", "week", "game_id",
        "player_id", "team"
    ]

    if df.duplicated(keys).any():
        raise ValueError("Duplicate player-game rows")

    df["season"] = pd.to_numeric(
        df["season"], errors="raise"
    )

    df["prior_games"] = pd.to_numeric(
        df["prior_games"], errors="coerce"
    )

    return df

def feature_columns(df):
    features = ["prior_games"]

    for metric in HISTORY_METRICS:
        features.extend([
            f"season_prior_{metric}",
            f"last5_{metric}",
            f"last5_vs_season_{metric}",
        ])

    for play_type in ["pass", "run"]:
        for metric in DEFENSE_METRICS:
            features.append(
                f"opp_{play_type}_{metric}"
            )

    missing = sorted(set(features) - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing pregame features: {missing}"
        )

    return features

def evaluate(y, prediction):
    return {
        "mae": round(
            float(mean_absolute_error(y, prediction)), 3
        ),
        "rmse": round(
            float(np.sqrt(mean_squared_error(y, prediction))),
            3
        ),
        "bias": round(
            float(np.mean(prediction - y)), 3
        ),
    }

def run():
    df = load_data()
    features = feature_columns(df)

    # Convert only explicitly approved pregame features.
    for column in features:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    reports = []
    predictions = []

    for role, metrics in TARGETS.items():

        # Role labels are observed from completed games.
        # This is a conditional historical backtest,
        # not yet a deployable pregame participation model.
        role_df = df[
            df["roles"].fillna("").str.split(",")
            .apply(lambda roles: role in roles)
        ].copy()

        # At least two previous appearances.
        role_df = role_df[
            role_df["prior_games"] >= 2
        ]

        for target in metrics:
            baseline_col = f"season_prior_{target}"

            subset = role_df.copy()

            subset[target] = pd.to_numeric(
                subset[target], errors="coerce"
            )

            subset[baseline_col] = pd.to_numeric(
                subset[baseline_col], errors="coerce"
            )

            # Evaluate model and baseline on identical rows.
            subset = subset.dropna(
                subset=[target, baseline_col]
            )

            train = subset[
                subset["season"].isin(TRAIN_SEASONS)
            ].copy()

            test = subset[
                subset["season"] == TEST_SEASON
            ].copy()

            if len(train) < 100 or len(test) < 20:
                print(
                    f"Skipping {role}/{target}: "
                    "insufficient historical data"
                )
                continue

            X_train = train[features].replace(
                [np.inf, -np.inf], np.nan
            )
            X_test = test[features].replace(
                [np.inf, -np.inf], np.nan
            )

            y_train = train[target]
            y_test = test[target]

            model = HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=10.0,
                random_state=42,
            )

            model.fit(X_train, y_train)

            model_pred = model.predict(X_test)

            # Physical count/yardage totals cannot be negative.
            model_pred = np.maximum(model_pred, 0)

            baseline_pred = np.maximum(
                test[baseline_col].to_numpy(), 0
            )

            model_scores = evaluate(
                y_test, model_pred
            )

            baseline_scores = evaluate(
                y_test, baseline_pred
            )

            improvement = (
                baseline_scores["mae"]
                - model_scores["mae"]
            )

            report = {
                "role": role,
                "target": target,
                "train_games": len(train),
                "test_games": len(test),
                "train_seasons": TRAIN_SEASONS,
                "test_season": TEST_SEASON,
                "model": model_scores,
                "baseline": baseline_scores,
                "mae_improvement": round(
                    improvement, 3
                ),
                "beats_baseline": improvement > 0,
            }

            reports.append(report)

            output = test[
                [
                    "season", "week", "game_id",
                    "player_id", "team", "opponent"
                ]
            ].copy()

            output["role"] = role
            output["target"] = target
            output["actual"] = y_test.to_numpy()
            output["model_prediction"] = model_pred
            output["baseline_prediction"] = baseline_pred

            predictions.append(output)

            print(
                f"{role} / {target}: "
                f"model MAE={model_scores['mae']}, "
                f"baseline MAE={baseline_scores['mae']}"
            )

    if not reports:
        raise ValueError(
            "No models trained. Check historical data."
        )

    report_path = OUT / "a3_backtest_report.json"

    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2)

    predictions_df = pd.concat(
        predictions, ignore_index=True
    )

    predictions_df.to_csv(
        OUT / "a3_backtest_predictions.csv",
        index=False
    )

    print("\nA3 PHASE 4 BACKTEST COMPLETE")
    print(f"Models evaluated: {len(reports)}")
    print(f"Report: {report_path}")
    print(
        "Predictions:",
        OUT / "a3_backtest_predictions.csv"
    )

    wins = sum(
        report["beats_baseline"]
        for report in reports
    )

    print(
        f"Models beating historical baseline: "
        f"{wins}/{len(reports)}"
    )

    print(
        "Results are experimental. "
        "Do not deploy without further validation."
    )

if __name__ == "__main__":
    run()
