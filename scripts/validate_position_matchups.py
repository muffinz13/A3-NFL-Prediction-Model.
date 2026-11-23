
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# A3 PHASE 4.3
# Position-specific defensive matchup experiment.
# Train: 2024-2025. Test: 2026.
# No sportsbook data is used.

DATA = Path("data/processed")
OUT = Path("data/model_results")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "RB": [
        "rush_yards", "carries",
        "rec_yards", "receptions", "targets"
    ],
    "WR": ["rec_yards", "receptions", "targets"],
    "TE": ["rec_yards", "receptions", "targets"],
}

METRICS = [
    "pass_yards", "completions", "pass_attempts",
    "rush_yards", "carries",
    "rec_yards", "receptions", "targets",
]

DEFENSE_METRICS = [
    "prior5_epa_allowed",
    "prior5_success_allowed",
    "prior5_yards_per_play_allowed",
    "prior5_explosive_rate_allowed",
]

POSITION_METRICS = [
    "carries", "rush_yards",
    "targets", "receptions", "rec_yards",
]

PLAYER_FEATURES = ["prior_games"]

for metric in METRICS:
    PLAYER_FEATURES += [
        f"season_prior_{metric}",
        f"last5_{metric}",
        f"last5_vs_season_{metric}",
    ]

GENERAL_FEATURES = []

for play_type in ["pass", "run"]:
    for metric in DEFENSE_METRICS:
        GENERAL_FEATURES.append(
            f"opp_{play_type}_{metric}"
        )

POSITION_FEATURES = [
    f"opp_position_prior5_{metric}"
    for metric in POSITION_METRICS
]


def evaluate(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(
            np.sqrt(np.mean(errors ** 2))
        ),
        "bias": float(np.mean(errors)),
        "overprediction_rate": float(
            np.mean(errors > 0)
        ),
    }


def predict(train, test, features, target):
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=10.0,
        random_state=42,
    )

    x_train = train[features].replace(
        [np.inf, -np.inf], np.nan
    )

    x_test = test[features].replace(
        [np.inf, -np.inf], np.nan
    )

    model.fit(x_train, train[target])

    return np.maximum(
        model.predict(x_test), 0
    )


def run():
    path = (
        DATA / "player_position_matchup_features.csv"
    )

    df = pd.read_csv(
        path,
        dtype={"player_id": str},
        low_memory=False,
    )

    required = (
        [
            "season", "week", "game_id",
            "player_id", "team", "opponent",
            "roster_position"
        ]
        + PLAYER_FEATURES
        + GENERAL_FEATURES
        + POSITION_FEATURES
        + list({
            target
            for targets in TARGETS.values()
            for target in targets
        })
    )

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    keys = [
        "season", "week", "game_id",
        "player_id", "team"
    ]

    if df.duplicated(keys).any():
        raise ValueError(
            "Duplicate player-game rows"
        )

    numeric = list(set(
        PLAYER_FEATURES
        + GENERAL_FEATURES
        + POSITION_FEATURES
        + [
            target
            for targets in TARGETS.values()
            for target in targets
        ]
    ))

    for column in numeric:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    reports = []
    predictions = []

    for position, targets in TARGETS.items():
        group = df[
            (df["roster_position"] == position)
            & (df["prior_games"] >= 2)
        ].copy()

        for target in targets:
            baseline_col = f"season_prior_{target}"

            subset = group.dropna(
                subset=[target, baseline_col]
            ).copy()

            train = subset[
                subset["season"].isin([2024, 2025])
            ].copy()

            test = subset[
                subset["season"] == 2026
            ].copy()

            if len(train) < 100 or len(test) < 20:
                print(
                    f"SKIPPED {position}/{target}: "
                    "insufficient data"
                )
                continue

            actual = test[target].to_numpy()

            baseline = np.maximum(
                test[baseline_col].to_numpy(), 0
            )

            player = predict(
                train, test,
                PLAYER_FEATURES, target
            )

            general = predict(
                train, test,
                PLAYER_FEATURES + GENERAL_FEATURES,
                target
            )

            position_model = predict(
                train, test,
                PLAYER_FEATURES
                + GENERAL_FEATURES
                + POSITION_FEATURES,
                target
            )

            scores = {
                "baseline": evaluate(actual, baseline),
                "player_only": evaluate(actual, player),
                "general_defense": evaluate(
                    actual, general
                ),
                "position_defense": evaluate(
                    actual, position_model
                ),
            }

            gain = (
                scores["general_defense"]["mae"]
                - scores["position_defense"]["mae"]
            )

            reports.append({
                "position": position,
                "target": target,
                "train_rows": len(train),
                "test_rows": len(test),
                "scores": scores,
                "position_feature_mae_gain": float(gain),
                "position_features_help": bool(gain > 0),
                "position_history_coverage_test": float(
                    test[
                        "opp_position_prior5_rec_yards"
                    ].notna().mean()
                ),
            })

            result = test[
                keys + ["opponent", "roster_position"]
            ].copy()

            result["target"] = target
            result["actual"] = actual
            result["baseline"] = baseline
            result["player_only"] = player
            result["general_defense"] = general
            result["position_defense"] = position_model

            predictions.append(result)

            print(
                f"{position}/{target}: "
                f"player={scores['player_only']['mae']:.3f}, "
                f"general={scores['general_defense']['mae']:.3f}, "
                f"position={scores['position_defense']['mae']:.3f}, "
                f"position_gain={gain:+.3f}"
            )

    if not reports:
        raise RuntimeError(
            "No models evaluated"
        )

    wins = sum(
        report["position_features_help"]
        for report in reports
    )

    summary = {
        "experiment": "A3 Phase 4.3",
        "training_seasons": [2024, 2025],
        "holdout_season": 2026,
        "evaluated_models": len(reports),
        "position_feature_wins": wins,
        "position_feature_losses_or_ties": (
            len(reports) - wins
        ),
        "limitations": [
            "Player-game rows are selected using observed participation.",
            "Season roster positions may be imperfect.",
            "Position defensive totals use observed offensive player production.",
            "This is one chronological holdout, not rolling validation.",
            "No individual coverage or route alignment data is included.",
            "Conservative thresholds are not calibrated here."
        ],
        "results": reports,
    }

    with open(
        OUT / "a3_position_validation.json", "w"
    ) as file:
        json.dump(summary, file, indent=2)

    pd.concat(
        predictions, ignore_index=True
    ).to_csv(
        OUT / "a3_position_validation_predictions.csv",
        index=False
    )

    print("\nA3 PHASE 4.3 COMPLETE")
    print(f"Models evaluated: {len(reports)}")
    print(f"Position features helped: {wins}")
    print(
        "Position features did not help:",
        len(reports) - wins
    )
    print(
        "Results are experimental. "
        "Do not promote models without further validation."
    )


if __name__ == "__main__":
    run()
