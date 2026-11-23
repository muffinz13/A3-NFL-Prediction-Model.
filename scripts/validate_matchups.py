
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path("data/processed")
OUT = Path("data/model_results")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = [2024, 2025]
TEST_YEAR = 2026

TARGETS = {
    "passer": [
        "pass_yards",
        "pass_attempts",
        "completions",
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

METRICS = [
    "pass_yards",
    "pass_attempts",
    "completions",
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

PLAYER_FEATURES = ["prior_games"]

for metric in METRICS:
    PLAYER_FEATURES += [
        f"season_prior_{metric}",
        f"last5_{metric}",
        f"last5_vs_season_{metric}",
    ]

MATCHUP_FEATURES = []

for play_type in ["pass", "run"]:
    for metric in DEFENSE_METRICS:
        MATCHUP_FEATURES.append(
            f"opp_{play_type}_{metric}"
        )


def score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    return {
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "rmse": round(
            float(np.sqrt(np.mean(errors ** 2))), 4
        ),
        "bias": round(float(np.mean(errors)), 4),
        "overprediction_rate": round(
            float(np.mean(errors > 0)), 4
        ),
        "within_10": round(
            float(np.mean(np.abs(errors) <= 10)), 4
        ),
    }


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=10.0,
        random_state=42,
    )


def fit_predict(train, test, features, target):
    model = make_model()

    x_train = train[features].replace(
        [np.inf, -np.inf], np.nan
    )

    x_test = test[features].replace(
        [np.inf, -np.inf], np.nan
    )

    model.fit(x_train, train[target])

    return np.maximum(model.predict(x_test), 0)


def run():
    path = DATA / "player_matchup_features.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        dtype={"player_id": str},
        low_memory=False,
    )

    required = (
        [
            "season", "week", "game_id",
            "player_id", "team", "opponent",
            "roles",
        ]
        + PLAYER_FEATURES
        + MATCHUP_FEATURES
        + list({
            target
            for targets in TARGETS.values()
            for target in targets
        })
    )

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    keys = [
        "season", "week", "game_id",
        "player_id", "team",
    ]

    if df.duplicated(keys).any():
        raise ValueError("Duplicate player-game records")

    numeric_columns = list(set(
        PLAYER_FEATURES
        + MATCHUP_FEATURES
        + [
            target
            for targets in TARGETS.values()
            for target in targets
        ]
    ))

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    reports = []
    predictions = []

    for role, targets in TARGETS.items():

        # IMPORTANT:
        # These are postgame observed roles.
        # Results are conditional on participation.
        role_mask = (
            df["roles"]
            .fillna("")
            .str.split(",")
            .apply(lambda values: role in values)
        )

        role_df = df[
            role_mask & (df["prior_games"] >= 2)
        ].copy()

        for target in targets:
            baseline_col = f"season_prior_{target}"

            subset = role_df.dropna(
                subset=[target, baseline_col]
            ).copy()

            train = subset[
                subset["season"].isin(TRAIN_YEARS)
            ].copy()

            test = subset[
                subset["season"] == TEST_YEAR
            ].copy()

            if len(train) < 100 or len(test) < 20:
                print(
                    f"Skipping {role}/{target}: "
                    "insufficient data"
                )
                continue

            # Identical train/test rows for all methods.
            actual = test[target].to_numpy()

            baseline = np.maximum(
                test[baseline_col].to_numpy(), 0
            )

            player_model = fit_predict(
                train,
                test,
                PLAYER_FEATURES,
                target,
            )

            matchup_model = fit_predict(
                train,
                test,
                PLAYER_FEATURES + MATCHUP_FEATURES,
                target,
            )

            baseline_score = score(actual, baseline)
            player_score = score(actual, player_model)
            matchup_score = score(actual, matchup_model)

            player_gain = (
                baseline_score["mae"]
                - player_score["mae"]
            )

            matchup_gain = (
                player_score["mae"]
                - matchup_score["mae"]
            )

            report = {
                "role": role,
                "target": target,
                "train_rows": len(train),
                "test_rows": len(test),
                "baseline": baseline_score,
                "player_only": player_score,
                "player_plus_matchup": matchup_score,
                "player_gain_mae": round(
                    player_gain, 4
                ),
                "defense_added_gain_mae": round(
                    matchup_gain, 4
                ),
                "defense_improves_accuracy": (
                    matchup_gain > 0
                ),
            }

            reports.append(report)

            result = test[keys + ["opponent"]].copy()

            result["role"] = role
            result["target"] = target
            result["actual"] = actual
            result["baseline"] = baseline
            result["player_only"] = player_model
            result["player_plus_matchup"] = matchup_model

            predictions.append(result)

            print(
                f"{role}/{target} | "
                f"baseline={baseline_score['mae']:.3f} | "
                f"player={player_score['mae']:.3f} | "
                f"matchup={matchup_score['mae']:.3f} | "
                f"defense gain={matchup_gain:+.3f}"
            )

    if not reports:
        raise RuntimeError(
            "No models evaluated. Check input data."
        )

    wins = sum(
        x["defense_improves_accuracy"]
        for x in reports
    )

    summary = {
        "experiment": "A3 Phase 4.1 defensive ablation",
        "train_years": TRAIN_YEARS,
        "test_year": TEST_YEAR,
        "model_count": len(reports),
        "defense_wins": wins,
        "defense_losses_or_ties": len(reports) - wins,
        "limitations": [
            "Player participation is selected after games.",
            "Roles are observed, not pregame roster positions.",
            "Only one fixed historical holdout is tested.",
            "Results do not establish causal matchup value.",
            "No calibrated conservative thresholds yet.",
        ],
        "results": reports,
    }

    with open(
        OUT / "a3_matchup_validation.json", "w"
    ) as file:
        json.dump(summary, file, indent=2)

    pd.concat(
        predictions, ignore_index=True
    ).to_csv(
        OUT / "a3_matchup_validation_predictions.csv",
        index=False,
    )

    print("\nA3 PHASE 4.1 COMPLETE")
    print(f"Models tested: {len(reports)}")
    print(f"Defensive features improved: {wins}")
    print(
        "Defensive features did not improve: "
        f"{len(reports) - wins}"
    )

    print(
        "These results are experimental, "
        "not verified pregame forecasts."
    )


if __name__ == "__main__":
    run()
