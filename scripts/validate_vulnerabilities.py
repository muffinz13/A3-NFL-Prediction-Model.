
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# A3 PHASE 4.5
# Chronological defensive vulnerability validation.
# This is an experiment, not a deployed A3 model.
# No sportsbook data is used.

DATA = Path("data/processed")
OUT = Path("data/model_results")
OUT.mkdir(parents=True, exist_ok=True)

INPUT = DATA / "player_vulnerability_matchup_features.csv"

TARGETS = {
    "RB": [
        "rush_yards", "carries",
        "rec_yards", "receptions", "targets",
    ],
    "WR": [
        "rec_yards", "receptions", "targets",
    ],
    "TE": [
        "rec_yards", "receptions", "targets",
    ],
}

METRICS = [
    "pass_yards", "completions", "pass_attempts",
    "rush_yards", "carries",
    "rec_yards", "receptions", "targets",
]

PLAYER_FEATURES = ["prior_games"]

for metric in METRICS:
    PLAYER_FEATURES.extend([
        f"season_prior_{metric}",
        f"last5_{metric}",
        f"last5_vs_season_{metric}",
    ])

GENERAL_FEATURES = []

for play_type in ["pass", "run"]:
    for metric in [
        "prior5_epa_allowed",
        "prior5_success_allowed",
        "prior5_yards_per_play_allowed",
        "prior5_explosive_rate_allowed",
    ]:
        GENERAL_FEATURES.append(
            f"opp_{play_type}_{metric}"
        )

POSITION_FEATURES = [
    f"opp_position_prior5_{metric}"
    for metric in [
        "carries", "rush_yards",
        "targets", "receptions", "rec_yards",
    ]
]

VULNERABILITY_STATS = [
    "pass_plays",
    "short_pass_attempts",
    "short_pass_completions",
    "short_pass_yards",
    "deep_pass_attempts",
    "explosive_passes",
    "rush_attempts",
    "rush_yards",
    "explosive_runs",
    "rb_targets",
    "rb_receptions",
    "rb_rec_yards",
    "te_targets",
    "te_receptions",
    "te_rec_yards",
]

VULNERABILITY_RATES = [
    "short_completion_rate",
    "short_yards_per_attempt",
    "explosive_pass_rate",
    "explosive_run_rate",
    "rush_yards_per_carry",
    "rb_yards_per_target",
    "te_yards_per_target",
]

VULNERABILITY_FEATURES = [
    f"prior5_{name}"
    for name in (
        VULNERABILITY_STATS
        + VULNERABILITY_RATES
    )
]

EXISTING_FEATURES = (
    PLAYER_FEATURES
    + GENERAL_FEATURES
    + POSITION_FEATURES
)

ALL_FEATURES = (
    EXISTING_FEATURES
    + VULNERABILITY_FEATURES
)

FOLDS = [
    {
        "train": [2024],
        "test": 2025,
    },
    {
        "train": [2024, 2025],
        "test": 2026,
    },
]


def score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "bias": float(np.mean(error)),
        "overprediction_rate": float(
            np.mean(error > 0)
        ),
        "overprediction_mae": (
            float(np.mean(error[error > 0]))
            if np.any(error > 0)
            else 0.0
        ),
        "underprediction_rate": float(
            np.mean(error < 0)
        ),
    }


def fit_predict(train, test, features, target):
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

    model.fit(
        x_train,
        train[target],
    )

    return np.maximum(
        model.predict(x_test),
        0,
    )


def run():
    df = pd.read_csv(
        INPUT,
        dtype={"player_id": str},
        low_memory=False,
    )

    required = [
        "season", "week", "game_id",
        "team", "opponent", "player_id",
        "roster_position", "prior_games",
    ] + ALL_FEATURES

    required += list({
        target
        for values in TARGETS.values()
        for target in values
    })

    missing = sorted(
        set(required) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    keys = [
        "season", "week", "game_id",
        "team", "player_id",
    ]

    if df.duplicated(keys).any():
        raise ValueError(
            "Duplicate player-game records"
        )

    numeric_columns = list(set(
        ALL_FEATURES
        + [
            target
            for values in TARGETS.values()
            for target in values
        ]
    ))

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    reports = []
    predictions = []

    for fold in FOLDS:
        train_years = fold["train"]
        test_year = fold["test"]

        print(
            f"\nTRAIN {train_years} "
            f"-> TEST {test_year}"
        )

        for position, targets in TARGETS.items():
            group = df[
                (df["roster_position"] == position)
                & (df["prior_games"] >= 2)
            ].copy()

            for target in targets:
                baseline_col = (
                    f"season_prior_{target}"
                )

                subset = group.dropna(
                    subset=[
                        target,
                        baseline_col,
                    ]
                ).copy()

                train = subset[
                    subset["season"].isin(
                        train_years
                    )
                ].copy()

                test = subset[
                    subset["season"] == test_year
                ].copy()

                if (
                    len(train) < 100
                    or len(test) < 20
                ):
                    print(
                        f"SKIPPED {position}/"
                        f"{target}: insufficient rows"
                    )
                    continue

                actual = (
                    test[target]
                    .to_numpy(dtype=float)
                )

                baseline = np.maximum(
                    test[baseline_col].to_numpy(
                        dtype=float
                    ),
                    0,
                )

                player = fit_predict(
                    train,
                    test,
                    PLAYER_FEATURES,
                    target,
                )

                existing = fit_predict(
                    train,
                    test,
                    EXISTING_FEATURES,
                    target,
                )

                vulnerability = fit_predict(
                    train,
                    test,
                    ALL_FEATURES,
                    target,
                )

                scores = {
                    "season_average": score(
                        actual, baseline
                    ),
                    "player_only": score(
                        actual, player
                    ),
                    "existing_matchup": score(
                        actual, existing
                    ),
                    "phase44_vulnerability": score(
                        actual, vulnerability
                    ),
                }

                gain = (
                    scores["existing_matchup"]["mae"]
                    - scores[
                        "phase44_vulnerability"
                    ]["mae"]
                )

                # Breakout is defined against the
                # player's pregame season average.
                # The minimum ensures tiny baselines
                # do not create trivial breakouts.
                minimum = (
                    20.0
                    if "yards" in target
                    else 3.0
                )

                breakout = (
                    (actual >= baseline * 1.5)
                    & (actual >= minimum)
                )

                breakout_scores = None

                if breakout.sum() >= 10:
                    breakout_scores = {
                        "games": int(
                            breakout.sum()
                        ),
                        "player_only": score(
                            actual[breakout],
                            player[breakout],
                        ),
                        "existing_matchup": score(
                            actual[breakout],
                            existing[breakout],
                        ),
                        "phase44_vulnerability": score(
                            actual[breakout],
                            vulnerability[breakout],
                        ),
                    }

                reports.append({
                    "train_seasons": train_years,
                    "test_season": test_year,
                    "position": position,
                    "target": target,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "scores": scores,
                    "phase44_mae_gain": float(
                        gain
                    ),
                    "phase44_helped": bool(
                        gain > 0
                    ),
                    "vulnerability_coverage": float(
                        test[
                            "prior5_explosive_pass_rate"
                        ].notna().mean()
                    ),
                    "breakout": breakout_scores,
                })

                result = test[
                    keys + [
                        "opponent",
                        "roster_position",
                    ]
                ].copy()

                result["target"] = target
                result["actual"] = actual
                result["season_average"] = baseline
                result["player_only"] = player
                result["existing_matchup"] = existing
                result["phase44_vulnerability"] = (
                    vulnerability
                )
                result["is_breakout"] = breakout

                predictions.append(result)

                print(
                    f"{position}/{target}: "
                    f"player="
                    f"{scores['player_only']['mae']:.3f} "
                    f"existing="
                    f"{scores['existing_matchup']['mae']:.3f} "
                    f"phase44="
                    f"{scores['phase44_vulnerability']['mae']:.3f} "
                    f"gain={gain:+.3f}"
                )

    if not reports:
        raise RuntimeError(
            "No Phase 4.5 evaluations completed"
        )

    wins = sum(
        r["phase44_helped"]
        for r in reports
    )

    # Pool all test predictions for a consistent
    # aggregate comparison on identical rows.
    combined = pd.concat(
        predictions,
        ignore_index=True,
    )

    overall = {
        model: score(
            combined["actual"],
            combined[model],
        )
        for model in [
            "season_average",
            "player_only",
            "existing_matchup",
            "phase44_vulnerability",
        ]
    }

    # Pooling all positions/targets is only a
    # diagnostic: yards and counts have different
    # scales. Per-target results remain primary.
    summary = {
        "experiment": "A3 Phase 4.5",
        "validation": "chronological expanding window",
        "folds": FOLDS,
        "comparisons": len(reports),
        "phase44_wins": wins,
        "phase44_losses_or_ties": (
            len(reports) - wins
        ),
        "overall_diagnostic": overall,
        "limitations": [
            "Observed player-game participation is used to select rows.",
            "The 2025 fold trains on only one season.",
            "No weekly active-roster or injury eligibility model is included.",
            "Season-level roster positions may be imperfect.",
            "No individual cornerback coverage or route alignment is measured.",
            "Breakout labels are retrospective and are diagnostic only.",
            "Breakout-only error does not measure breakout detection precision or recall.",
            "Overprediction rate alone is not a calibrated threshold probability.",
            "Pooled MAE combines targets with different units.",
            "Play eligibility and feature definitions still require audit.",
            "These results do not establish live pregame forecasting accuracy.",
        ],
        "results": reports,
    }

    report_path = (
        OUT / "a3_vulnerability_validation.json"
    )

    prediction_path = (
        OUT
        / "a3_vulnerability_validation_predictions.csv"
    )

    with open(report_path, "w") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    combined.to_csv(
        prediction_path,
        index=False,
    )

    print("\n============================")
    print("A3 PHASE 4.5 COMPLETE")
    print("============================")
    print(f"Comparisons: {len(reports)}")
    print(f"Phase 4.4 wins: {wins}")
    print(
        "Losses/ties:",
        len(reports) - wins,
    )
    print(f"Test predictions: {len(combined):,}")
    print(f"Report: {report_path}")
    print(f"Predictions: {prediction_path}")
    print(
        "Experimental results only. "
        "Do not promote models automatically."
    )


if __name__ == "__main__":
    run()
