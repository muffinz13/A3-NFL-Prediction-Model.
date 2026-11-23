
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# A3 PHASE 4.6
# Position-specific defensive feature selection.
#
# 2024: feature-selection training
# 2025: feature-selection validation
# 2024-2025: final model training
# 2026: untouched final evaluation
#
# Experimental only. No sportsbook inputs.

DATA = Path("data/processed")
OUT = Path("data/model_results")
OUT.mkdir(parents=True, exist_ok=True)

INPUT = DATA / "player_vulnerability_matchup_features.csv"

TARGETS = {
    "RB": [
        "rush_yards", "carries",
        "rec_yards", "receptions", "targets"
    ],
    "WR": [
        "rec_yards", "receptions", "targets"
    ],
    "TE": [
        "rec_yards", "receptions", "targets"
    ],
}

METRICS = [
    "pass_yards", "completions", "pass_attempts",
    "rush_yards", "carries",
    "rec_yards", "receptions", "targets",
]

PLAYER = ["prior_games"]

for metric in METRICS:
    PLAYER += [
        f"season_prior_{metric}",
        f"last5_{metric}",
        f"last5_vs_season_{metric}",
    ]

GENERAL = [
    f"opp_{play}_{metric}"
    for play in ["pass", "run"]
    for metric in [
        "prior5_epa_allowed",
        "prior5_success_allowed",
        "prior5_yards_per_play_allowed",
        "prior5_explosive_rate_allowed",
    ]
]

POSITION = [
    f"opp_position_prior5_{metric}"
    for metric in [
        "carries", "rush_yards",
        "targets", "receptions", "rec_yards"
    ]
]

BASE = PLAYER + GENERAL + POSITION

GROUPS = {
    "short_pass": [
        "short_pass_attempts",
        "short_pass_completions",
        "short_pass_yards",
        "short_completion_rate",
        "short_yards_per_attempt",
    ],
    "deep_pass": [
        "pass_plays",
        "deep_pass_attempts",
        "explosive_passes",
        "explosive_pass_rate",
    ],
    "rushing": [
        "rush_attempts",
        "rush_yards",
        "explosive_runs",
        "explosive_run_rate",
        "rush_yards_per_carry",
    ],
    "rb_receiving": [
        "rb_targets",
        "rb_receptions",
        "rb_rec_yards",
        "rb_yards_per_target",
    ],
    "te_receiving": [
        "te_targets",
        "te_receptions",
        "te_rec_yards",
        "te_yards_per_target",
    ],
}

GROUPS = {
    name: [f"prior5_{x}" for x in values]
    for name, values in GROUPS.items()
}

KEYS = [
    "season", "week", "game_id",
    "team", "player_id"
]


def evaluate(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "overprediction_rate": float(
            np.mean(error > 0)
        ),
        "overprediction_mae": float(
            np.mean(error[error > 0])
        ) if np.any(error > 0) else 0.0,
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
    df = pd.read_csv(
        INPUT,
        dtype={"player_id": str},
        low_memory=False,
    )

    all_group_features = [
        feature
        for features in GROUPS.values()
        for feature in features
    ]

    all_targets = sorted(set(
        target
        for targets in TARGETS.values()
        for target in targets
    ))

    required = (
        KEYS
        + ["roster_position", "prior_games"]
        + BASE
        + all_group_features
        + all_targets
    )

    missing = sorted(
        set(required) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df.duplicated(KEYS).any():
        raise ValueError(
            "Duplicate player-game rows"
        )

    for column in set(
        BASE + all_group_features + all_targets
    ):
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
            group_target = group.dropna(
                subset=[
                    target,
                    f"season_prior_{target}"
                ]
            )

            train24 = group_target[
                group_target["season"] == 2024
            ].copy()

            valid25 = group_target[
                group_target["season"] == 2025
            ].copy()

            train_full = group_target[
                group_target["season"].isin(
                    [2024, 2025]
                )
            ].copy()

            test26 = group_target[
                group_target["season"] == 2026
            ].copy()

            if (
                len(train24) < 100
                or len(valid25) < 20
                or len(test26) < 20
            ):
                print(
                    f"SKIP {position}/{target}: "
                    "insufficient data"
                )
                continue

            # Select one defensive group using
            # 2025 only. The 2026 outcomes are
            # never used in this decision.

            base_valid = predict(
                train24, valid25, BASE, target
            )

            base_mae = evaluate(
                valid25[target], base_valid
            )["mae"]

            selected_group = None
            selected_features = []
            best_gain = 0.0
            group_results = {}

            for name, features in GROUPS.items():
                candidate = predict(
                    train24,
                    valid25,
                    BASE + features,
                    target,
                )

                candidate_mae = evaluate(
                    valid25[target],
                    candidate,
                )["mae"]

                gain = base_mae - candidate_mae

                group_results[name] = {
                    "validation_mae": candidate_mae,
                    "validation_gain": gain,
                }

                if gain > best_gain:
                    best_gain = gain
                    selected_group = name
                    selected_features = features

            # Refit after selection.
            # Compare on the same 2026 test rows.

            player_pred = predict(
                train_full,
                test26,
                PLAYER,
                target,
            )

            base_pred = predict(
                train_full,
                test26,
                BASE,
                target,
            )

            selected_pred = predict(
                train_full,
                test26,
                BASE + selected_features,
                target,
            )

            actual = test26[target].to_numpy(
                dtype=float
            )

            baseline = np.maximum(
                test26[
                    f"season_prior_{target}"
                ].to_numpy(dtype=float),
                0,
            )

            scores = {
                "season_average": evaluate(
                    actual, baseline
                ),
                "player_only": evaluate(
                    actual, player_pred
                ),
                "existing_matchup": evaluate(
                    actual, base_pred
                ),
                "selected_vulnerability": evaluate(
                    actual, selected_pred
                ),
            }

            gain26 = (
                scores["existing_matchup"]["mae"]
                - scores[
                    "selected_vulnerability"
                ]["mae"]
            )

            # Diagnostic breakout subset.
            # These are known after the game;
            # not a prospective detection test.

            minimum = (
                20.0 if "yards" in target
                else 3.0
            )

            breakout = (
                (actual >= baseline * 1.5)
                & (actual >= minimum)
            )

            breakout_result = None

            if breakout.sum() >= 10:
                breakout_result = {
                    "games": int(breakout.sum()),
                    "existing_mae": evaluate(
                        actual[breakout],
                        base_pred[breakout]
                    )["mae"],
                    "selected_mae": evaluate(
                        actual[breakout],
                        selected_pred[breakout]
                    )["mae"],
                }

            reports.append({
                "position": position,
                "target": target,
                "selected_group": selected_group,
                "selected_features": selected_features,
                "validation_2025_gain": best_gain,
                "validation_groups": group_results,
                "test_2026_rows": len(test26),
                "test_2026_gain": gain26,
                "test_2026_helped": bool(
                    gain26 > 0
                ),
                "scores": scores,
                "breakout_diagnostic": breakout_result,
            })

            output = test26[
                KEYS + ["roster_position"]
            ].copy()

            output["target"] = target
            output["actual"] = actual
            output["season_average"] = baseline
            output["player_only"] = player_pred
            output["existing_matchup"] = base_pred
            output["selected_vulnerability"] = (
                selected_pred
            )
            output["selected_group"] = (
                selected_group or "none"
            )
            output["is_breakout"] = breakout

            predictions.append(output)

            print(
                f"{position}/{target}: "
                f"group={selected_group or 'none'}, "
                f"2025_gain={best_gain:+.3f}, "
                f"2026_gain={gain26:+.3f}"
            )

    if not reports:
        raise RuntimeError(
            "No Phase 4.6 evaluations completed"
        )

    wins = sum(
        r["test_2026_helped"]
        for r in reports
    )

    summary = {
        "experiment": "A3 Phase 4.6",
        "selection_train": [2024],
        "selection_validation": 2025,
        "final_train": [2024, 2025],
        "final_test": 2026,
        "evaluations": len(reports),
        "wins": wins,
        "losses_or_ties": len(reports) - wins,
        "limitations": [
            "Player-game selection uses observed participation.",
            "Feature groups are selected using one validation season.",
            "2026 is a single final holdout.",
            "No individual coverage or alignment data.",
            "Breakout subset uses known outcomes.",
            "No prospective breakout classification metrics.",
            "Pregame active-player eligibility is not modeled.",
            "Conservative thresholds are not calibrated.",
            "Do not automatically promote selected features.",
        ],
        "results": reports,
    }

    report_path = (
        OUT / "a3_feature_selection_validation.json"
    )

    prediction_path = (
        OUT / "a3_feature_selection_predictions.csv"
    )

    with open(report_path, "w") as file:
        json.dump(summary, file, indent=2)

    pd.concat(
        predictions, ignore_index=True
    ).to_csv(
        prediction_path,
        index=False,
    )

    print("\nA3 PHASE 4.6 COMPLETE")
    print("Evaluations:", len(reports))
    print("2026 improvements:", wins)
    print(
        "2026 losses/ties:",
        len(reports) - wins
    )
    print("Report:", report_path)
    print("Predictions:", prediction_path)
    print(
        "Experimental only. "
        "Existing A3 models unchanged."
    )


if __name__ == "__main__":
    run()
