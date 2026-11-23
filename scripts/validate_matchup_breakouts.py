
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    brier_score_loss,
    roc_auc_score,
)

# A3 PHASE 4.7C
# Experimental defensive matchup and
# opportunity breakout validation.
#
# Train: 2024
# Validate: 2025
# Final train: 2024-2025
# Holdout: 2026
#
# No sportsbook lines.
# No changes to incumbent A3 models.

BASE = Path("data/processed")
OUT = Path("data/model_results")
OUT.mkdir(parents=True, exist_ok=True)

POOL = BASE / "pregame_player_pool.csv"
DEFENSE = (
    BASE / "defense_position_game_features.csv"
)
VULNERABILITY = (
    BASE / "defense_vulnerability_game_features.csv"
)

REPORT = OUT / "a3_matchup_breakout_validation.json"
PREDICTIONS = OUT / "a3_matchup_breakout_predictions.csv"

TARGETS = [
    ("QB", "pass_attempts"),
    ("RB", "carries"),
    ("RB", "targets"),
    ("WR", "targets"),
    ("TE", "targets"),
]

METRICS = [
    "pass_attempts",
    "carries",
    "targets",
    "receptions",
    "pass_yards",
    "rush_yards",
    "rec_yards",
]

KEYS = [
    "season", "week", "game_id",
    "team", "player_id",
]

DEF_KEYS = [
    "season", "week", "game_id",
    "opponent",
]

PLAYER_FEATURES = [
    "prior_games",
    "weeks_since_appearance",
    "recently_seen",
] + [
    f"prior_mean_{m}" for m in METRICS
]

POSITION_FEATURES = [
    f"opp_position_prior5_{m}"
    for m in [
        "carries",
        "rush_yards",
        "targets",
        "receptions",
        "rec_yards",
    ]
]

VULNERABILITY_FEATURES = [
    "prior5_pass_plays",
    "prior5_short_pass_attempts",
    "prior5_short_completion_rate",
    "prior5_short_yards_per_attempt",
    "prior5_deep_pass_attempts",
    "prior5_explosive_pass_rate",
    "prior5_rush_attempts",
    "prior5_rush_yards",
    "prior5_explosive_run_rate",
    "prior5_rush_yards_per_carry",
    "prior5_rb_targets",
    "prior5_rb_yards_per_target",
    "prior5_te_targets",
    "prior5_te_yards_per_target",
]


def load_data():
    pool = pd.read_csv(
        POOL, dtype={"player_id": str},
        low_memory=False,
    )

    defense = pd.read_csv(
        DEFENSE, low_memory=False,
    )

    vulnerability = pd.read_csv(
        VULNERABILITY, low_memory=False,
    )

    if pool.duplicated(KEYS).any():
        raise ValueError("Duplicate player-game pool rows")

    pos_keys = DEF_KEYS + ["roster_position"]

    if defense.duplicated(pos_keys).any():
        raise ValueError("Duplicate position defense rows")

    if vulnerability.duplicated(DEF_KEYS).any():
        raise ValueError("Duplicate vulnerability rows")

    needed = (
        pos_keys + POSITION_FEATURES
    )

    missing = set(needed) - set(defense.columns)
    if missing:
        raise ValueError(
            f"Missing position columns: {sorted(missing)}"
        )

    needed = DEF_KEYS + VULNERABILITY_FEATURES
    missing = set(needed) - set(vulnerability.columns)
    if missing:
        raise ValueError(
            f"Missing vulnerability columns: {sorted(missing)}"
        )

    pool = pool.merge(
        defense[pos_keys + POSITION_FEATURES],
        on=pos_keys,
        how="left",
        validate="many_to_one",
    )

    pool = pool.merge(
        vulnerability[
            DEF_KEYS + VULNERABILITY_FEATURES
        ],
        on=DEF_KEYS,
        how="left",
        validate="many_to_one",
    )

    pool["recently_seen"] = (
        pool["recently_seen"]
        .astype(str)
        .str.lower()
        .map({
            "true": 1, "false": 0,
            "1": 1, "0": 0,
        })
    )

    for column in (
        PLAYER_FEATURES
        + POSITION_FEATURES
        + VULNERABILITY_FEATURES
        + [f"actual_{m}" for m in METRICS]
    ):
        pool[column] = pd.to_numeric(
            pool[column], errors="coerce"
        )

    return pool


def matrix(df, features):
    return (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )


def workload_forecast(train, test, target, features):
    y = train[f"actual_{target}"].to_numpy(float)
    positive = (y > 0).astype(int)

    if len(np.unique(positive)) < 2:
        probability = np.full(
            len(test), float(positive[0])
        )
    else:
        classifier = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=10,
            min_samples_leaf=40,
            l2_regularization=10,
            random_state=42,
        )

        classifier.fit(
            matrix(train, features), positive
        )

        probability = classifier.predict_proba(
            matrix(test, features)
        )[:, 1]

    active = train[
        train[f"actual_{target}"] > 0
    ]

    if len(active) < 50:
        raise ValueError(
            f"Insufficient positive examples: {target}"
        )

    regressor = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=12,
        min_samples_leaf=25,
        l2_regularization=10,
        random_state=42,
    )

    regressor.fit(
        matrix(active, features),
        active[f"actual_{target}"],
    )

    conditional = np.maximum(
        regressor.predict(matrix(test, features)),
        0,
    )

    return probability * conditional


def breakout_label(df, target):
    actual = df[f"actual_{target}"].to_numpy(float)
    usual = df[f"prior_mean_{target}"].to_numpy(float)

    # Meaningful workload breakout:
    # at least 25% above prior average
    # AND at least two extra opportunities.
    threshold = np.maximum(
        usual * 1.25,
        usual + 2.0,
    )

    return (actual >= threshold).astype(int)


def breakout_probability(train, test, target, features):
    y = breakout_label(train, target)

    if len(np.unique(y)) < 2:
        return np.full(len(test), float(y[0]))

    classifier = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=10,
        min_samples_leaf=40,
        l2_regularization=10,
        random_state=42,
    )

    classifier.fit(
        matrix(train, features), y
    )

    return classifier.predict_proba(
        matrix(test, features)
    )[:, 1]


def evaluate(train, test, target):
    baseline = workload_forecast(
        train, test, target, PLAYER_FEATURES
    )

    matchup_features = (
        PLAYER_FEATURES
        + POSITION_FEATURES
        + VULNERABILITY_FEATURES
    )

    challenger = workload_forecast(
        train, test, target, matchup_features
    )

    probability = breakout_probability(
        train, test, target, matchup_features
    )

    labels = breakout_label(test, target)

    actual = test[
        f"actual_{target}"
    ].to_numpy(float)

    baseline_mae = mean_absolute_error(
        actual, baseline
    )

    challenger_mae = mean_absolute_error(
        actual, challenger
    )

    # Brier score is descriptive here.
    # Probabilities have not been
    # separately calibrated.
    breakout = {
        "actual_rate": float(labels.mean()),
        "mean_predicted_probability": float(
            probability.mean()
        ),
        "brier_score": float(
            brier_score_loss(labels, probability)
        ),
        "constant_rate_brier": float(
            brier_score_loss(
                labels,
                np.full(
                    len(labels),
                    float(breakout_label(
                        train, target
                    ).mean()),
                ),
            )
        ),
    }

    if len(np.unique(labels)) == 2:
        breakout["auc"] = float(
            roc_auc_score(labels, probability)
        )
    else:
        breakout["auc"] = None

    report = {
        "rows": int(len(test)),
        "baseline_mae": float(baseline_mae),
        "matchup_mae": float(challenger_mae),
        "mae_improvement": float(
            baseline_mae - challenger_mae
        ),
        "baseline_bias": float(
            np.mean(baseline - actual)
        ),
        "matchup_bias": float(
            np.mean(challenger - actual)
        ),
        "breakout": breakout,
    }

    return report, baseline, challenger, probability


def run():
    df = load_data()

    results = []
    prediction_frames = []

    for position, target in TARGETS:
        part = df[
            df["roster_position"] == position
        ].copy()

        part = part.dropna(
            subset=[
                f"actual_{target}",
                f"prior_mean_{target}",
            ]
        )

        train24 = part[part.season == 2024]
        valid25 = part[part.season == 2025]
        train_full = part[
            part.season.isin([2024, 2025])
        ]
        test26 = part[part.season == 2026]

        if min(
            len(train24), len(valid25), len(test26)
        ) < 30:
            print(f"Skipping {position}/{target}")
            continue

        validation, _, _, _ = evaluate(
            train24, valid25, target
        )

        holdout, baseline, challenger, prob = (
            evaluate(train_full, test26, target)
        )

        actual = test26[
            f"actual_{target}"
        ].to_numpy(float)

        output = test26[
            KEYS + [
                "player_name",
                "roster_position",
                "opponent",
            ]
        ].copy()

        output["statistic"] = target
        output["actual"] = actual
        output["opportunity_baseline"] = baseline
        output["matchup_projection"] = challenger
        output["usual_opportunity"] = test26[
            f"prior_mean_{target}"
        ].to_numpy(float)

        output["projected_vs_usual"] = (
            challenger - output["usual_opportunity"]
        )

        output["breakout_probability_uncalibrated"] = prob
        output["actual_breakout"] = breakout_label(
            test26, target
        )

        prediction_frames.append(output)

        results.append({
            "position": position,
            "target": target,
            "validation_2025": validation,
            "holdout_2026": holdout,
        })

        print(
            f"{position}/{target} "
            f"2025 MAE gain: "
            f"{validation['mae_improvement']:+.3f} | "
            f"2026 MAE gain: "
            f"{holdout['mae_improvement']:+.3f}"
        )

    if not results:
        raise RuntimeError("No comparisons completed")

    report = {
        "phase": "4.7C",
        "objective": (
            "Test whether prior-game defensive matchup "
            "features improve player opportunity forecasts."
        ),
        "holdout_year": 2026,
        "comparisons": len(results),
        "holdout_improvements": sum(
            r["holdout_2026"]["mae_improvement"] > 0
            for r in results
        ),
        "results": results,
        "limitations": [
            "Historical candidate pool omits some players.",
            "No verified game-day inactive or injury feed.",
            "No individual cornerback or coverage assignments.",
            "No receiver route or alignment data.",
            "No team play-volume projection.",
            "Breakout probabilities are uncalibrated.",
            "Breakout means workload, not yardage breakout.",
            "2026 is an incomplete season.",
            "This is not a live pregame prediction service.",
            "Incumbent A3 models remain unchanged.",
        ],
    }

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_csv(PREDICTIONS, index=False)

    print("\nA3 PHASE 4.7C COMPLETE")
    print(
        "Holdout improvements:",
        report["holdout_improvements"],
        "of",
        report["comparisons"],
    )
    print("Report:", REPORT)
    print("Predictions:", PREDICTIONS)
    print("Existing A3 models remain unchanged.")


if __name__ == "__main__":
    run()
