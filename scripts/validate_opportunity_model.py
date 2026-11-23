
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
    mean_squared_error,
    brier_score_loss,
    roc_auc_score,
)

# ==========================================
# A3 PHASE 4.7B
# PREGAME OPPORTUNITY PREDICTION
# ==========================================
#
# 2024: Training
# 2025: Validation
# 2024-2025: Final training
# 2026: Independent holdout
#
# No sportsbook information.
# Existing A3 models remain unchanged.
#
# This is an experimental historical backtest,
# not a live game-day availability model.

DATA = Path("data/processed")
OUT = Path("data/model_results")

OUT.mkdir(parents=True, exist_ok=True)

INPUT = DATA / "pregame_player_pool.csv"

OUTPUT_REPORT = (
    OUT / "a3_opportunity_validation.json"
)

OUTPUT_PREDICTIONS = (
    OUT / "a3_opportunity_predictions.csv"
)

TARGETS = {
    "QB": "pass_attempts",
    "RB": "carries",
    "WR": "targets",
    "TE": "targets",
}

# RB receiving targets are also evaluated.
EXTRA_TARGETS = {
    "RB": ["targets"],
}

METRICS = [
    "pass_attempts",
    "carries",
    "targets",
    "receptions",
    "pass_yards",
    "rush_yards",
    "rec_yards",
]

FEATURES = [
    "prior_games",
    "weeks_since_appearance",
    "recently_seen",
]

FEATURES += [
    f"prior_mean_{metric}"
    for metric in METRICS
]

KEYS = [
    "season",
    "week",
    "game_id",
    "team",
    "player_id",
]


# ==========================================
# UTILITIES
# ==========================================

def clean_x(frame):
    return (
        frame[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "bias": float(np.mean(error)),
        "overprediction_rate": float(
            np.mean(error > 0)
        ),
    }


def classification_metrics(actual, probability):
    actual = np.asarray(actual, dtype=int)
    probability = np.asarray(
        probability,
        dtype=float,
    )

    result = {
        "brier": float(
            brier_score_loss(actual, probability)
        ),
        "actual_participation_rate": float(
            np.mean(actual)
        ),
        "mean_predicted_probability": float(
            np.mean(probability)
        ),
    }

    if len(np.unique(actual)) == 2:
        result["auc"] = float(
            roc_auc_score(actual, probability)
        )
    else:
        result["auc"] = None

    return result


# ==========================================
# PARTICIPATION MODEL
# ==========================================

def predict_participation(train, test):
    y = (
        train["observed_any_opportunity"]
        .astype(int)
    )

    # Handle training subsets with only
    # one observed participation class.
    if y.nunique() < 2:
        return np.full(
            len(test),
            float(y.iloc[0]),
        )

    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=10,
        min_samples_leaf=40,
        l2_regularization=10,
        random_state=42,
    )

    model.fit(clean_x(train), y)

    return model.predict_proba(
        clean_x(test)
    )[:, 1]


# ==========================================
# CONDITIONAL WORKLOAD MODEL
# ==========================================

def predict_workload(train, test, target):
    # Only train workload size on players
    # with a positive opportunity count.
    active = train[
        train[f"actual_{target}"] > 0
    ].copy()

    if len(active) < 50:
        return np.full(
            len(test),
            np.nan,
        )

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=12,
        min_samples_leaf=25,
        l2_regularization=10,
        random_state=42,
    )

    model.fit(
        clean_x(active),
        active[f"actual_{target}"],
    )

    prediction = model.predict(
        clean_x(test)
    )

    return np.maximum(prediction, 0)


# ==========================================
# COMBINED OPPORTUNITY FORECAST
# ==========================================

def forecast(train, test, target):
    # This is the probability of any offensive
    # opportunity, not necessarily a positive
    # count for the specific target.
    #
    # We therefore fit an additional target-
    # specific positive-opportunity classifier.

    y = (
        train[f"actual_{target}"] > 0
    ).astype(int)

    if y.nunique() < 2:
        probability = np.full(
            len(test),
            float(y.iloc[0]),
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

        classifier.fit(clean_x(train), y)

        probability = classifier.predict_proba(
            clean_x(test)
        )[:, 1]

    conditional = predict_workload(
        train,
        test,
        target,
    )

    expected = probability * conditional

    return probability, conditional, expected


# ==========================================
# HISTORICAL BASELINE
# ==========================================

def historical_baseline(test, target):
    return np.maximum(
        test[f"prior_mean_{target}"]
        .fillna(0)
        .to_numpy(dtype=float),
        0,
    )


# ==========================================
# EVALUATION
# ==========================================

def evaluate_split(train, test, target):
    probability, conditional, expected = (
        forecast(train, test, target)
    )

    if np.isnan(expected).any():
        raise RuntimeError(
            f"Missing workload forecasts for {target}"
        )

    actual = test[
        f"actual_{target}"
    ].to_numpy(dtype=float)

    baseline = historical_baseline(
        test,
        target,
    )

    actual_positive = (
        actual > 0
    ).astype(int)

    report = {
        "rows": int(len(test)),
        "historical_baseline": regression_metrics(
            actual,
            baseline,
        ),
        "opportunity_model": regression_metrics(
            actual,
            expected,
        ),
        "positive_opportunity": (
            classification_metrics(
                actual_positive,
                probability,
            )
        ),
    }

    report["mae_improvement"] = (
        report["historical_baseline"]["mae"]
        - report["opportunity_model"]["mae"]
    )

    return (
        report,
        probability,
        conditional,
        expected,
        baseline,
    )


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def run():
    df = pd.read_csv(
        INPUT,
        dtype={"player_id": str},
        low_memory=False,
    )

    required = (
        KEYS
        + ["roster_position"]
        + ["observed_any_opportunity"]
        + FEATURES
        + [
            f"actual_{metric}"
            for metric in METRICS
        ]
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
            "Duplicate player-game records"
        )

    df["recently_seen"] = (
        df["recently_seen"]
        .astype(str)
        .str.lower()
        .map({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        })
    )

    numeric = set(FEATURES)

    numeric.update(
        f"actual_{metric}"
        for metric in METRICS
    )

    numeric.add("observed_any_opportunity")

    for column in numeric:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df["observed_any_opportunity"].isna().any():
        raise ValueError(
            "Missing participation labels"
        )

    reports = []
    predictions = []

    for position, primary in TARGETS.items():
        targets = (
            [primary]
            + EXTRA_TARGETS.get(position, [])
        )

        subset = df[
            df["roster_position"] == position
        ].copy()

        for target in targets:
            subset_target = subset.dropna(
                subset=[
                    f"actual_{target}",
                    f"prior_mean_{target}",
                ]
            ).copy()

            train24 = subset_target[
                subset_target["season"] == 2024
            ].copy()

            valid25 = subset_target[
                subset_target["season"] == 2025
            ].copy()

            train_full = subset_target[
                subset_target["season"].isin(
                    [2024, 2025]
                )
            ].copy()

            test26 = subset_target[
                subset_target["season"] == 2026
            ].copy()

            if (
                len(train24) < 100
                or len(valid25) < 30
                or len(test26) < 30
            ):
                print(
                    f"SKIPPED {position}/{target}: "
                    "insufficient data"
                )
                continue

            # 2025 validation.
            validation, _, _, _, _ = (
                evaluate_split(
                    train24,
                    valid25,
                    target,
                )
            )

            # 2026 untouched holdout.
            (
                holdout,
                probability,
                conditional,
                expected,
                baseline,
            ) = evaluate_split(
                train_full,
                test26,
                target,
            )

            actual = test26[
                f"actual_{target}"
            ].to_numpy(dtype=float)

            # Opportunity mismatch is measured
            # against historical workload.
            # It is NOT a sportsbook-line gap.
            difference = expected - baseline

            ratio = expected / np.maximum(
                baseline,
                1.0,
            )

            output = test26[
                KEYS + [
                    "player_name",
                    "roster_position",
                    "opponent",
                ]
            ].copy()

            output["target"] = target
            output["actual"] = actual
            output["historical_average"] = baseline

            output[
                "positive_opportunity_probability"
            ] = probability

            output[
                "conditional_workload"
            ] = conditional

            output[
                "expected_opportunity"
            ] = expected

            output[
                "opportunity_difference"
            ] = difference

            output[
                "opportunity_ratio"
            ] = ratio

            predictions.append(output)

            reports.append({
                "position": position,
                "target": target,
                "validation_2025": validation,
                "holdout_2026": holdout,
            })

            print(
                f"{position}/{target}: "
                f"2025 MAE gain "
                f"{validation['mae_improvement']:+.3f}, "
                f"2026 MAE gain "
                f"{holdout['mae_improvement']:+.3f}"
            )

    if not reports:
        raise RuntimeError(
            "No opportunity evaluations completed"
        )

    improvements = sum(
        item["holdout_2026"]["mae_improvement"] > 0
        for item in reports
    )

    summary = {
        "experiment": "A3 Phase 4.7B",
        "training_2024": True,
        "validation_2025": True,
        "final_training": [2024, 2025],
        "holdout_2026": True,
        "comparisons": len(reports),
        "holdout_improvements": improvements,
        "holdout_losses_or_ties": (
            len(reports) - improvements
        ),
        "results": reports,
        "limitations": [
            "Historical candidates are not official active rosters.",
            "New players and first appearances can be missed.",
            "No verified injury or inactive status is used.",
            "No current-game defensive matchup features are used.",
            "No offensive snap projections are used.",
            "No team-level pass or rush volume forecast is used.",
            "Player transfers can cause candidate omissions.",
            "2026 is an incomplete season.",
            "A positive-opportunity classifier is not an active-status classifier.",
            "No conservative thresholds are calibrated.",
            "This experiment does not predict breakout probability.",
            "Existing A3 models must not be automatically replaced.",
        ],
    }

    with open(
        OUTPUT_REPORT,
        "w",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    pd.concat(
        predictions,
        ignore_index=True,
    ).to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print("\n================================")
    print("A3 PHASE 4.7B COMPLETE")
    print("================================")

    print(
        f"Comparisons: {len(reports)}"
    )

    print(
        f"2026 improvements: {improvements}"
    )

    print(
        "2026 losses or ties:",
        len(reports) - improvements,
    )

    print(
        f"Report: {OUTPUT_REPORT}"
    )

    print(
        f"Predictions: {OUTPUT_PREDICTIONS}"
    )

    print(
        "\nExisting A3 models remain unchanged."
    )


if __name__ == "__main__":
    run()
