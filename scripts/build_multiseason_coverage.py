
from pathlib import Path
import json

import numpy as np
import pandas as pd
import requests

# A3 PHASE 4.8B
# Historical defensive coverage features.
# Only previous games contribute to pregame data.
# 2026 is not assumed available.

DATA = Path("data")
OUT = DATA / "processed"
REPORTS = DATA / "model_results"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEASONS = [2024, 2025]

COVERAGES = [
    "COVER_0", "COVER_1", "COVER_2",
    "COVER_3", "COVER_4", "COVER_6",
    "2_MAN",
]

ROUTES = [
    "SLANT", "SHALLOW CROSS/DRAG",
    "IN/DIG", "GO", "POST", "SCREEN",
]

COLUMNS = [
    "nflverse_game_id",
    "play_id",
    "possession_team",
    "route",
    "defense_man_zone_type",
    "defense_coverage_type",
    "was_pressure",
]


def safe_name(value):
    return "".join(
        c.lower() if c.isalnum() else "_"
        for c in value
    ).strip("_")


def download(year):
    path = DATA / f"pbp_participation_{year}.parquet"

    if not path.exists():
        url = (
            "https://github.com/nflverse/nflverse-data/"
            "releases/download/pbp_participation/"
            f"pbp_participation_{year}.parquet"
        )

        response = requests.get(url, timeout=180)
        response.raise_for_status()
        path.write_bytes(response.content)

    frame = pd.read_parquet(
        path, columns=COLUMNS
    )

    return frame


def prepare(frame, year):
    frame = frame.copy()

    if frame.duplicated(
        ["nflverse_game_id", "play_id"]
    ).any():
        raise ValueError(
            f"Duplicate participation plays: {year}"
        )

    parts = frame[
        "nflverse_game_id"
    ].astype(str).str.split("_", expand=True)

    if parts.shape[1] != 4:
        raise ValueError("Unexpected game ID format")

    frame["season"] = pd.to_numeric(
        parts[0], errors="raise"
    ).astype(int)

    frame["week"] = pd.to_numeric(
        parts[1], errors="raise"
    ).astype(int)

    if not frame["season"].eq(year).all():
        raise ValueError("Season mismatch")

    away = parts[2]
    home = parts[3]

    frame["team"] = frame["possession_team"]

    frame["opponent"] = np.where(
        frame["team"].eq(away),
        home,
        np.where(
            frame["team"].eq(home),
            away,
            None,
        ),
    )

    if frame["opponent"].isna().any():
        raise ValueError("Invalid possession team")

    frame["coverage_known"] = (
        frame["defense_coverage_type"]
        .notna().astype(int)
    )

    frame["route_known"] = (
        frame["route"].notna().astype(int)
    )

    frame["pressure_known"] = (
        frame["was_pressure"].notna().astype(int)
    )

    frame["pressure"] = (
        frame["was_pressure"]
        .astype(str)
        .str.lower()
        .eq("true")
        .astype(int)
    )

    for coverage in COVERAGES:
        frame[
            "coverage_" + safe_name(coverage)
        ] = (
            frame["defense_coverage_type"]
            .eq(coverage).astype(int)
        )

    for route in ROUTES:
        frame[
            "route_" + safe_name(route)
        ] = (
            frame["route"].eq(route).astype(int)
        )

    return frame


def run():
    frames = []
    audits = {}

    for year in SEASONS:
        frame = prepare(download(year), year)
        frames.append(frame)

        audits[str(year)] = {
            "plays": int(len(frame)),
            "games": int(
                frame["nflverse_game_id"].nunique()
            ),
            "coverage_labeled": int(
                frame["coverage_known"].sum()
            ),
            "route_labeled": int(
                frame["route_known"].sum()
            ),
        }

    df = pd.concat(
        frames, ignore_index=True
    )

    counts = [
        "coverage_known",
        "route_known",
        "pressure_known",
        "pressure",
    ]

    counts += [
        "coverage_" + safe_name(c)
        for c in COVERAGES
    ]

    counts += [
        "route_" + safe_name(r)
        for r in ROUTES
    ]

    keys = [
        "season",
        "week",
        "nflverse_game_id",
        "opponent",
    ]

    games = df.groupby(
        keys, as_index=False
    )[counts].sum()

    play_counts = (
        df.groupby(keys)
        .size()
        .rename("plays")
        .reset_index()
    )

    games = games.merge(
        play_counts,
        on=keys,
        validate="one_to_one",
    )

    games = games.rename(
        columns={"nflverse_game_id": "game_id"}
    )

    games = games.sort_values(
        ["season", "opponent", "week", "game_id"]
    ).reset_index(drop=True)

    group = games.groupby(
        ["season", "opponent"],
        sort=False,
    )

    games["prior_defense_games"] = (
        group.cumcount()
    )

    for column in ["plays"] + counts:
        games[
            "prior5_sum_" + column
        ] = group[column].transform(
            lambda s: (
                s.shift(1)
                .rolling(5, min_periods=2)
                .sum()
            )
        )

    coverage_den = (
        games["prior5_sum_coverage_known"]
        .replace(0, np.nan)
    )

    route_den = (
        games["prior5_sum_route_known"]
        .replace(0, np.nan)
    )

    for coverage in COVERAGES:
        column = (
            "coverage_" + safe_name(coverage)
        )

        games[
            "prior5_rate_" + column
        ] = (
            games["prior5_sum_" + column]
            / coverage_den
        )

    for route in ROUTES:
        column = "route_" + safe_name(route)

        games[
            "prior5_rate_" + column
        ] = (
            games["prior5_sum_" + column]
            / route_den
        )

    games["prior5_pressure_rate"] = (
        games["prior5_sum_pressure"]
        / games["prior5_sum_pressure_known"]
        .replace(0, np.nan)
    )

    # Missingness is informative.
    # These rates show label completeness,
    # not defensive effectiveness.

    games["prior5_coverage_label_rate"] = (
        coverage_den
        / games["prior5_sum_plays"]
        .replace(0, np.nan)
    )

    games["prior5_route_label_rate"] = (
        route_den
        / games["prior5_sum_plays"]
        .replace(0, np.nan)
    )

    pregame = [
        c for c in games.columns
        if c.startswith("prior5_")
    ]

    first = games["prior_defense_games"].eq(0)

    if games.loc[
        first, pregame
    ].notna().any().any():
        raise ValueError(
            "First-game leakage detected"
        )

    unique_keys = [
        "season", "week",
        "game_id", "opponent",
    ]

    if games.duplicated(unique_keys).any():
        raise ValueError(
            "Duplicate defensive game keys"
        )

    output = (
        OUT / "defense_coverage_pregame_2024_2025.csv"
    )

    games.to_csv(output, index=False)

    report = {
        "phase": "4.8B",
        "seasons": SEASONS,
        "audit": audits,
        "defensive_game_rows": int(len(games)),
        "pregame_feature_count": len(pregame),
        "limitations": [
            "2026 participation data not yet available.",
            "Missing coverage and route labels.",
            "Route labels are not full receiver route assignments.",
            "No individual defender coverage assignments.",
            "Coverage features are previous-game only.",
            "No predictive improvement established yet.",
            "Existing A3 models remain unchanged.",
        ],
    }

    report_path = (
        REPORTS / "a3_coverage_multiseason_audit.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2)
    )

    print("A3 PHASE 4.8B DATA BUILD COMPLETE")
    print(json.dumps(audits, indent=2))
    print("Defensive game rows:", len(games))
    print("Pregame features:", len(pregame))
    print("Output:", output)
    print("Report:", report_path)


if __name__ == "__main__":
    run()
