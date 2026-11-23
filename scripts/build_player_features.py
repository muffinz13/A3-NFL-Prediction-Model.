
from pathlib import Path
import pandas as pd

# A3 PHASE 2 — PLAYER PERFORMANCE FEATURES
# Uses existing NFL play-by-play Parquet files.

SEASONS = [2024, 2025, 2026]
DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "pass_attempts",
    "completions",
    "pass_yards",
    "pass_tds",
    "interceptions",
    "carries",
    "rush_yards",
    "rush_tds",
    "targets",
    "receptions",
    "rec_yards",
    "rec_tds",
]

ROLE_CONFIG = {
    "passer": {
        "id": "passer_player_id",
        "name": "passer_player_name",
    },
    "rusher": {
        "id": "rusher_player_id",
        "name": "rusher_player_name",
    },
    "receiver": {
        "id": "receiver_player_id",
        "name": "receiver_player_name",
    },
}

REQUIRED = [
    "season",
    "week",
    "game_id",
    "season_type",
    "posteam",
    "play_type",
    "pass_attempt",
    "complete_pass",
    "passing_yards",
    "pass_touchdown",
    "interception",
    "rush_attempt",
    "rushing_yards",
    "rush_touchdown",
    "receiving_yards",
    "touchdown",
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
]

def numeric(df, column):
    if column not in df.columns:
        raise ValueError(f"Required column missing: {column}")

    return pd.to_numeric(
        df[column], errors="coerce"
    )

def prepare(df):
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing PBP columns: {missing}")

    df = df[
        (df["season_type"] == "REG")
        & df["posteam"].notna()
        & df["play_type"].isin(["run", "pass"])
    ].copy()

    # Exclude plays nullified by penalties and two-point tries.
    if "no_play" in df.columns:
        df = df[numeric(df, "no_play").fillna(0) == 0]

    if "two_point_attempt" in df.columns:
        df = df[
            numeric(df, "two_point_attempt").fillna(0) == 0
        ]

    for column in [
        "pass_attempt", "complete_pass",
        "passing_yards", "pass_touchdown",
        "interception", "rush_attempt",
        "rushing_yards", "rush_touchdown",
        "receiving_yards", "touchdown",
    ]:
        df[column] = numeric(df, column)

    # A missing flag is treated as false, but missing
    # yardage on a qualifying play remains missing.
    flags = [
        "pass_attempt", "complete_pass",
        "pass_touchdown", "interception",
        "rush_attempt", "rush_touchdown",
        "touchdown",
    ]
    df[flags] = df[flags].fillna(0)

    return df

def build_role(df, role):
    cfg = ROLE_CONFIG[role]
    player_id = cfg["id"]
    player_name = cfg["name"]

    if player_name not in df.columns:
        raise ValueError(
            f"Missing player-name column: {player_name}"
        )

    part = df[df[player_id].notna()].copy()

    part["player_id"] = part[player_id].astype(str)
    part["player_name"] = part[player_name]
    part["team"] = part["posteam"]
    part["role"] = role

    for metric in METRICS:
        part[metric] = 0.0

    if role == "passer":
        part["pass_attempts"] = part["pass_attempt"]
        part["completions"] = part["complete_pass"]
        part["pass_yards"] = part["passing_yards"]
        part["pass_tds"] = part["pass_touchdown"]
        part["interceptions"] = part["interception"]

    elif role == "rusher":
        part["carries"] = part["rush_attempt"]
        part["rush_yards"] = part["rushing_yards"]
        part["rush_tds"] = part["rush_touchdown"]

    elif role == "receiver":
        # A receiver target requires a pass attempt
        # with a known intended receiver.
        part["targets"] = part["pass_attempt"]
        part["receptions"] = part["complete_pass"]
        part["rec_yards"] = part["receiving_yards"]

        # Receiving touchdowns on completed passes.
        part["rec_tds"] = (
            (part["complete_pass"] == 1)
            & (part["touchdown"] == 1)
        ).astype(float)

    keys = [
        "season", "week", "game_id",
        "player_id", "team", "role"
    ]

    result = (
        part.groupby(keys, as_index=False)
        .agg(
            player_name=("player_name", "first"),
            **{m: (m, "sum") for m in METRICS}
        )
    )

    return result

def build():
    all_roles = []

    for season in SEASONS:
        path = DATA_DIR / f"play_by_play_{season}.parquet"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the A3 download workflow."
            )

        print(f"Processing season {season}")
        df = prepare(pd.read_parquet(path))

        for role in ROLE_CONFIG:
            all_roles.append(build_role(df, role))

    role_games = pd.concat(
        all_roles, ignore_index=True
    )

    # Combine roles for players who both rush and receive,
    # or quarterbacks who both pass and rush.
    keys = [
        "season", "week", "game_id",
        "player_id", "team"
    ]

    games = (
        role_games.groupby(keys, as_index=False)
        .agg(
            player_name=("player_name", "first"),
            roles=("role", lambda x: ",".join(sorted(set(x)))),
            **{m: (m, "sum") for m in METRICS}
        )
    )

    games = games.sort_values(
        ["season", "week", "game_id", "player_id"]
    ).reset_index(drop=True)

    games.to_csv(
        OUT_DIR / "player_game_stats.csv",
        index=False
    )

    # Calculate strictly pregame historical features.
    # First game of a season has no season history.
    # Last-five uses the previous five appearances.
    features = games.copy()

    features = features.sort_values(
        ["player_id", "season", "week", "game_id"]
    ).reset_index(drop=True)

    group = features.groupby(
        ["player_id", "season"], sort=False
    )

    features["prior_games"] = group.cumcount()

    for metric in METRICS:
        features[f"season_prior_{metric}"] = (
            group[metric].transform(
                lambda s: s.shift(1).expanding(
                    min_periods=1
                ).mean()
            )
        )

        features[f"last5_{metric}"] = (
            group[metric].transform(
                lambda s: s.shift(1).rolling(
                    window=5,
                    min_periods=2
                ).mean()
            )
        )

        features[f"last5_vs_season_{metric}"] = (
            features[f"last5_{metric}"]
            - features[f"season_prior_{metric}"]
        )

    features.to_csv(
        OUT_DIR / "player_pregame_features.csv",
        index=False
    )

    # Sanity checks.
    if games.empty:
        raise ValueError("No player games generated")

    if games.duplicated(keys).any():
        raise ValueError("Duplicate player-game rows")

    first_games = features["prior_games"] == 0
    for metric in METRICS:
        if features.loc[
            first_games, f"season_prior_{metric}"
        ].notna().any():
            raise ValueError(
                f"Future-data leakage detected: {metric}"
            )

    print("A3 player features generated successfully")
    print(f"Player-game rows: {len(games):,}")
    print(f"Pregame feature rows: {len(features):,}")
    print("Outputs:")
    print("  data/processed/player_game_stats.csv")
    print("  data/processed/player_pregame_features.csv")

if __name__ == "__main__":
    build()
