
from pathlib import Path
import pandas as pd

DATA = Path("data")
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

SEASONS = [2024, 2025, 2026]

def require(df, columns, name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing: {missing}")

def build_schedule():
    frames = []

    for year in SEASONS:
        path = DATA / f"play_by_play_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(
            path,
            columns=[
                "season", "week", "game_id",
                "season_type", "posteam", "defteam"
            ]
        )

        df = df[
            (df["season_type"] == "REG")
            & df["posteam"].notna()
            & df["defteam"].notna()
            & (df["posteam"] != df["defteam"])
        ]

        games = df[
            ["season", "week", "game_id",
             "posteam", "defteam"]
        ].drop_duplicates()

        games = games.rename(columns={
            "posteam": "team",
            "defteam": "opponent"
        })

        frames.append(games)

    schedule = pd.concat(frames, ignore_index=True)

    keys = ["season", "week", "game_id", "team"]

    if schedule.duplicated(keys).any():
        raise ValueError(
            "Ambiguous opponent mapping in schedule"
        )

    return schedule

def build():
    players = pd.read_csv(
        OUT / "player_pregame_features.csv",
        dtype={"player_id": str}
    )

    defense = pd.read_csv(
        OUT / "defense_pregame_features.csv"
    )

    schedule = build_schedule()

    player_keys = [
        "season", "week", "game_id",
        "team", "player_id"
    ]

    require(
        players,
        player_keys + ["prior_games"],
        "Player features"
    )

    require(
        defense,
        [
            "season", "week", "game_id",
            "defteam", "play_type",
            "prior5_epa_allowed",
            "prior5_success_allowed",
            "prior5_yards_per_play_allowed",
            "prior5_explosive_rate_allowed"
        ],
        "Defensive features"
    )

    if players.duplicated(player_keys).any():
        raise ValueError("Duplicate player-game rows")

    matchup = players.merge(
        schedule,
        on=["season", "week", "game_id", "team"],
        how="left",
        validate="many_to_one"
    )

    if matchup["opponent"].isna().any():
        raise ValueError(
            "Some player games lack an opponent"
        )

    metrics = [
        "prior5_epa_allowed",
        "prior5_success_allowed",
        "prior5_yards_per_play_allowed",
        "prior5_explosive_rate_allowed"
    ]

    # One defensive record per game and play type.
    defense_keys = [
        "season", "week", "game_id",
        "defteam", "play_type"
    ]

    if defense.duplicated(defense_keys).any():
        raise ValueError("Duplicate defensive records")

    # Attach passing and rushing defensive features.
    for play_type in ["pass", "run"]:
        subset = defense[
            defense["play_type"] == play_type
        ].copy()

        subset = subset.drop(columns=["play_type"])

        subset = subset.rename(columns={
            "defteam": "opponent",
            **{
                metric: f"opp_{play_type}_{metric}"
                for metric in metrics
            }
        })

        matchup = matchup.merge(
            subset[
                ["season", "week", "game_id",
                 "opponent"]
                + [
                    f"opp_{play_type}_{m}"
                    for m in metrics
                ]
            ],
            on=[
                "season", "week", "game_id",
                "opponent"
            ],
            how="left",
            validate="many_to_one"
        )

    # Missing prior-five statistics are expected
    # early in the season. Never replace with zero.
    for play_type in ["pass", "run"]:
        column = (
            f"opp_{play_type}_"
            "prior5_yards_per_play_allowed"
        )

        matchup[
            f"opp_{play_type}_history_available"
        ] = matchup[column].notna()

    # Matchup indicators are descriptive,
    # not yet calibrated player predictions.
    matchup["pass_defense_epa_signal"] = (
        matchup["opp_pass_prior5_epa_allowed"]
    )

    matchup["run_defense_epa_signal"] = (
        matchup["opp_run_prior5_epa_allowed"]
    )

    matchup = matchup.sort_values(
        ["season", "week", "game_id", "player_id"]
    )

    output = OUT / "player_matchup_features.csv"
    matchup.to_csv(output, index=False)

    print("A3 matchup features generated")
    print(f"Rows: {len(matchup):,}")
    print(
        "Pass-defense history available:",
        matchup[
            "opp_pass_history_available"
        ].mean()
    )
    print(
        "Run-defense history available:",
        matchup[
            "opp_run_history_available"
        ].mean()
    )
    print(f"Output: {output}")

if __name__ == "__main__":
    build()
