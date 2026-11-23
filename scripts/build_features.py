from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [2024, 2025, 2026]
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"

def build_features():
    all_games = []

    for season in SEASONS:
        print(f"Processing NFL season {season}")
        df = pd.read_parquet(URL.format(season))

        required = [
            "season", "week", "game_id", "season_type",
            "posteam", "defteam", "play_type", "epa",
            "success", "yards_gained"
        ]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        df = df[
            (df["season_type"] == "REG")
            & df["posteam"].notna()
            & df["defteam"].notna()
            & df["play_type"].isin(["run", "pass"])
        ].copy()

        df["epa"] = pd.to_numeric(df["epa"], errors="coerce")
        df["success"] = pd.to_numeric(df["success"], errors="coerce")
        df["yards_gained"] = pd.to_numeric(
            df["yards_gained"], errors="coerce"
        )

        df["explosive"] = (
            ((df["play_type"] == "pass") & (df["yards_gained"] >= 20))
            | ((df["play_type"] == "run") & (df["yards_gained"] >= 10))
        ).astype(int)

        games = (
            df.groupby(
                ["season", "week", "game_id", "defteam", "play_type"]
            )
            .agg(
                plays=("epa", "size"),
                epa_allowed=("epa", "mean"),
                success_allowed=("success", "mean"),
                yards_per_play_allowed=("yards_gained", "mean"),
                explosive_rate_allowed=("explosive", "mean"),
            )
            .reset_index()
        )

        all_games.append(games)

    result = pd.concat(all_games, ignore_index=True)

    result.to_csv(
        OUTPUT_DIR / "defense_game_features.csv",
        index=False
    )

    result = result.sort_values(
        ["defteam", "season", "play_type", "week", "game_id"]
    )

    metrics = [
        "epa_allowed",
        "success_allowed",
        "yards_per_play_allowed",
        "explosive_rate_allowed",
    ]

    for metric in metrics:
        result[f"prior5_{metric}"] = (
            result.groupby(["defteam", "season", "play_type"])[metric]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=5, min_periods=2
                ).mean()
            )
        )

    result.to_csv(
        OUTPUT_DIR / "defense_pregame_features.csv",
        index=False
    )

    print("A3 defensive features successfully generated.")
    print(f"Processed {len(result)} defense/game/play-type rows.")

if __name__ == "__main__":
    build_features()
