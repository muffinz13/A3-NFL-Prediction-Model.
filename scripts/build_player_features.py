from pathlib import Path
import pandas as pd

SEASONS = [2024, 2025, 2026]
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://github.com/nflverse/nflverse-data/"
    "releases/download/player_stats/"
    "player_stats_{year}.csv"
)

METRICS = {
    "passing_yards": "pass_yards",
    "passing_tds": "pass_tds",
    "interceptions": "interceptions",
    "rushing_yards": "rush_yards",
    "carries": "carries",
    "rushing_tds": "rush_tds",
    "receptions": "receptions",
    "targets": "targets",
    "receiving_yards": "rec_yards",
    "receiving_tds": "rec_tds",
}

def load_data():
    frames = []

    for year in SEASONS:
        url = BASE.format(year=year)
        print(f"Downloading player stats: {year}")
        df = pd.read_csv(url)
        df["season"] = year
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def build():
    df = load_data()

    required = [
        "player_id", "player_name", "position",
        "team", "season", "week"
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[df["position"].isin(
        ["QB", "RB", "WR", "TE"]
    )].copy()

    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()

    df["week"] = pd.to_numeric(
        df["week"], errors="coerce"
    )
    df = df.dropna(subset=["week"])
    df["week"] = df["week"].astype(int)

    available = {}

    for source, target in METRICS.items():
        if source in df.columns:
            df[target] = pd.to_numeric(
                df[source], errors="coerce"
            ).fillna(0)
            available[target] = source
        else:
            df[target] = 0
            print(f"WARNING: {source} unavailable")

    df = df.sort_values(
        ["player_id", "season", "week"]
    ).reset_index(drop=True)

    grouped = df.groupby(
        ["player_id", "season"], sort=False
    )

    for metric in METRICS.values():
        df[f"season_prior_{metric}"] = (
            grouped[metric].transform(
                lambda s: s.shift(1).expanding(
                    min_periods=1
                ).mean()
            )
        )

        df[f"last5_{metric}"] = (
            grouped[metric].transform(
                lambda s: s.shift(1).rolling(
                    5, min_periods=2
                ).mean()
            )
        )

    for metric in [
        "pass_yards", "rush_yards",
        "rec_yards", "receptions", "targets"
    ]:
        df[f"last5_vs_season_{metric}"] = (
            df[f"last5_{metric}"]
            - df[f"season_prior_{metric}"]
        )

    df.to_csv(
        OUT / "player_pregame_features.csv",
        index=False
    )

    print("Player features generated successfully.")
    print(f"Rows: {len(df)}")
    print(f"Metrics available: {available}")

if __name__ == "__main__":
    build()
