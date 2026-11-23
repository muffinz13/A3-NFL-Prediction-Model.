
from pathlib import Path
import json
import numpy as np
import pandas as pd

# A3 PHASE 4.7A
# Pregame player eligibility and opportunity foundation.
#
# Experimental historical candidate generation.
# Does NOT claim to know official active/inactive status.
# Does NOT change the existing A3 prediction model.

DATA = Path("data")
PROCESSED = DATA / "processed"
RESULTS = DATA / "model_results"

PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]

METRICS = [
    "pass_attempts",
    "completions",
    "pass_yards",
    "carries",
    "rush_yards",
    "targets",
    "receptions",
    "rec_yards",
]

KEYS = [
    "season",
    "week",
    "game_id",
    "team",
    "player_id",
]


def load():
    players = pd.read_csv(
        PROCESSED / "player_game_stats.csv",
        dtype={"player_id": str},
        low_memory=False,
    )

    roster = pd.read_csv(
        DATA / "a3_roster_positions_2024_2026.csv",
        dtype={"player_id": str},
    )

    roster = roster[
        ["season", "player_id", "roster_position"]
    ].copy()

    roster["roster_position"] = (
        roster["roster_position"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if roster.duplicated(
        ["season", "player_id"]
    ).any():
        raise ValueError(
            "Duplicate roster player-season IDs"
        )

    players = players.merge(
        roster,
        on=["season", "player_id"],
        how="left",
        validate="many_to_one",
    )

    players = players[
        players["roster_position"].isin(POSITIONS)
    ].copy()

    for metric in METRICS:
        players[metric] = pd.to_numeric(
            players[metric],
            errors="coerce",
        ).fillna(0)

    return players


def build_schedule():
    frames = []

    for season in [2024, 2025, 2026]:
        path = DATA / f"play_by_play_{season}.parquet"

        pbp = pd.read_parquet(
            path,
            columns=[
                "season",
                "week",
                "game_id",
                "season_type",
                "posteam",
                "defteam",
            ],
        )

        pbp = pbp[
            (pbp["season_type"] == "REG")
            & pbp["posteam"].notna()
            & pbp["defteam"].notna()
        ].copy()

        schedule = pbp[
            [
                "season",
                "week",
                "game_id",
                "posteam",
                "defteam",
            ]
        ].drop_duplicates()

        schedule = schedule.rename(
            columns={
                "posteam": "team",
                "defteam": "opponent",
            }
        )

        frames.append(schedule)

    schedule = pd.concat(
        frames,
        ignore_index=True,
    ).drop_duplicates()

    if schedule.duplicated(
        ["season", "week", "game_id", "team"]
    ).any():
        raise ValueError(
            "Ambiguous team-game schedule"
        )

    return schedule


def build():
    players = load()
    schedule = build_schedule()

    # Each player must have appeared previously
    # in the same season to become a candidate.
    #
    # Only previous weeks are used.
    # The upcoming game's actual player statistics
    # are never used to choose candidates.

    candidates = []

    for (season, week), games in schedule.groupby(
        ["season", "week"],
        sort=True,
    ):
        history = players[
            (players["season"] == season)
            & (players["week"] < week)
        ].copy()

        if history.empty:
            continue

        history = history.sort_values(
            ["player_id", "week", "game_id"]
        )

        # Most recent team is the known team
        # from the player's prior appearances.
        latest = history.groupby(
            "player_id",
            as_index=False,
        ).tail(1)

        latest = latest[
            [
                "player_id",
                "player_name",
                "team",
                "roster_position",
                "week",
            ]
        ].rename(
            columns={
                "week": "last_appearance_week"
            }
        )

        # Previous appearances only.
        aggregates = history.groupby(
            "player_id",
            as_index=False,
        ).agg(
            prior_games=("game_id", "nunique"),
            **{
                f"prior_mean_{m}": (m, "mean")
                for m in METRICS
            },
        )

        latest = latest.merge(
            aggregates,
            on="player_id",
            how="left",
            validate="one_to_one",
        )

        pool = games.merge(
            latest,
            on="team",
            how="inner",
        )

        pool["weeks_since_appearance"] = (
            week - pool["last_appearance_week"]
        )

        # Avoid silently treating a long-absent
        # player as a confirmed active player.
        pool["recently_seen"] = (
            pool["weeks_since_appearance"] <= 3
        )

        candidates.append(pool)

    if not candidates:
        raise RuntimeError(
            "No pregame candidates generated"
        )

    pool = pd.concat(
        candidates,
        ignore_index=True,
    )

    if pool.duplicated(KEYS).any():
        raise ValueError(
            "Duplicate pregame candidate rows"
        )

    # Attach current-game outcomes ONLY AFTER
    # the pregame candidate list is frozen.
    # These columns are labels, not predictors.

    outcomes = players[
        KEYS + METRICS
    ].copy()

    outcomes["observed_participation"] = 1

    pool = pool.merge(
        outcomes,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    pool["observed_participation"] = (
        pool["observed_participation"]
        .fillna(0)
        .astype(int)
    )

    for metric in METRICS:
        pool[f"actual_{metric}"] = (
            pool[metric].fillna(0)
        )

    pool = pool.drop(
        columns=METRICS
    )

    # A zero label means no recorded involvement
    # in the source player-game statistics.
    # It does NOT establish whether the player
    # was inactive, injured, or dressed.

    pool["observed_any_opportunity"] = (
        (
            pool[
                [
                    "actual_pass_attempts",
                    "actual_carries",
                    "actual_targets",
                ]
            ].sum(axis=1)
        ) > 0
    ).astype(int)

    pool = pool.sort_values(
        KEYS
    ).reset_index(drop=True)

    output = (
        PROCESSED
        / "pregame_player_pool.csv"
    )

    pool.to_csv(
        output,
        index=False,
    )

    # Measure the share of observed participants
    # captured by the historical candidate pool.

    actual = players[
        KEYS
    ].drop_duplicates()

    coverage = actual.merge(
        pool[KEYS].assign(in_pool=1),
        on=KEYS,
        how="left",
    )

    coverage["in_pool"] = (
        coverage["in_pool"]
        .fillna(0)
        .astype(int)
    )

    seasonal = {}

    for season, part in coverage.groupby(
        "season"
    ):
        seasonal[str(season)] = {
            "actual_player_games": int(len(part)),
            "captured": int(part["in_pool"].sum()),
            "coverage": float(part["in_pool"].mean()),
        }

    report = {
        "phase": "4.7A",
        "candidate_rows": int(len(pool)),
        "observed_participation_rate": float(
            pool["observed_participation"].mean()
        ),
        "observed_opportunity_rate": float(
            pool["observed_any_opportunity"].mean()
        ),
        "seasonal_coverage": seasonal,
        "limitations": [
            "First appearances of a season cannot be predicted.",
            "Newly signed players may be missed.",
            "Team assignment uses last observed team.",
            "Historical roster position may not match game-day role.",
            "No official game-day inactive list is integrated.",
            "No injury report is integrated.",
            "No snap-count participation is integrated.",
            "Candidates include players who may be inactive.",
            "Missing recorded participation is labeled zero.",
            "This is a candidate-pool audit, not a validated forecast.",
        ],
    }

    report_path = (
        RESULTS
        / "a3_pregame_pool_report.json"
    )

    with open(
        report_path,
        "w",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\n================================")
    print("A3 PHASE 4.7A COMPLETE")
    print("================================")

    print(
        f"Pregame candidate rows: {len(pool):,}"
    )

    for season, info in seasonal.items():
        print(
            f"{season} observed-player coverage: "
            f"{info['coverage']:.2%}"
        )

    print(f"Player pool: {output}")
    print(f"Report: {report_path}")

    print(
        "\nExisting A3 models remain unchanged."
    )


if __name__ == "__main__":
    build()
