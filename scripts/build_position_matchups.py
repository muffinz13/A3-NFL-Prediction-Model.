
from pathlib import Path
import pandas as pd

# A3 PHASE 4.2
# Position-specific defensive matchup features.
# Every defensive feature uses previous games only.

DATA = Path("data")
PROCESSED = DATA / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

ROSTER_FILE = DATA / "a3_roster_positions_2024_2026.csv"
MATCHUP_FILE = PROCESSED / "player_matchup_features.csv"
PLAYER_FILE = PROCESSED / "player_game_stats.csv"

POSITIONS = ["QB", "RB", "WR", "TE"]

METRICS = [
    "carries",
    "rush_yards",
    "targets",
    "receptions",
    "rec_yards",
]

KEYS = ["season", "week", "game_id"]


def load_inputs():
    roster = pd.read_csv(
        ROSTER_FILE,
        dtype={"player_id": str},
    )

    matchup = pd.read_csv(
        MATCHUP_FILE,
        dtype={"player_id": str},
        low_memory=False,
    )

    players = pd.read_csv(
        PLAYER_FILE,
        dtype={"player_id": str},
        low_memory=False,
    )

    roster_keys = ["season", "player_id"]

    if roster.duplicated(roster_keys).any():
        raise ValueError("Duplicate player-season roster IDs")

    roster["roster_position"] = (
        roster["roster_position"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    roster = roster[
        roster["roster_position"].isin(POSITIONS)
    ].copy()

    for metric in METRICS:
        players[metric] = pd.to_numeric(
            players[metric], errors="coerce"
        ).fillna(0)

    return roster, matchup, players


def build():
    roster, matchup, players = load_inputs()

    # Match players to their season-level position.
    roster_info = roster[
        ["season", "player_id", "roster_position"]
    ]

    players = players.merge(
        roster_info,
        on=["season", "player_id"],
        how="left",
        validate="many_to_one",
    )

    # The matchup file contains the opponent for
    # every offensive team/game.
    schedule = matchup[
        KEYS + ["team", "opponent"]
    ].drop_duplicates()

    if schedule.duplicated(
        KEYS + ["team"]
    ).any():
        raise ValueError("Ambiguous schedule mapping")

    players = players.merge(
        schedule,
        on=KEYS + ["team"],
        how="left",
        validate="many_to_one",
    )

    if players["opponent"].isna().any():
        raise ValueError("Missing opponent mapping")

    # Aggregate production allowed by a defense
    # to each offensive position.
    observed = players[
        players["roster_position"].isin(POSITIONS)
    ].groupby(
        KEYS + ["opponent", "roster_position"],
        as_index=False,
    )[METRICS].sum()

    # Construct a complete defense/game/position
    # grid so zero production is not mistaken
    # for missing defensive history.
    defense_games = schedule[
        KEYS + ["opponent"]
    ].drop_duplicates()

    positions = pd.DataFrame({
        "roster_position": POSITIONS
    })

    defense_games["_join"] = 1
    positions["_join"] = 1

    grid = defense_games.merge(
        positions,
        on="_join",
    ).drop(columns="_join")

    defense = grid.merge(
        observed,
        on=KEYS + ["opponent", "roster_position"],
        how="left",
        validate="one_to_one",
    )

    defense[METRICS] = defense[
        METRICS
    ].fillna(0)

    defense = defense.sort_values(
        [
            "opponent",
            "season",
            "roster_position",
            "week",
            "game_id",
        ]
    ).reset_index(drop=True)

    group = defense.groupby(
        ["opponent", "season", "roster_position"],
        sort=False,
    )

    defense["prior_defense_games"] = (
        group.cumcount()
    )

    for metric in METRICS:
        defense[f"opp_position_prior5_{metric}"] = (
            group[metric].transform(
                lambda s: s.shift(1).rolling(
                    5, min_periods=2
                ).mean()
            )
        )

    # Add player position to original matchup data.
    enriched = matchup.merge(
        roster_info,
        on=["season", "player_id"],
        how="left",
        validate="many_to_one",
    )

    # Attach only pregame opponent statistics.
    pregame_columns = [
        f"opp_position_prior5_{m}"
        for m in METRICS
    ]

    enriched = enriched.merge(
        defense[
            KEYS
            + ["opponent", "roster_position"]
            + ["prior_defense_games"]
            + pregame_columns
        ],
        on=KEYS + ["opponent", "roster_position"],
        how="left",
        validate="many_to_one",
    )

    # Historical production of the current game
    # must not be included in pregame features.
    first_games = (
        defense["prior_defense_games"] == 0
    )

    if defense.loc[
        first_games, pregame_columns
    ].notna().any().any():
        raise ValueError(
            "Future-data leakage in defensive history"
        )

    original_keys = (
        KEYS + ["team", "player_id"]
    )

    if enriched.duplicated(original_keys).any():
        raise ValueError(
            "Duplicate enriched player-game records"
        )

    defense.to_csv(
        PROCESSED / "defense_position_game_features.csv",
        index=False,
    )

    enriched.to_csv(
        PROCESSED / "player_position_matchup_features.csv",
        index=False,
    )

    print("A3 PHASE 4.2 FEATURES COMPLETE")
    print(f"Defensive position rows: {len(defense):,}")
    print(f"Player matchup rows: {len(enriched):,}")

    print(
        "Player position coverage:",
        round(
            enriched["roster_position"].notna().mean(),
            4,
        ),
    )

    print(
        "Receiving matchup history coverage:",
        round(
            enriched[
                "opp_position_prior5_rec_yards"
            ].notna().mean(),
            4,
        ),
    )

    print(
        "Outputs:",
        "defense_position_game_features.csv",
        "player_position_matchup_features.csv",
    )


if __name__ == "__main__":
    build()
