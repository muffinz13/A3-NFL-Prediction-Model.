
"""
A3 Phase 4.9
2026-Only Matchup and Trap Intelligence

Uses:
- 2026 player statistics
- 2026 defensive allowances by position
- Recent 3-game and 5-game trends
- Historical game-script tendencies within 2026

Does NOT use 2024 or 2025 statistics.

Experimental challenger only.
Does not replace the incumbent A3 model.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE = Path("data/processed")
OUT = Path("data/model_results")

OUT.mkdir(parents=True, exist_ok=True)

POOL = BASE / "pregame_player_pool.csv"
PBP = Path("data/play_by_play_2026.parquet")

OUTPUT = OUT / "a3_2026_matchup_predictions.csv"
REPORT = OUT / "a3_2026_matchup_audit.json"

METRICS = [
    "pass_attempts",
    "pass_yards",
    "carries",
    "rush_yards",
    "targets",
    "receptions",
    "rec_yards",
]

TASKS = {
    "QB": [
        "pass_attempts",
        "pass_yards",
        "rush_yards",
    ],
    "RB": [
        "carries",
        "rush_yards",
        "targets",
        "receptions",
        "rec_yards",
    ],
    "WR": [
        "targets",
        "receptions",
        "rec_yards",
    ],
    "TE": [
        "targets",
        "receptions",
        "rec_yards",
    ],
}

KEYS = [
    "season",
    "week",
    "game_id",
    "team",
    "player_id",
]


def rolling_prior(
    frame,
    by,
    metrics,
    windows=(3, 5),
):
    frame = frame.sort_values(
        by + ["week", "game_id"]
    ).copy()

    grp = frame.groupby(
        by,
        sort=False,
    )

    for metric in metrics:

        frame[
            "prior_season_" + metric
        ] = grp[metric].transform(
            lambda s: (
                s.shift()
                .expanding(min_periods=1)
                .mean()
            )
        )

        for n in windows:

            frame[
                f"prior{n}_" + metric
            ] = grp[metric].transform(
                lambda s: (
                    s.shift()
                    .rolling(
                        n,
                        min_periods=1,
                    )
                    .mean()
                )
            )

    frame["prior_games"] = grp.cumcount()

    return frame


def load_2026_pbp():

    if not PBP.exists():
        raise FileNotFoundError(PBP)

    available_columns = set(
        pq.read_schema(PBP).names
    )

    requested_columns = [
        "season",
        "week",
        "game_id",
        "season_type",
        "posteam",
        "play_type",
        "no_play",
        "two_point_attempt",
        "score_differential",
    ]

    required_columns = [
        "season",
        "week",
        "game_id",
        "season_type",
        "posteam",
        "play_type",
        "score_differential",
    ]

    missing = (
        set(required_columns)
        - available_columns
    )

    if missing:
        raise ValueError(
            "Missing required PBP columns: "
            + str(sorted(missing))
        )

    columns = [
        column
        for column in requested_columns
        if column in available_columns
    ]

    pbp = pd.read_parquet(
        PBP,
        columns=columns,
    )

    # NFLverse releases may omit no_play.
    # Missing is treated as zero for
    # compatibility, not as proof that
    # every play was officially valid.
    if "no_play" not in pbp.columns:
        pbp["no_play"] = 0

    if "two_point_attempt" not in pbp.columns:
        pbp["two_point_attempt"] = 0

    return pbp


def run():

    pool = pd.read_csv(
        POOL,
        dtype={"player_id": str},
        low_memory=False,
    )

    required = (
        KEYS
        + [
            "opponent",
            "roster_position",
            "player_name",
        ]
        + [
            "actual_" + m
            for m in METRICS
        ]
    )

    missing = set(required) - set(
        pool.columns
    )

    if missing:
        raise ValueError(
            "Missing pool columns: "
            + str(sorted(missing))
        )

    pool = pool.loc[
        pool.season.eq(2026)
        & pool.roster_position.isin(TASKS)
    ].copy()

    if pool.empty:
        raise ValueError(
            "Empty 2026 candidate pool"
        )

    if pool.duplicated(KEYS).any():
        raise ValueError(
            "Duplicate player-game rows"
        )

    for m in METRICS:
        pool[
            "actual_" + m
        ] = pd.to_numeric(
            pool["actual_" + m],
            errors="coerce",
        )

    if pool[
        ["actual_" + m for m in METRICS]
    ].isna().any().any():
        raise ValueError(
            "Missing actual player statistics"
        )

    # ---------------------------------
    # 2026 defensive positional totals
    # ---------------------------------

    bypos = (
        pool.groupby(
            [
                "season",
                "week",
                "game_id",
                "opponent",
                "roster_position",
            ],
            as_index=False,
        )[
            [
                "actual_" + m
                for m in METRICS
            ]
        ].sum()
    )

    bypos = bypos.rename(
        columns={
            "opponent": "defense",
            "roster_position": "position",
            **{
                "actual_" + m:
                "allowed_" + m
                for m in METRICS
            },
        }
    )

    defense_games = (
        pool[
            [
                "season",
                "week",
                "game_id",
                "opponent",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "opponent": "defense"
            }
        )
    )

    positions = pd.DataFrame(
        {
            "position": list(TASKS)
        }
    )

    defense_games["_k"] = 1
    positions["_k"] = 1

    full = defense_games.merge(
        positions,
        on="_k",
    ).drop(columns="_k")

    bypos = full.merge(
        bypos,
        on=[
            "season",
            "week",
            "game_id",
            "defense",
            "position",
        ],
        how="left",
        validate="one_to_one",
    )

    allow_cols = [
        "allowed_" + m
        for m in METRICS
    ]

    bypos[allow_cols] = (
        bypos[allow_cols].fillna(0)
    )

    bypos = rolling_prior(
        bypos,
        [
            "season",
            "defense",
            "position",
        ],
        allow_cols,
    )

    bypos = bypos.rename(
        columns={
            "defense": "opponent",
            "position": "roster_position",
        }
    )

    predef = [
        c
        for c in bypos
        if c.startswith("prior")
    ]

    bypos = bypos.rename(
        columns={
            c: "def_" + c
            for c in predef
        }
    )

    df = pool.merge(
        bypos[
            [
                "season",
                "week",
                "game_id",
                "opponent",
                "roster_position",
            ]
            + [
                "def_" + c
                for c in predef
            ]
        ],
        on=[
            "season",
            "week",
            "game_id",
            "opponent",
            "roster_position",
        ],
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------
    # 2026 player performance trends
    # ---------------------------------

    df = rolling_prior(
        df,
        [
            "season",
            "player_id",
        ],
        [
            "actual_" + m
            for m in METRICS
        ],
    )

    # ---------------------------------
    # 2026 game-script intelligence
    # ---------------------------------

    pbp = load_2026_pbp()

    pbp = pbp.loc[
        pbp.season.eq(2026)
        & pbp.season_type.eq("REG")
        & pbp.posteam.notna()
        & pbp.play_type.isin(
            ["pass", "run"]
        )
    ].copy()

    valid_play = (
        pd.to_numeric(
            pbp["no_play"],
            errors="coerce",
        )
        .fillna(0)
        .eq(0)
    )

    valid_two_point = (
        pd.to_numeric(
            pbp["two_point_attempt"],
            errors="coerce",
        )
        .fillna(0)
        .eq(0)
    )

    pbp = pbp.loc[
        valid_play & valid_two_point
    ].copy()

    pbp["is_pass"] = (
        pbp.play_type.eq("pass")
        .astype(float)
    )

    score_diff = pd.to_numeric(
        pbp.score_differential,
        errors="coerce",
    )

    pbp["leading_14"] = (
        score_diff.ge(14)
        .astype(float)
    )

    pbp["trailing_14"] = (
        score_diff.le(-14)
        .astype(float)
    )

    pbp["pass_leading_14"] = (
        pbp.is_pass
        * pbp.leading_14
    )

    off = (
        pbp.groupby(
            [
                "season",
                "week",
                "game_id",
                "posteam",
            ],
            as_index=False,
        )
        .agg(
            plays=("is_pass", "size"),
            passes=("is_pass", "sum"),
            lead_plays=(
                "leading_14",
                "sum",
            ),
            lead_passes=(
                "pass_leading_14",
                "sum",
            ),
            trail_plays=(
                "trailing_14",
                "sum",
            ),
        )
    )

    off = off.rename(
        columns={
            "posteam": "team"
        }
    )

    off = rolling_prior(
        off,
        [
            "season",
            "team",
        ],
        [
            "plays",
            "passes",
            "lead_plays",
            "lead_passes",
            "trail_plays",
        ],
    )

    offcols = [
        c
        for c in off
        if c.startswith("prior")
    ]

    df = df.merge(
        off[
            [
                "season",
                "week",
                "game_id",
                "team",
            ]
            + offcols
        ],
        on=[
            "season",
            "week",
            "game_id",
            "team",
        ],
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------
    # A3 player projections
    # ---------------------------------

    rows = []

    for position, targets in TASKS.items():

        part = df.loc[
            df.roster_position.eq(position)
        ].copy()

        for metric in targets:

            actual = "actual_" + metric

            usual = part[
                "prior_season_" + actual
            ]

            recent = part[
                "prior5_" + actual
            ]

            def5 = part[
                "def_prior5_allowed_"
                + metric
            ]

            defseason = part[
                "def_prior_season_allowed_"
                + metric
            ]

            # Bounded defensive trend
            # adjustment, not a verified
            # player-specific matchup model.

            factor = np.where(
                (defseason > 0)
                & def5.notna(),
                np.clip(
                    def5 / defseason,
                    0.8,
                    1.2,
                ),
                1.0,
            )

            factor = (
                1
                + 0.6 * (factor - 1)
            )

            central = np.maximum(
                (
                    0.55 * usual
                    + 0.45 * recent
                )
                * factor,
                0,
            )

            # Uncalibrated threshold.
            comfort = np.floor(
                np.maximum(
                    central * 0.7,
                    0,
                )
            )

            missing_history = (
                part.prior_games.lt(3)
                | part.def_prior_games.lt(3)
            )

            central = np.where(
                missing_history,
                np.nan,
                central,
            )

            comfort = np.where(
                missing_history,
                np.nan,
                comfort,
            )

            passrate = (
                part.prior5_passes
                / part.prior5_plays.replace(
                    0,
                    np.nan,
                )
            )

            leadshare = (
                part.prior5_lead_plays
                / part.prior5_plays.replace(
                    0,
                    np.nan,
                )
            )

            trap = np.where(
                (
                    position
                    in ("QB", "WR", "TE")
                )
                & (leadshare > 0.20),
                "POTENTIAL_LEAD_SCRIPT",
                "NONE",
            )

            out = part[
                KEYS
                + [
                    "opponent",
                    "player_name",
                    "roster_position",
                ]
            ].copy()

            out["stat"] = metric

            out["actual"] = (
                part[actual].to_numpy()
            )

            out[
                "season_average_pregame"
            ] = usual.to_numpy()

            out[
                "last5_average_pregame"
            ] = recent.to_numpy()

            out[
                "opponent_allowed_position_last5"
            ] = def5.to_numpy()

            out[
                "opponent_allowed_position_season"
            ] = defseason.to_numpy()

            out[
                "projection_heuristic"
            ] = central

            out[
                "comfort_threshold_uncalibrated"
            ] = comfort

            out[
                "matchup_adjustment"
            ] = factor

            out[
                "team_prior5_pass_rate"
            ] = passrate.to_numpy()

            out[
                "team_prior5_big_lead_play_share"
            ] = leadshare.to_numpy()

            out["trap_flag"] = trap

            out["status"] = np.where(
                missing_history,
                "INSUFFICIENT_2026_HISTORY",
                "RESEARCH_ONLY",
            )

            rows.append(out)

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    # ---------------------------------
    # Retrospective validation
    # ---------------------------------

    usable = result.loc[
        result.status.eq(
            "RESEARCH_ONLY"
        )
    ].copy()

    metrics = {}

    for (
        pos,
        stat,
    ), grp in usable.groupby(
        [
            "roster_position",
            "stat",
        ]
    ):

        if len(grp) < 10:
            continue

        y = grp.actual.to_numpy(float)

        pred = (
            grp.projection_heuristic
            .to_numpy(float)
        )

        base = (
            grp.season_average_pregame
            .to_numpy(float)
        )

        metrics[
            pos + "_" + stat
        ] = {
            "rows": int(len(grp)),
            "season_baseline_mae": float(
                np.mean(abs(base - y))
            ),
            "challenger_mae": float(
                np.mean(abs(pred - y))
            ),
            "challenger_bias": float(
                np.mean(pred - y)
            ),
            "overprediction_rate": float(
                np.mean(pred > y)
            ),
        }

    audit = {
        "phase": "4.9",
        "year": 2026,
        "data_rows": len(result),
        "usable_rows": len(usable),
        "metrics": metrics,
        "promoted": False,
        "limitations": [
            "Retrospective research only",
            "No live injury verification",
            "2026-only data",
            "Candidate pool may omit players",
            "No verified slot/outside assignments",
            "No individual defender matchups",
            "Blowout flag is not a probability",
            "Comfort thresholds are uncalibrated",
            "Incumbent A3 remains unchanged",
            "Player active status is unverified",
            "Missing no_play defaults to zero",
        ],
    }

    REPORT.write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print(
        "A3 2026 MATCHUP RESEARCH COMPLETE"
    )

    print(
        json.dumps(
            {
                "rows": len(result),
                "usable": len(usable),
                "metrics": metrics,
            },
            indent=2,
        )
    )

    print(
        "Incumbent A3 unchanged."
    )


if __name__ == "__main__":
    run()
