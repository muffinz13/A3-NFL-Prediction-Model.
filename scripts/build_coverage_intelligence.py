from pathlib import Path
import json
import numpy as np
import pandas as pd

# A3 Phase 4.8A: audited 2025 coverage and route data.
# Produces prior-game defensive tendencies; never uses current-game
# coverage as a pregame predictor. No defender assignment inference.

SOURCE = Path('data/pbp_participation_2025.csv')
OUT = Path('data/processed')
REPORTS = Path('data/model_results')
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

COVERAGES = ['COVER_0','COVER_1','COVER_2','COVER_3','COVER_4','COVER_6','2_MAN']
ROUTES = ['SLANT','SHALLOW CROSS/DRAG','IN/DIG','GO','POST','SCREEN']

def safe_name(value):
    return ''.join(c.lower() if c.isalnum() else '_' for c in value).strip('_')

def run():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Missing {SOURCE}. Add the uploaded CSV at this path.')
    columns = ['nflverse_game_id','play_id','possession_team','route',
               'defense_man_zone_type','defense_coverage_type',
               'offense_players','defense_players','was_pressure']
    df = pd.read_csv(SOURCE, usecols=columns, low_memory=False)
    if df.duplicated(['nflverse_game_id','play_id']).any():
        raise ValueError('Duplicate game/play identifiers in participation file')
    parts = df.nflverse_game_id.astype(str).str.split('_', expand=True)
    if parts.shape[1] != 4:
        raise ValueError('Unexpected nflverse_game_id format')
    df['season'] = pd.to_numeric(parts[0], errors='raise').astype(int)
    df['week'] = pd.to_numeric(parts[1], errors='raise').astype(int)
    away, home = parts[2], parts[3]
    df['team'] = df.possession_team
    df['opponent'] = np.where(df.team.eq(away), home,
                       np.where(df.team.eq(home), away, None))
    if df.opponent.isna().any():
        raise ValueError('Possession team inconsistent with game ID')
    df['coverage_known'] = df.defense_coverage_type.notna().astype(int)
    df['route_known'] = df.route.notna().astype(int)
    df['pressure_known'] = df.was_pressure.notna().astype(int)
    df['pressure'] = df.was_pressure.astype(str).str.lower().eq('true').astype(int)
    for coverage in COVERAGES:
        df['coverage_'+safe_name(coverage)] = df.defense_coverage_type.eq(coverage).astype(int)
    for route in ROUTES:
        df['route_'+safe_name(route)] = df.route.eq(route).astype(int)
    cols = ['coverage_known','route_known','pressure_known','pressure'] + [
        'coverage_'+safe_name(x) for x in COVERAGES
    ] + ['route_'+safe_name(x) for x in ROUTES]
    keys = ['season','week','nflverse_game_id','opponent']
    games = df.groupby(keys, as_index=False)[cols].sum()
    games['plays'] = df.groupby(keys).size().to_numpy()
    games = games.sort_values(['opponent','season','week','nflverse_game_id']).reset_index(drop=True)
    grp = games.groupby(['season','opponent'], sort=False)
    games['prior_defense_games'] = grp.cumcount()
    # Prior-five-game *sums* make denominators consistent despite missing labels.
    for col in ['plays'] + cols:
        games['prior5_sum_'+col] = grp[col].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).sum()
        )
    den = games['prior5_sum_coverage_known'].replace(0, np.nan)
    for coverage in COVERAGES:
        col = 'coverage_'+safe_name(coverage)
        games['prior5_rate_'+col] = games['prior5_sum_'+col] / den
    route_den = games['prior5_sum_route_known'].replace(0, np.nan)
    for route in ROUTES:
        col = 'route_'+safe_name(route)
        games['prior5_rate_'+col] = games['prior5_sum_'+col] / route_den
    games['prior5_pressure_rate'] = (
        games['prior5_sum_pressure'] /
        games['prior5_sum_pressure_known'].replace(0, np.nan)
    )
    prior_cols = [c for c in games if c.startswith('prior5_')]
    if games.loc[games.prior_defense_games.eq(0), prior_cols].notna().any().any():
        raise ValueError('Pregame leakage in first defensive games')
    output = OUT / 'defense_coverage_pregame_2025.csv'
    games.to_csv(output, index=False)
    report = {
        'phase':'4.8A', 'source':str(SOURCE),
        'rows':int(len(df)), 'games':int(df.nflverse_game_id.nunique()),
        'coverage_labeled':int(df.coverage_known.sum()),
        'route_labeled':int(df.route_known.sum()),
        'defensive_game_rows':int(len(games)),
        'coverage_values':{str(k):int(v) for k,v in df.defense_coverage_type.value_counts().items()},
        'limitations':[
            'Coverage/route labels missing on many plays; rates conditional on labeled plays.',
            'Route is a play-level field, not every receiver route.',
            'Defenders on field do not identify individual coverage assignments.',
            'Only 2025 data; 2024 training and 2026 testing require matching source years.',
            'No player-specific coverage model or prediction improvement established.',
            'No existing A3 model is changed.'
        ]
    }
    path = REPORTS / 'a3_coverage_audit_2025.json'
    path.write_text(json.dumps(report, indent=2))
    print('A3 PHASE 4.8A COMPLETE')
    print(json.dumps({k:report[k] for k in ['rows','games','coverage_labeled','route_labeled','defensive_game_rows']}, indent=2))
    print('Outputs:',output,path)

if __name__ == '__main__':
    run()
