from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path('data/processed')
OUT = Path('data/model_results')
OUT.mkdir(parents=True, exist_ok=True)
POOL = BASE / 'pregame_player_pool.csv'
COVERAGE = BASE / 'defense_coverage_pregame_2024_2025.csv'
REPORT = OUT / 'a3_coverage_48c_validation.json'
PREDICTIONS = OUT / 'a3_coverage_48c_predictions.csv'
KEYS = ['season', 'week', 'game_id', 'opponent']
METRICS = ['pass_attempts', 'carries', 'targets', 'receptions', 'pass_yards', 'rush_yards', 'rec_yards']
PLAYER = ['prior_games', 'weeks_since_appearance', 'recently_seen'] + ['prior_mean_' + x for x in METRICS]
# Exclude route and pressure fields pending source-quality review.
COV = ['prior5_rate_coverage_' + x for x in ['cover_0','cover_1','cover_2','cover_3','cover_4','cover_6','2_man']]
COV += ['prior5_coverage_label_rate']
TASKS = [(p, t) for p in ('WR', 'TE') for t in ('targets', 'receptions', 'rec_yards')]

def matrix(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).astype(float)

def forecast(train, test, target, features):
    y = train['actual_' + target].to_numpy(dtype=float)
    pos = (y > 0).astype(int)
    if len(np.unique(pos)) < 2:
        prob = np.full(len(test), float(pos[0]))
    else:
        cls = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=10, min_samples_leaf=40, l2_regularization=10, random_state=42)
        cls.fit(matrix(train, features), pos)
        prob = cls.predict_proba(matrix(test, features))[:, 1]
    active = train.loc[y > 0]
    if len(active) < 50:
        raise ValueError(f'Insufficient positive training rows for {target}')
    reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.04, max_leaf_nodes=12, min_samples_leaf=25, l2_regularization=10, random_state=42)
    reg.fit(matrix(active, features), active['actual_' + target])
    return np.maximum(prob * np.maximum(reg.predict(matrix(test, features)), 0), 0)

def metrics(y, pred):
    error = pred - y
    return {'mae': float(mean_absolute_error(y, pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'bias': float(error.mean()),
            'overprediction_rate': float((error > 0).mean()),
            'mean_positive_overprediction': float(error[error > 0].mean()) if (error > 0).any() else 0.0}

def run():
    pool = pd.read_csv(POOL, dtype={'player_id': str}, low_memory=False)
    coverage = pd.read_csv(COVERAGE, low_memory=False)
    required = ['roster_position', 'player_id'] + KEYS + PLAYER + ['actual_' + t for _, t in TASKS]
    for name, frame, cols in [('pool', pool, required), ('coverage', coverage, KEYS + COV)]:
        missing = sorted(set(cols) - set(frame.columns))
        if missing:
            raise ValueError(f'{name} missing columns: {missing}')
    if pool.duplicated(['season','week','game_id','team','player_id']).any():
        raise ValueError('Duplicate player rows')
    if coverage.duplicated(KEYS).any():
        raise ValueError('Duplicate coverage game rows')
    coverage = coverage[KEYS + COV].copy()
    for col in COV:
        coverage[col] = pd.to_numeric(coverage[col], errors='coerce')
    # Restrict to the two supported seasons, preventing accidental use of future records.
    pool = pool.loc[pool.season.isin([2024, 2025])].copy()
    pool = pool.merge(coverage, on=KEYS, how='left', validate='many_to_one', indicator=True)
    if (pool['_merge'] != 'both').any():
        raise ValueError('Unmatched player/defense game keys; inspect game ID formats before training')
    pool.drop(columns=['_merge'], inplace=True)
    pool['recently_seen'] = pool['recently_seen'].astype(str).str.lower().map({'true':1,'false':0,'1':1,'0':0})
    for col in PLAYER + ['actual_' + t for _, t in TASKS]:
        pool[col] = pd.to_numeric(pool[col], errors='coerce')
    # Only use defenses with at least two prior games of coverage data.
    # Equal rows for baseline/challenger ensure a fair comparison.
    usable = pool[COV[0]].notna() & pool['prior5_coverage_label_rate'].gt(0)
    pool = pool.loc[usable].copy()
    results, predictions = [], []
    for position, target in TASKS:
        sub = pool.loc[pool.roster_position.eq(position)].dropna(subset=['actual_' + target, 'prior_mean_' + target]).copy()
        train = sub.loc[sub.season.eq(2024)].copy()
        test = sub.loc[sub.season.eq(2025)].copy()
        if len(train) < 100 or len(test) < 30:
            raise ValueError(f'Insufficient {position}/{target}: train={len(train)}, test={len(test)}')
        base = forecast(train, test, target, PLAYER)
        enhanced = forecast(train, test, target, PLAYER + COV)
        y = test['actual_' + target].to_numpy(float)
        bm, cm = metrics(y, base), metrics(y, enhanced)
        # Relative breakout: 25% and at least 2 additional units over pregame mean.
        usual = test['prior_mean_' + target].to_numpy(float)
        breakout = y >= np.maximum(usual * 1.25, usual + 2)
        breakout_metrics = None
        if breakout.sum() >= 10:
            breakout_metrics = {'rows': int(breakout.sum()),
                                'baseline_mae': float(mean_absolute_error(y[breakout], base[breakout])),
                                'coverage_mae': float(mean_absolute_error(y[breakout], enhanced[breakout]))}
        results.append({'position': position, 'target': target, 'train_rows': len(train), 'test_rows': len(test),
                        'baseline': bm, 'coverage': cm, 'mae_gain': bm['mae'] - cm['mae'],
                        'breakout_subset': breakout_metrics})
        keep = ['season','week','game_id','team','player_id','opponent']
        if 'player_name' in test.columns:
            keep.append('player_name')
        pred = test[keep].copy()
        pred['position'] = position
        pred['stat'] = target
        pred['actual'] = y
        pred['usual'] = usual
        pred['baseline_prediction'] = base
        pred['coverage_prediction'] = enhanced
        pred['actual_breakout'] = breakout
        predictions.append(pred)
        print(f'{position}/{target}: 2025 MAE baseline={bm["mae"]:.3f}, coverage={cm["mae"]:.3f}, gain={bm["mae"]-cm["mae"]:+.3f}')
    report = {'phase':'4.8C', 'train_season':2024, 'evaluation_season':2025,
              'comparison':'Identical two-stage models; baseline player history vs player history plus defensive coverage',
              'comparisons':len(results), 'mae_improvements':sum(x['mae_gain'] > 0 for x in results),
              'results':results, 'limitations':[
                  '2025 is a development evaluation, not an untouched final holdout.',
                  'Player pool is historical candidate generation, not a verified active roster.',
                  'First two defensive games of each season are excluded.',
                  'Coverage missingness and source definitions require further audit.',
                  'Route and pressure features excluded due to data-quality concerns.',
                  'No receiver-level route or defender assignment is available.',
                  'No 2026 coverage validation is possible from this source yet.',
                  'No automatic model promotion; 2026 independent holdout required.'
              ]}
    REPORT.write_text(json.dumps(report, indent=2))
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTIONS, index=False)
    print(f'PHASE 4.8C COMPLETE: {report["mae_improvements"]}/{len(results)} MAE improvements')
    print('Report:', REPORT, 'Predictions:', PREDICTIONS)
    print('Incumbent A3 unchanged.')

if __name__ == '__main__':
    run()
