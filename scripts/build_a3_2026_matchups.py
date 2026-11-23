
"""A3 Phase 4.9: 2026-only, leakage-safe matchup research and forecasts.

Historical 2024/2025 data is not used. This is a challenger, not incumbent A3.
Outputs pregame retrospective predictions for completed 2026 games; it does not
assert live injury, active roster, coverage, or sportsbook knowledge.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path('data/processed')
OUT = Path('data/model_results')
OUT.mkdir(parents=True, exist_ok=True)
POOL = BASE / 'pregame_player_pool.csv'
PBP = Path('data/play_by_play_2026.parquet')
OUTPUT = OUT / 'a3_2026_matchup_predictions.csv'
REPORT = OUT / 'a3_2026_matchup_audit.json'
METRICS = ['pass_attempts','pass_yards','carries','rush_yards','targets','receptions','rec_yards']
TASKS = {'QB':['pass_attempts','pass_yards','rush_yards'], 'RB':['carries','rush_yards','targets','receptions','rec_yards'], 'WR':['targets','receptions','rec_yards'], 'TE':['targets','receptions','rec_yards']}
KEYS = ['season','week','game_id','team','player_id']


def rolling_prior(frame, by, metrics, windows=(3,5)):
    frame = frame.sort_values(by+['week','game_id']).copy()
    grp = frame.groupby(by, sort=False)
    for metric in metrics:
        frame['prior_season_'+metric] = grp[metric].transform(lambda s: s.shift().expanding(min_periods=1).mean())
        for n in windows:
            frame[f'prior{n}_'+metric] = grp[metric].transform(lambda s: s.shift().rolling(n,min_periods=1).mean())
    frame['prior_games'] = grp.cumcount()
    return frame


def run():
    pool = pd.read_csv(POOL, dtype={'player_id':str},low_memory=False)
    required = KEYS + ['opponent','roster_position','player_name'] + ['actual_'+m for m in METRICS]
    missing = set(required)-set(pool.columns)
    if missing:
        raise ValueError('Missing pool columns: '+str(sorted(missing)))
    pool = pool.loc[pool.season.eq(2026) & pool.roster_position.isin(TASKS)].copy()
    if pool.empty or pool.duplicated(KEYS).any():
        raise ValueError('Empty or duplicate 2026 candidate pool')
    for m in METRICS:
        pool['actual_'+m] = pd.to_numeric(pool['actual_'+m],errors='coerce')
    if pool[['actual_'+m for m in METRICS]].isna().any().any():
        raise ValueError('Missing actual stats in candidate pool')
    # Defensive allowance totals use observed 2026 player performances only.
    # All position groups are derived from player-game pool, not 2024/25.
    bypos = (pool.groupby(['season','week','game_id','opponent','roster_position'],as_index=False)
             [['actual_'+m for m in METRICS]].sum())
    bypos = bypos.rename(columns={'opponent':'defense', 'roster_position':'position', **{'actual_'+m:'allowed_'+m for m in METRICS}})
    # Include zero-allowance games, avoiding selection of only games with candidates.
    defense_games = pool[['season','week','game_id','opponent']].drop_duplicates().rename(columns={'opponent':'defense'})
    positions = pd.DataFrame({'position':list(TASKS)})
    defense_games['_k']=1; positions['_k']=1
    full = defense_games.merge(positions,on='_k').drop(columns='_k')
    bypos = full.merge(bypos,on=['season','week','game_id','defense','position'],how='left',validate='one_to_one')
    allow_cols = ['allowed_'+m for m in METRICS]
    bypos[allow_cols]=bypos[allow_cols].fillna(0)
    bypos = rolling_prior(bypos,['season','defense','position'],allow_cols)
    bypos = bypos.rename(columns={'defense':'opponent','position':'roster_position'})
    predef = [c for c in bypos if c.startswith('prior')]
    bypos = bypos.rename(columns={c:'def_'+c for c in predef})
    df = pool.merge(bypos[['season','week','game_id','opponent','roster_position']+['def_'+c for c in predef]],
                    on=['season','week','game_id','opponent','roster_position'],how='left',validate='many_to_one')
    df = rolling_prior(df,['season','player_id'],['actual_'+m for m in METRICS])
    # Game-script proxy uses only past 2026 offensive plays. No inferred live spread.
    if not PBP.exists():
        raise FileNotFoundError(PBP)
    pbp = pd.read_parquet(PBP,columns=['season','week','game_id','season_type','posteam','play_type','no_play','two_point_attempt','score_differential'])
    pbp=pbp.loc[pbp.season.eq(2026)&pbp.season_type.eq('REG')&pbp.posteam.notna()&pbp.play_type.isin(['pass','run'])].copy()
    pbp=pbp.loc[pd.to_numeric(pbp.no_play,errors='coerce').fillna(0).eq(0)&pd.to_numeric(pbp.two_point_attempt,errors='coerce').fillna(0).eq(0)]
    pbp['is_pass']=pbp.play_type.eq('pass').astype(float)
    pbp['leading_14']=pd.to_numeric(pbp.score_differential,errors='coerce').ge(14).astype(float)
    pbp['trailing_14']=pd.to_numeric(pbp.score_differential,errors='coerce').le(-14).astype(float)
    pbp['pass_leading_14']=pbp.is_pass*pbp.leading_14
    off = pbp.groupby(['season','week','game_id','posteam'],as_index=False).agg(plays=('is_pass','size'),passes=('is_pass','sum'),lead_plays=('leading_14','sum'),lead_passes=('pass_leading_14','sum'),trail_plays=('trailing_14','sum'))
    off=off.rename(columns={'posteam':'team'})
    off=rolling_prior(off,['season','team'],['plays','passes','lead_plays','lead_passes','trail_plays'])
    offcols=[c for c in off if c.startswith('prior')]
    df=df.merge(off[['season','week','game_id','team']+offcols],on=['season','week','game_id','team'],how='left',validate='many_to_one')
    # Same-week stats from other games must never enter predictors: all group
    # histories are lagged, and no game outcomes enter features for that game.
    rows=[]
    for position, targets in TASKS.items():
        part=df.loc[df.roster_position.eq(position)].copy()
        for metric in targets:
            actual='actual_'+metric
            usual=part['prior_season_'+actual]
            recent=part['prior5_'+actual]
            def5=part['def_prior5_allowed_'+metric]
            # Shrink defensive position allowances toward 2026 league average.
            # Uses the full 2026 average ONLY as a retrospective reference; this
            # would leak future games. Therefore do NOT use it in projections.
            # Instead use current opponent-to-previous-own-season ratio, bounded.
            defseason=part['def_prior_season_allowed_'+metric]
            factor=np.where((defseason>0)&def5.notna(),np.clip(def5/defseason,0.8,1.2),1.0)
            # Avoid optimistic double-counting: limited +/- 12% adjustment.
            factor=1+0.6*(factor-1)
            central=np.maximum((0.55*usual+0.45*recent)*factor,0)
            # Conservative threshold is a heuristic, NOT a calibrated probability.
            comfort=np.floor(np.maximum(central*0.7,0))
            missing_history=part.prior_games.lt(3)|part.def_prior_games.lt(3)
            central=np.where(missing_history,np.nan,central)
            comfort=np.where(missing_history,np.nan,comfort)
            # Flag potential script traps using own prior passing tendency and
            # share of offensive plays while leading by 14+, not forecast spread.
            passrate=part.prior5_passes/part.prior5_plays.replace(0,np.nan)
            leadshare=part.prior5_lead_plays/part.prior5_plays.replace(0,np.nan)
            trap=np.where((position in ('QB','WR','TE')) & (leadshare>0.20), 'POTENTIAL_LEAD_SCRIPT', 'NONE')
            out=part[KEYS+['opponent','player_name','roster_position']].copy()
            out['stat']=metric
            out['actual']=part[actual].to_numpy()
            out['season_average_pregame']=usual.to_numpy()
            out['last5_average_pregame']=recent.to_numpy()
            out['opponent_allowed_position_last5']=def5.to_numpy()
            out['opponent_allowed_position_season']=defseason.to_numpy()
            out['projection_heuristic']=central
            out['comfort_threshold_uncalibrated']=comfort
            out['matchup_adjustment']=factor
            out['team_prior5_pass_rate']=passrate.to_numpy()
            out['team_prior5_big_lead_play_share']=leadshare.to_numpy()
            out['trap_flag']=trap
            out['status']=np.where(missing_history,'INSUFFICIENT_2026_HISTORY','RESEARCH_ONLY')
            rows.append(out)
    result=pd.concat(rows,ignore_index=True)
    result.to_csv(OUTPUT,index=False)
    usable=result.loc[result.status.eq('RESEARCH_ONLY')].copy()
    metrics={}
    for (pos,stat),grp in usable.groupby(['roster_position','stat']):
        if len(grp)<10:continue
        y=grp.actual.to_numpy(float)
        pred=grp.projection_heuristic.to_numpy(float)
        base=grp.season_average_pregame.to_numpy(float)
        metrics[pos+'_'+stat]={'rows':int(len(grp)),'season_baseline_mae':float(np.mean(abs(base-y))), 'challenger_mae':float(np.mean(abs(pred-y))), 'challenger_bias':float(np.mean(pred-y)), 'overprediction_rate':float(np.mean(pred>y))}
    audit={'phase':'4.9','year':2026,'data_rows':len(result),'usable_rows':len(usable),'metrics':metrics,'promoted':False,
           'limitations':['Retrospective research: no live game schedule or injury verification','2026-only; prior seasons excluded','Position allowed totals rely on candidate pool coverage; missing first appearances can bias totals','No verified slot/outside or individual coverage assignments','Blowout flag is historical lead exposure, not a projected blowout probability','Comfort threshold is uncalibrated and must not be presented as a reliable probability','Projection heuristic is experimental; incumbent A3 unchanged','Player active status is not verified','Potential week-to-week data corrections can change historical figures']}
    REPORT.write_text(json.dumps(audit,indent=2))
    print('A3 2026 MATCHUP RESEARCH COMPLETE',json.dumps({'rows':len(result),'usable':len(usable),'metrics':metrics},indent=2))
    print('Incumbent A3 unchanged.')

if __name__=='__main__':run()
