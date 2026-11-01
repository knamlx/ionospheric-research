"""Daily IonoAutoML replication for the controlled interstation foF2 comparison."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import ast, json, time, hashlib, math, warnings, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

SOURCE=Path(__file__).resolve().parent/'source'
ROOT=Path(__file__).resolve().parent

def load_original_functions():
    names={'as_utc_timestamp','split_walk_forward_window','select_features','metric_record','default_model_params','make_estimator'}
    tree=ast.parse((SOURCE/'run_ml_baselines.py').read_text(encoding='utf-8'))
    chosen=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(chosen)==len(names)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(SOURCE/'run_ml_baselines.py'),'exec',flags=__import__('__future__').annotations.compiler_flag),globals())
load_original_functions()

def prepare(out):
    out=Path(out)
    if (out/'data/datasets.joblib').exists() and (out/'config.json').exists():
        return json.loads((out/'config.json').read_text(encoding='utf-8')),pd.read_pickle(out/'data/panel.pkl')
    for d in ['data','checkpoints','figures','models','source']:(out/d).mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'configs/experiments/fof2_fast_ml_24h.json').read_text(encoding='utf-8'))
    fc=json.loads((SOURCE/'configs/feature_engineering.json').read_text(encoding='utf-8'))
    cfg.update(experiment_id='interstation_walkforward_15min',horizons=['15min'],test_start='2026-01-01T00:00:00Z',test_end='2026-04-01T00:00:00Z',stations=['EB040','DB049','PQ052'])
    cfg['automl']['enabled']=False
    cfg['automl']['n_trials']=0
    cfg['protocol_notes']=['Daily rolling refit uses already-observed 2026 data after first day, NOT frozen training on 2025','Five original estimators/parameters, 7/21/28-day windows; no tuning, no early stopping','Only foF2 histories and calendar; no other ionospheric states/drivers, to isolate interstation information','CS rules follow IonoAutoML: >=50 or -1 or missing; positive foF2 <=30 MHz','Original daily split on completed-bin start, purge 15min ensures training target interval ends at test-day start','Day-ahead batch contains rolling 15min forecasts: test-day past observations available per forecast; no model refit within day']
    (out/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    import shutil
    for p in [SOURCE/'run_ml_baselines.py',SOURCE/'build_features.py',SOURCE/'normalize_time_grid.py',SOURCE/'configs/experiments/fof2_fast_ml_24h.json',SOURCE/'configs/feature_engineering.json',Path(__file__)]:shutil.copy2(p,out/'source'/p.name)
    index=pd.date_range('2025-01-01','2026-04-01',freq='15min',inclusive='left',tz='UTC')
    panel=pd.DataFrame(index=index)
    manifest=[];inventory=[]
    for station in cfg['stations']:
        records=[];rawrows=0;bad=0
        for period in ['train','test']:
            for p in sorted((ROOT/'archive_2025_2026'/period/f'station={station}').rglob('*.txt')):
                data=p.read_bytes()
                manifest.append(dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                for line in data.decode('utf-8-sig',errors='replace').splitlines():
                    if not line or line.startswith('#'):continue
                    parts=line.split(maxsplit=3)
                    try:
                        cs=float(parts[1]);v=float(parts[2]);rawrows+=1
                        if (cs>=50 or cs==-1 or math.isnan(cs)) and 0<v<=30:records.append((parts[0],v))
                        else:bad+=1
                    except (ValueError,IndexError):continue
        df=pd.DataFrame(records,columns=['time','foF2'])
        df.time=pd.to_datetime(df.time,utc=True)
        duplicates=int(df.time.duplicated().sum())
        df=df.sort_values('time').drop_duplicates('time',keep='last').set_index('time')
        panel[station]=df.foF2.resample('15min').mean().reindex(index)
        inventory.append(dict(station=station,raw_numeric_rows=rawrows,rejected=bad,duplicate_timestamps=duplicates,valid_bins=int(panel[station].notna().sum())))
    panel.index.name='time_utc'
    panel.to_csv(out/'data/observations_15min.csv')
    panel.to_pickle(out/'data/panel.pkl')
    panel.notna().resample('MS').sum().to_csv(out/'data/monthly_coverage.csv')
    pd.DataFrame(inventory).to_csv(out/'data/inventory.csv',index=False)
    (out/'data/sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    datasets={}
    for gap in [0,30,60]:
        local,remote=features(panel,gap,fc)
        datasets[gap]=(local,remote)
        local.to_csv(out/f'data/features_local_gap{gap}.csv')
        remote.to_csv(out/f'data/features_interstation_gap{gap}.csv')
    joblib.dump(datasets,out/'data/datasets.joblib')
    audit=causal_checks(panel,fc,datasets)
    (out/'data/causality_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return cfg,panel

def features(panel,gap,fc):
    # Calendar formulas and lag durations are taken from build_features.py.
    x=pd.DataFrame(index=panel.index)
    t=panel.index
    for n,phase in [('hour',t.hour/24),('month',t.month/12),('doy',t.dayofyear/366)]:
        x[n+'_sin']=np.sin(2*np.pi*phase);x[n+'_cos']=np.cos(2*np.pi*phase)
    visible=panel.EB040.shift(gap//15)
    x['foF2']=visible;x['foF2_state']=visible
    for lag in fc['lags']:
        k=int(pd.Timedelta(lag)/pd.Timedelta('15min'))
        x['foF2_lag_'+lag]=visible.shift(k)
    for lag in fc['diff_lags']:
        k=int(pd.Timedelta(lag)/pd.Timedelta('15min'))
        x['foF2_diff_'+lag]=visible-visible.shift(k)
    x['last_available']=visible.ffill(limit=96)
    stamp=pd.Series(panel.index,index=panel.index).where(panel.EB040.notna()).shift(gap//15).ffill(limit=96)
    x['age_minutes']=(pd.Series(panel.index,index=panel.index)-stamp).dt.total_seconds()/60
    z=x.copy()
    for s in ['DB049','PQ052']:
        z[s+'_state']=panel[s]
        for lag in fc['lags']:
            z[s+'_lag_'+lag]=panel[s].shift(int(pd.Timedelta(lag)/pd.Timedelta('15min')))
    return x,z

def causal_checks(panel,fc,datasets):
    checks=[]
    idx=panel.index.get_loc(pd.Timestamp('2026-02-01',tz='UTC'))
    for gap in [0,30,60]:
        altered=panel.copy()
        altered.iloc[idx-gap//15+1:,altered.columns.get_loc('EB040')]=10000
        altered.iloc[idx+1:,altered.columns.get_loc('DB049')]=10000
        altered.iloc[idx+1:,altered.columns.get_loc('PQ052')]=10000
        a,b=features(altered,gap,fc)
        pd.testing.assert_series_equal(a.iloc[idx],datasets[gap][0].iloc[idx])
        pd.testing.assert_series_equal(b.iloc[idx],datasets[gap][1].iloc[idx])
        checks.append(dict(gap_minutes=gap,hidden_local_and_future_remote_invariance=True))
    return checks

G={}
def init_worker(out):
    out=Path(out)
    G.update(out=out,cfg=json.loads((out/'config.json').read_text(encoding='utf-8')),panel=pd.read_pickle(out/'data/panel.pkl'),datasets=joblib.load(out/'data/datasets.joblib'))

def task(args):
    daystr,window,gap=args
    out=G['out'];cfg=G['cfg'];panel=G['panel'];local,multi=G['datasets'][gap]
    name=f'{daystr}_w{window}_g{gap}'
    done=out/'checkpoints'/f'{name}.json'
    if done.exists():return json.loads(done.read_text(encoding='utf-8'))
    tic=time.perf_counter()
    try:
        day=pd.Timestamp(daystr,tz='UTC')
        frame=pd.DataFrame({'time_utc':panel.index},index=panel.index)
        tr,_,te,bounds=split_walk_forward_window(frame,current_day=day,train_days=window,forecast_horizon='15min',dataset_step='15min')
        y=panel.EB040.shift(-1)
        available=pd.Series(True,index=panel.index)
        for pair in G['datasets'].values():available &= pair[0].last_available.notna()
        trainidx=tr.index[y.reindex(tr.index).notna() & available.reindex(tr.index)]
        alltestidx=te.index[y.reindex(te.index).notna() & available.reindex(te.index) & (te.index+pd.Timedelta('30min')<=pd.Timestamp('2026-04-01',tz='UTC'))]
        shared=panel[['DB049','PQ052']].notna().all(axis=1).reindex(alltestidx)
        assert len(trainidx)>=cfg['minimum_train_rows']
        assert (trainidx+pd.Timedelta('30min')).max()<=day
        assert not set(trainidx).intersection(alltestidx)
        metadata=dict(task=name,day=daystr,window=window,gap=gap,n_train=len(trainidx),n_test_all=len(alltestidx),n_test_shared=int(shared.sum()),train_first=str(trainidx.min()),train_last=str(trainidx.max()),train_target_end=str((trainidx+pd.Timedelta('30min')).max()),features={},warnings=[])
        if len(alltestidx)<2:
            metadata.update(status='skipped_no_test',seconds=time.perf_counter()-tic)
            done.write_text(json.dumps(metadata,indent=2),encoding='utf-8');return metadata
        pred=pd.DataFrame({'time_utc':alltestidx,'actual':y.reindex(alltestidx).values,'shared':shared.values})
        pred['persistence']=local.last_available.reindex(alltestidx).values
        for kind,x in [('local',local),('interstation',multi)]:
            columns=select_features(x.loc[trainidx],list(x.columns),cfg['minimum_feature_coverage'])
            metadata['features'][kind]=columns
            imp=SimpleImputer(strategy='median',keep_empty_features=True)
            xt=pd.DataFrame(imp.fit_transform(x.loc[trainidx,columns]),columns=columns,index=trainidx)
            xe=pd.DataFrame(imp.transform(x.loc[alltestidx,columns]),columns=columns,index=alltestidx)
            metadata[kind+'_imputer_medians']={c:float(v) for c,v in zip(columns,imp.statistics_)}
            for model_name in cfg['models']:
                start=time.perf_counter()
                estimator=make_estimator(model_name,cfg)
                # Disable CatBoost side-effect files; learning settings are unchanged.
                if model_name=='CatBoost':estimator.set_params(allow_writing_files=False)
                with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                    warnings.simplefilter('always')
                    estimator.fit(xt,y.loc[trainidx])
                    values=estimator.predict(xe)
                metadata['warnings'] += [dict(model=model_name,kind=kind,message=str(w.message)) for w in caught]
                assert np.isfinite(values).all()
                pred[kind+'__'+model_name]=values
                metadata[kind+'__'+model_name+'_seconds']=time.perf_counter()-start
                # Retain the last test day's models as independently replayable examples.
                if daystr=='2026-03-31':
                    bundle=dict(columns=columns,imputer=imp,estimator=estimator,params=estimator.get_params(),train_index=trainidx,test_index=alltestidx)
                    path=out/'models'/f'{name}_{kind}_{model_name}.joblib'
                    joblib.dump(bundle,path,compress=3)
                    restored=joblib.load(path)
                    np.testing.assert_allclose(restored['estimator'].predict(xe),values,rtol=0,atol=0)
        pred.to_csv(out/'checkpoints'/f'{name}.csv',index=False)
        metadata.update(status='completed',seconds=time.perf_counter()-tic)
        done.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        return metadata
    except Exception:
        error=dict(task=name,status='failed',error=traceback.format_exc())
        (out/'checkpoints'/f'{name}.error.json').write_text(json.dumps(error),encoding='utf-8')
        return error

def train_all(out,workers=6):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    out=Path(out);start=time.perf_counter()
    existing=list((out/'checkpoints').glob('*.json'))
    if len(existing)==810 and not list((out/'checkpoints').glob('*.error.json')):
        cached=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(existing)]
        assert all(r['status'] in ['completed','skipped_no_test'] for r in cached)
        print('Loaded 810 completed/explicitly skipped checkpoints; original training runtime retained.',flush=True)
        return cached
    days=pd.date_range('2026-01-01','2026-04-01',freq='D',inclusive='left')
    tasks=[(d.strftime('%Y-%m-%d'),w,g) for d in days for w in [7,21,28] for g in [0,30,60]]
    finished=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(out),)) as pool:
        pending=[pool.submit(task,t) for t in tasks]
        for future in as_completed(pending):
            r=future.result();finished.append(r)
            if len(finished)%9==0 or r['status']=='failed':
                print(f"{len(finished)}/{len(tasks)} splits; elapsed={time.perf_counter()-start:.1f}s; {r['task']} {r['status']}",flush=True)
            (out/'progress.json').write_text(json.dumps(dict(done=len(finished),total=len(tasks),failed=sum(r['status']=='failed' for r in finished),seconds=time.perf_counter()-start)),encoding='utf-8')
    errors=[r for r in finished if r['status']=='failed']
    assert not errors,errors[:2]
    (out/'runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,workers=workers,splits=len(tasks),completed=sum(r['status']=='completed' for r in finished),skipped=sum(r['status'].startswith('skipped') for r in finished)),indent=2),encoding='utf-8')
    return finished
