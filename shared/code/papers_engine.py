"""Extended fixed-parameter daily validation for two foF2 manuscripts."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,time,hashlib,math,traceback,warnings,shutil,importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
from paths import baseline_directory
BASE=baseline_directory()
SOURCE_BASE=ROOT/'vendor'
spec=importlib.util.spec_from_file_location('legacy_papers',SOURCE_BASE/'walkforward_engine.py')
LEGACY=importlib.util.module_from_spec(spec);spec.loader.exec_module(LEGACY)
STATIONS=['EB040','DB049','PQ052']
GAPS=[0,15,30,60,120]

def feature_frames(panel,target,gap,fc):
    donors=sorted(set(STATIONS)-{target})
    mapped=panel[[target]+donors].copy();mapped.columns=['EB040','DB049','PQ052']
    local,multi=LEGACY.features(mapped,gap,fc)
    rename={c:(donors[0]+c[len('DB049'):] if c.startswith('DB049_') else donors[1]+c[len('PQ052'):] if c.startswith('PQ052_') else c) for c in multi.columns}
    return local,multi.rename(columns=rename)

def prepare(out):
    out=Path(out)
    for name in ['data','source','checkpoints','models','figures','manuscripts','notebooks']:(out/name).mkdir(exist_ok=True)
    old=pd.read_pickle(BASE/'data/panel.pkl')
    grid=pd.date_range(old.index.min(),pd.Timestamp('2026-07-01',tz='UTC'),freq='15min',inclusive='left')
    panel=old.reindex(grid);panel.index.name='time_utc'
    sources=json.loads((BASE/'data/sources.json').read_text(encoding='utf-8'))
    for station in STATIONS:
        rows=[]
        for p in sorted((out/'archive_confirmation/test'/f'station={station}').rglob('*.txt')):
            b=p.read_bytes();assert ('URSI-Code '+station).encode() in b
            sources.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
            for line in b.decode('utf-8-sig',errors='replace').splitlines():
                if not line or line.startswith('#'):continue
                parts=line.split(maxsplit=3)
                try:
                    cs,v=float(parts[1]),float(parts[2])
                    if (cs>=50 or cs==-1 or math.isnan(cs)) and 0<v<=30:rows.append((parts[0],v))
                except (ValueError,IndexError):pass
        assert rows,station
        frame=pd.DataFrame(rows,columns=['time','foF2']);frame.time=pd.to_datetime(frame.time,utc=True)
        frame=frame[(frame.time>=pd.Timestamp('2026-04-01',tz='UTC'))&(frame.time<pd.Timestamp('2026-07-01',tz='UTC'))]
        values=frame.sort_values('time').drop_duplicates('time',keep='last').set_index('time').foF2.resample('15min').mean()
        panel.loc[values.index,station]=values
    pd.testing.assert_frame_equal(panel.loc[old.index],old,check_freq=False)
    panel.to_pickle(out/'data/panel.pkl');panel.to_csv(out/'data/observations_15min.csv',index_label='time_utc',float_format='%.17g')
    panel.notna().resample('MS').sum().to_csv(out/'data/monthly_valid_bins.csv')
    panel.notna().resample('MS').mean().mul(100).to_csv(out/'data/monthly_coverage_percent.csv')
    (out/'data/sources.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False),encoding='utf-8')
    cfg=json.loads((SOURCE_BASE/'config.json').read_text(encoding='utf-8'))
    plan=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    cfg.update(models=plan['models_default'],horizons=['15min'],gaps=GAPS,stations=STATIONS)
    (out/'model_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    fc=json.loads((SOURCE_BASE/'source/feature_engineering.json').read_text(encoding='utf-8'))
    checks=[]
    for target in STATIONS:
        datasets={g:feature_frames(panel,target,g,fc) for g in GAPS}
        joblib.dump(datasets,out/f'data/datasets_{target}.joblib')
        # Controlled 120-minute failure. All hidden values, including intermediate
        # ones between forecast checkpoints, are changed simultaneously.
        start=pd.Timestamp('2026-04-15 06:00',tz='UTC')
        hidden=panel.copy();hidden.loc[(hidden.index>=start)&(hidden.index<start+pd.Timedelta('120min')),target]=10000
        for elapsed in [15,30,60,120]:
            origin=start+pd.Timedelta(minutes=elapsed-15)
            a,b=feature_frames(hidden,target,elapsed,fc)
            pd.testing.assert_series_equal(a.loc[origin],datasets[elapsed][0].loc[origin])
            pd.testing.assert_series_equal(b.loc[origin],datasets[elapsed][1].loc[origin])
            expected_anchor=start-pd.Timedelta('15min')
            observed=panel[target].loc[expected_anchor]
            np.testing.assert_allclose(a.loc[origin,'foF2_state'],observed,equal_nan=True)
            checks.append(dict(target=target,outage_start=str(start),elapsed=elapsed,whole_episode_hidden_invariance=True,anchor=str(expected_anchor)))
    (out/'data/continuous_outage_causality.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    for p in [Path(__file__),SOURCE_BASE/'source/run_ml_baselines.py',SOURCE_BASE/'source/build_features.py',SOURCE_BASE/'source/feature_engineering.json',SOURCE_BASE/'source/fof2_fast_ml_24h.json']:
        shutil.copy2(p,out/'source'/p.name)
    return panel

G={}
def init(out):
    out=Path(out);G.update(out=out,panel=pd.read_pickle(out/'data/panel.pkl'),target=None)

def task(args):
    phase,target,daystr,window,gap=args
    out=G['out'];name=f'{phase}_{target}_{daystr}_w{window}_g{gap}'
    marker=out/'checkpoints'/f'{name}.json'
    cfg=json.loads((out/'model_config.json').read_text(encoding='utf-8'))
    models=cfg['models']
    if marker.exists():return json.loads(marker.read_text(encoding='utf-8'))
    start=time.perf_counter()
    try:
        if G['target']!=target:
            G['datasets']=joblib.load(out/f'data/datasets_{target}.joblib');G['target']=target
        datasets=G['datasets'];panel=G['panel'];local,multi=datasets[gap]
        donors=sorted(set(STATIONS)-{target});y=panel[target].shift(-1)
        available=pd.Series(True,index=panel.index)
        # Keep the original training eligibility unchanged to reuse exact EB040
        # fits. Fair five-gap scoring is imposed separately and transparently.
        for g in [0,30,60]:available &= datasets[g][0].last_available.notna()
        all_gaps=pd.Series(True,index=panel.index)
        for pair in datasets.values():all_gaps &= pair[0].last_available.notna()
        day=pd.Timestamp(daystr,tz='UTC');end=pd.Timestamp('2026-04-01' if phase=='exploratory' else '2026-07-01',tz='UTC')
        frame=pd.DataFrame({'time_utc':panel.index},index=panel.index)
        tr,_,te,bounds=LEGACY.split_walk_forward_window(frame,current_day=day,train_days=window,forecast_horizon='15min',dataset_step='15min')
        trainidx=tr.index[y.reindex(tr.index).notna()&available.reindex(tr.index)]
        testidx=te.index[y.reindex(te.index).notna()&available.reindex(te.index)&(te.index+pd.Timedelta('30min')<=end)]
        meta=dict(task=name,phase=phase,target=target,donors=donors,day=daystr,window=window,gap=gap,models=models,n_train=len(trainidx),n_test=len(testidx),features={},medians={},warnings=[])
        if len(trainidx)<48 or len(testidx)<2 or y.loc[trainidx].nunique()<2:
            meta.update(status='skipped',reason='insufficient_train' if len(trainidx)<48 else 'insufficient_test' if len(testidx)<2 else 'constant_train_target',seconds=time.perf_counter()-start)
            marker.write_text(json.dumps(meta,indent=2),encoding='utf-8');return meta
        assert (trainidx+pd.Timedelta('30min')).max()<=day
        meta.update(train_start=str(trainidx.min()),train_end=str(trainidx.max()),last_train_target_end=str((trainidx+pd.Timedelta('30min')).max()))
        pred=pd.DataFrame({'time_utc':testidx,'actual':y.loc[testidx].values,'shared':panel[donors].notna().all(axis=1).loc[testidx].values,'all_gaps_history':all_gaps.loc[testidx].values,'persistence':local.last_available.loc[testidx].values})
        delayed_shared=pd.Series(True,index=panel.index)
        for steps in [0,1,2]:delayed_shared &= panel[donors].shift(steps).notna().all(axis=1)
        pred['shared_delays']=delayed_shared.loc[testidx].values
        reused=False
        if phase=='exploratory' and target=='EB040' and gap in [0,30,60]:
            oldname=f'{daystr}_w{window}_g{gap}'
            oldmeta=json.loads((BASE/'checkpoints'/f'{oldname}.json').read_text(encoding='utf-8'))
            if oldmeta['status']=='completed':
                old=pd.read_csv(BASE/'checkpoints'/f'{oldname}.csv',parse_dates=['time_utc'])
                assert len(trainidx)==oldmeta['n_train']
                assert pd.DatetimeIndex(old.time_utc).equals(testidx)
                np.testing.assert_allclose(old.actual,pred.actual,rtol=0,atol=1e-12)
                for kind in ['local','interstation']:
                    for model in models:
                        column=kind+'__'+model
                        values=old[column].astype(np.float32).astype(float).values if model=='XGBoost' else old[column].values
                        pred[column]=values
                    meta['features'][kind]=oldmeta['features'][kind]
                    meta['medians'][kind]=oldmeta[kind+'_imputer_medians']
                reused=True;meta['reused_original']=oldname
                if daystr=='2026-03-31' and window==21:
                    for kind in ['local','interstation']:
                        for model in models:
                            source=BASE/'models'/f'{oldname}_{kind}_{model}.joblib'
                            if source.exists():shutil.copy2(source,out/'models'/f'{name}_{kind}_{model}.joblib')
        if not reused:
            for kind,x in [('local',local),('interstation',multi)]:
                columns=LEGACY.select_features(x.loc[trainidx],list(x.columns),.3)
                imp=SimpleImputer(strategy='median',keep_empty_features=True)
                xt=pd.DataFrame(imp.fit_transform(x.loc[trainidx,columns]),columns=columns,index=trainidx)
                xe=pd.DataFrame(imp.transform(x.loc[testidx,columns]),columns=columns,index=testidx)
                meta['features'][kind]=columns;meta['medians'][kind]={c:float(v) for c,v in zip(columns,imp.statistics_)}
                for model in models:
                    tic=time.perf_counter();estimator=LEGACY.make_estimator(model,cfg)
                    if model=='CatBoost':estimator.set_params(allow_writing_files=False)
                    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                        warnings.simplefilter('always');estimator.fit(xt,y.loc[trainidx]);values=estimator.predict(xe)
                    pred[kind+'__'+model]=np.asarray(values,dtype=np.float64)
                    meta['warnings'] += [dict(kind=kind,model=model,message=str(w.message)) for w in caught]
                    meta[kind+'__'+model+'_seconds']=time.perf_counter()-tic
                    if phase=='confirmation' and kind=='interstation':
                        for steps in [1,2]:
                            delayed=x.copy()
                            for c in columns:
                                if any(c.startswith(s+'_') for s in donors):delayed[c]=x[c].shift(steps)
                            xd=pd.DataFrame(imp.transform(delayed.loc[testidx,columns]),columns=columns,index=testidx)
                            pred[f'delay{steps*15}__{model}']=np.asarray(estimator.predict(xd),dtype=np.float64)
                    if window==21 and daystr in ['2026-03-31','2026-06-30']:
                        bundle=dict(columns=columns,imputer=imp,estimator=estimator,params=estimator.get_params(),train_index=trainidx,test_index=testidx,target=target,gap=gap)
                        path=out/'models'/f'{name}_{kind}_{model}.joblib';joblib.dump(bundle,path,compress=3)
                        np.testing.assert_array_equal(joblib.load(path)['estimator'].predict(xe),values)
        assert all(np.isfinite(pred[k+'__'+m]).all() for k in ['local','interstation'] for m in models)
        pred.to_csv(out/'checkpoints'/f'{name}.csv',index=False,float_format='%.17g')
        meta.update(status='completed',reused=reused,seconds=time.perf_counter()-start)
        marker.write_text(json.dumps(meta,indent=2),encoding='utf-8');return meta
    except Exception:
        record=dict(task=name,status='failed',error=traceback.format_exc())
        (out/'checkpoints'/f'{name}.error.json').write_text(json.dumps(record),encoding='utf-8');return record

def run(out,workers=12):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    out=Path(out);start=time.perf_counter();tasks=[]
    for target in STATIONS:
        tasks += [('exploratory',target,d.strftime('%Y-%m-%d'),w,g) for d in pd.date_range('2026-01-01','2026-04-01',inclusive='left') for w in [7,21,28] for g in GAPS]
        tasks += [('confirmation',target,d.strftime('%Y-%m-%d'),21,g) for d in pd.date_range('2026-04-01','2026-07-01',inclusive='left') for g in GAPS]
    results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(str(out),)) as pool:
        for future in as_completed([pool.submit(task,t) for t in tasks]):
            r=future.result();results.append(r)
            state=dict(done=len(results),total=len(tasks),failed=sum(x['status']=='failed' for x in results),skipped=sum(x['status']=='skipped' for x in results),reused=sum(x.get('reused',False) for x in results),seconds=time.perf_counter()-start,latest=r['task'])
            (out/'progress.json').write_text(json.dumps(state),encoding='utf-8')
            if len(results)%15==0 or r['status']=='failed':print(json.dumps(state),flush=True)
    assert not [r for r in results if r['status']=='failed']
    (out/'runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,workers=workers,tasks=len(tasks),completed=sum(r['status']=='completed' for r in results),skipped=sum(r['status']=='skipped' for r in results),reused=sum(r.get('reused',False) for r in results)),indent=2),encoding='utf-8')
