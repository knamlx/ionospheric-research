"""Post-training audit and reproducible paper analysis; never refits models."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,hashlib,itertools,time
import numpy as np
import pandas as pd
import joblib
from paths import experiment_directory
OUT=experiment_directory()
A=OUT/'final_analysis'
MODELS=['LinearRegression','ElasticNet','RandomForest','XGBoost','CatBoost']
STATIONS=['EB040','DB049','PQ052'];GAPS=[0,15,30,60,120]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def raw_data_audit():
    import math
    panel=pd.read_pickle(OUT/'data/panel.pkl');sources=json.loads((OUT/'data/sources.json').read_text(encoding='utf-8'))
    summaries=[]
    for station in STATIONS:
        rows=[];raw=accepted=bad_quality=bad_value=malformed=0
        for entry in sources:
            path=Path(entry['path'])
            if f'station={station}' not in str(path):continue
            assert digest(path)==entry['sha256']
            text=path.read_text(encoding='utf-8-sig')
            assert 'URSI-Code '+station in text and '# foF2[MHz]' in text
            for line in text.splitlines():
                if not line or line.startswith('#'):continue
                raw+=1
                try:
                    parts=line.split(maxsplit=3);cs=float(parts[1]);v=float(parts[2]);timestamp=pd.Timestamp(parts[0])
                except (ValueError,IndexError):malformed+=1;continue
                if not(cs>=50 or cs==-1 or math.isnan(cs)):bad_quality+=1;continue
                if not(0<v<=30):bad_value+=1;continue
                rows.append((timestamp,v));accepted+=1
        f=pd.DataFrame(rows,columns=['time','foF2']);f.time=pd.to_datetime(f.time,utc=True)
        duplicates=int(f.time.duplicated().sum())
        # pandas stable sort retains archive ordering for duplicate observations.
        s=f.sort_values('time',kind='stable').drop_duplicates('time',keep='last').set_index('time').foF2.resample('15min').mean().reindex(panel.index)
        np.testing.assert_allclose(s,panel[station],rtol=0,atol=1e-12,equal_nan=True)
        summaries.append(dict(station=station,raw_rows=raw,accepted_rows=accepted,rejected_quality=bad_quality,rejected_value=bad_value,missing_or_unparseable_rows=malformed,duplicate_timestamps=duplicates,valid_15min_bins=int(s.notna().sum()),raw_to_panel_equal=True))
    pd.DataFrame(summaries).to_csv(A/'raw_data_checks.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

def parameter_audit():
    import papers_engine as engine
    cfg=json.loads((OUT/'model_config.json').read_text(encoding='utf-8'))
    assert not cfg['automl']['enabled'] and cfg['automl']['n_trials']==0 and not cfg['hyperparameter_spaces']
    expected={}
    for model in MODELS:
        e=engine.LEGACY.make_estimator(model,cfg)
        if model=='CatBoost':e.set_params(allow_writing_files=False)
        expected[model]=e.get_params()
    rows=[]
    for p in sorted((OUT/'models').glob('*.joblib')):
        model=next(m for m in MODELS if p.stem.endswith('_'+m));bundle=joblib.load(p)
        actual=bundle['estimator'].get_params()
        assert json.dumps(actual,sort_keys=True,default=str)==json.dumps(expected[model],sort_keys=True,default=str),p
        rows.append(dict(file=p.name,model=model,fixed_source_parameters_equal=True))
    assert len(rows)==300
    pd.DataFrame(rows).to_csv(A/'parameter_checks.csv',index=False)
    write_json(A/'effective_training_protocol.json',dict(archived_data_start='2025-01-01',archived_data_end_exclusive='2026-07-01',exploratory=['2026-01-01','2026-04-01'],confirmation=['2026-04-01','2026-07-01'],rolling_windows_exploratory=[7,21,28],rolling_window_confirmation=21,models=MODELS,parameter_search=False,parameters=expected,minimum_train_rows=48,minimum_test_rows=2,minimum_train_feature_coverage=.3,imputation='median fitted only on training window',target='next 15-minute foF2 mean',note='model_config retains inherited historical path/date labels; actual effective dates are verified against every checkpoint and listed here'))
    print('FIXED PARAMETER CHECKS PASSED',len(rows),flush=True)

def audit():
    import papers_engine as engine
    panel=pd.read_pickle(OUT/'data/panel.pkl')
    assert panel.index.is_unique and panel.index.tz is not None
    assert (panel.index.to_series().diff().dropna()==pd.Timedelta('15min')).all()
    assert ((panel>0)&(panel<=30)|panel.isna()).all().all()
    sources=json.loads((OUT/'data/sources.json').read_text(encoding='utf-8'))
    for s in sources:assert digest(s['path'])==s['sha256'],s['path']
    expected=set()
    for target in STATIONS:
        for phase,start,end,windows in [('exploratory','2026-01-01','2026-04-01',[7,21,28]),('confirmation','2026-04-01','2026-07-01',[21])]:
            for d,w,g in itertools.product(pd.date_range(start,end,inclusive='left'),windows,GAPS):
                expected.add(f'{phase}_{target}_{d:%Y-%m-%d}_w{w}_g{g}')
    paths=sorted((OUT/'checkpoints').glob('*.json'))
    assert {p.stem for p in paths}==expected
    records=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    assert all(m['status'] in ['completed','skipped'] for m in records)
    frames=[];warnings=[];model_replays=[];input_hashes={}
    for p in paths:input_hashes[str(p)]=digest(p)
    fc=json.loads((OUT/'source/feature_engineering.json').read_text(encoding='utf-8'))
    checked=0;replays=0;causal=[]
    for target in STATIONS:
        ds=joblib.load(OUT/f'data/datasets_{target}.joblib')
        for gap in GAPS:
            fresh=engine.feature_frames(panel,target,gap,fc)
            for k in [0,1]:pd.testing.assert_frame_equal(fresh[k],ds[gap][k])
            # Changing unavailable local data and all future donor data cannot affect origin features.
            origin=pd.Timestamp('2026-05-15 12:00',tz='UTC')
            changed=panel.copy()
            changed.loc[changed.index>origin-pd.Timedelta(minutes=gap),target]=10000
            changed.loc[changed.index>origin,[s for s in STATIONS if s!=target]]=10000
            mutated=engine.feature_frames(changed,target,gap,fc)
            for k in [0,1]:pd.testing.assert_series_equal(mutated[k].loc[origin],ds[gap][k].loc[origin])
            causal.append(dict(target=target,gap=gap,passed=True))
        eligible=pd.Series(True,index=panel.index)
        for g in [0,30,60]:eligible &= ds[g][0].last_available.notna()
        allg=pd.Series(True,index=panel.index)
        for g in GAPS:allg &= ds[g][0].last_available.notna()
        y=panel[target].shift(-1);donors=[s for s in STATIONS if s!=target]
        cache={}
        for m in [r for r in records if r['target']==target]:
            day=pd.Timestamp(m['day'],tz='UTC');w=m['window'];g=m['gap']
            end=pd.Timestamp('2026-04-01' if m['phase']=='exploratory' else '2026-07-01',tz='UTC')
            key=(m['day'],w)
            if key not in cache:
                tr=panel.index[(panel.index>=day-pd.Timedelta(days=w))&(panel.index<=day-pd.Timedelta('30min'))&y.notna()&eligible]
                te=panel.index[(panel.index>=day)&(panel.index<day+pd.Timedelta('1D'))&(panel.index+pd.Timedelta('30min')<=end)&y.notna()&eligible]
                cache[key]=(tr,te)
            tr,te=cache[key]
            assert len(tr)==m['n_train'] and len(te)==m['n_test'],m['task']
            reason='insufficient_train' if len(tr)<48 else 'insufficient_test' if len(te)<2 else 'constant_train_target' if y.loc[tr].nunique()<2 else None
            if m['status']=='skipped':assert m['reason']==reason;continue
            assert reason is None and (tr+pd.Timedelta('30min')).max()<=day
            assert str(tr.min())==m['train_start'] and str(tr.max())==m['train_end']
            p=OUT/'checkpoints'/f"{m['task']}.csv";input_hashes[str(p)]=digest(p)
            f=pd.read_csv(p,parse_dates=['time_utc'],float_precision='round_trip')
            assert pd.DatetimeIndex(f.time_utc).equals(te) and not f.time_utc.duplicated().any()
            np.testing.assert_allclose(f.actual,y.loc[te],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(f.shared,panel[donors].notna().all(axis=1).loc[te])
            np.testing.assert_array_equal(f.all_gaps_history,allg.loc[te])
            delayok=pd.Series(True,index=panel.index)
            for step in [0,1,2]:delayok &= panel[donors].shift(step).notna().all(axis=1)
            np.testing.assert_array_equal(f.shared_delays,delayok.loc[te])
            np.testing.assert_allclose(f.persistence,ds[g][0].last_available.loc[te],rtol=0,atol=1e-12)
            for kind,x in [('local',ds[g][0]),('interstation',ds[g][1])]:
                cols=list(x.columns[x.loc[tr].notna().mean()>=.3]);assert cols==m['features'][kind]
                med=x.loc[tr,cols].median().fillna(0)
                np.testing.assert_allclose([m['medians'][kind][c] for c in cols],med,rtol=1e-12,atol=1e-12)
                for model in MODELS:
                    c=kind+'__'+model;assert np.isfinite(f[c]).all()
                    bundlepath=OUT/'models'/f"{m['task']}_{kind}_{model}.joblib"
                    if bundlepath.exists():
                        bundle=joblib.load(bundlepath);assert bundle['columns']==cols
                        assert pd.DatetimeIndex(bundle['train_index']).equals(tr)
                        xe=pd.DataFrame(bundle['imputer'].transform(x.loc[te,cols]),columns=cols,index=te)
                        actual=np.asarray(bundle['estimator'].predict(xe),dtype=float)
                        diff=float(np.max(np.abs(actual-f[c].to_numpy())))
                        assert diff<1e-11,(bundlepath,diff)
                        for steps in ([1,2] if m['phase']=='confirmation' and kind=='interstation' else []):
                            xd=x.loc[:,cols].copy()
                            for col in cols:
                                if any(col.startswith(s+'_') for s in donors):xd[col]=x[col].shift(steps)
                            xe2=pd.DataFrame(bundle['imputer'].transform(xd.loc[te]),columns=cols,index=te)
                            np.testing.assert_allclose(bundle['estimator'].predict(xe2),f[f'delay{steps*15}__{model}'],rtol=0,atol=1e-11)
                        model_replays.append(dict(file=bundlepath.name,max_absolute_difference=diff));replays+=1
            if m['phase']=='confirmation':
                assert all(np.isfinite(f[f'delay{d}__{model}']).all() for d in [15,30] for model in MODELS)
            for warning in m['warnings']:warnings.append(dict(task=m['task'],**warning))
            for k in ['phase','target','day','window','gap']:f[k]=m[k]
            frames.append(f);checked+=1
            if checked%300==0:print('AUDIT',checked,'completed splits; replayed models',replays,flush=True)
        print('AUDIT TARGET FINISHED',target,flush=True)
    data=pd.concat(frames,ignore_index=True)
    # Some windows can fail minimum train count. Compare on the intersection,
    # preserving raw rows and reporting the removed origin count explicitly.
    matching=[]
    data['matched_windows']=False
    for (phase,target),part in data.groupby(['phase','target']):
        sets=[set(zip(f.time_utc,f.actual)) for _,f in part.groupby(['window','gap'])]
        common=set.intersection(*sets)
        data.loc[part.index,'matched_windows']=[(t,y) in common for t,y in zip(part.time_utc,part.actual)]
        matching.append(dict(phase=phase,target=target,common_origins=len(common),union_origins=len(set.union(*sets)),excluded_origins=len(set.union(*sets))-len(common)))
    pd.DataFrame(matching).to_csv(A/'window_matching.csv',index=False)
    data.to_pickle(A/'predictions.pkl')
    pd.DataFrame(records).drop(columns=['features','medians','warnings'],errors='ignore').to_csv(A/'split_audit.csv',index=False)
    pd.DataFrame(warnings,columns=['task','kind','model','message']).to_csv(A/'warnings.csv',index=False)
    skips=pd.DataFrame([m for m in records if m['status']=='skipped'])
    skips.groupby(['phase','target','reason']).agg(tasks=('task','size'),days=('day','nunique')).reset_index().to_csv(A/'skip_summary.csv',index=False)
    pd.DataFrame(model_replays).to_csv(A/'model_replay.csv',index=False)
    pd.DataFrame(causal).to_csv(A/'causality_checks.csv',index=False)
    write_json(A/'input_hashes.json',input_hashes)
    result=dict(passed=True,expected_tasks=len(expected),completed=checked,skipped=len(skips),errors=0,source_hashes_verified=len(sources),model_replays=replays,warnings=len(warnings),causal_checks=len(causal),train_medians_and_feature_selection='every completed split',matching_timestamps='intersection of all windows and gaps; excluded origins reported',prediction_rows=len(data),created=pd.Timestamp.now().isoformat())
    write_json(A/'checks.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def bootstrap(days,local,multi,block,reps=2000,seed=42):
    """Circular moving calendar-day blocks, paired sums/counts; retain absent days."""
    date=pd.DatetimeIndex(pd.to_datetime(days,utc=True)).normalize()
    f=pd.DataFrame({'day':date,'l':local,'m':multi})
    daily=f.groupby('day').agg(l=('l','sum'),m=('m','sum'),n=('l','size'))
    phase_start=pd.Timestamp('2026-01-01' if date.min().month<4 else '2026-04-01',tz='UTC')
    phase_end=pd.Timestamp('2026-04-01' if date.min().month<4 else '2026-07-01',tz='UTC')
    grid=pd.date_range(phase_start,phase_end,freq='D',inclusive='left')
    arr=daily.reindex(grid,fill_value=0)[['l','m','n']].to_numpy();n=len(arr)
    rng=np.random.default_rng(seed);starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    sums=arr[ix].sum(axis=1);valid=sums[:,2]>0;sums=sums[valid]
    gain=(sums[:,0]-sums[:,1])/sums[:,2]
    pct=100*(sums[:,0]-sums[:,1])/sums[:,0]
    return dict(ci_low=np.quantile(gain,.025),ci_high=np.quantile(gain,.975),simultaneous_low=np.quantile(gain,.05/6),simultaneous_high=np.quantile(gain,1-.05/6),gain_pct_low=np.quantile(pct,.025),gain_pct_high=np.quantile(pct,.975),valid_repetitions=len(gain))

def statistics():
    data=pd.read_pickle(A/'predictions.pkl');rows=[];daily=[];effects=[];coverage=[];delayrows=[];outagerows=[];monthly=[]
    tests=[]
    dates=pd.DatetimeIndex(['2026-01-01','2026-01-10','2026-03-31'],tz='UTC')
    for block in [1,3,7]:
        z=bootstrap(dates,np.array([2.,2.,2.]),np.array([1.,1.,1.]),block)
        assert z['ci_low']==z['ci_high']==1
        z=bootstrap(dates,np.array([2.,1.,3.]),np.array([2.,1.,3.]),block)
        assert z['ci_low']==z['ci_high']==0
        tests.append(dict(block_days=block,known_constant_effect=True,identical_predictions_zero_interval=True,missing_calendar_days_retained=True))
    write_json(A/'bootstrap_checks.json',tests)
    if 'matched_windows' not in data:
        data['matched_windows']=False
        for _,part in data.groupby(['phase','target']):
            common=set.intersection(*[set(f.time_utc) for _,f in part.groupby(['window','gap'])])
            data.loc[part.index,'matched_windows']=part.time_utc.isin(common)
    for scope in ['shared','all_targets']:
        eligible=data.matched_windows & data.all_gaps_history & (data.shared if scope=='shared' else True)
        for keys,f in data[eligible].groupby(['phase','target','window','gap']):
            phase,target,window,gap=keys;meta=dict(scope=scope,phase=phase,target=target,window=window,gap=gap)
            ngrid=(90 if phase=='exploratory' else 91)*96-1
            coverage.append(dict(**meta,n=len(f),days=f.day.nunique(),grid=ngrid,coverage_percent=100*len(f)/ngrid))
            for model in MODELS+['Persistence']:
                kinds=['persistence'] if model=='Persistence' else ['local','interstation']
                for kind in kinds:
                    pred=f.persistence if kind=='persistence' else f[kind+'__'+model]
                    err=pred-f.actual
                    rows.append(dict(**meta,kind=kind,model=model,n=len(f),days=f.day.nunique(),mae=err.abs().mean(),rmse=np.sqrt(np.mean(err**2)),bias=err.mean()))
                    df=pd.DataFrame({'day':f.day,'ae':err.abs(),'se':err**2,'month':f.time_utc.dt.strftime('%Y-%m')})
                    for day,z in df.groupby('day'):daily.append(dict(**meta,kind=kind,model=model,day=day,n=len(z),mae=z.ae.mean()))
                    for month,z in df.groupby('month'):monthly.append(dict(**meta,kind=kind,model=model,month=month,n=len(z),mae=z.ae.mean(),rmse=np.sqrt(z.se.mean())))
                if model=='Persistence':continue
                l=(f['local__'+model]-f.actual).abs();m=(f['interstation__'+model]-f.actual).abs()
                for block in [1,3,7]:effects.append(dict(**meta,model=model,block_days=block,n=len(f),days=f.day.nunique(),local_mae=l.mean(),interstation_mae=m.mean(),gain_mhz=l.mean()-m.mean(),gain_percent=100*(l.mean()-m.mean())/l.mean(),**bootstrap(f.time_utc,l,m,block)))
            if phase=='confirmation':
                delayed=f[f.shared_delays]
                for model in MODELS:
                    for delay in [0,15,30]:
                        col='interstation__'+model if delay==0 else f'delay{delay}__{model}'
                        ae=(delayed[col]-delayed.actual).abs();lae=(delayed['local__'+model]-delayed.actual).abs()
                        delayrows.append(dict(**meta,model=model,delay_minutes=delay,n=len(delayed),days=delayed.day.nunique(),mae=ae.mean(),local_mae=lae.mean(),gain_percent=100*(lae.mean()-ae.mean())/lae.mean()))
                if gap in [15,30,60,120]:
                    # Origin is outage_start + elapsed - 15 min; target starts at elapsed.
                    elapsed=gap;start=f.time_utc-pd.Timedelta(minutes=elapsed-15)
                    mask=(start.dt.hour.isin([6,18]))&(start.dt.minute==0)
                    episode=f[mask].copy();episode['outage_start']=start[mask];episode['elapsed_minutes']=elapsed
                    outagerows.append(episode.assign(scope=scope))
    for name,records in [('metrics',rows),('daily_metrics',daily),('effects',effects),('coverage',coverage),('delay_sensitivity',delayrows),('monthly_metrics',monthly)]:
        pd.DataFrame(records).to_csv(A/f'{name}.csv',index=False,float_format='%.17g')
    month=pd.DataFrame(monthly)
    month=month[month.kind.isin(['local','interstation'])].pivot(index=['scope','phase','target','window','gap','model','month'],columns='kind',values='mae').reset_index()
    month['gain_percent']=100*(month.local-month.interstation)/month.local
    month.to_csv(A/'monthly_effects.csv',index=False,float_format='%.17g')
    episodes=pd.concat(outagerows,ignore_index=True)
    # Strict paired four-checkpoint episodes, same dates across all gaps/models.
    complete=episodes.groupby(['scope','target','outage_start']).elapsed_minutes.transform('nunique')==4
    episodes=episodes[complete].copy();episodes.to_pickle(A/'continuous_outages.pkl')
    episode_metrics=[];episode_effects=[]
    for keys,f in episodes.groupby(['scope','target','elapsed_minutes']):
        scope,target,elapsed=keys
        meta=dict(scope=scope,target=target,elapsed_minutes=elapsed,episodes=len(f),days=f.day.nunique())
        for model in MODELS:
            l=(f['local__'+model]-f.actual).abs();m=(f['interstation__'+model]-f.actual).abs()
            episode_metrics.append(dict(**meta,model=model,local_mae=l.mean(),interstation_mae=m.mean(),gain_percent=100*(l.mean()-m.mean())/l.mean()))
            for block in [1,3,7]:episode_effects.append(dict(**meta,model=model,block_days=block,gain_mhz=l.mean()-m.mean(),**bootstrap(f.time_utc,l,m,block)))
    pd.DataFrame(episode_metrics).to_csv(A/'outage_metrics.csv',index=False)
    pd.DataFrame(episode_effects).to_csv(A/'outage_effects.csv',index=False)
    e=pd.DataFrame(effects)
    primary=e[(e.phase=='confirmation')&(e.scope=='shared')&(((e.model=='LinearRegression')&(e.gap==0))|((e.model=='CatBoost')&(e.gap==60)))].copy()
    primary['article']=np.where(primary.gap==0,1,2);primary.to_csv(A/'primary_confirmation.csv',index=False)
    print('STATISTICS FINISHED',len(rows),'metrics;',len(effects),'bootstrap comparisons;',len(episodes),'outage checkpoints',flush=True)

def figures():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager,ticker,dates as mdates
    from matplotlib.text import Text
    mpl.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.labelsize':14,'axes.titlesize':14,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,'figure.titlesize':14,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'axes.spines.top':True,'axes.spines.right':True})
    dest=A/'figures';dest.mkdir(exist_ok=True);audits=[];captions=[]
    colors=['#0072B2','#D55E00'];abbr=['LR','EN','RF','XGB','CB']
    markers=['o','s','^','D','v'];line_styles=['-','--','-.',':',(0,(5,2,1,2))]
    daily=pd.read_csv(A/'daily_metrics.csv');effects=pd.read_csv(A/'effects.csv');metrics=pd.read_csv(A/'metrics.csv')
    def style(ax):
        ax.set_axisbelow(True);ax.grid(axis='y',color='#d5d5d5',lw=.6)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5));ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2));ax.grid(axis='y',which='minor',color='#efefef',lw=.4)
        precision=3 if max(abs(v) for v in ax.get_ylim())<.05 else 2
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,p:format(v,f'.{precision}f').replace('.',',')))
    def save(fig,name,caption):
        fig.canvas.draw()
        texts=[t for t in fig.findobj(Text) if t.get_text()]
        assert all(abs(t.get_fontsize()-14)<.01 for t in texts)
        assert all(t.get_fontproperties().get_name()=='Times New Roman' for t in texts)
        for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=600)
        audits.append(dict(name=name,text_items=len(texts),font='Times New Roman',points=14,width_cm=fig.get_figwidth()*2.54,height_cm=fig.get_figheight()*2.54))
        captions.append(f'{name}: {caption}');plt.close(fig)
    # Three vertically stacked panels preserve 14pt at manuscript width.
    for phase in ['exploratory','confirmation']:
        for gap in GAPS:
            f=daily[(daily.scope=='shared')&(daily.phase==phase)&(daily.window==21)&(daily.gap==gap)]
            fig,axes=plt.subplots(3,1,figsize=(19/2.54,24/2.54),layout='constrained')
            for j,(ax,target) in enumerate(zip(axes,STATIONS)):
                for ki,kind in enumerate(['local','interstation']):
                    z=f[(f.target==target)&(f.kind==kind)]
                    arrays=[z[z.model==model].mae.to_numpy() for model in MODELS]
                    box=ax.boxplot(arrays,positions=np.arange(5)+(ki-.5)*.3,widths=.26,patch_artist=True,manage_ticks=False,flierprops=dict(marker='+',markersize=4,markeredgecolor=colors[ki]),medianprops=dict(color='black',lw=1.1))
                    for patch in box['boxes']:patch.set_facecolor(colors[ki]);patch.set_alpha(.5);patch.set_hatch(['///','..'][ki])
                ax.set_xticks(range(5),abbr);ax.set_ylabel('MAE, МГц');ax.set_title(f'{"абв"[j]}) {target}');style(ax)
            handles=[mpl.patches.Patch(facecolor=colors[k],alpha=.5,hatch=['///','..'][k],label=t) for k,t in enumerate(['Локальная','Межстанционная'])]
            fig.legend(handles=handles,loc='outside upper center',ncol=2)
            save(fig,f'box_{phase}_w21_g{gap}',f'Распределения дневной MAE; {phase}, окно 21 сутки, отсутствие локальных данных {gap} мин. Одинаковые моменты для моделей, окон и пропусков; свежие данные обеих дополнительных станций. Усики 1,5 IQR, выбросы +. Дни с малым числом точек имеют такой же вес в boxplot, как остальные дни; основные MAE взвешены по точкам.')
    # Primary contrasts with sensitivity to calendar block length.
    primary=pd.read_csv(A/'primary_confirmation.csv')
    for article in [1,2]:
        f=primary[primary.article==article];fig,ax=plt.subplots(figsize=(19/2.54,11/2.54),layout='constrained')
        for k,block in enumerate([1,3,7]):
            z=f[f.block_days==block].set_index('target').reindex(STATIONS);pos=np.arange(3)+(k-1)*.17
            ax.errorbar(pos,z.gain_mhz,yerr=[z.gain_mhz-z.simultaneous_low,z.simultaneous_high-z.gain_mhz],fmt=markers[k],capsize=4,label=f'Блок {block} сут.')
        ax.axhline(0,color='black',ls='--',lw=1);ax.set_xticks(range(3),STATIONS);ax.set_ylabel('Снижение MAE, МГц');style(ax);ax.legend(loc='best')
        save(fig,f'primary_article{article}',f'Подтверждающая выборка апрель–июнь 2026; статья {article}. Положительный эффект означает меньшую MAE межстанционной модели. Парный круговой bootstrap календарных блоков 1/3/7 суток, 2000 повторов. Интервалы 98,33% для каждого из трёх контрастов, номинальное семейное покрытие 95% по Бонферрони; малое число блоков и нестационарность ограничивают интерпретацию.')
    # Gain versus gap, all five models, station-specific.
    f=effects[(effects.phase=='confirmation')&(effects.scope=='shared')&(effects.block_days==7)]
    fig,axes=plt.subplots(3,1,figsize=(19/2.54,25/2.54),layout='constrained')
    for j,(ax,target) in enumerate(zip(axes,STATIONS)):
        for mi,(model,label) in enumerate(zip(MODELS,abbr)):
            z=f[(f.target==target)&(f.model==model)].sort_values('gap');ax.plot(z.gap,z.gain_percent,marker=markers[mi],linestyle=line_styles[mi],markersize=6,label=label)
        ax.axhline(0,color='black',lw=.8,ls='--');ax.set_title(f'{"абв"[j]}) {target}');ax.set_xticks(GAPS);ax.set_ylabel('Снижение MAE, %');style(ax)
    axes[-1].set_xlabel('Нет локальных данных, мин');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=5)
    save(fig,'confirmation_gain_all_models','Снижение MAE относительно локальной модели того же алгоритма, апрель–июнь, окно 21 сутки. Общая выборка пяти пропусков; свежие дополнительные наблюдения. Линии соединяют отдельные сценарии, а не непрерывную оценку зависимости.')
    # Explicit paired continuous episodes.
    om=pd.read_csv(A/'outage_metrics.csv');fig,axes=plt.subplots(3,1,figsize=(19/2.54,24/2.54),layout='constrained')
    for j,(ax,target) in enumerate(zip(axes,STATIONS)):
        z=om[(om.scope=='shared')&(om.target==target)&(om.model=='CatBoost')].sort_values('elapsed_minutes')
        for k,(col,label) in enumerate([('local_mae','Локальная'),('interstation_mae','Межстанционная')]):ax.plot(z.elapsed_minutes,z[col],marker=['s','^'][k],linestyle=['--','-'][k],markersize=6,color=colors[k],label=label)
        ax.set_title(f'{"абв"[j]}) {target}, эпизодов: {int(z.episodes.iloc[0]) if len(z) else 0}');ax.set_xticks([15,30,60,120]);ax.set_ylabel('MAE, МГц');style(ax)
    axes[-1].set_xlabel('От начала отключения, мин');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    save(fig,'continuous_outages_catboost','CatBoost, окно 21 сутки; непрерывные синтетические отключения на 120 мин с началом 06:00 и 18:00 UTC в апреле–июне. Одинаковые полные эпизоды на всех четырёх контрольных отметках. Последняя доступная локальная отметка фиксирована перед отключением; прогнозируемый интервал начинается на указанной минуте. Используются заранее обученные модели соответствующего пропуска, переключаемые по длительности отключения.')
    delay=pd.read_csv(A/'delay_sensitivity.csv');fig,axes=plt.subplots(3,1,figsize=(19/2.54,24/2.54),layout='constrained')
    for j,(ax,target) in enumerate(zip(axes,STATIONS)):
        z=delay[(delay.scope=='shared')&(delay.target==target)&(delay.model=='CatBoost')&(delay.gap==60)].sort_values('delay_minutes')
        ax.plot(z.delay_minutes,z.mae,marker='^',linestyle='-',markersize=6,color=colors[1],label='Межстанционная');ax.plot(z.delay_minutes,np.full(len(z),z.local_mae.iloc[0]),marker='s',linestyle='--',markersize=6,color=colors[0],label='Локальная');ax.set_xticks([0,15,30]);ax.set_title(f'{"абв"[j]}) {target}, n = {int(z.n.iloc[0])}');ax.set_ylabel('MAE, МГц');style(ax)
    axes[-1].set_xlabel('Задержка дополнительных данных, мин');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles[::-1],labels[::-1],loc='outside upper center',ncol=2)
    save(fig,'donor_delay_catboost_g60','Входная устойчивость CatBoost к задержке дополнительных данных 0/15/30 мин, апрель–июнь, локальный пропуск 60 мин. Одна фиксированная модель, обученная без задержки; сдвигаются только дополнительные признаки, переобучения нет. Общая выборка с наблюдениями обоих доноров при всех задержках. Пунктир — локальная MAE.')
    # Coverage is unsmoothed and without imputation.
    panel=pd.read_pickle(OUT/'data/panel.pkl');cov=panel.notna().resample('MS').mean().mul(100).T
    fig,ax=plt.subplots(figsize=(26/2.54,8/2.54),layout='constrained');im=ax.imshow(cov,aspect='auto',vmin=0,vmax=100,cmap='YlGnBu')
    ax.set_yticks(range(3),cov.index);ax.set_xticks(range(cov.shape[1]),[d.strftime('%m.%y') for d in cov.columns],rotation=60);fig.colorbar(im,ax=ax,label='Покрытие, %')
    save(fig,'coverage_2025_2026','Покрытие 15-минутной сетки после фильтра качества, январь 2025 — июнь 2026. Пропуски наблюдений не восстанавливаются. Архив 2025 обеспечивает историю; ежедневные модели используют только последние 7/21/28 суток.')
    # Exploratory window sensitivity, no retrospective best-window selection.
    f=effects[(effects.phase=='exploratory')&(effects.scope=='shared')&(effects.block_days==7)&(effects.gap==60)]
    fig,axes=plt.subplots(3,1,figsize=(19/2.54,24/2.54),layout='constrained')
    for j,(ax,target) in enumerate(zip(axes,STATIONS)):
        for mi,(model,label) in enumerate(zip(MODELS,abbr)):
            z=f[(f.target==target)&(f.model==model)].sort_values('window');ax.plot(z.window,z.gain_percent,marker=markers[mi],linestyle=line_styles[mi],markersize=6,label=label)
        ax.set_xticks([7,21,28]);ax.axhline(0,color='black',ls='--',lw=.8);ax.set_title(f'{"абв"[j]}) {target}');ax.set_ylabel('Снижение MAE, %');style(ax)
    axes[-1].set_xlabel('Окно обучения, суток');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=5)
    save(fig,'exploratory_window_sensitivity_g60','Чувствительность к длине окна 7/21/28 суток, январь–март, локальный пропуск 60 мин. Все пять алгоритмов; общие моменты сравнения. Подтверждающий период использует фиксированное окно 21 сутки.')
    # A predetermined date: never select a visually favourable example.
    data=pd.read_pickle(A/'predictions.pkl');fig,axes=plt.subplots(3,1,figsize=(23/2.54,25/2.54),layout='constrained')
    for j,(ax,target) in enumerate(zip(axes,STATIONS)):
        z=data[(data.phase=='confirmation')&(data.target==target)&(data.window==21)&(data.gap==60)&(data.day=='2026-05-15')].copy()
        idx=pd.date_range('2026-05-15',periods=96,freq='15min',tz='UTC');z=z.set_index('time_utc').reindex(idx)
        ax.plot(z.index,z.actual,color='black',marker='o',markerfacecolor='white',markersize=5,markevery=6,label='Наблюдения')
        for k,kind in enumerate(['local','interstation']):ax.plot(z.index,z[kind+'__CatBoost'],color=colors[k],marker=['s','^'][k],linestyle=['--','-.'][k],markersize=5,markevery=6,label=['Локальная','Межстанционная'][k],alpha=.85)
        ax.set_title(f'{"абв"[j]}) {target}');ax.set_ylabel('foF2, МГц');ax.xaxis.set_major_locator(mdates.HourLocator(interval=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));style(ax)
    axes[-1].set_xlabel('15.05.2026, UTC, начало входного интервала');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=3)
    save(fig,'forecast_example_20260515','Пример CatBoost на заранее фиксированной календарной дате 15 мая 2026, окно 21 сутки и локальный пропуск 60 мин; прогноз на следующий 15-минутный интервал. Дата не отбиралась по качеству; пропущенные моменты не соединяются. Это отдельные прогнозы скользящего сценария, а не двухчасовой непрерывный эпизод.')
    write_json(A/'font_and_export_audit.json',audits)
    (A/'captions.md').write_text('\n\n'.join(captions),encoding='utf-8')
    print('FIGURES FINISHED',len(audits),'figures /',len(audits)*3,'exports',flush=True)

def report():
    checks=json.loads((A/'checks.json').read_text(encoding='utf-8'));primary=pd.read_csv(A/'primary_confirmation.csv')
    data=pd.read_pickle(A/'predictions.pkl');diagnostic=[]
    for keys,f in data[data.matched_windows & data.all_gaps_history].groupby(['phase','target','window','gap']):
        for kind,model in itertools.product(['local','interstation'],MODELS):
            p=f[kind+'__'+model];ae=(p-f.actual).abs()
            diagnostic.append(dict(phase=keys[0],target=keys[1],window=keys[2],gap=keys[3],kind=kind,model=model,n=len(f),negative_predictions=int((p<0).sum()),above30_predictions=int((p>30).sum()),min_prediction=p.min(),max_prediction=p.max(),median_absolute_error=ae.median(),p95_absolute_error=ae.quantile(.95),max_absolute_error=ae.max()))
    diagnostic=pd.DataFrame(diagnostic);diagnostic.to_csv(A/'prediction_diagnostics.csv',index=False)
    lines=['# Итоговые проверки и анализ двух статей','',f"Проверено {checks['expected_tasks']} заданий: выполнено {checks['completed']}, пропущено {checks['skipped']}, ошибок 0. Повторно воспроизведено {checks['model_replays']} сохранённых моделей. Признаки и медианы проверены во всех выполненных разбиениях.",'','## Протокол','', 'Цель — среднее foF2 следующего 15-минутного интервала. Входная временная отметка означает начало последнего завершённого интервала. Прогноз выпускается через 15 мин после этой отметки; целевой интервал заканчивается ещё через 15 мин. Метки train полностью доступны к полуночи тестового дня. Ежедневное переобучение без подбора гиперпараметров, пять алгоритмов IonoAutoML с фиксированными параметрами. Архив весь 2025 год, анализ январь–март 2026 с окнами 7/21/28 суток; независимая по ошибкам подтверждающая выборка апрель–июнь 2026, окно 21 сутки. В подтверждающем периоде train ежедневно обновляется на прошлых наблюдениях, включая предыдущие дни 2026; это последовательная проверка, а не замороженная модель 2025 года.','', 'Три цели EB040/DB049/PQ052, остальные две станции — фиксированные доноры. Проверяются пропуски 0/15/30/60/120 мин. Выборка shared требует свежих измерений обоих доноров; all_targets допускает заполнение дополнительных признаков медианами train. Оба варианта требуют одинаковой доступной локальной истории для всех пяти пропусков. Покрытие обязательно оценивается отдельно от MAE.','', '## Подтверждающие контрасты','', 'Положительное снижение MAE означает пользу дополнительных станций. Ниже интервалы с поправкой на три станции в пределах каждой статьи, круговой bootstrap календарных блоков 7 суток, 2000 повторов. Интервалы описательные: короткий период, нестационарность и небольшое число блоков ограничивают точность покрытия.']
    for article in [1,2]:
        lines+=['',f'### Статья {article}','']
        for _,r in primary[(primary.article==article)&(primary.block_days==7)].iterrows():
            verdict='интервал выше нуля' if r.simultaneous_low>0 else 'интервал ниже нуля' if r.simultaneous_high<0 else 'интервал включает ноль'
            lines.append(f"- {r.target}, {r.model}, пропуск {int(r.gap)} мин: MAE {r.local_mae:.4f} → {r.interstation_mae:.4f} МГц; снижение {r.gain_percent:.2f}%; интервал эффекта [{r.simultaneous_low:.6f}; {r.simultaneous_high:.6f}] МГц — {verdict}; {int(r.n)} точек за {int(r.days)} суток.")
    lines+=['','При полной локальной истории скорректированный семисуточный интервал выше нуля только у PQ052; у EB040 и DB049 он включает ноль. При пропуске 60 мин CatBoost показывает положительный скорректированный интервал на всех трёх станциях. Поэтому вывод о компенсации недостатка локальной истории сильнее, чем вывод об универсальном улучшении обычного прогноза.','',f"Диагностика выходов на all_targets: отрицательных прогнозов {diagnostic.negative_predictions.sum()}, выше 30 МГц {diagnostic.above30_predictions.sum()} (сумма по моделям, окнам и сценариям, не число уникальных моментов). Ограничение или обрезка прогнозов не применялись. Диапазоны прогнозов, 95-й процентиль и максимум абсолютной ошибки сохранены в prediction_diagnostics.csv.",'', '## Численные результаты дополнительных проверок','']
    outage=pd.read_csv(A/'outage_metrics.csv');delay=pd.read_csv(A/'delay_sensitivity.csv');metric=pd.read_csv(A/'metrics.csv');month=pd.read_csv(A/'monthly_effects.csv')
    for target in STATIONS:
        e=outage[(outage.scope=='shared')&(outage.target==target)&(outage.model=='CatBoost')].set_index('elapsed_minutes')
        d=delay[(delay.scope=='shared')&(delay.target==target)&(delay.model=='CatBoost')&(delay.gap==60)].set_index('delay_minutes')
        q=metric[(metric.scope=='shared')&(metric.phase=='confirmation')&(metric.target==target)&(metric.model=='CatBoost')]
        local_full=q[(q.gap==0)&(q.kind=='local')].mae.iloc[0];remote_gap=q[(q.gap==60)&(q.kind=='interstation')].mae.iloc[0]
        monthly=month[(month.scope=='shared')&(month.phase=='confirmation')&(month.target==target)&(month.model=='CatBoost')&(month.gap==60)].sort_values('month')
        values='; '.join(f'{r.month}: {r.gain_percent:.1f}%' for _,r in monthly.iterrows())
        lines.append(f"- {target}: полных эпизодов со свежими донорами {int(e.loc[60,'episodes'])}; снижение MAE CatBoost на отметках 60/120 мин — {e.loc[60,'gain_percent']:.1f}% / {e.loc[120,'gain_percent']:.1f}%. При задержке доноров на 30 мин снижение относительно локальной модели остаётся {d.loc[30,'gain_percent']:.1f}% на общей для задержек выборке n={int(d.loc[30,'n'])}. Месячные эффекты при пропуске 60 мин: {values}. Межстанционная MAE при 60 мин {remote_gap:.4f} МГц против локальной MAE без пропуска {local_full:.4f} МГц: добавление станций не восстанавливает точность полной локальной истории.")
    skips=pd.read_csv(A/'skip_summary.csv');lines+=['','## Пропуски заданий','',skips.to_string(index=False),'','Пропущенное задание не считается нулевой ошибкой и не входит в метрики. Причины восстановлены из исходной сетки для каждого задания. Число заданий отличается от числа уникальных дней: один день повторяется для нескольких окон и сценариев.','', '## Непрерывные отключения и задержки','', 'Синтетические эпизоды стартуют ежедневно в 06:00 и 18:00 UTC и длятся 120 мин. Проверка вмешательством заменяет все скрытые локальные значения и подтверждает неизменность входных признаков. Для каждого времени 15/30/60/120 мин используется модель соответствующего сценария; это переключение обученных моделей, не одна модель с замороженным состоянием. Для сравнения оставлены только эпизоды с четырьмя контрольными точками; реальные сбои связи таким экспериментом не подтверждаются. Метки за пределами доступного эпизода не восстанавливаются.', '', 'Задержки дополнительных входов 15/30 мин проверены без переобучения исходной модели. Для всех трёх задержек используется одна общая выборка. Это проверка устойчивости входов, не симуляция документированных задержек GIRO.', '', '## Ограничения','', '- Измерительное время в архиве не доказывает время фактической публикации; доступны лучшие архивные значения с возможными поздними исправлениями.','- Три региональные европейские станции и два квартала не доказывают глобальную переносимость или годовую устойчивость.','- Общая выборка может отличаться от режима эксплуатации; анализ покрытия и all_targets приведён отдельно.','- Статистические интервалы вторичных сравнений не скорректированы на все алгоритмы, окна и пропуски; нельзя объявлять каждый из них отдельным подтверждённым открытием.','- MAE/RMSE измеряют прогноз; DTW и фильтрация не подменяют эти метрики. На рисунках показаны реальные рассчитанные ошибки, не данные из образцов пользователя.','', '## Файлы','', 'metrics.csv, monthly_metrics.csv, daily_metrics.csv — точечные и дневные ошибки; effects.csv — парные интервалы блоков 1/3/7 суток; coverage.csv — покрытие; primary_confirmation.csv — заранее фиксированные основные контрасты; continuous_outages.pkl, outage_metrics.csv, outage_effects.csv — полные эпизоды; delay_sensitivity.csv — задержки; split_audit.csv, skip_summary.csv, model_replay.csv, causality_checks.csv, checks.json — проверки; figures/ — PNG 600 dpi, PDF и SVG; captions.md — подписи.']
    (A/'ОТЧЁТ.md').write_text('\n'.join(lines),encoding='utf-8')
    # Original inputs remain immutable.
    hashes=json.loads((A/'input_hashes.json').read_text(encoding='utf-8'))
    assert all(digest(p)==h for p,h in hashes.items())
    write_json(A/'output_hashes.json',{str(p.relative_to(A)):digest(p) for p in A.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
    print('\n'.join(lines[:22]),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=[name for name, value in globals().items() if callable(value) and getattr(value, '__module__', None)==__name__ and name not in ['digest','write_json']])
    args=parser.parse_args()
    if not OUT.is_dir(): parser.error('Set IONOSPHERIC_EXPERIMENT to a local experiment copy containing data, models and checkpoints')
    A.mkdir(exist_ok=True)
    globals()[args.action]()
