"""Second-round S6E10 experiments, with fold-safe target encoding.

Feature ideas are credited in SOURCES.md. Public notebooks are read as references,
never executed. Competition validation labels do not enter any feature fit.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from train import read_data, encode_target, validate_submission, COMPETITION

ROOT = Path(__file__).resolve().parent
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
NUM = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
CACHE = ROOT / 'experiments' / 'base.joblib'


def emit(value):
    print(json.dumps(value), flush=True)


def prepare():
    started = time.time()
    CACHE.parent.mkdir(exist_ok=True)
    tr, ts, sample, raw, y = read_data(ROOT / 'data')
    og = pd.read_csv(ROOT / 'data-original' / 'data.csv')
    # Remove exact feature matches from the public source to avoid memorization.
    competition_hash = pd.util.hash_pandas_object(pd.concat([tr[raw], ts[raw]]), index=False)
    overlap = pd.util.hash_pandas_object(og[raw], index=False).isin(competition_hash)
    og = og.loc[~overlap].reset_index(drop=True)
    oy = encode_target(og.satisfaction)
    emit({'stage': 'prepare', 'original_rows_used': len(og), 'original_matches_removed': int(overlap.sum())})
    full = pd.concat([tr[raw], ts[raw]], ignore_index=True)
    base = full.copy()
    for c in CATS:
        levels = sorted(set(full[c]) | set(og[c]))
        base[c] = pd.Categorical(base[c], categories=levels)
        og[c] = pd.Categorical(og[c], categories=levels)
    for c in raw:
        if c not in CATS:
            base[c] = base[c].astype('float32')
        base['count_' + c] = full[c].map(full[c].value_counts(dropna=False)).astype('float32')
        means = og.groupby(c, observed=True).satisfaction.mean()
        base['original_mean_' + c] = full[c].map(means).astype('float32').fillna(float(oy.mean()))
    for c in NUM:
        for modulus in (10, 100):
            base[f'digit_{c}_{modulus}'] = (full[c] % modulus).astype('float32')
    orig_lg = lgb.LGBMClassifier(n_estimators=600, learning_rate=.05, num_leaves=31,
                                max_depth=8, min_child_samples=20, subsample=.8,
                                subsample_freq=1, colsample_bytree=.5, reg_lambda=5,
                                reg_alpha=.05, random_state=123, n_jobs=12, verbosity=-1)
    orig_lg.fit(og[raw], oy)
    base['original_lgb_probability'] = orig_lg.predict_proba(base[raw])[:, 1].astype('float32')
    orig_cb = CatBoostClassifier(iterations=1200, learning_rate=.07, depth=7,
                                task_type='GPU', random_seed=42, verbose=300,
                                allow_writing_files=False)
    cb_og, cb_full = og[raw].copy(), base[raw].copy()
    for c in CATS:
        cb_og[c] = cb_og[c].cat.codes.astype('int32')
        cb_full[c] = cb_full[c].cat.codes.astype('int32')
    cb_og, cb_full = cb_og.fillna(-999), cb_full.fillna(-999)
    orig_cb.fit(cb_og, oy, cat_features=CATS)
    base['original_cat_probability'] = orig_cb.predict_proba(cb_full)[:, 1].astype('float32')
    emit({'external_lgb_auc_on_competition': roc_auc_score(y, base.original_lgb_probability.iloc[:len(tr)]),
          'external_cat_auc_on_competition': roc_auc_score(y, base.original_cat_probability.iloc[:len(tr)])})
    joblib.dump({'base': base, 'tr': tr, 'ts': ts, 'sample': sample, 'raw': raw, 'y': y,
                 'original_removed': int(overlap.sum()), 'original_rows': len(og)}, CACHE, compress=3)
    emit({'stage': 'prepared', 'features': len(base.columns), 'seconds': round(time.time()-started, 2)})


def make_keys(full, raw, wide=False):
    keys = {}
    fd = full['Flight Distance']
    keys['fd'] = fd
    keys['fd10'] = fd // 10
    keys['fd_mod1000'] = fd % 1000
    keys['age_class'] = full.Age.astype(str) + '|' + full.Class.astype(str)
    keys['fd_class'] = fd.astype(str) + '|' + full.Class.astype(str)
    if wide:
        for c in raw:
            keys['raw_' + c] = full[c]
        for width in (100, 250, 500):
            keys[f'fd_bin{width}'] = fd // width
        keys['age5'] = full.Age // 5
        keys['delay10'] = full['Departure Delay in Minutes'].fillna(-999) // 10
        pairs = [('Class','Type of Travel'), ('Customer Type','Type of Travel'),
                 ('Class','Customer Type'), ('Gender','Class'),
                 ('Inflight wifi service','Online boarding'),
                 ('Inflight wifi service','Type of Travel'), ('Online boarding','Type of Travel'),
                 ('Seat comfort','Inflight entertainment'), ('Class','Inflight wifi service'),
                 ('Class','Online boarding'), ('Flight Distance','Type of Travel'), ('Flight Distance','Age')]
        for a,b in pairs:
            keys[a+'__'+b] = full[a].astype(str)+'|'+full[b].astype(str)
    # Factorization uses feature values only; unknown labels never enter vocabulary.
    return pd.DataFrame({name: pd.factorize(values, sort=False)[0].astype('int32')
                         for name, values in keys.items()})


def encoded_fold(base, keys, fit_idx, val_idx, n, y, wide):
    xa = base.iloc[fit_idx].reset_index(drop=True).copy()
    xv = base.iloc[val_idx].reset_index(drop=True).copy()
    xt = base.iloc[n:].reset_index(drop=True).copy()
    for smooth in (['auto', 100.] if wide else ['auto']):
        encoder = TargetEncoder(target_type='binary', smooth=smooth, cv=5,
                                shuffle=True, random_state=42)
        blocks = [encoder.fit_transform(keys.iloc[fit_idx], y.iloc[fit_idx]),
                  encoder.transform(keys.iloc[val_idx]), encoder.transform(keys.iloc[n:])]
        names = [f'te_{smooth}_{c}' for c in keys.columns]
        xa, xv, xt = [pd.concat([frame, pd.DataFrame(block.astype('float32'), columns=names)], axis=1)
                      for frame, block in zip((xa,xv,xt), blocks)]
    if wide:
        # Unlabeled route profiles fit only to this outer fold's training rows.
        fit = base.iloc[fit_idx]
        measures = [c for c in base.columns[:21] if c not in CATS and c != 'Flight Distance']
        route_stats = fit.groupby('Flight Distance', observed=True)[measures].agg(['mean','std'])
        counts = fit['Flight Distance'].value_counts()
        for frame in (xa,xv,xt):
            additions = {'route_count': frame['Flight Distance'].map(counts).fillna(0).astype('float32')}
            for c in measures:
                mean = frame['Flight Distance'].map(route_stats[(c,'mean')]).fillna(fit[c].mean())
                additions['route_mean_'+c] = mean.astype('float32')
                additions['route_std_'+c] = frame['Flight Distance'].map(route_stats[(c,'std')]).fillna(fit[c].std()).astype('float32')
                additions['route_residual_'+c] = (frame[c]-mean).astype('float32')
            frame[ list(additions) ] = pd.DataFrame(additions, index=frame.index)
    return xa,xv,xt


def run(mode, folds=3, threads=12, iterations=4000):
    started = time.time()
    output = ROOT / 'experiments' / mode
    output.mkdir(parents=True, exist_ok=True)
    data = joblib.load(CACHE)
    configuration = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'cache_sha256': hashlib.sha256(CACHE.read_bytes()).hexdigest(),
                     'mode': mode, 'folds': folds, 'iterations': iterations, 'seed': 42}
    config_path = output / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError('Cached experiment configuration differs; use a fresh experiment directory.')
    config_path.write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    base,tr,ts,sample,raw,y = (data[c] for c in ('base','tr','ts','sample','raw','y'))
    n = len(tr)
    wide = mode == 'route'
    keys = make_keys(pd.concat([tr[raw],ts[raw]],ignore_index=True),raw,wide)
    oof = np.full(n,np.nan)
    prediction = np.zeros(len(ts))
    fold_scores, importances = [], []
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for f,(a,b) in enumerate(split.split(tr,y),1):
        record = output/f'fold-{f}.npz'
        if record.exists():
            saved=np.load(record)
            if not np.array_equal(saved['val_idx'],b): raise ValueError('Cached split mismatch')
            oof[b]=saved['val_pred']; prediction+=saved['test_pred']/folds
            fold_scores.append(json.loads((output/f'fold-{f}.json').read_text()))
            emit({'mode':mode,'fold':f,'restored':True})
            continue
        xa,xv,xt = encoded_fold(base,keys,a,b,n,y,wide)
        emit({'mode':mode,'fold':f,'features':len(xa.columns),'stage':'training'})
        if mode == 'cat':
            for frame in (xa,xv,xt):
                for c in CATS: frame[c]=frame[c].cat.codes.astype('int32')
                frame.fillna(-999,inplace=True)
            model=CatBoostClassifier(iterations=iterations, learning_rate=.05, depth=7,
                                     l2_leaf_reg=3, border_count=254, task_type='GPU',
                                     eval_metric='AUC', random_seed=42+f, verbose=500,
                                     allow_writing_files=False)
            model.fit(xa,y.iloc[a],cat_features=CATS,eval_set=(xv,y.iloc[b]),
                      early_stopping_rounds=200)
            best=int(model.get_best_iteration()+1)
            importance=model.feature_importances_
        else:
            params=dict(n_estimators=iterations,learning_rate=.025,n_jobs=threads,
                        objective='binary',reg_lambda=5 if wide else 1,
                        reg_alpha=.05 if wide else 0,random_state=42+f,
                        min_child_samples=20 if wide else 100,
                        num_leaves=63 if wide else 127,max_depth=8 if wide else -1,
                        colsample_bytree=.85 if wide else .5,subsample=.9 if wide else .8,
                        subsample_freq=1,verbosity=-1,deterministic=True,force_col_wise=True)
            model=lgb.LGBMClassifier(**params)
            model.fit(xa,y.iloc[a],eval_set=[(xv,y.iloc[b])],eval_metric='auc',
                      callbacks=[lgb.early_stopping(180,verbose=False),lgb.log_evaluation(500)])
            best=int(model.best_iteration_)
            importance=model.booster_.feature_importance('gain')
        pv=model.predict_proba(xv)[:,1]; pt=model.predict_proba(xt)[:,1]
        score={'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),'iterations':best}
        oof[b]=pv; prediction+=pt/folds; fold_scores.append(score)
        np.savez_compressed(record,val_idx=b,val_pred=pv,test_pred=pt)
        (output/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
        importances.append(pd.DataFrame({'feature':xa.columns,'gain':importance}))
        emit({'mode':mode,**score,'seconds':round(time.time()-started,1)})
        del xa,xv,xt,model
        gc.collect()
    if not np.isfinite(oof).all(): raise ValueError('OOF incomplete')
    by_id=pd.Series(prediction,index=ts.id)
    submission=sample.copy(); submission.satisfaction=submission.id.map(by_id)
    validate_submission(submission,sample)
    path=output/'submission.csv'; submission.to_csv(path,index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(output/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':prediction}).to_csv(output/'test_predictions.csv',index=False)
    if importances:
        pd.concat(importances).groupby('feature').gain.mean().sort_values(ascending=False).to_csv(output/'feature_importance.csv')
    report={'competition':COMPETITION,'mode':mode,'folds':folds,'seed':42,
            'fold_scores':fold_scores,'selected_blend':{'auc':float(roc_auc_score(y,oof))},
            'input_sha256':{name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                            for name in ('train.csv','test.csv','sample_submission.csv')},
            'external_source':'arseniyshutko/binary-aviation-satisfaction-129k',
            'external_sha256':hashlib.sha256((ROOT/'data-original'/'data.csv').read_bytes()).hexdigest(),
            'original_matches_removed':data['original_removed'],'original_rows_used':data['original_rows'],
            'submission_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'seconds':round(time.time()-started,2),
            'validation_note':'Outer validation labels excluded from feature fitting; train TE uses 5 inner folds.'}
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    emit(report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','compact','route','cat'])
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--threads',type=int,default=12)
    p.add_argument('--iterations',type=int,default=4000)
    args=p.parse_args()
    prepare() if args.mode=='prepare' else run(args.mode,args.folds,args.threads,args.iterations)
