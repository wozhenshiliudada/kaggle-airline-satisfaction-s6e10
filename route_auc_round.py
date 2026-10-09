"""Controlled raw numeric categorical-twin ablation on existing route features.

Only feature values enter categorical vocabularies. All outer folds, route/TE
features, LightGBM settings and early stopping match improve.py mode route.
Idea source: evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96102 and
megayak/s6e10-honest-oof-stack. Public notebook code is not executed.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from improve import ROOT, CACHE, CATS, make_keys, encoded_fold, emit
from train import COMPETITION, validate_submission


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(mode):
    started = time.time()
    data = joblib.load(CACHE)
    base, tr, ts, sample, raw, y = (data[c] for c in ('base','tr','ts','sample','raw','y'))
    n = len(tr)
    full = pd.concat([tr[raw], ts[raw]], ignore_index=True)
    keys = make_keys(full, raw, wide=True)
    twins = []
    levels = {c: sorted(full[c].dropna().unique().tolist()) for c in twins}
    output = ROOT / 'experiments' / 'route-auc'
    output.mkdir(parents=True, exist_ok=True)
    config = {'script_sha256': digest(Path(__file__)), 'cache_sha256': digest(CACHE),
              'seed':42, 'folds':3, 'twins':twins, 'iterations':3500, 'threads':8, 'metric':'auc', 'first_metric_only':True}
    configpath=output/'configuration.json'
    if configpath.exists() and json.loads(configpath.read_text()) != config:
        raise ValueError('Experiment configuration differs')
    configpath.write_text(json.dumps(config, indent=2), encoding='utf-8')
    oof = np.full(n, np.nan)
    test_pred = np.zeros(len(ts))
    scores, importance = [], []
    label_invariance = False
    for fold, (a,b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(tr,y),1):
        xa,xv,xt = encoded_fold(base,keys,a,b,n,y,True)
        if fold == 1:
            changed_y=y.copy()
            changed_y.iloc[b]=1-changed_y.iloc[b]
            altered=encoded_fold(base,keys,a,b,n,changed_y,True)
            for original, changed in zip((xa,xv,xt), altered):
                pd.testing.assert_frame_equal(original, changed, check_exact=True)
            label_invariance=True
            del altered, changed_y
            emit({'mode':mode, 'outer_validation_label_flip_invariance':'PASS'})
        for frame in (xa,xv,xt):
            for c in twins:
                frame['categorical_' + c] = pd.Categorical(frame[c], categories=levels[c])
        emit({'mode':mode,'fold':fold,'features':len(xa.columns),'stage':'training'})
        model=lgb.LGBMClassifier(n_estimators=3500, learning_rate=.025, n_jobs=8,
              objective='binary', reg_lambda=5, reg_alpha=.05, random_state=42+fold,
              min_child_samples=20, num_leaves=63, max_depth=8, colsample_bytree=.85,
              subsample=.9, subsample_freq=1, verbosity=-1, deterministic=True,
              force_col_wise=True, metric='auc')
        model.fit(xa,y.iloc[a],eval_set=[(xv,y.iloc[b])],eval_metric='auc',
              callbacks=[lgb.early_stopping(180,first_metric_only=True,verbose=False),lgb.log_evaluation(500)])
        pv=model.predict_proba(xv)[:,1]
        pt=model.predict_proba(xt)[:,1]
        oof[b]=pv; test_pred += pt/3
        record={'fold':fold,'auc':float(roc_auc_score(y.iloc[b],pv)),
                'iterations':int(model.best_iteration_)}
        scores.append(record)
        np.savez_compressed(output/f'fold-{fold}.npz',val_idx=b,val_pred=pv,test_pred=pt)
        (output/f'fold-{fold}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        importance.append(pd.DataFrame({'feature':xa.columns,'gain':model.booster_.feature_importance('gain')}))
        emit({'mode':mode,**record,'seconds':round(time.time()-started,1)})
        del xa,xv,xt,model
        gc.collect()
    if not np.isfinite(oof).all() or not np.isfinite(test_pred).all():
        raise ValueError('Predictions incomplete')
    sub=sample.copy()
    sub.satisfaction=sub.id.map(pd.Series(test_pred,index=ts.id))
    validate_submission(sub,sample)
    path=output/'submission.csv'; sub.to_csv(path,index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(output/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':test_pred}).to_csv(output/'test_predictions.csv',index=False)
    pd.concat(importance).groupby('feature').gain.mean().sort_values(ascending=False).to_csv(output/'feature_importance.csv')
    report={'competition':COMPETITION,'mode':'route-auc','model_description':'Route LightGBM AUC-only early stopping',
       'folds':3,'seed':42,'fold_scores':scores,'selected_blend':{'auc':float(roc_auc_score(y,oof))},
       'comparison_baseline_oof_auc':0.9610085911957561,
       'categorical_twins':twins,'input_sha256':{c:digest(ROOT/'data'/c) for c in ('train.csv','test.csv','sample_submission.csv')},
       'external_sha256':digest(ROOT/'data-original'/'data.csv'),
       'external_source':'arseniyshutko/binary-aviation-satisfaction-129k',
       'original_matches_removed':data['original_removed'],'original_rows_used':data['original_rows'],
       'submission_sha256':digest(path),'script_sha256':digest(Path(__file__)),'cache_sha256':digest(CACHE),
       'seconds':round(time.time()-started,2),'outer_validation_label_flip_invariance':label_invariance,
       'validation_note':'Same features and 3-fold seed42 as route baseline; validation labels excluded from feature fitting; train TE uses 5 inner folds. Only change is AUC-only early stopping rather than stopping on either AUC or default logloss.',
       'sources':['https://www.kaggle.com/code/evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96102','https://www.kaggle.com/code/megayak/s6e10-honest-oof-stack']}
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    emit(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['auc'])
    run(parser.parse_args().mode)
