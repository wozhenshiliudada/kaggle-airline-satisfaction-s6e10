"""Add wide route/interaction features to the completed conditional CatBoost.

All five folds, seeds and training hyperparameters match cat-conditional-5fold.
This preserves the previous script/configuration fingerprints unchanged.
"""
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from improve import ROOT, CACHE, make_keys, encoded_fold, emit
from context_round import context_keys, context_blocks, add_context, sha
from train import read_data, COMPETITION, validate_submission


def make_frames(base, keys, ckeys, a, b, n, y):
    frames = encoded_fold(base, keys, a, b, n, y, True)
    blocks = context_blocks(ckeys, a, b, n, y)
    add_context(frames, blocks, ckeys, a, b, n)
    return frames


def run():
    start = time.time(); out = ROOT/'experiments'/'cat-wide-conditional-5fold'
    out.mkdir(exist_ok=True)
    reference = ROOT/'experiments'/'cat-conditional-5fold'
    ref_config = json.loads((reference/'configuration.json').read_text())
    ref_metrics = json.loads((reference/'metrics.json').read_text())
    hashes = {k:sha(ROOT/'data'/k) for k in ('train.csv','test.csv','sample_submission.csv')}
    external = sha(ROOT/'data-original'/'data.csv')
    if hashes != ref_config['input_sha256'] or external != ref_config['external_sha256']:
        raise ValueError('Reference input files changed')
    params = ref_config['params']
    config = {'script_sha256':sha(Path(__file__)), 'base_feature_source_sha256':sha(ROOT/'improve.py'),
              'context_feature_source_sha256':sha(ROOT/'context_round.py'), 'cache_sha256':sha(CACHE),
              'reference_configuration_sha256':sha(reference/'configuration.json'),
              'params':params, 'input_sha256':hashes, 'external_sha256':external,
              'folds':5, 'seed':42, 'early_stopping_rounds':200, 'wide':True}
    cp = out/'configuration.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Changed wide CatBoost experiment; use a fresh folder')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    data = joblib.load(CACHE)
    base,tr,ts,sample,raw,y = (data[k] for k in ('base','tr','ts','sample','raw','y'))
    official = read_data(ROOT/'data')
    if not tr.id.equals(official[0].id) or not ts.id.equals(official[1].id):
        raise ValueError('Cached IDs differ from official files')
    pd.testing.assert_series_equal(y,official[4]); del official
    full = pd.concat([tr[raw],ts[raw]],ignore_index=True)
    keys = make_keys(full,raw,True); ckeys = context_keys(full); n = len(tr)
    oof = np.full(n,np.nan); pred = np.zeros(len(ts)); scores = []
    for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y),1):
        with np.load(reference/f'fold-{f}.npz') as saved:
            if not np.array_equal(saved['val_idx'],b):raise ValueError('Reference folds differ')
        path = out/f'fold-{f}.npz'
        if path.exists():
            with np.load(path) as saved:
                if not np.array_equal(saved['val_idx'],b):raise ValueError('Cached fold differs')
                pv,pt = saved['val_pred'],saved['test_pred']
            score = json.loads((out/f'fold-{f}.json').read_text())
        else:
            frames = make_frames(base,keys,ckeys,a,b,n,y)
            if f == 1:
                changed = y.copy(); changed.iloc[b] = 1-changed.iloc[b]
                check = make_frames(base,keys,ckeys,a,b,n,changed)
                for actual,perturbed in zip(frames,check):
                    pd.testing.assert_frame_equal(actual,perturbed,check_exact=True)
                del check;gc.collect();emit({'wide_cat_validation_label_perturbation':'PASS'})
            xa,xv,xt = frames; cats = list(xa.select_dtypes(include='category').columns)
            for frame in frames:
                for c in cats:frame[c] = frame[c].cat.codes.astype('int32')
                frame.fillna(-999,inplace=True)
            emit({'mode':'wide-conditional','fold':f,'columns':len(xa.columns),'stage':'training','device':'GPU:0'})
            model = CatBoostClassifier(**params,random_seed=42+f)
            model.fit(xa,y.iloc[a],cat_features=cats,eval_set=(xv,y.iloc[b]),early_stopping_rounds=200)
            pv = model.predict_proba(xv)[:,1]; pt = model.predict_proba(xt)[:,1]
            if not np.isfinite(pv).all() or not np.isfinite(pt).all():raise ValueError('Nonfinite probabilities')
            score = {'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),
                     'iterations':int(model.get_best_iteration()+1),'columns':len(xa.columns),
                     'delta_against_compact_conditional':float(roc_auc_score(y.iloc[b],pv)-ref_metrics['fold_scores'][f-1]['auc']),
                     'seconds_elapsed':round(time.time()-start,2)}
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            pd.DataFrame({'feature':xa.columns,'importance':model.feature_importances_}).to_csv(out/f'importance-{f}.csv',index=False)
            del frames,xa,xv,xt,model;gc.collect()
        oof[b] = pv; pred += pt/5; scores.append(score);emit(score)
    if not np.isfinite(oof).all():raise ValueError('Incomplete OOF')
    sub = sample.copy();sub.satisfaction = pred;validate_submission(sub,sample)
    sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(out/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':pred}).to_csv(out/'test_predictions.csv',index=False)
    metrics = {'competition':COMPETITION,'folds':5,'seed':42,'model_description':'Wide conditional GPU CatBoost',
               'input_sha256':hashes,'external_sha256':external,'fold_scores':scores,
               'selected_blend':{'auc':float(roc_auc_score(y,oof))},'submission_sha256':sha(out/'submission.csv'),
               'original_rows_used':data['original_rows'],'original_matches_removed':data['original_removed'],
               'seconds':round(time.time()-start,2),
               'validation_note':'Same five folds, seeds and parameters as conditional CatBoost control; adds wide target encodings and outer-train-only unlabeled route statistics. Inner fivefold crossfitting excludes outer validation labels from feature construction. Outer validation labels used for AUC early stopping.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');emit(metrics)


if __name__ == '__main__':run()
