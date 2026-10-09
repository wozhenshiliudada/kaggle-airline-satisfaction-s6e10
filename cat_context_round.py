"""Matched compact/control and context CatBoost, full five-fold GPU training."""
import argparse
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
from improve import ROOT,CACHE,make_keys,encoded_fold,emit
from context_round import context_keys,context_blocks,add_context,sha
from train import read_data,COMPETITION,validate_submission


def run(mode):
    start=time.time();out=ROOT/'experiments'/f'cat-{mode}-5fold'
    out.mkdir(exist_ok=True)
    data=joblib.load(CACHE)
    base,tr,ts,sample,raw,y=(data[k] for k in ('base','tr','ts','sample','raw','y'))
    hashes={k:sha(ROOT/'data'/k) for k in ('train.csv','test.csv','sample_submission.csv')}
    reference=json.loads((ROOT/'experiments'/'cat'/'metrics.json').read_text())
    if hashes!=reference['input_sha256'] or sha(ROOT/'data-original'/'data.csv')!=reference['external_sha256']:
        raise ValueError('CatBoost reference inputs changed')
    official=read_data(ROOT/'data')
    if not tr.id.equals(official[0].id) or not ts.id.equals(official[1].id):raise ValueError('CatBoost cached IDs changed')
    pd.testing.assert_series_equal(y,official[4]);del official
    params=dict(iterations=5000,learning_rate=.05,depth=7,l2_leaf_reg=3,
                border_count=254,task_type='GPU',devices='0',thread_count=4,
                eval_metric='AUC',verbose=1000,allow_writing_files=False)
    config={'script_sha256':sha(Path(__file__)),'base_feature_source_sha256':sha(ROOT/'improve.py'),
            'context_feature_source_sha256':sha(ROOT/'context_round.py'),'cache_sha256':sha(CACHE),
            'params':params,'input_sha256':hashes,'external_sha256':reference['external_sha256'],
            'mode':mode,'folds':5,'seed':42,'early_stopping_rounds':200}
    cp=out/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Changed cached CatBoost configuration')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    full=pd.concat([tr[raw],ts[raw]],ignore_index=True)
    keys=make_keys(full,raw,False);ckeys=context_keys(full) if mode=='conditional' else None
    n=len(tr);oof=np.full(n,np.nan);pred=np.zeros(len(ts));scores=[]
    for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y),1):
        path=out/f'fold-{f}.npz'
        if path.exists():
            saved=np.load(path)
            if not np.array_equal(saved['val_idx'],b):raise ValueError('Cached CatBoost split differs')
            pv,pt=saved['val_pred'],saved['test_pred'];score=json.loads((out/f'fold-{f}.json').read_text())
        else:
            xa,xv,xt=encoded_fold(base,keys,a,b,n,y,False)
            if mode=='conditional':
                blocks=context_blocks(ckeys,a,b,n,y)
                if f==1:
                    changed=y.copy();changed.iloc[b]=1-changed.iloc[b]
                    check=context_blocks(ckeys,a,b,n,changed)
                    for actual,perturbed in zip(blocks,check):pd.testing.assert_frame_equal(actual,perturbed,check_exact=True)
                    del check;emit({'cat_context_validation_label_perturbation':'PASS'})
                add_context((xa,xv,xt),blocks,ckeys,a,b,n);del blocks
            cats=list(xa.select_dtypes(include='category').columns)
            for frame in (xa,xv,xt):
                for c in cats:frame[c]=frame[c].cat.codes.astype('int32')
                frame.fillna(-999,inplace=True)
            emit({'mode':mode,'fold':f,'stage':'training','columns':len(xa.columns),'task_type':'GPU','devices':'0'})
            model=CatBoostClassifier(**params,random_seed=42+f)
            model.fit(xa,y.iloc[a],cat_features=cats,eval_set=(xv,y.iloc[b]),early_stopping_rounds=200)
            pv=model.predict_proba(xv)[:,1];pt=model.predict_proba(xt)[:,1]
            if not np.isfinite(pv).all() or not np.isfinite(pt).all():raise ValueError('Nonfinite CatBoost probabilities')
            score={'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),
                   'iterations':int(model.get_best_iteration()+1),'columns':len(xa.columns),
                   'seconds_elapsed':round(time.time()-start,2)}
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            pd.DataFrame({'feature':xa.columns,'importance':model.feature_importances_}).to_csv(out/f'importance-{f}.csv',index=False)
            del xa,xv,xt,model;gc.collect()
        oof[b]=pv;pred+=pt/5;scores.append(score);emit(dict(mode=mode,**score))
    if not np.isfinite(oof).all():raise ValueError('Incomplete CatBoost OOF')
    sub=sample.copy();sub.satisfaction=pred;validate_submission(sub,sample)
    sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(out/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':pred}).to_csv(out/'test_predictions.csv',index=False)
    metrics={'competition':COMPETITION,'folds':5,'seed':42,'mode':mode,
             'model_description':f'GPU CatBoost {mode} matched fivefold',
             'input_sha256':hashes,'external_sha256':reference['external_sha256'],
             'fold_scores':scores,'selected_blend':{'auc':float(roc_auc_score(y,oof))},
             'submission_sha256':sha(out/'submission.csv'),'seconds':round(time.time()-start,2),
             'original_rows_used':data['original_rows'],'original_matches_removed':data['original_removed'],
             'validation_note':'Control/conditional models share identical five folds, random seeds and CatBoost hyperparameters. Five inner folds for target features; outer validation labels excluded from feature fit, used only for AUC early stopping.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');emit(metrics)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['control','conditional']);run(p.parse_args().mode)
