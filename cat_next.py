"""Deeper GPU CatBoost screening with unchanged compact feature preparation."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from improve import ROOT,CACHE,CATS,make_keys,encoded_fold,emit
from train import read_data,COMPETITION,validate_submission


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    start=time.time();out=ROOT/'experiments'/f'cat-depth{args.depth}-{args.folds}fold'
    out.mkdir(exist_ok=True)
    data=joblib.load(CACHE)
    base,tr,ts,sample,raw,y=(data[c] for c in ('base','tr','ts','sample','raw','y'))
    hashes={n:sha(ROOT/'data'/n) for n in ('train.csv','test.csv','sample_submission.csv')}
    reference=json.loads((ROOT/'experiments'/'cat'/'metrics.json').read_text())
    official=read_data(ROOT/'data')
    if reference['input_sha256']!=hashes or not tr.id.equals(official[0].id) or not ts.id.equals(official[1].id):
        raise ValueError('Cached input IDs or fingerprints differ')
    pd.testing.assert_series_equal(y,official[4]);del official
    params=dict(iterations=4500,learning_rate=.04,depth=args.depth,l2_leaf_reg=5,
                border_count=254,task_type='GPU',devices='0',thread_count=4,
                eval_metric='AUC',verbose=500,allow_writing_files=False)
    config=dict(script_sha256=sha(Path(__file__)),feature_script_sha256=sha(ROOT/'improve.py'),
                cache_sha256=sha(CACHE),folds=args.folds,seed=42,params=params)
    cp=out/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Cache config changed')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    keys=make_keys(pd.concat([tr[raw],ts[raw]],ignore_index=True),raw,False)
    oof=np.full(len(tr),np.nan);pred=np.zeros(len(ts));scores=[]
    for f,(a,b) in enumerate(StratifiedKFold(args.folds,shuffle=True,random_state=42).split(tr,y),1):
        path=out/f'fold-{f}.npz'
        if path.exists():
            saved=np.load(path)
            if not np.array_equal(saved['val_idx'],b):raise ValueError('Cached fold differs')
            pv,pt=saved['val_pred'],saved['test_pred']
            score=json.loads((out/f'fold-{f}.json').read_text())
        else:
            xa,xv,xt=encoded_fold(base,keys,a,b,len(tr),y,False)
            if f==1:
                alt=y.copy();alt.iloc[b]=1-alt.iloc[b]
                check=encoded_fold(base,keys,a,b,len(tr),alt,False)
                for first,second in zip((xa,xv,xt),check):pd.testing.assert_frame_equal(first,second,check_exact=True)
                del check
                emit({'validation_label_perturbation':'PASS'})
            for frame in (xa,xv,xt):
                for c in CATS:frame[c]=frame[c].cat.codes.astype('int32')
                frame.fillna(-999,inplace=True)
            emit({'fold':f,'stage':'training','params':params})
            model=CatBoostClassifier(**params,random_seed=42+f)
            model.fit(xa,y.iloc[a],cat_features=CATS,eval_set=(xv,y.iloc[b]),early_stopping_rounds=200)
            if model.get_all_params()['task_type']!='GPU':raise ValueError('GPU not used')
            pv=model.predict_proba(xv)[:,1];pt=model.predict_proba(xt)[:,1]
            score={'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),
                   'iterations':int(model.get_best_iteration()+1),'elapsed_seconds':round(time.time()-start,2)}
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            del model,xa,xv,xt;gc.collect()
        oof[b]=pv;pred+=pt/args.folds;scores.append(score);emit(score)
        if args.first_fold:
            best=pd.read_csv(ROOT/'outputs-round4'/'oof.csv').prediction.to_numpy()[b]
            old=pd.read_csv(ROOT/'experiments'/'cat'/'oof.csv').prediction.to_numpy()[b]
            emit({'baseline_auc':float(roc_auc_score(y.iloc[b],best)),
                  'replace_cat_auc':float(roc_auc_score(y.iloc[b],best+(pv-old)/6)),
                  'addition_auc':{str(w):float(roc_auc_score(y.iloc[b],(1-w)*best+w*pv)) for w in (.1,.25,.5)}})
            return
    if not np.isfinite(oof).all():raise ValueError('Incomplete OOF')
    sub=sample.copy();sub.satisfaction=pred;validate_submission(sub,sample)
    path=out/'submission.csv';sub.to_csv(path,index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(out/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':pred}).to_csv(out/'test_predictions.csv',index=False)
    metrics=dict(competition=COMPETITION,folds=args.folds,seed=42,
                 model_description=f'CatBoost depth{args.depth}',selected_blend={'auc':float(roc_auc_score(y,oof))},
                 fold_scores=scores,input_sha256=hashes,external_sha256=reference['external_sha256'],
                 submission_sha256=sha(path),seconds=round(time.time()-start,2))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');emit(metrics)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--depth',type=int,default=8)
    p.add_argument('--folds',type=int,default=3);p.add_argument('--first-fold',action='store_true')
    run(p.parse_args())
