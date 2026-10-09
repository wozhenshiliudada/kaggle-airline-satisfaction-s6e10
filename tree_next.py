"""Five-fold compact LightGBM with AUC-only stopping, preserving old models."""
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
from improve import ROOT,CACHE,make_keys,encoded_fold,emit
from train import read_data,COMPETITION,validate_submission


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    start=time.time();out=ROOT/'experiments'/'compact-auc-5fold'
    out.mkdir(exist_ok=True)
    data=joblib.load(CACHE)
    base,tr,ts,sample,raw,y=(data[c] for c in ('base','tr','ts','sample','raw','y'))
    hashes={n:sha(ROOT/'data'/n) for n in ('train.csv','test.csv','sample_submission.csv')}
    reference=json.loads((ROOT/'experiments'/'compact'/'metrics.json').read_text())
    if reference['input_sha256'] != hashes:raise ValueError('Input fingerprint differs')
    official=read_data(ROOT/'data')
    if not tr.id.equals(official[0].id) or not ts.id.equals(official[1].id):
        raise ValueError('Cached IDs differ')
    pd.testing.assert_series_equal(y,official[4]);del official
    params=dict(n_estimators=4500,learning_rate=.025,n_jobs=8,objective='binary',
                reg_lambda=1,reg_alpha=0,min_child_samples=100,num_leaves=127,
                max_depth=-1,colsample_bytree=.5,subsample=.8,subsample_freq=1,
                verbosity=-1,deterministic=True,force_col_wise=True,metric='auc')
    config={'script_sha256':sha(Path(__file__)), 'cache_sha256':sha(CACHE),
            'feature_script_sha256':sha(ROOT/'improve.py'),'params':params,'folds':5,'seed':42}
    cp=out/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Cache config changed')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    keys=make_keys(pd.concat([tr[raw],ts[raw]],ignore_index=True),raw,False)
    oof=np.full(len(tr),np.nan);pred=np.zeros(len(ts));scores=[]
    for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y),1):
        path=out/f'fold-{f}.npz'
        if path.exists():
            saved=np.load(path)
            if not np.array_equal(saved['val_idx'],b):raise ValueError('Fold differs')
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
            emit({'fold':f,'stage':'training'})
            model=lgb.LGBMClassifier(**params,random_state=42+f)
            model.fit(xa,y.iloc[a],eval_set=[(xv,y.iloc[b])],eval_metric='auc',
                      callbacks=[lgb.early_stopping(180,first_metric_only=True,verbose=False),lgb.log_evaluation(1000)])
            pv=model.predict_proba(xv)[:,1];pt=model.predict_proba(xt)[:,1]
            score={'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),
                   'iterations':int(model.best_iteration_),'elapsed_seconds':round(time.time()-start,2)}
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            del model,xa,xv,xt;gc.collect()
        oof[b]=pv;pred+=pt/5;scores.append(score);emit(score)
    if not np.isfinite(oof).all():raise ValueError('Incomplete OOF')
    sub=sample.copy();sub.satisfaction=pred;validate_submission(sub,sample)
    path=out/'submission.csv';sub.to_csv(path,index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(out/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':pred}).to_csv(out/'test_predictions.csv',index=False)
    metrics={'competition':COMPETITION,'folds':5,'seed':42,'model_description':'Compact LightGBM 5fold AUC early stop',
             'selected_blend':{'auc':float(roc_auc_score(y,oof))},'fold_scores':scores,
             'input_sha256':hashes,'external_sha256':reference['external_sha256'],
             'submission_sha256':sha(path),'seconds':round(time.time()-start,2)}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');emit(metrics)


if __name__=='__main__':run()
