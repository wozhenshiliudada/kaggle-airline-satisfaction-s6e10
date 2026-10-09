"""GPU XGBoost comparison on the same second-round features and outer folds."""
import gc
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from improve import ROOT, CACHE, CATS, make_keys, encoded_fold, emit
from train import validate_submission, COMPETITION

started=time.time()
data=joblib.load(CACHE)
base,tr,ts,sample,raw,y=(data[c] for c in ('base','tr','ts','sample','raw','y'))
keys=make_keys(pd.concat([tr[raw],ts[raw]],ignore_index=True),raw)
output=ROOT/'experiments'/'xgb'; output.mkdir(parents=True,exist_ok=True)
oof=np.full(len(tr),np.nan); prediction=np.zeros(len(ts)); scores=[]
for f,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(tr,y),1):
    xa,xv,xt=encoded_fold(base,keys,a,b,len(tr),y,False)
    for frame in (xa,xv,xt):
        for c in CATS: frame[c]=frame[c].cat.codes.astype('int32')
    emit({'mode':'xgb','fold':f,'stage':'training'})
    model=XGBClassifier(n_estimators=4000,learning_rate=.025,max_depth=6,
                        min_child_weight=5,subsample=.8,colsample_bytree=.65,
                        reg_lambda=5,max_bin=1024,tree_method='hist',device='cuda',
                        n_jobs=8,eval_metric='auc',early_stopping_rounds=180,
                        random_state=42+f)
    model.fit(xa,y.iloc[a],eval_set=[(xv,y.iloc[b])],verbose=500)
    # Do not describe a silent CPU fallback as GPU execution.
    config=json.loads(model.get_booster().save_config())
    device=config['learner']['generic_param']['device']
    pv=model.predict_proba(xv)[:,1]; pt=model.predict_proba(xt)[:,1]
    oof[b]=pv; prediction+=pt/3
    score={'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),
           'iterations':int(model.best_iteration+1),'device':device}
    scores.append(score)
    np.savez_compressed(output/f'fold-{f}.npz',val_idx=b,val_pred=pv,test_pred=pt)
    (output/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
    emit({**score,'seconds':round(time.time()-started,1)})
    del model,xa,xv,xt
    gc.collect()
if not np.isfinite(oof).all(): raise ValueError('OOF incomplete')
submission=sample.copy(); submission.satisfaction=submission.id.map(pd.Series(prediction,index=ts.id))
validate_submission(submission,sample)
path=output/'submission.csv'; submission.to_csv(path,index=False)
pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(output/'oof.csv',index=False)
pd.DataFrame({'id':ts.id,'prediction':prediction}).to_csv(output/'test_predictions.csv',index=False)
report={'competition':COMPETITION,'mode':'xgb','folds':3,'seed':42,'fold_scores':scores,
        'selected_blend':{'auc':float(roc_auc_score(y,oof))},
        'input_sha256':{name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                        for name in ('train.csv','test.csv','sample_submission.csv')},
        'external_sha256':hashlib.sha256((ROOT/'data-original'/'data.csv').read_bytes()).hexdigest(),
        'original_matches_removed':data['original_removed'],
        'submission_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        'seconds':round(time.time()-started,2)}
(output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
emit(report)
