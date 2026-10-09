"""Original-only XGBoost feature added to the matched conditional LightGBM."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from auxiliary_round import AUXDIR, config_file, hashes
from context_round import ROOT, SERVICES, sha, context_keys, context_blocks, add_context
from improve import CACHE, make_keys, encoded_fold
from nn_round import report
from train import read_data, validate_submission, COMPETITION

UNCERT = ROOT/'experiments/round11-auxiliary-uncertainty'
AGG = ROOT/'experiments/round12-teacher-features'
warnings.simplefilter('ignore', pd.errors.PerformanceWarning)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def build():
    source=ROOT/'experiments/original-xgb-600rounds'
    sm=load(source/'metrics.json');ih=hashes()
    if sm['configuration']['input_sha256']!=ih or sm['configuration']['original_sha256']!=sha(ROOT/'data-original/data.csv'):
        raise ValueError('Original teacher input changed')
    if sha(source/'probability.npy')!=sm['probability_sha256']:
        raise ValueError('Original teacher probability changed')
    config_file(AGG,dict(script_sha256=sha(Path(__file__)),input_sha256=ih,
        teacher_sha256=sm['probability_sha256'],label_used=False,original_matches_removed=sm['configuration']['overlaps_removed']))
    if (AGG/'features.pkl').exists():
        if sha(AGG/'features.pkl')!=load(AGG/'metrics.json')['features_sha256']:
            raise ValueError('Teacher feature cache changed')
        return
    probability=np.load(source/'probability.npy')
    shape=load(UNCERT/'metrics.json')['shape']
    if probability.shape!=(shape[0],) or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
        raise ValueError('Invalid original teacher predictions')
    frame=pd.DataFrame({'original_xgb_logit':logit(np.clip(probability,1e-6,1-1e-6)).astype('float32')})
    frame.to_pickle(AGG/'features.pkl')
    save(AGG/'metrics.json',dict(input_sha256=ih,shape=list(frame.shape),features_sha256=sha(AGG/'features.pkl'),
        teacher_sha256=sm['probability_sha256'],label_used=False,
        validation_note='Competition labels not loaded. Existing original-only XGBoost predictions; exact original/competition overlaps excluded.'))
    report(dict(stage='original_teacher_feature',shape=list(frame.shape)))


def digest(frame):
    metadata={'columns':list(frame),'dtypes':[str(t) for t in frame.dtypes],
        'categories':{c:list(frame[c].cat.categories) for c in frame.select_dtypes(include='category')}}
    values=pd.util.hash_pandas_object(frame,index=True).to_numpy()
    return (metadata,hashlib.sha256(values.tobytes()).hexdigest())


def rebuilt(tree, old5, context, aux, tabm, travel):
    ctx=expit(.5*logit(np.clip(tree,1e-6,1-1e-6))+.5*logit(np.clip(context,1e-6,1-1e-6)))
    old9=.9*(.75*(.5*old5+.5*ctx)+.25*aux)+.1*tabm
    return .9*old9+.1*travel


def screen(folder,mode,b,pv,y):
    paths={'base':'outputs-round11','old5':'outputs-round5','lgb':'experiments/round9-lgb-auxiliary-5fold',
        'context':'experiments/realmlp-context-5fold-8ens','aux':'experiments/realmlp-route-auxiliary-all-5fold-8ens',
        'tabm':'experiments/round9-tabm-auxiliary-5fold','travel':'experiments/round11-experts-5fold'}
    p={k:pd.read_csv(ROOT/v/'oof.csv',usecols=['prediction']).prediction.to_numpy()[b] for k,v in paths.items()}
    args=[p[k] for k in ('old5','context','aux','tabm','travel')]
    if np.max(abs(rebuilt(p['lgb'],*args)-p['base']))>1e-12:
        raise ValueError('Current official baseline formula mismatch')
    candidates={str(w):(1-w)*p['base']+w*pv for w in (.1,.25,.5)}
    candidates['replace']=rebuilt(pv,*args)
    result=dict(scope='first_fold_only',mode=mode,baseline_auc=float(roc_auc_score(y.iloc[b],p['base'])),
        model_auc=float(roc_auc_score(y.iloc[b],pv)),blend_auc={k:float(roc_auc_score(y.iloc[b],v)) for k,v in candidates.items()})
    result['continue_full']=max(result['blend_auc'].values())>result['baseline_auc']
    save(folder/'first-fold-screen.json',result);report(result)


def run(args):
    start=time.time();tr,ts,sample,raw,y=read_data(ROOT/'data');ih=hashes()
    ref=ROOT/'experiments/round9-lgb-auxiliary-5fold';tm=load(ref/'metrics.json');am=load(AUXDIR/'metrics.json')
    if tm['input_sha256']!=ih or am['configuration']['input_sha256']!=ih:
        raise ValueError('Input mismatch')
    if sha(ROOT/'data-original/data.csv')!=tm['external_sha256'] or sha(AUXDIR/'features.pkl')!=am['features_sha256']:
        raise ValueError('Source changed')
    data=joblib.load(CACHE)
    if not data['tr'].id.equals(tr.id) or not data['ts'].id.equals(ts.id):raise ValueError('Cached rows changed')
    pd.testing.assert_series_equal(data['y'],y)
    full=data.pop('base');del data;gc.collect()
    aux=pd.read_pickle(AUXDIR/'features.pkl')
    if not aux.index.equals(pd.RangeIndex(len(tr)+len(ts))):raise ValueError('Auxiliary rows changed')
    full=pd.concat([full,aux],axis=1);del aux
    extra=None
    if args.mode=='teacher':
        ag=load(AGG/'metrics.json');extra=sha(AGG/'features.pkl')
        if ag['input_sha256']!=ih or ag['features_sha256']!=extra:raise ValueError('Coherence features changed')
        added=pd.read_pickle(AGG/'features.pkl')
        if not added.index.equals(full.index):raise ValueError('Coherence rows changed')
        full=pd.concat([full,added],axis=1);del added
    raw_full=pd.concat([tr[raw],ts[raw]],ignore_index=True)
    keys=make_keys(raw_full,raw,True);ckeys=context_keys(raw_full);gate=raw_full['Class'].to_numpy()
    del raw_full;gc.collect()
    def prepare(a,b,target):
        result=encoded_fold(full,keys,a,b,len(tr),target,True)
        add_context(result,context_blocks(ckeys,a,b,len(tr),target),ckeys,a,b,len(tr))
        return result
    params=load(ref/'configuration.json')['params']
    out=ROOT/'experiments'/('round12-'+args.mode+'-5fold')
    config_file(out,dict(script_sha256=sha(Path(__file__)),input_sha256=ih,external_sha256=tm['external_sha256'],
        auxiliary_sha256=am['features_sha256'],teacher_features_sha256=extra,mode=args.mode,folds=5,seed=42,params=params,
        source_code_sha256={p:sha(ROOT/p) for p in ('improve.py','context_round.py')}))
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y))
    oof=np.full(len(tr),np.nan);test=np.zeros(len(ts));scores=[]
    for f,(a,b) in enumerate(folds,1):
        with np.load(ref/f'fold-{f}.npz') as z:
            if not np.array_equal(z['val_idx'],b):raise ValueError('Reference fold mismatch')
        path=out/f'fold-{f}.npz'
        if path.exists():
            with np.load(path) as z:
                if not np.array_equal(z['val_idx'],b):raise ValueError('Checkpoint fold mismatch')
                pv,pt=z['val_pred'],z['test_pred']
            score=load(out/f'fold-{f}.json')
        else:
            xa,xv,xt=prepare(a,b,y)
            if f==1:
                expected=[digest(frame) for frame in (xa,xv,xt)]
                del xa,xv,xt;gc.collect()
                alternate=y.copy();alternate.iloc[b]=1-alternate.iloc[b]
                changed=prepare(a,b,alternate)
                if [digest(frame) for frame in changed]!=expected:raise ValueError('Validation labels affect features')
                del changed,alternate;gc.collect()
                xa,xv,xt=prepare(a,b,y)
                if [digest(frame) for frame in (xa,xv,xt)]!=expected:raise ValueError('Feature generation nondeterministic')
                report(dict(mode=args.mode,validation_label_perturbation='PASS',check='sequential_frame_digests'))
            report(dict(stage='training',mode=args.mode,fold=f,features=len(xa.columns)))
            pv=np.full(len(b),np.nan);pt=np.full(len(ts),np.nan);details=[];gains=[]
            groups=sorted(np.unique(gate)) if args.mode=='cabin' else ['all']
            for group in groups:
                ma=np.ones(len(a),bool) if group=='all' else gate[a]==group
                mb=np.ones(len(b),bool) if group=='all' else gate[b]==group
                mt=np.ones(len(ts),bool) if group=='all' else gate[len(tr):]==group
                fit=lgb.Dataset(xa.loc[ma],y.iloc[a].to_numpy()[ma]);val=lgb.Dataset(xv.loc[mb],y.iloc[b].to_numpy()[mb],reference=fit)
                model=lgb.train(dict(params,seed=42+f),fit,num_boost_round=3500,valid_sets=[val],valid_names=['val'],
                    callbacks=[lgb.early_stopping(180,first_metric_only=True,verbose=False),lgb.log_evaluation(1000)])
                pv[mb]=model.predict(xv.loc[mb]);pt[mt]=model.predict(xt.loc[mt])
                details.append(dict(group=group,rows=int(ma.sum()),iterations=model.best_iteration))
                gains.append(model.feature_importance('gain'))
                del model,fit,val;gc.collect()
            if not all(np.isfinite(p).all() and p.min()>=0 and p.max()<=1 for p in (pv,pt)):
                raise ValueError('Invalid predictions')
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            pd.DataFrame(dict(feature=list(xa),gain=np.mean(gains,axis=0))).to_csv(out/f'importance-{f}.csv',index=False)
            score=dict(fold=f,auc=float(roc_auc_score(y.iloc[b],pv)),features=len(xa.columns),
                experts=details,seconds_elapsed=round(time.time()-start,2))
            save(out/f'fold-{f}.json',score);del xa,xv,xt;gc.collect()
        oof[b]=pv;test+=pt/5;scores.append(score);report(dict(mode=args.mode,**score))
        if args.first_fold:screen(out,args.mode,b,pv,y);return
    sub=sample.copy();sub.satisfaction=test;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=oof)).to_csv(out/'oof.csv',index=False)
    m=dict(competition=COMPETITION,folds=5,seed=42,mode=args.mode,input_sha256=ih,
        external_sha256=tm['external_sha256'],submission_sha256=sha(out/'submission.csv'),
        selected_blend=dict(auc=float(roc_auc_score(y,oof))),fold_scores=scores,seconds=round(time.time()-start,2),
        validation_note='Matched five folds and safe inner target encoding. Covariate-only auxiliary features transductive. Historical blend contains mixed folds; not nested validation.')
    save(out/'metrics.json',m);report(m)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['build','teacher']);parser.add_argument('--first-fold',action='store_true')
    args=parser.parse_args();build() if args.mode=='build' else run(args)
