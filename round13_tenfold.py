"""Larger outer training coverage for the existing route/auxiliary RealMLP."""
import argparse
import gc
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from auxiliary_round import AUXDIR,config_file,hashes
from auxiliary_neural_round import TEACHERDIR,route_keys,frames
from context_round import ROOT,sha
from improve import CACHE
from nn_round import prepare_frames,predict_chunks,report
from round12_models import digest,rebuilt
from train import read_data,validate_submission,COMPETITION


def screen(out,b,pv,y):
    paths=dict(base='outputs-round11',old5='outputs-round5',lgb='experiments/round9-lgb-auxiliary-5fold',
        context='experiments/realmlp-context-5fold-8ens',aux='experiments/realmlp-route-auxiliary-all-5fold-8ens',
        tabm='experiments/round9-tabm-auxiliary-5fold',travel='experiments/round11-experts-5fold')
    p={k:pd.read_csv(ROOT/v/'oof.csv',usecols=['prediction']).prediction.to_numpy()[b] for k,v in paths.items()}
    args=[p[k] for k in ('old5','context','aux','tabm','travel')]
    if np.max(abs(rebuilt(p['lgb'],*args)-p['base']))>1e-12:raise ValueError('Current baseline formula changed')
    replacement=rebuilt(p['lgb'],p['old5'],p['context'],pv,p['tabm'],p['travel'])
    candidates={str(w):(1-w)*p['base']+w*pv for w in (.1,.25,.5)};candidates['replace']=replacement
    result=dict(scope='first_native_tenfold_block',baseline_auc=float(roc_auc_score(y.iloc[b],p['base'])),
        model_auc=float(roc_auc_score(y.iloc[b],pv)),blend_auc={k:float(roc_auc_score(y.iloc[b],v)) for k,v in candidates.items()})
    result['continue_full']=max(result['blend_auc'].values())>result['baseline_auc']
    (out/'first-fold-screen.json').write_text(json.dumps(result,indent=2),encoding='utf-8');report(result)


def run(args):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4);start=time.time();tr,ts,sample,raw,y=read_data(ROOT/'data');ih=hashes()
    ref=ROOT/'experiments/realmlp-route-auxiliary-all-5fold-8ens'
    rc=json.loads((ref/'configuration.json').read_text(encoding='utf-8'))
    am=json.loads((AUXDIR/'metrics.json').read_text(encoding='utf-8'))
    tm=json.loads((TEACHERDIR/'metrics.json').read_text(encoding='utf-8'))
    if any(v!=ih for v in (rc['input_sha256'],am['configuration']['input_sha256'],tm['configuration']['input_sha256'])):
        raise ValueError('Source inputs changed')
    if sha(AUXDIR/'features.pkl')!=am['features_sha256'] or sha(TEACHERDIR/'probability.npy')!=tm['probability_sha256']:
        raise ValueError('Feature sources changed')
    if sha(ROOT/'data-original/data.csv')!=tm['configuration']['original_sha256']:raise ValueError('Original source changed')
    full,keys=prepare_frames(tr,ts,raw);rkeys=route_keys(full)
    data=joblib.load(CACHE)
    if not data['tr'].id.equals(tr.id) or not data['ts'].id.equals(ts.id):raise ValueError('Cached rows changed')
    pd.testing.assert_series_equal(data['y'],y)
    for c in ('original_lgb_probability','original_cat_probability'):
        full[c]=logit(data['base'][c].clip(1e-6,1-1e-6)).astype('float32')
    del data;gc.collect()
    full['original_xgb_logit']=logit(np.clip(np.load(TEACHERDIR/'probability.npy'),1e-6,1-1e-6)).astype('float32')
    for c in ('Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes'):
        full['route_count_'+c]=full[c].map(full[c].value_counts()).astype('float32')
    aux=pd.read_pickle(AUXDIR/'features.pkl')
    if not aux.index.equals(full.index):raise ValueError('Auxiliary rows changed')
    full=pd.concat([full,aux],axis=1);del aux;gc.collect()
    params=dict(rc['params'],n_ens=args.ensemble,verbosity=0)
    out=ROOT/'experiments'/f'round13-realmlp-route-auxiliary-10fold-{args.ensemble}ens'
    config_file(out,dict(script_sha256=sha(Path(__file__)),input_sha256=ih,params=params,folds=10,seed=42,
        reference_configuration_sha256=sha(ref/'configuration.json'),external_sha256=tm['configuration']['original_sha256'],
        auxiliary_sha256=am['features_sha256'],original_teacher_sha256=tm['probability_sha256'],
        source_code_sha256={p:sha(ROOT/p) for p in ('nn_round.py','auxiliary_neural_round.py','round12_models.py')},
        base_cache_sha256=sha(CACHE)))
    cats=list(full.select_dtypes(include='category'));oof=np.full(len(tr),np.nan);test=np.zeros(len(ts));scores=[]
    folds=list(StratifiedKFold(10,shuffle=True,random_state=42).split(tr,y))
    for f,(a,b) in enumerate(folds,1):
        path=out/f'fold-{f}.npz'
        if path.exists():
            with np.load(path) as z:
                if not np.array_equal(z['val_idx'],b):raise ValueError('Cached split changed')
                pv,pt=z['val_pred'],z['test_pred']
            score=json.loads((out/f'fold-{f}.json').read_text(encoding='utf-8'))
        else:
            prepared=frames(full,raw,keys,rkeys,a,b,len(tr),y)
            if f==1:
                expected=[digest(frame) for frame in prepared];del prepared;gc.collect()
                changed=y.copy();changed.iloc[b]=1-changed.iloc[b]
                check=frames(full,raw,keys,rkeys,a,b,len(tr),changed)
                if [digest(frame) for frame in check]!=expected:raise ValueError('Validation labels affect features')
                del check,changed;gc.collect()
                prepared=frames(full,raw,keys,rkeys,a,b,len(tr),y)
                if [digest(frame) for frame in prepared]!=expected:raise ValueError('Nondeterministic features')
                report(dict(validation_label_perturbation='PASS',native_folds=10))
            xa,xv,xt=prepared
            report(dict(stage='training',fold=f,native_folds=10,members=args.ensemble,features=len(xa.columns),rows=len(a)))
            model=RealMLP_TD_Classifier(**dict(params,random_state=42+f))
            model.fit(xa,y.iloc[a],X_val=xv,y_val=y.iloc[b],cat_col_names=cats)
            trainer=model.alg_interface_.min_trainer_kwargs
            if str(model.alg_interface_.device)!='cuda:0' or trainer['accelerator']!='gpu' or trainer['devices']!=[0]:
                raise RuntimeError('GPU trainer mismatch')
            pv,pt=predict_chunks(model,xv),predict_chunks(model,xt)
            if not all(np.isfinite(p).all() and p.min()>=0 and p.max()<=1 for p in (pv,pt)):
                raise ValueError('Invalid predictions')
            score=dict(fold=f,auc=float(roc_auc_score(y.iloc[b],pv)),features=len(xa.columns),
                members=args.ensemble,training_rows=len(a),elapsed_seconds=round(time.time()-start,2))
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            del prepared,xa,xv,xt,model;gc.collect();torch.cuda.empty_cache()
        oof[b]=pv;test+=pt/10;scores.append(score);report(score)
        if args.first_fold:screen(out,b,pv,y);return
    sub=sample.copy();sub.satisfaction=test;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=oof)).to_csv(out/'oof.csv',index=False)
    m=dict(competition=COMPETITION,folds=10,seed=42,input_sha256=ih,
        external_sha256=tm['configuration']['original_sha256'],submission_sha256=sha(out/'submission.csv'),
        selected_blend=dict(auc=float(roc_auc_score(y,oof))),fold_scores=scores,members_per_fold=args.ensemble,
        seconds=round(time.time()-start,2),
        validation_note='Ten native folds, inner target encoding still five folds; validation labels excluded. Auxiliary covariates transductive, original exact overlaps excluded. Not nested validation of the historical mixed-fold blend.')
    (out/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8');report(m)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--first-fold',action='store_true');parser.add_argument('--ensemble',type=int,choices=[4,8],default=8)
    run(parser.parse_args())
