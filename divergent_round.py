"""Three fixed structural experiments; preserve all previous configurations."""
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from auxiliary_round import AUXDIR, config_file, hashes
from auxiliary_neural_round import TEACHERDIR, route_keys, frames
from context_round import ROOT, sha, context_keys, context_blocks, add_context, SERVICES
from improve import CACHE, CATS, make_keys, encoded_fold
from nn_round import prepare_frames, predict_chunks, report
from train import read_data, validate_submission, COMPETITION

UNCERTAINTY = ROOT/'experiments/round11-auxiliary-uncertainty'


def build():
    # Deliberately exclude satisfaction and id when reconstructing auxiliary outputs.
    tr=pd.read_csv(ROOT/'data/train.csv',usecols=lambda c:c not in ('id','satisfaction'))
    ts=pd.read_csv(ROOT/'data/test.csv',usecols=lambda c:c!='id')
    full=pd.concat([tr,ts],ignore_index=True)
    del tr,ts
    for c in CATS:full[c]=pd.Categorical(full[c]).codes.astype('float32')
    full=full.astype('float32');full['Arrival Delay in Minutes']=full['Arrival Delay in Minutes'].fillna(-1)
    aux_m=json.loads((AUXDIR/'metrics.json').read_text(encoding='utf-8'))
    if aux_m['configuration']['input_sha256']!=hashes():raise ValueError('Auxiliary input mismatch')
    config_file(UNCERTAINTY,dict(script_sha256=sha(Path(__file__)),input_sha256=hashes(),
        auxiliary_configuration_sha256=sha(AUXDIR/'configuration.json'),label_used=False))
    cached=UNCERTAINTY/'features.pkl'
    if cached.exists():
        m=json.loads((UNCERTAINTY/'metrics.json').read_text(encoding='utf-8'))
        if m['features_sha256']!=sha(cached):raise ValueError('Uncertainty fingerprint mismatch')
        return
    additions={}; checkpoints={}
    splits=list(KFold(5,shuffle=True,random_state=0).split(full))
    for c in SERVICES+['Class','Type of Travel','Customer Type']:
        values=np.sort(full[c].unique());target=np.searchsorted(values,full[c].to_numpy())
        p=np.full((len(full),len(values)),np.nan,dtype='float32')
        for f,(_,b) in enumerate(splits,1):
            path=AUXDIR/(c.replace('/','_').replace(' ','_')+f'-fold-{f}.npz')
            with np.load(path) as z:
                if not np.array_equal(z['val_idx'],b):raise ValueError('Auxiliary fold exclusion mismatch')
                p[b]=z['probability']
            checkpoints[str(path.relative_to(ROOT))]=sha(path)
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-5):raise ValueError('Invalid auxiliary probabilities')
        safe=np.clip(p,1e-7,1.)
        additions['uncert_entropy_'+c]=(-(safe*np.log(safe)).sum(axis=1)/np.log(len(values))).astype('float32')
        largest=np.partition(p,-2,axis=1)[:,-2:]
        additions['uncert_margin_'+c]=(largest[:,1]-largest[:,0]).astype('float32')
        if c in SERVICES:
            mean=p@values;std=np.sqrt(np.maximum(p@(values**2)-mean**2,0))
            additions['uncert_std_'+c]=std.astype('float32')
            additions['uncert_residual_z_'+c]=((full[c].to_numpy()-mean)/(std+.1)).astype('float32')
            tails=np.cumsum(p[:,::-1],axis=1)[:,::-1]
            additions['uncert_tail_'+c]=tails[np.arange(len(full)),target].astype('float32')
        report(dict(stage='uncertainty_features',column=c))
    result=pd.DataFrame(additions)
    if not np.isfinite(result.to_numpy()).all() or not result.index.equals(pd.RangeIndex(len(full))):raise ValueError('Invalid uncertainty matrix')
    result.to_pickle(cached)
    m=dict(input_sha256=hashes(),shape=list(result.shape),features_sha256=sha(cached),
        checkpoint_sha256=checkpoints,label_used=False,
        validation_note='Only raw covariates and previously row-excluding auxiliary probabilities. Transductive train/test covariates; no satisfaction labels.')
    (UNCERTAINTY/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    report(dict(stage='uncertainty_complete',shape=m['shape'],features_sha256=m['features_sha256']))


def run(args):
    start=time.time();tr,ts,sample,raw,y=read_data(ROOT/'data')
    ih=hashes();ref=ROOT/'experiments/round9-lgb-auxiliary-5fold'
    tree_config=json.loads((ref/'configuration.json').read_text(encoding='utf-8'))
    tree_m=json.loads((ref/'metrics.json').read_text(encoding='utf-8'))
    aux_m=json.loads((AUXDIR/'metrics.json').read_text(encoding='utf-8'))
    if tree_m['input_sha256']!=ih or aux_m['configuration']['input_sha256']!=ih or sha(AUXDIR/'features.pkl')!=aux_m['features_sha256']:
        raise ValueError('Input mismatch')
    if sha(ROOT/'data-original/data.csv')!=tree_m['external_sha256']:raise ValueError('External fingerprint mismatch')
    data=joblib.load(CACHE)
    if not data['tr'].id.equals(tr.id) or not data['ts'].id.equals(ts.id):raise ValueError('Cached row mismatch')
    pd.testing.assert_series_equal(data['y'],y)
    aux=pd.read_pickle(AUXDIR/'features.pkl')
    if not aux.index.equals(pd.RangeIndex(len(tr)+len(ts))):raise ValueError('Auxiliary row mismatch')
    source_code={p:sha(ROOT/p) for p in ('improve.py','context_round.py','nn_round.py','auxiliary_neural_round.py')}
    params=dict(tree_config['params'])
    extra_hash=None
    if args.mode in ('uncertainty','experts'):
        full=data['base'].copy();full[aux.columns]=aux
        raw_full=pd.concat([tr[raw],ts[raw]],ignore_index=True)
        keys=make_keys(raw_full,raw,True);ckeys=context_keys(raw_full)
        if args.mode=='uncertainty':
            um=json.loads((UNCERTAINTY/'metrics.json').read_text(encoding='utf-8'))
            extra_hash=sha(UNCERTAINTY/'features.pkl')
            if um['input_sha256']!=ih or um['features_sha256']!=extra_hash:raise ValueError('Uncertainty cache mismatch')
            u=pd.read_pickle(UNCERTAINTY/'features.pkl');full[u.columns]=u;del u
        gate=raw_full['Type of Travel'].to_numpy()
        del raw_full
        def prepare(a,b,target):
            result=encoded_fold(full,keys,a,b,len(tr),target,True)
            add_context(result,context_blocks(ckeys,a,b,len(tr),target),ckeys,a,b,len(tr))
            return result
    else:
        import torch
        from pytabkit import RealMLP_TD_Classifier
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(4)
        full,keys=prepare_frames(tr,ts,raw);rkeys=route_keys(full);ckeys=context_keys(full)
        for c in ('original_lgb_probability','original_cat_probability'):
            full[c]=logit(data['base'][c].clip(1e-6,1-1e-6)).astype('float32')
        teacher_m=json.loads((TEACHERDIR/'metrics.json').read_text(encoding='utf-8'))
        if sha(TEACHERDIR/'probability.npy')!=teacher_m['probability_sha256']:raise ValueError('Teacher fingerprint mismatch')
        full['original_xgb_logit']=logit(np.clip(np.load(TEACHERDIR/'probability.npy'),1e-6,1-1e-6)).astype('float32')
        for c in ('Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes'):
            full['route_count_'+c]=full[c].map(full[c].value_counts()).astype('float32')
        full[aux.columns]=aux
        nn_config=json.loads((ROOT/'experiments/realmlp-route-auxiliary-all-5fold-8ens/configuration.json').read_text(encoding='utf-8'))
        params=dict(nn_config['params'],verbosity=0)
        def prepare(a,b,target):
            result=frames(full,raw,keys,rkeys,a,b,len(tr),target)
            blocks=context_blocks(ckeys,a,b,len(tr),target)
            for frame,block in zip(result,blocks):frame[block.columns]=block
            return result
    del data,aux
    out=ROOT/'experiments'/('round11-'+args.mode+'-5fold')
    config_file(out,dict(script_sha256=sha(Path(__file__)),input_sha256=ih,params=params,
        source_code_sha256=source_code,external_sha256=tree_m['external_sha256'],
        auxiliary_sha256=aux_m['features_sha256'],uncertainty_sha256=extra_hash,
        mode=args.mode,folds=5,seed=42))
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y))
    oof=np.full(len(tr),np.nan);prediction=np.zeros(len(ts));scores=[]
    for f,(a,b) in enumerate(folds,1):
        with np.load(ref/f'fold-{f}.npz') as z:
            if not np.array_equal(z['val_idx'],b):raise ValueError('Reference fold mismatch')
        path=out/f'fold-{f}.npz'
        if path.exists():
            with np.load(path) as z:
                if not np.array_equal(z['val_idx'],b):raise ValueError('Cache fold mismatch')
                pv,pt=z['val_pred'],z['test_pred']
            score=json.loads((out/f'fold-{f}.json').read_text(encoding='utf-8'))
        else:
            xa,xv,xt=prepare(a,b,y)
            if f==1:
                alternate=y.copy();alternate.iloc[b]=1-alternate.iloc[b]
                perturbed=prepare(a,b,alternate)
                for actual,changed in zip((xa,xv,xt),perturbed):pd.testing.assert_frame_equal(actual,changed,check_exact=True)
                del perturbed;gc.collect();report(dict(mode=args.mode,validation_label_perturbation='PASS'))
            report(dict(stage='training',mode=args.mode,fold=f,features=len(xa.columns)))
            details=[]
            if args.mode!='joint':
                pv=np.full(len(b),np.nan);pt=np.full(len(ts),np.nan)
                groups=sorted(np.unique(gate)) if args.mode=='experts' else ['all']
                for group in groups:
                    ma=np.ones(len(a),dtype=bool) if group=='all' else gate[a]==group
                    mb=np.ones(len(b),dtype=bool) if group=='all' else gate[b]==group
                    mt=np.ones(len(ts),dtype=bool) if group=='all' else gate[len(tr):]==group
                    fit=lgb.Dataset(xa.loc[ma],y.iloc[a].to_numpy()[ma]);val=lgb.Dataset(xv.loc[mb],y.iloc[b].to_numpy()[mb],reference=fit)
                    model=lgb.train(dict(params,seed=42+f),fit,num_boost_round=3500,valid_sets=[val],valid_names=['val'],
                        callbacks=[lgb.early_stopping(180,first_metric_only=True,verbose=False),lgb.log_evaluation(1000)])
                    pv[mb]=model.predict(xv.loc[mb]);pt[mt]=model.predict(xt.loc[mt])
                    details.append(dict(group=group,rows=int(ma.sum()),iterations=model.best_iteration))
                    del model,fit,val;gc.collect()
            else:
                cats=list(xa.select_dtypes(include='category').columns)
                model=RealMLP_TD_Classifier(**dict(params,random_state=42+f))
                model.fit(xa,y.iloc[a],X_val=xv,y_val=y.iloc[b],cat_col_names=cats)
                trainer=model.alg_interface_.min_trainer_kwargs
                if str(model.alg_interface_.device)!='cuda:0' or trainer['accelerator']!='gpu':raise RuntimeError('Neural device mismatch')
                pv,pt=predict_chunks(model,xv),predict_chunks(model,xt);del model
            if not all(np.isfinite(p).all() and p.min()>=0 and p.max()<=1 for p in (pv,pt)):raise ValueError('Invalid probability')
            score=dict(fold=f,auc=float(roc_auc_score(y.iloc[b],pv)),features=len(xa.columns),
                seconds_elapsed=round(time.time()-start,2),experts=details)
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            del xa,xv,xt;gc.collect()
            if args.mode=='joint':torch.cuda.empty_cache()
        oof[b]=pv;prediction+=pt/5;scores.append(score);report(dict(mode=args.mode,**score))
        if args.first_fold:
            baseline=pd.read_csv(ROOT/'outputs-round9/oof.csv').prediction.to_numpy()[b]
            old5=pd.read_csv(ROOT/'outputs-round5/oof.csv').prediction.to_numpy()[b]
            nl=pd.read_csv(ref/'oof.csv').prediction.to_numpy()[b]
            nc=pd.read_csv(ROOT/'experiments/realmlp-context-5fold-8ens/oof.csv').prediction.to_numpy()[b]
            na=pd.read_csv(ROOT/'experiments/realmlp-route-auxiliary-all-5fold-8ens/oof.csv').prediction.to_numpy()[b]
            tm=pd.read_csv(ROOT/'experiments/round9-tabm-auxiliary-5fold/oof.csv').prediction.to_numpy()[b]
            def rebuilt(lgbp,auxp):
                ctx=expit(.5*logit(np.clip(lgbp,1e-6,1-1e-6))+.5*logit(np.clip(nc,1e-6,1-1e-6)))
                return .9*(.75*(.5*old5+.5*ctx)+.25*auxp)+.1*tm
            if np.max(abs(rebuilt(nl,na)-baseline))>1e-12:raise ValueError('Official baseline formula mismatch')
            replace=rebuilt(nl,pv) if args.mode=='joint' else rebuilt(pv,na)
            candidates={str(w):(1-w)*baseline+w*pv for w in (.1,.25,.5)};candidates['replace']=replace
            screen=dict(scope='first_fold_only',mode=args.mode,model_auc=score['auc'],
                baseline_auc=float(roc_auc_score(y.iloc[b],baseline)),
                blend_auc={k:float(roc_auc_score(y.iloc[b],p)) for k,p in candidates.items()})
            screen['continue_full']=max(screen['blend_auc'].values())>screen['baseline_auc']
            (out/'first-fold-screen.json').write_text(json.dumps(screen,indent=2),encoding='utf-8');report(screen);return
    sub=sample.copy();sub.satisfaction=prediction;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=oof)).to_csv(out/'oof.csv',index=False)
    metrics=dict(competition=COMPETITION,folds=5,seed=42,model_description='Divergent '+args.mode,
        input_sha256=ih,external_sha256=tree_m['external_sha256'],submission_sha256=sha(out/'submission.csv'),
        selected_blend=dict(auc=float(roc_auc_score(y,oof))),fold_scores=scores,seconds=round(time.time()-start,2),
        validation_note='Matched outer folds and inner target encoding; validation labels excluded from features. Auxiliary reconstruction is transductive. Historical blend uses mixed fold counts; not nested validation.')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');report(metrics)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','uncertainty','experts','joint']);p.add_argument('--first-fold',action='store_true')
    args=p.parse_args();build() if args.mode=='build' else run(args)
