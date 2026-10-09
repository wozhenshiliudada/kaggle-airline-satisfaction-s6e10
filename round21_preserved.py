"""Prospective fixed simplex ranking combiner preserving the native best model.

Keeps the ten-fold auxiliary network and round13 prediction as a mandatory
majority member. Reused full OOF and meta CV are adaptive and not nested.
This is a compatibility experiment; no round16 holdout re-scoring occurs.
"""
from datetime import datetime,timezone
import argparse
import time
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from round18_joint import ROOT,sha,save,load,idxsha,MEMBERS
from train import validate_submission,COMPETITION
from evaluate_context import paired

OUT=ROOT/'experiments/round21-preserved'
ANALYSIS=ROOT/'analyses/round21-preserved'
POOL={'best':'outputs-round13',**{n:'experiments/'+d for n,d in MEMBERS.items()}}
POOL['aux']='experiments/round13-realmlp-route-auxiliary-10fold-8ens'


def msg(**x):print(__import__('json').dumps(x),flush=True)


def freeze():
    if (OUT/'protocol.json').exists():raise ValueError('Already frozen')
    previous=[]
    for r,variants in [(19,['aux_route_embedding','raw_route_embedding']),
                       (20,['matched_control','route_onehot'])]:
        name='round19-route-tabm' if r==19 else 'round20-route-control'
        for v in variants:
            result=load(ROOT/'analyses'/name/(v+'-pilot.json'))
            if result['continue_full']:raise ValueError('Complete eligible previous pilot first')
            previous.append(dict(round=r,variant=v,outcome='Rejected by frozen pilot gate',
                                 best_delta=max(k['delta'] for k in result['candidates'])))
    save(ROOT/'evidence/round19-20-outcomes.json',dict(outcomes=previous,
            official_submission_eligible=False,official_submission_attempted=False))
    sources={}
    for n,d in POOL.items():
        folder=ROOT/d
        files=[folder/'metrics.json',folder/'oof.csv',folder/'submission.csv']
        if n!='best':files += [folder/'configuration.json',*sorted(folder.glob('fold-*.npz'))]
        sources[n]={str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for f in files}
    p=dict(round=21,created_at_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
        input_sha256=load(ROOT/'outputs-round13/metrics.json')['input_sha256'],members=POOL,
        native_source_sha256=sources,
        protected_sha256={s:sha(ROOT/s) for s in ['best-submission.json','last-submission.json',
            'team-request.json','outputs-round13/oof.csv','outputs-round13/submission.csv']},
        method='Nonnegative logit simplex with best-model weight at least .5',
        optimization=dict(pair_count=2000000,pair_seed=2026100621,l2_to_best=1e-3,
            max_iter=160,weight_bounds=[[.5,1.]]+[[0.,.5]]*8,sum_weights=1.),
        meta_cv=dict(folds=5,seeds=[42,142,242]),
        gates=dict(mean_delta=1e-5,each_repeat_positive=True,every_meta_block_positive=True),
        held_rule='Do not separately rescore H1/H2. No pristine labels remain.',
        limitations='Reused historical mixed-fold OOF, adaptive hypotheses and meta CV on '
            'overlapping base fits. Not fully nested or an independent generalization estimate. '
            'Native auxiliary keeps ten folds; other members keep their own five folds. '
            'Full-data coefficients are refit only after the crossfit gate passes.',
        previous_outcomes_sha256=sha(ROOT/'evidence/round19-20-outcomes.json'))
    save(OUT/'protocol.json',p);msg(stage='frozen',round=21,protocol_sha256=sha(OUT/'protocol.json'))


def library(p):
    if sha(__file__)!=p['script_sha256']:raise ValueError('Frozen source changed')
    for f,h in p['protected_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('Protected state changed')
    for f,h in p['input_sha256'].items():
        if sha(ROOT/'data'/f)!=h:raise ValueError('Input changed')
    tr=pd.read_csv(ROOT/'data/train.csv',usecols=['id','satisfaction'])
    sample=pd.read_csv(ROOT/'data/sample_submission.csv');y=tr.satisfaction.to_numpy(dtype='int8')
    P=[];T=[];audit={}
    for name,d in p['members'].items():
        folder=ROOT/d
        for f,h in p['native_source_sha256'][name].items():
            if sha(ROOT/f)!=h:raise ValueError('Native artifact changed: '+f)
        m=load(folder/'metrics.json')
        if m['input_sha256']!=p['input_sha256']:raise ValueError('Native inputs differ')
        o=pd.read_csv(folder/'oof.csv');s=pd.read_csv(folder/'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError('OOF alignment')
        if sha(folder/'submission.csv')!=m['submission_sha256']:raise ValueError('Native test fingerprint')
        validate_submission(s,sample)
        pred=o.prediction.to_numpy();test=s.satisfaction.to_numpy()
        if not np.isfinite(pred).all() or pred.min()<0 or pred.max()>1:raise ValueError('Native probability')
        if name!='best':
            nf=10 if name=='aux' else 5
            folds=StratifiedKFold(nf,shuffle=True,random_state=42).split(tr,y)
            rt=np.zeros(len(test))
            for f,(a,b) in enumerate(folds,1):
                with np.load(folder/f'fold-{f}.npz') as z:
                    if not np.array_equal(z['val_idx'],b):raise ValueError('Native fold mismatch')
                    if np.max(abs(z['val_pred']-pred[b]))>1e-12:raise ValueError('Native OOF reconstruction')
                    rt += z['test_pred']/nf
            if np.max(abs(rt-test))>1e-7:raise ValueError('Native test reconstruction')
            audit[name]=dict(native_folds=nf,reconstruction_max_error=float(np.max(abs(rt-test))))
        P.append(pred);T.append(test)
    save(OUT/'native-audit.json',audit)
    return tr,sample,y,np.column_stack(P),np.column_stack(T)


def fit(X,y,q,seed):
    rng=np.random.default_rng(seed);pos=X[y==1];neg=X[y==0]
    d=pos[rng.integers(len(pos),size=q['pair_count'])]-neg[rng.integers(len(neg),size=q['pair_count'])]
    initial=np.r_[1.,np.zeros(X.shape[1]-1)]
    def objective(w):
        margin=d@w;offset=w-initial
        return (float(np.logaddexp(0.,-margin).mean()+q['l2_to_best']*(offset@offset)),
                -(d.T@expit(-margin))/len(d)+2*q['l2_to_best']*offset)
    result=minimize(objective,initial,jac=True,method='SLSQP',bounds=q['weight_bounds'],
        constraints=[dict(type='eq',fun=lambda w:w.sum()-1.,jac=lambda w:np.ones_like(w))],
        options=dict(maxiter=q['max_iter'],ftol=1e-10))
    if not result.success or abs(result.x.sum()-1)>1e-7:raise ValueError('Simplex optimizer failed: '+result.message)
    if result.x[0]<.5-1e-7 or result.x.min()<-1e-7:raise ValueError('Preservation bounds failed')
    return result.x


def run():
    p=load(OUT/'protocol.json')
    if (ANALYSIS/'crossfit.json').exists():raise ValueError('Crossfit already evaluated')
    tr,sample,y,P,T=library(p);X=logit(np.clip(P,1e-6,1-1e-6));base=P[:,0]
    baseline=float(roc_auc_score(y,base));results=[];allpred=[];t=time.time()
    with threadpool_limits(limits=4):
        for seed in p['meta_cv']['seeds']:
            pred=np.full(len(y),np.nan);weights=[];deltas=[]
            for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y),1):
                w=fit(X[a],y[a],p['optimization'],p['optimization']['pair_seed']+f)
                pred[b]=expit(X[b]@w);weights.append(w.tolist())
                delta=float(roc_auc_score(y[b],pred[b])-roc_auc_score(y[b],base[b]));deltas.append(delta)
                msg(stage='meta_fold',seed=seed,fold=f,delta=delta,best_weight=float(w[0]),seconds=time.time()-t)
            score=float(roc_auc_score(y,pred));results.append(dict(seed=seed,auc=score,delta=score-baseline,
                        block_deltas=deltas,weights=weights));allpred.append(pred)
    mean=float(np.mean([r['delta'] for r in results]));worst=min(min(r['block_deltas']) for r in results)
    eligible=mean>=p['gates']['mean_delta'] and min(r['delta'] for r in results)>0 and worst>0
    result=dict(competition=COMPETITION,folds=5,input_sha256=p['input_sha256'],baseline_auc=baseline,
        mean_delta=mean,worst_meta_block_delta=worst,repeats=results,eligible=eligible,
        validation_note=p['limitations'],protocol_sha256=sha(OUT/'protocol.json'),seconds=time.time()-t)
    np.savez_compressed(OUT/'crossfit-predictions.npz',predictions=np.asarray(allpred))
    if eligible:
        with threadpool_limits(limits=4):w=fit(X,y,p['optimization'],p['optimization']['pair_seed'])
        pt=expit(logit(np.clip(T,1e-6,1-1e-6))@w)
        oof=np.mean(allpred,axis=0);auc=float(roc_auc_score(y,oof))
        out=ROOT/'outputs-round21';out.mkdir(exist_ok=True)
        sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample)
        sub.to_csv(out/'submission.csv',index=False)
        pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=oof)).to_csv(out/'oof.csv',index=False)
        result.update(selected_blend=dict(auc=auc,candidate='preserved simplex'),final_weights=w.tolist(),
            paired_fixed_prediction=paired(y,base,oof),submission_sha256=sha(out/'submission.csv'),
            external_sha256=load(ROOT/'outputs-round13/metrics.json')['external_sha256'],
            model_description='Preserved native tenfold auxiliary network with simplex ranking fusion')
        save(out/'metrics.json',result)
    save(ANALYSIS/'crossfit.json',result);msg(stage='complete',eligible=eligible,mean_delta=mean,worst_block_delta=worst)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);args=a.parse_args()
    if args.action=='freeze':freeze()
    else:run()
