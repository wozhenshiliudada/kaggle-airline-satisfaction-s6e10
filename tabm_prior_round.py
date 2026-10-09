"""Post-round9 fixed calibration diagnostic with no held-out target labels.

Per-fold intercept maps mean unlabeled TEST prediction to outer TRAIN prior.
The same map applies to validation and test probabilities. This is explicitly
transductive and assumes no large class-prior shift between train and test.
Three fixed blend weights (10/25/50%) compare against the saved round9 candidate.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT, sha
from auxiliary_round import hashes, config_file
from evaluate_context import paired
from train import read_data, validate_submission, COMPETITION


def run():
    tr, ts, sample, raw, ys = read_data(ROOT / 'data')
    y = ys.to_numpy()
    folders = dict(tabm=ROOT/'experiments/round9-tabm-auxiliary-5fold',
        lgb=ROOT/'experiments/round9-lgb-auxiliary-5fold', context=ROOT/'experiments/realmlp-context-5fold-8ens',
        old5=ROOT/'outputs-round5', aux=ROOT/'experiments/realmlp-route-auxiliary-all-5fold-8ens',
        baseline=ROOT/'outputs-round9')
    predictions, meta = {}, {}
    for name, folder in folders.items():
        m = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        if m['competition'] != COMPETITION or m['input_sha256'] != hashes():
            raise ValueError('Input fingerprint mismatch')
        if sha(folder/'submission.csv') != m['submission_sha256']:
            raise ValueError('Prediction fingerprint mismatch')
        o = pd.read_csv(folder/'oof.csv'); s = pd.read_csv(folder/'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):
            raise ValueError('OOF alignment mismatch')
        validate_submission(s,sample)
        predictions[name] = (o.prediction.to_numpy(),s.satisfaction.to_numpy())
        meta[name] = m
    out = ROOT/'experiments/round10-tabm-prior-normalized-5fold'
    config = dict(script_sha256=sha(Path(__file__)), input_sha256=hashes(),
        source_submission_sha256={k:m['submission_sha256'] for k,m in meta.items()},
        rule='Set mean test probability to outer training prevalence via a logit intercept',
        blend_weights=[.1,.25,.5], folds=5, seed=42,
        transductive=True, held_out_labels_used=False)
    config_file(out,config)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    oof=np.full(len(tr),np.nan); test=np.zeros(len(ts)); records=[]
    for f,(a,b) in enumerate(folds,1):
        source=folders['tabm']/f'fold-{f}.npz'
        with np.load(source) as z:
            if not np.array_equal(z['val_idx'],b) or not np.allclose(z['val_pred'],predictions['tabm'][0][b],rtol=0,atol=1e-12):
                raise ValueError('Source fold mismatch')
            pv,pt=z['val_pred'],z['test_pred']
        prior=float(y[a].mean())
        lt=logit(np.clip(pt.astype('float64'),1e-7,1-1e-7))
        lv=logit(np.clip(pv.astype('float64'),1e-7,1-1e-7))
        offset=float(brentq(lambda v:float(expit(lt+v).mean())-prior,-4.,4.))
        changed=y.copy();changed[b]=1-changed[b]
        if float(changed[a].mean()) != prior:
            raise ValueError('Validation label perturbation changed calibration')
        nv,nt=expit(lv+offset),expit(lt+offset)
        if abs(nt.mean()-prior)>1e-10:
            raise ValueError('Prior normalization mismatch')
        if abs(roc_auc_score(y[b],nv)-roc_auc_score(y[b],pv))>1e-9:
            raise ValueError('Monotone calibration changed within-fold AUC')
        oof[b]=nv;test+=nt/5
        np.savez_compressed(out/f'fold-{f}.npz',val_idx=b,val_pred=nv,test_pred=nt)
        records.append(dict(fold=f,outer_training_prior=prior,logit_intercept=offset,
            raw_test_mean=float(pt.mean()),normalized_test_mean=float(nt.mean()),
            raw_validation_mean=float(pv.mean()),normalized_validation_mean=float(nv.mean()),
            source_checkpoint_sha256=sha(source),validation_label_perturbation='PASS'))
    pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=oof)).to_csv(out/'oof.csv',index=False)
    sub=sample.copy();sub.satisfaction=test;validate_submission(sub,sample)
    sub.to_csv(out/'submission.csv',index=False)
    model_m=dict(competition=COMPETITION,folds=5,seed=42,input_sha256=hashes(),
        selected_blend=dict(auc=float(roc_auc_score(y,oof))),
        submission_sha256=sha(out/'submission.csv'),fold_scores=records,
        validation_note='Only test covariates and outer-training prevalence fit each calibration map. Validation labels excluded. Transductive; assumes stable train/test class prevalence.')
    (out/'metrics.json').write_text(json.dumps(model_m,indent=2),encoding='utf-8')
    def mix(a,b,w):return tuple((1-w)*a[i]+w*b[i] for i in range(2))
    context=tuple(expit(.5*logit(np.clip(predictions['lgb'][i],1e-6,1-1e-6))+.5*logit(np.clip(predictions['context'][i],1e-6,1-1e-6))) for i in range(2))
    tree_replaced=mix(mix(predictions['old5'],context,.5),predictions['aux'],.25)
    baseline=predictions['baseline']
    if max(float(np.max(abs(mix(tree_replaced,predictions['tabm'],.1)[i]-baseline[i]))) for i in range(2))>1e-12:
        raise ValueError('Round9 baseline formula mismatch')
    def score(p):return dict(auc=float(roc_auc_score(y,p)),block_auc=[float(roc_auc_score(y[b],p[b])) for _,b in folds])
    bs=score(baseline[0]); candidates={str(w):mix(tree_replaced,(oof,test),w) for w in (.1,.25,.5)}
    rows=[]
    for w,(p,pt) in candidates.items():
        s=score(p);d=[aa-bb for aa,bb in zip(s['block_auc'],bs['block_auc'])]
        rows.append(dict(candidate='normalized_tabm_'+w,weight=float(w),**s,delta=s['auc']-bs['auc'],
            fold_deltas=d,all_blocks_positive=bool(min(d)>0)))
    eligible=[r for r in rows if r['all_blocks_positive'] and r['delta']>0]
    selected=max(eligible,key=lambda r:r['auc']) if eligible else None
    leave=[];chosen=baseline[0].copy()
    for f,(_,b) in enumerate(folds):
        a=np.concatenate([folds[i][1] for i in range(5) if i!=f])
        pick=None;best=roc_auc_score(y[a],baseline[0][a])
        for r in rows:
            if min(v for i,v in enumerate(r['fold_deltas']) if i!=f)<=0:continue
            p=candidates[str(r['weight'])][0];auc=roc_auc_score(y[a],p[a])
            if auc>best:pick=r;best=auc
        p=baseline[0] if pick is None else candidates[str(pick['weight'])][0]
        chosen[b]=p[b]
        leave.append(dict(held_out_block=f+1,selected='baseline' if pick is None else pick['candidate'],
            delta=float(roc_auc_score(y[b],p[b])-bs['block_auc'][f])))
    result=dict(competition=COMPETITION,folds=5,seed=42,input_sha256=hashes(),
        model_description='Auxiliary LightGBM plus prior-normalized route TabM',
        external_sha256=meta['lgb']['external_sha256'],baseline=bs,
        model_scores=dict(raw_tabm=score(predictions['tabm'][0]),normalized_tabm=score(oof)),
        normalization=records,coarse_scan=rows,selected_blend=selected,
        selection_stability=dict(auc=float(roc_auc_score(y,chosen)),held_out_blocks=leave),
        source_submission_sha256={k:m['submission_sha256'] for k,m in meta.items()},
        script_sha256=sha(Path(__file__)),
        validation_note='Three weights fixed before normalization results. Per-fold test covariates and training class prior fit intercept; no held-out labels. Historical OOF reused and transductive, not nested CV. Stable class prevalence assumption must be verified by official scoring.')
    if selected:
        p,pt=candidates[str(selected['weight'])]
        dest=ROOT/'outputs-round10';dest.mkdir(exist_ok=True)
        final=sample.copy();final.satisfaction=pt;validate_submission(final,sample)
        final.to_csv(dest/'submission.csv',index=False)
        pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=p)).to_csv(dest/'oof.csv',index=False)
        result['submission_sha256']=sha(dest/'submission.csv')
        result['paired_against_previous_best']=paired(y,baseline[0],p)
        (dest/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'analyses/round10-prior-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
