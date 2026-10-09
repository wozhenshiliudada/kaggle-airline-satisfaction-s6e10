"""Evaluate fixed divergent candidates against the current round9 baseline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT,sha
from auxiliary_round import hashes
from evaluate_context import paired
from train import read_data,validate_submission,COMPETITION


def run():
    tr,ts,sample,raw,ys=read_data(ROOT/'data');y=ys.to_numpy();ih=hashes()
    paths=dict(baseline='outputs-round9',old5='outputs-round5',lgb='experiments/round9-lgb-auxiliary-5fold',
        context='experiments/realmlp-context-5fold-8ens',aux='experiments/realmlp-route-auxiliary-all-5fold-8ens',
        tabm='experiments/round9-tabm-auxiliary-5fold')
    modes=[]
    for mode in ('uncertainty','experts','joint'):
        folder='experiments/round11-'+mode+'-5fold'
        if (ROOT/folder/'metrics.json').exists():paths[mode]=folder;modes.append(mode)
    if not modes:raise ValueError('No full five-fold divergent models completed')
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    preds,metadata={},{}
    for name,relative in paths.items():
        folder=ROOT/relative;m=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        if m['competition']!=COMPETITION or m['input_sha256']!=ih:raise ValueError('Input mismatch: '+name)
        if sha(folder/'submission.csv')!=m['submission_sha256']:raise ValueError('Output fingerprint mismatch: '+name)
        o=pd.read_csv(folder/'oof.csv');s=pd.read_csv(folder/'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError('OOF alignment mismatch')
        validate_submission(s,sample)
        p=o.prediction.to_numpy()
        if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid OOF probability')
        if name not in ('baseline','old5'):
            for f,(_,b) in enumerate(folds,1):
                with np.load(folder/f'fold-{f}.npz') as z:
                    if not np.array_equal(z['val_idx'],b) or not np.allclose(z['val_pred'],p[b],rtol=0,atol=1e-12):raise ValueError('Saved fold mismatch')
        preds[name]=(p,s.satisfaction.to_numpy());metadata[name]=m
    def mix(a,b,w):return tuple((1-w)*a[i]+w*b[i] for i in range(2))
    def rebuilt(lg,aux):
        ctx=tuple(expit(.5*logit(np.clip(lg[i],1e-6,1-1e-6))+.5*logit(np.clip(preds['context'][i],1e-6,1-1e-6))) for i in range(2))
        return mix(mix(mix(preds['old5'],ctx,.5),aux,.25),preds['tabm'],.1)
    base=preds['baseline'];reb=rebuilt(preds['lgb'],preds['aux'])
    error=[float(np.max(abs(base[i]-reb[i]))) for i in range(2)]
    if max(error)>1e-12:raise ValueError('Official baseline formula changed')
    def score(p):return dict(auc=float(roc_auc_score(y,p)),block_auc=[float(roc_auc_score(y[b],p[b])) for _,b in folds])
    bs=score(base[0]);candidates={}
    for mode in modes:
        for w in (.1,.25,.5):candidates[f'{mode}_weight{w}']=mix(base,preds[mode],w)
        candidates[mode+'_replace']=rebuilt(preds['lgb'],preds[mode]) if mode=='joint' else rebuilt(preds[mode],preds['aux'])
    if len(modes)==3:
        average=tuple(np.mean([preds[mode][i] for mode in modes],axis=0) for i in range(2))
        for w in (.1,.25):candidates[f'new_average_weight{w}']=mix(base,average,w)
    rows=[]
    for name,(p,pt) in candidates.items():
        s=score(p);d=[a-b for a,b in zip(s['block_auc'],bs['block_auc'])]
        rows.append(dict(candidate=name,**s,delta=s['auc']-bs['auc'],fold_deltas=d,all_blocks_positive=bool(min(d)>0),
            correlation=float(np.corrcoef(p,base[0])[0,1])))
    eligible=[r for r in rows if r['all_blocks_positive'] and r['delta']>0]
    selected=max(eligible,key=lambda r:r['auc']) if eligible else None
    chosen=base[0].copy();leave=[]
    for f,(_,b) in enumerate(folds):
        a=np.concatenate([folds[i][1] for i in range(5) if i!=f]);best=roc_auc_score(y[a],base[0][a]);pick=None
        for r in rows:
            if min(v for i,v in enumerate(r['fold_deltas']) if i!=f)<=0:continue
            auc=roc_auc_score(y[a],candidates[r['candidate']][0][a])
            if auc>best:best=auc;pick=r['candidate']
        p=base[0] if pick is None else candidates[pick][0];chosen[b]=p[b]
        leave.append(dict(held_out_block=f+1,selected=pick or 'baseline',delta=float(roc_auc_score(y[b],p[b])-bs['block_auc'][f])))
    result=dict(competition=COMPETITION,folds=5,seed=42,completed_modes=modes,input_sha256=ih,
        external_sha256=metadata['lgb']['external_sha256'],model_description='Divergent structural features and experts',
        baseline=bs,baseline_reconstruction_max_difference=error,model_scores={k:score(preds[k][0]) for k in modes},
        coarse_scan=sorted(rows,key=lambda r:r['auc'],reverse=True),selected_blend=selected,
        selection_stability=dict(auc=float(roc_auc_score(y,chosen)),held_out_blocks=leave),
        source_submission_sha256={k:m['submission_sha256'] for k,m in metadata.items()},script_sha256=sha(Path(__file__)),
        validation_note='Fixed candidates before training; full supervised five-fold predictions with safe target encoding. Auxiliary covariates transductive. Reused historical OOF contains mixed fold counts and adaptive selection; leave-block diagnostic is not nested base-model validation.')
    if selected:
        out=ROOT/'outputs-round11';out.mkdir(exist_ok=True);p,pt=candidates[selected['candidate']]
        sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
        pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=p)).to_csv(out/'oof.csv',index=False)
        result['submission_sha256']=sha(out/'submission.csv');result['paired_against_previous_best']=paired(y,base[0],p)
        (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'analyses/round11-divergent-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
