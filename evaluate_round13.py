"""Fixed tenfold neural comparisons against the officially scored round11 baseline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT, sha
from auxiliary_round import hashes
from evaluate_context import paired
from round12_models import rebuilt
from train import read_data, validate_submission, COMPETITION


def run(ensemble=8):
    tr,ts,sample,raw,ys=read_data(ROOT/'data');y=ys.to_numpy();ih=hashes()
    paths=dict(baseline='outputs-round11',old5='outputs-round5',lgb='experiments/round9-lgb-auxiliary-5fold',
        context='experiments/realmlp-context-5fold-8ens',aux='experiments/realmlp-route-auxiliary-all-5fold-8ens',
        tabm='experiments/round9-tabm-auxiliary-5fold',travel='experiments/round11-experts-5fold',
        uncertainty='experiments/round11-uncertainty-5fold')
    paths.pop('uncertainty')
    paths['new']=f'experiments/round13-realmlp-route-auxiliary-10fold-{ensemble}ens'
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    preds,metadata={},{}
    for name,relative in paths.items():
        folder=ROOT/relative;m=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        if m['competition']!=COMPETITION or m['input_sha256']!=ih:raise ValueError('Input mismatch: '+name)
        if sha(folder/'submission.csv')!=m['submission_sha256']:raise ValueError('Source output changed: '+name)
        o=pd.read_csv(folder/'oof.csv');s=pd.read_csv(folder/'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError('OOF rows changed')
        validate_submission(s,sample)
        p=o.prediction.to_numpy()
        if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid OOF probability')
        if name not in ('baseline','old5'):
            native_folds=list(StratifiedKFold(10,shuffle=True,random_state=42).split(tr,ys)) if name=='new' else folds
            for f,(_,b) in enumerate(native_folds,1):
                with np.load(folder/f'fold-{f}.npz') as z:
                    if not np.array_equal(z['val_idx'],b) or not np.allclose(z['val_pred'],p[b],rtol=0,atol=1e-12):
                        raise ValueError('Saved outer fold changed: '+name)
        preds[name]=(p,s.satisfaction.to_numpy());metadata[name]=m
    def mix(a,b,w):return tuple((1-w)*a[i]+w*b[i] for i in range(2))
    def replace(p):
        return tuple(rebuilt(preds['lgb'][i],preds['old5'][i],preds['context'][i],p[i],preds['tabm'][i],preds['travel'][i]) for i in range(2))
    base=preds['baseline'];reb=replace(preds['aux'])
    error=[float(np.max(abs(base[i]-reb[i]))) for i in range(2)]
    if max(error)>1e-12:raise ValueError('Official baseline formula changed')
    def score(p):return dict(auc=float(roc_auc_score(y,p)),block_auc=[float(roc_auc_score(y[b],p[b])) for _,b in folds])
    bs=score(base[0]);candidates={f'neural_weight{w}':mix(base,preds['new'],w) for w in (.1,.25,.5)}
    candidates['neural_replace']=replace(preds['new'])
    rows=[]
    for name,(p,pt) in candidates.items():
        s=score(p);d=[a-b for a,b in zip(s['block_auc'],bs['block_auc'])]
        rows.append(dict(candidate=name,**s,delta=s['auc']-bs['auc'],fold_deltas=d,
            all_blocks_positive=bool(min(d)>0),correlation=float(np.corrcoef(p,base[0])[0,1])))
    eligible=[row for row in rows if row['all_blocks_positive'] and row['delta']>0]
    selected=max(eligible,key=lambda row:row['auc']) if eligible else None
    chosen=base[0].copy();leave=[]
    for f,(_,b) in enumerate(folds):
        a=np.concatenate([folds[i][1] for i in range(5) if i!=f]);best=roc_auc_score(y[a],base[0][a]);pick=None
        for row in rows:
            if min(v for i,v in enumerate(row['fold_deltas']) if i!=f)<=0:continue
            auc=roc_auc_score(y[a],candidates[row['candidate']][0][a])
            if auc>best:best=auc;pick=row['candidate']
        p=base[0] if pick is None else candidates[pick][0];chosen[b]=p[b]
        leave.append(dict(held_out_block=f+1,selected=pick or 'baseline',
            delta=float(roc_auc_score(y[b],p[b])-bs['block_auc'][f])))
    result=dict(competition=COMPETITION,folds=10,native_folds=10,evaluation_blocks=5,seed=42,members_per_fold=ensemble,new_model_directory=paths['new'],input_sha256=ih,
        external_sha256=metadata['lgb']['external_sha256'],model_description='Tenfold route auxiliary RealMLP with historical mixed-fold blend',
        baseline=bs,baseline_reconstruction_max_difference=error,model_scores={'new':score(preds['new'][0])},
        coarse_scan=sorted(rows,key=lambda row:row['auc'],reverse=True),selected_blend=selected,
        selection_stability=dict(auc=float(roc_auc_score(y,chosen)),held_out_blocks=leave),
        source_submission_sha256={k:m['submission_sha256'] for k,m in metadata.items()},
        script_sha256=sha(Path(__file__)),
        feature_code_sha256={p:sha(ROOT/p) for p in ('round13_tenfold.py','nn_round.py','auxiliary_neural_round.py','round12_models.py')},
        validation_note='Four fixed candidates for tenfold neural predictions. Five old evaluation blocks, historical three/five-fold members, safe inner target encoding and transductive auxiliary covariates. Sequential hypotheses and OOF selection remain adaptive; leave-block diagnosis is not nested model validation.')
    if selected:
        out=ROOT/'outputs-round13';out.mkdir(exist_ok=True);p,pt=candidates[selected['candidate']]
        sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
        pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=p)).to_csv(out/'oof.csv',index=False)
        result['submission_sha256']=sha(out/'submission.csv');result['paired_against_previous_best']=paired(y,base[0],p)
        (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'analyses/round13-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--ensemble',type=int,choices=[4,8],default=8)
    run(parser.parse_args().ensemble)
