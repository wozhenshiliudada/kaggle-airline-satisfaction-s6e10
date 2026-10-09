"""Eight predeclared fusion candidates, checked against preserved official best."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT, sha
from evaluate_bagging_round import FOLDERS
from evaluate_context import paired
from train import read_data, COMPETITION, validate_submission


def run():
    tr, ts, sample, raw, ys = read_data(ROOT / 'data'); y = ys.to_numpy()
    hashes = {c: sha(ROOT / 'data' / c) for c in ('train.csv','test.csv','sample_submission.csv')}
    folds = list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    paths = {k: FOLDERS[k] for k in ('baseline','compact','route','cat_old','plain42','tuned','context42','lgb_context')}
    for mode in ('control','all'):
        paths[mode] = f'experiments/realmlp-route-auxiliary-{mode}-5fold-8ens'
    predictions = {}; metadata = {}
    for name, relative in paths.items():
        folder = ROOT / relative
        m = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        if m['competition'] != COMPETITION or m['input_sha256'] != hashes:
            raise ValueError('Input mismatch: ' + name)
        o = pd.read_csv(folder / 'oof.csv'); s = pd.read_csv(folder / 'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):
            raise ValueError('OOF alignment mismatch: ' + name)
        validate_submission(s,sample)
        if sha(folder / 'submission.csv') != m['submission_sha256']:
            raise ValueError('Prediction fingerprint mismatch: ' + name)
        p, pt = o.prediction.to_numpy(), s.satisfaction.to_numpy()
        if not np.isfinite(p).all() or p.min()<0 or p.max()>1:
            raise ValueError('Invalid OOF probabilities')
        if name in ('control','all','plain42'):
            for f,(_,b) in enumerate(folds,1):
                with np.load(folder / f'fold-{f}.npz') as saved:
                    if not np.array_equal(saved['val_idx'],b) or not np.allclose(saved['val_pred'],p[b],rtol=0,atol=1e-12):
                        raise ValueError('Five-fold cache mismatch')
        predictions[name]=(p,pt);metadata[name]=m
    def score(p):
        return dict(auc=float(roc_auc_score(y,p)),block_auc=[float(roc_auc_score(y[b],p[b])) for _,b in folds])
    def mix(a,b,w):return tuple((1-w)*a[i]+w*b[i] for i in range(2))
    def lmean(a,b):return tuple(expit(.5*logit(np.clip(a[i],1e-6,1-1e-6))+.5*logit(np.clip(b[i],1e-6,1-1e-6))) for i in range(2))
    tree=tuple(np.mean([predictions[k][i] for k in ('compact','route','cat_old')],axis=0) for i in range(2))
    context=lmean(predictions['lgb_context'],predictions['context42'])
    def replace_plain(member):return mix(mix(lmean(tree,member),predictions['tuned'],.1),context,.5)
    reconstructed=replace_plain(predictions['plain42'])
    reconstruction=[float(np.max(abs(reconstructed[i]-predictions['baseline'][i]))) for i in range(2)]
    if max(reconstruction)>1e-12:raise ValueError('Original baseline reconstruction differs')
    base=predictions['baseline']; baseline=score(base[0])
    candidates={}
    for name in ('control','all'):
        candidates[name+'_replace_plain']=replace_plain(predictions[name])
        for w in (.1,.25,.5):candidates[f'{name}_weight{w}']=mix(base,predictions[name],w)
    comparisons=[]
    for name,(p,pt) in candidates.items():
        s=score(p);delta=[a-b for a,b in zip(s['block_auc'],baseline['block_auc'])]
        comparisons.append(dict(candidate=name,**s,delta=s['auc']-baseline['auc'],fold_deltas=delta,
                    all_blocks_positive=bool(min(delta)>0),correlation=float(np.corrcoef(p,base[0])[0,1]),
                    mean_absolute_change=float(np.mean(abs(p-base[0])))))
    eligible=[r for r in comparisons if r['all_blocks_positive'] and r['delta']>0]
    selected=max(eligible,key=lambda r:r['auc']) if eligible else None
    result=dict(competition=COMPETITION,folds=5,seed=42,script_sha256=sha(Path(__file__)),model_description='Route neural + feature reconstruction',
                input_sha256=hashes,external_sha256=metadata['baseline']['external_sha256'],
                baseline=baseline,baseline_reconstruction_max_difference=reconstruction,
                model_scores={k:score(predictions[k][0]) for k in ('plain42','control','all')},
                matched_auxiliary_comparison=dict(paired=paired(y,predictions['control'][0],predictions['all'][0]),
                     fold_deltas=[a-b for a,b in zip(score(predictions['all'][0])['block_auc'],score(predictions['control'][0])['block_auc'])]),
                coarse_scan=sorted(comparisons,key=lambda r:r['auc'],reverse=True),selected_blend=selected,
                source_submission_sha256={k:m['submission_sha256'] for k,m in metadata.items()},
                validation_note='Eight fixed candidates declared before neural results: control/all each at 10/25/50 percent probability, or replace plain neural in preserved formula. Historical baseline uses three/five-fold members. Selection on reused OOF is exploratory, not nested CV. Auxiliary features are transductive, satisfaction excluded. No third-party prediction files.')
    if selected:
        p,pt=candidates[selected['candidate']]
        result['paired_against_previous_best']=paired(y,base[0],p)
        out=ROOT/'outputs-round8';out.mkdir(exist_ok=True)
        sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample)
        sub.to_csv(out/'submission.csv',index=False)
        pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=p)).to_csv(out/'oof.csv',index=False)
        result['submission_sha256']=sha(out/'submission.csv')
        (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'analyses'/'auxiliary-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
