"""Coarse probability-scale blend comparison, using only self-trained members."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from train import read_data,COMPETITION,validate_submission

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    tr,ts,sample,raw,y=read_data(ROOT/'data')
    hashes={n:sha(ROOT/'data'/n) for n in ('train.csv','test.csv','sample_submission.csv')}
    arrays={};tests={}
    paths={n:ROOT/'experiments'/n for n in ('compact','route-auc','cat','realmlp-5fold-8ens')}
    for name,folder in paths.items():
        m=json.loads((folder/'metrics.json').read_text())
        if m['competition']!=COMPETITION or m['input_sha256']!=hashes:raise ValueError('Input mismatch')
        o=pd.read_csv(folder/'oof.csv');t=pd.read_csv(folder/'test_predictions.csv')
        if not o.id.equals(tr.id) or not t.id.equals(ts.id) or not np.array_equal(o.satisfaction,y):
            raise ValueError('OOF or test alignment differs')
        arrays[name]=o.prediction.to_numpy();tests[name]=t.prediction.to_numpy()
        for p in (arrays[name],tests[name]):
            if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid probabilities')
    tree=np.mean([arrays[n] for n in ('compact','route-auc','cat')],axis=0)
    tt=np.mean([tests[n] for n in ('compact','route-auc','cat')],axis=0)
    nn=arrays['realmlp-5fold-8ens'];nt=tests['realmlp-5fold-8ens']
    def transform(p,kind):
        if kind=='logit':return logit(np.clip(p,1e-6,1-1e-6))
        if kind=='rank':return (rankdata(p)-.5)/len(p)
        return p
    folds=list(StratifiedKFold(3,shuffle=True,random_state=42).split(tr,y))
    baseline=(tree+nn)/2
    comparisons=[];predictions={};test_predictions={}
    for kind in ('probability','logit','rank'):
        for w in (.25,.5,.75):
            p=(1-w)*transform(tree,kind)+w*transform(nn,kind)
            pt=(1-w)*transform(tt,kind)+w*transform(nt,kind)
            if kind=='logit':p=expit(p);pt=expit(pt)
            key=f'{kind}-{w}'
            score=dict(transform=kind,neural_weight=w,auc=float(roc_auc_score(y,p)),
                       block_auc=[float(roc_auc_score(y.iloc[b],p[b])) for a,b in folds])
            comparisons.append(score);predictions[key]=p;test_predictions[key]=pt
    selected=max(comparisons,key=lambda c:c['auc'])
    key=f"{selected['transform']}-{selected['neural_weight']}"
    base_blocks=[float(roc_auc_score(y.iloc[b],baseline[b])) for a,b in folds]
    result=dict(competition=COMPETITION,folds=3,seed=42,
                model_description='Probability scale RealMLP tree blend',
                selected_blend=selected,comparisons=comparisons,
                baseline_auc=float(roc_auc_score(y,baseline)),
                block_improvements=[a-b for a,b in zip(selected['block_auc'],base_blocks)],
                input_sha256=hashes,external_sha256=sha(ROOT/'data-original'/'data.csv'),
                script_sha256=sha(Path(__file__)),
                validation_note='Nine declared candidates selected on same OOF; selection bias exists. Three evaluation blocks use tree folds, neural fits use five folds. Not nested CV. Rank transform uses unlabeled predictions separately for OOF and test.')
    out=ROOT/'outputs-round5-calibration';out.mkdir(exist_ok=True)
    sub=sample.copy();sub.satisfaction=test_predictions[key];validate_submission(sub,sample)
    sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':predictions[key]}).to_csv(out/'oof.csv',index=False)
    result['submission_sha256']=sha(out/'submission.csv')
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
