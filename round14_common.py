"""Frozen round14 source validation, baseline reconstruction, and screening."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from auxiliary_round import hashes
from context_round import ROOT, sha
from round12_models import rebuilt
from train import COMPETITION, read_data, validate_submission

PATHS = dict(baseline='outputs-round13', old5='outputs-round5',
    lgb='experiments/round9-lgb-auxiliary-5fold', context='experiments/realmlp-context-5fold-8ens',
    aux='experiments/round13-realmlp-route-auxiliary-10fold-8ens',
    tabm='experiments/round9-tabm-auxiliary-5fold', travel='experiments/round11-experts-5fold')
BASE_SHA = 'b3451e0310cf0892ebd89cab80f400a5c0fe7fe3725ff93965e0499bf9225f93'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sources(tr, sample, ys):
    ih=hashes(); y=ys.to_numpy(); preds={}; metadata={}; fingerprints={}
    for name, relative in PATHS.items():
        folder=ROOT/relative; m=load(folder/'metrics.json')
        if m['competition']!=COMPETITION or m['input_sha256']!=ih:
            raise ValueError('Source input mismatch: '+name)
        if sha(folder/'submission.csv')!=m['submission_sha256']:
            raise ValueError('Source prediction changed: '+name)
        o=pd.read_csv(folder/'oof.csv'); s=pd.read_csv(folder/'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):
            raise ValueError('Source OOF row mismatch: '+name)
        validate_submission(s,sample); p=o.prediction.to_numpy(); pt=s.satisfaction.to_numpy()
        if not np.isfinite(p).all() or p.min()<0 or p.max()>1:
            raise ValueError('Invalid OOF: '+name)
        if name not in ('baseline','old5'):
            n=10 if name=='aux' else 5; test=np.zeros(len(sample))
            for f,(_,b) in enumerate(StratifiedKFold(n,shuffle=True,random_state=42).split(tr,ys),1):
                with np.load(folder/f'fold-{f}.npz') as z:
                    if not np.array_equal(z['val_idx'],b) or not np.allclose(z['val_pred'],p[b],rtol=0,atol=1e-12):
                        raise ValueError('Source native fold changed: '+name)
                    test+=z['test_pred']/n
            if not np.allclose(test,pt,rtol=0,atol=1e-12):
                raise ValueError('Source test averaging changed: '+name)
        preds[name]=(p,pt); metadata[name]=m
        fingerprints[relative+'/submission.csv']=m['submission_sha256']
        fingerprints[relative+'/oof.csv']=sha(folder/'oof.csv')
    if metadata['baseline']['submission_sha256']!=BASE_SHA:
        raise ValueError('Official baseline changed')
    reconstructed=reconstruct(preds)
    error=[float(np.max(abs(reconstructed[i]-preds['baseline'][i]))) for i in (0,1)]
    if max(error)>1e-12: raise ValueError('Baseline formula changed')
    return preds,metadata,fingerprints,error


def reconstruct(preds, mode=None, new=None):
    return tuple(rebuilt(new[i] if mode=='bins' else preds['lgb'][i], preds['old5'][i],
        preds['context'][i],preds['aux'][i],new[i] if mode=='tabm' else preds['tabm'][i],
        preds['travel'][i]) for i in (0,1))


def model_candidates(preds, mode, new):
    base=preds['baseline']
    result={mode+'_weight'+str(w):tuple((1-w)*base[i]+w*new[i] for i in (0,1)) for w in (.05,.1,.25)}
    result[mode+'_replace']=reconstruct(preds,mode,new)
    return result


def screen(out,mode,b,pv,y):
    tr,ts,sample,raw,ys=read_data(ROOT/'data')
    preds,metadata,fingerprints,error=sources(tr,sample,ys)
    short={k:(v[0][b],v[0][b]) for k,v in preds.items()}
    candidates=model_candidates(short,mode,(pv,pv)); by=y.iloc[b].to_numpy()
    baseline=float(roc_auc_score(by,short['baseline'][0]))
    result=dict(scope='first_fivefold_block_only',mode=mode,baseline_auc=baseline,
        model_auc=float(roc_auc_score(by,pv)),blend_auc={k:float(roc_auc_score(by,v[0])) for k,v in candidates.items()},
        baseline_reconstruction_max_difference=error)
    result['continue_full']=max(result['blend_auc'].values())>baseline
    save(out/'first-fold-screen.json',result);print(json.dumps(result,indent=2),flush=True)
