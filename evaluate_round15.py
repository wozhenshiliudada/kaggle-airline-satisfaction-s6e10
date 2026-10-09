"""Four fixed tenfold tree candidates against the scored round13 baseline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT,sha
from auxiliary_round import hashes
from evaluate_context import paired
from round14_common import sources,load,save,model_candidates
from train import read_data,validate_submission,COMPETITION


def run():
    tr,ts,sample,raw,ys=read_data(ROOT/'data');y=ys.to_numpy();ih=hashes()
    preds,metadata,fingerprints,error=sources(tr,sample,ys);base=preds['baseline']
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    folder=ROOT/'experiments/round15-lgb-10fold';screen=load(folder/'first-fold-screen.json')
    candidates={};status='stopped_after_first_fold';new_auc=None
    if screen['continue_full']:
        m=load(folder/'metrics.json');cfg=load(folder/'configuration.json')
        if m['competition']!=COMPETITION or m['input_sha256']!=ih or cfg['script_sha256']!=sha(ROOT/'round15_models.py'):
            raise ValueError('Native inputs or code changed')
        if m['folds']!=10 or cfg['folds']!=10:raise ValueError('Native fold count changed')
        for name,digest in cfg['source_code_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Native dependency changed: '+name)
        if sha(folder/'submission.csv')!=m['submission_sha256']:raise ValueError('Native submission changed')
        o=pd.read_csv(folder/'oof.csv');s=pd.read_csv(folder/'submission.csv');validate_submission(s,sample)
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError('Native OOF alignment changed')
        p=o.prediction.to_numpy();pt=s.satisfaction.to_numpy();test=np.zeros(len(ts))
        if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid native OOF')
        for f,(_,b) in enumerate(StratifiedKFold(10,shuffle=True,random_state=42).split(tr,ys),1):
            with np.load(folder/f'fold-{f}.npz') as z:
                if not np.array_equal(z['val_idx'],b) or not np.allclose(z['val_pred'],p[b],rtol=0,atol=1e-12):
                    raise ValueError('Native checkpoint changed')
                test+=z['test_pred']/10
            fingerprints[str((folder/f'fold-{f}.npz').relative_to(ROOT))]=sha(folder/f'fold-{f}.npz')
        if not np.allclose(test,pt,rtol=0,atol=1e-12):raise ValueError('Native test mean changed')
        for name in ('configuration.json','metrics.json','oof.csv','submission.csv'):
            fingerprints[str((folder/name).relative_to(ROOT))]=sha(folder/name)
        candidates={k.replace('bins','tree'):v for k,v in model_candidates(preds,'bins',(p,pt)).items()}
        status='completed_ten_folds';new_auc=float(roc_auc_score(y,p))
    def score(p):return dict(auc=float(roc_auc_score(y,p)),block_auc=[float(roc_auc_score(y[b],p[b])) for _,b in folds])
    bs=score(base[0]);rows=[]
    for name,(p,pt) in candidates.items():
        if not all(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in (p,pt)):
            raise ValueError('Invalid candidate')
        s=score(p);d=[x-z for x,z in zip(s['block_auc'],bs['block_auc'])]
        rows.append(dict(candidate=name,**s,delta=s['auc']-bs['auc'],fold_deltas=d,all_blocks_positive=min(d)>0))
    selected=max((r for r in rows if r['delta']>=1e-5 and r['all_blocks_positive']),key=lambda r:r['auc'],default=None)
    chosen=base[0].copy();leave=[]
    for f,(_,b) in enumerate(folds):
        a=np.concatenate([folds[i][1] for i in range(5) if i!=f]);best=roc_auc_score(y[a],base[0][a])+1e-5;pick=None
        for row in rows:
            if min(v for i,v in enumerate(row['fold_deltas']) if i!=f)<=0:continue
            auc=roc_auc_score(y[a],candidates[row['candidate']][0][a])
            if auc>best:best=auc;pick=row['candidate']
        p=base[0] if pick is None else candidates[pick][0];chosen[b]=p[b]
        leave.append(dict(held_out_block=f+1,selected=pick or 'baseline',delta=float(roc_auc_score(y[b],p[b])-bs['block_auc'][f])))
    stability=dict(auc=float(roc_auc_score(y,chosen)),held_out_blocks=leave,all_held_blocks_positive=all(r['delta']>0 for r in leave))
    if selected and not stability['all_held_blocks_positive']:selected=None
    result=dict(competition=COMPETITION,folds=10,native_folds=10,evaluation_blocks=5,historical_native_folds=[3,5,10],seed=42,
        input_sha256=ih,external_sha256=metadata['lgb']['external_sha256'],model_description='Tenfold auxiliary LightGBM replacement with historical mixed-fold blend',
        source_status=dict(tree=status),baseline=bs,baseline_reconstruction_max_difference=error,model_scores=dict(tree=new_auc),
        coarse_scan=sorted(rows,key=lambda r:r['auc'],reverse=True),selected_blend=selected,selection_stability=stability,
        source_fingerprints=fingerprints,script_sha256=sha(Path(__file__)),
        feature_code_sha256={p:sha(ROOT/p) for p in ('round15_models.py','round14_common.py','round12_models.py')},
        validation_note='Four fixed candidates, delta >=1e-5, five old evaluation blocks and all leave-block holds positive. New tree tenfold with inner fivefold TE, unlabeled transductive auxiliary features; historical mixed 3/5/10-fold blend, adaptive sequential selection, not fully nested validation.')
    if selected:
        out=ROOT/'outputs-round15';out.mkdir(exist_ok=True);p,pt=candidates[selected['candidate']]
        sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
        pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=p)).to_csv(out/'oof.csv',index=False)
        result['submission_sha256']=sha(out/'submission.csv');result['paired_against_previous_best']=paired(y,base[0],p)
        save(out/'metrics.json',result)
    save(ROOT/'analyses/round15-comparison.json',result)
    print(json.dumps({k:result[k] for k in ('source_status','model_scores','baseline','coarse_scan','selected_blend','selection_stability')},indent=2),flush=True)


if __name__=='__main__':run()
