"""Twelve fixed research-inspired candidates; no post-result weight search."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr,ndtri,expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT,sha
from auxiliary_round import hashes
from evaluate_context import paired
from round14_common import sources,load,save,model_candidates
from train import read_data,validate_submission,COMPETITION


def rank_candidates(preds):
    # Fit a label-free CDF on OOF values, then reuse the same mapping on test.
    components={k:preds[k] for k in ('old5','aux','tabm','travel')}
    components['ctx']=tuple(expit(.5*logit(np.clip(preds['lgb'][i],1e-6,1-1e-6))+
        .5*logit(np.clip(preds['context'][i],1e-6,1-1e-6))) for i in (0,1))
    weights=dict(old5=.30375,ctx=.30375,aux=.2025,tabm=.09,travel=.1)
    if abs(sum(weights.values())-1)>1e-12:raise ValueError('Component weights changed')
    uniform=[np.zeros_like(preds['baseline'][i]) for i in (0,1)]
    normal=[np.zeros_like(preds['baseline'][i]) for i in (0,1)]
    normalized={};summaries={}
    for name, pair in dict(components,baseline=preds['baseline']).items():
        values,counts=np.unique(pair[0],return_counts=True)
        u=(np.cumsum(counts)-.5*counts)/len(pair[0])
        transformed=tuple(np.clip(np.interp(p,values,u),1e-6,1-1e-6) for p in pair)
        summaries[name]=dict(unique_values=len(values),reference_rows=len(pair[0]),label_used=False)
        if name=='baseline':normalized=transformed
        else:
            for i in (0,1):
                uniform[i]+=weights[name]*transformed[i]
                normal[i]+=weights[name]*ndtri(transformed[i])
    candidates={}
    for w in (.1,.25):
        candidates['uniform_weight'+str(w)]=tuple((1-w)*normalized[i]+w*uniform[i] for i in (0,1))
        candidates['probit_weight'+str(w)]=tuple(ndtr((1-w)*ndtri(normalized[i])+w*normal[i]) for i in (0,1))
    return candidates,summaries,normalized[0]


def run(rank_only=False):
    tr,ts,sample,raw,ys=read_data(ROOT/'data');y=ys.to_numpy();ih=hashes()
    preds,metadata,fingerprints,error=sources(tr,sample,ys)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    candidates,cdf_summary,normalized_base=rank_candidates(preds)
    base=preds['baseline']
    if abs(roc_auc_score(y,normalized_base)-roc_auc_score(y,base[0]))>1e-12:
        raise ValueError('Baseline normalization changed ranking')
    status={};new_meta={}
    if not rank_only:
        for mode in ('bins','tabm'):
            folder=ROOT/'experiments'/('round14-'+mode+'-5fold')
            screening=load(folder/'first-fold-screen.json')
            if not screening['continue_full']:
                status[mode]='stopped_after_first_fold';continue
            if not (folder/'metrics.json').exists():
                raise ValueError('Passing experiment not complete: '+mode)
            m=load(folder/'metrics.json');cfg=load(folder/'configuration.json')
            if m['input_sha256']!=ih or m['competition']!=COMPETITION or cfg['script_sha256']!=sha(ROOT/'round14_models.py'):
                raise ValueError('New model inputs or training code changed')
            if sha(folder/'submission.csv')!=m['submission_sha256']:
                raise ValueError('New prediction fingerprint changed')
            for name,digest in cfg['source_code_sha256'].items():
                if sha(ROOT/name)!=digest:raise ValueError('New dependency changed: '+name)
            if cfg['base_cache_sha256']!=sha(ROOT/'experiments/base.joblib'):
                raise ValueError('Feature cache changed')
            o=pd.read_csv(folder/'oof.csv');s=pd.read_csv(folder/'submission.csv');validate_submission(s,sample)
            if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError('New OOF alignment changed')
            p=o.prediction.to_numpy();pt=s.satisfaction.to_numpy();test=np.zeros(len(ts))
            if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid new OOF')
            for f,(_,b) in enumerate(folds,1):
                with np.load(folder/f'fold-{f}.npz') as z:
                    if not np.array_equal(z['val_idx'],b) or not np.allclose(z['val_pred'],p[b],rtol=0,atol=1e-12):
                        raise ValueError('New checkpoint changed')
                    test+=z['test_pred']/5
                fingerprints[str((folder/f'fold-{f}.npz').relative_to(ROOT))]=sha(folder/f'fold-{f}.npz')
            if not np.allclose(test,pt,rtol=0,atol=1e-12):raise ValueError('New test mean changed')
            for name in ('configuration.json','metrics.json','oof.csv','submission.csv'):
                fingerprints[str((folder/name).relative_to(ROOT))]=sha(folder/name)
            candidates.update(model_candidates(preds,mode,(p,pt)));new_meta[mode]=m;status[mode]='completed_five_folds'
    def score(p):
        return dict(auc=float(roc_auc_score(y,p)),block_auc=[float(roc_auc_score(y[b],p[b])) for _,b in folds])
    bs=score(base[0]);rows=[]
    for name,(p,pt) in candidates.items():
        if not all(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in (p,pt)):
            raise ValueError('Invalid candidate probabilities')
        s=score(p);d=[x-z for x,z in zip(s['block_auc'],bs['block_auc'])]
        rows.append(dict(candidate=name,**s,delta=s['auc']-bs['auc'],fold_deltas=d,
            all_blocks_positive=min(d)>0,correlation=float(np.corrcoef(p,base[0])[0,1])))
    selected=max((r for r in rows if r['delta']>=1e-5 and r['all_blocks_positive']),key=lambda r:r['auc'],default=None)
    chosen=base[0].copy();leave=[]
    for f,(_,b) in enumerate(folds):
        a=np.concatenate([folds[i][1] for i in range(5) if i!=f]);base_auc=roc_auc_score(y[a],base[0][a]);best=base_auc+1e-5;pick=None
        for row in rows:
            if min(v for i,v in enumerate(row['fold_deltas']) if i!=f)<=0:continue
            auc=roc_auc_score(y[a],candidates[row['candidate']][0][a])
            if auc>best:best=auc;pick=row['candidate']
        p=base[0] if pick is None else candidates[pick][0];chosen[b]=p[b]
        leave.append(dict(held_out_block=f+1,selected=pick or 'baseline',delta=float(roc_auc_score(y[b],p[b])-bs['block_auc'][f])))
    stability=dict(auc=float(roc_auc_score(y,chosen)),held_out_blocks=leave,all_held_blocks_positive=all(r['delta']>0 for r in leave))
    if selected and not stability['all_held_blocks_positive']:selected=None
    result=dict(competition=COMPETITION,folds=5,evaluation_blocks=5,native_folds=5,historical_native_folds=[3,5,10],seed=42,
        input_sha256=ih,external_sha256=metadata['lgb']['external_sha256'],
        model_description='Research-inspired blend with historical mixed-fold members',
        rank_only=rank_only,source_status=status,baseline=bs,baseline_reconstruction_max_difference=error,
        model_scores={k:m['selected_blend']['auc'] for k,m in new_meta.items()},
        coarse_scan=sorted(rows,key=lambda r:r['auc'],reverse=True),selected_blend=selected,
        selection_stability=stability,cdf_summary=cdf_summary,source_fingerprints=fingerprints,
        script_sha256=sha(Path(__file__)),
        feature_code_sha256={p:sha(ROOT/p) for p in ('round14_common.py','round14_models.py','round12_models.py')},
        validation_note='Twelve fixed candidates at most, global delta >=1e-5, all five blocks positive and all leave-block holds positive. CDF fit on OOF predictions without labels. Historical mixed 3/5/10-fold models and transductive auxiliary features; adaptive selection, not fully nested validation.')
    if selected and not rank_only:
        out=ROOT/'outputs-round14';out.mkdir(exist_ok=True);p,pt=candidates[selected['candidate']]
        sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
        pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=p)).to_csv(out/'oof.csv',index=False)
        result['submission_sha256']=sha(out/'submission.csv');result['paired_against_previous_best']=paired(y,base[0],p)
        save(out/'metrics.json',result)
    save(ROOT/'analyses'/('round14-rank-comparison.json' if rank_only else 'round14-comparison.json'),result)
    print(json.dumps({k:result[k] for k in ('rank_only','source_status','baseline','coarse_scan','selected_blend','selection_stability')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rank-only',action='store_true')
    run(parser.parse_args().rank_only)
