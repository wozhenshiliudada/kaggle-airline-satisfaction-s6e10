"""Aligned full-OOF comparisons and predeclared coarse combinations.

No leaderboard labels or other participants' predictions are loaded. All new
individual models share five folds. The historical blend contains 3/5-fold
members, so its five evaluation blocks are not a jointly fitted five-fold CV.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT, sha
from train import read_data, COMPETITION, validate_submission

FOLDERS = {'tree_control':'experiments/context-control-5fold',
           'tree_conditional':'experiments/context-conditional-5fold',
           'tree_offset':'experiments/context-offset-5fold',
           'neural_control':'experiments/realmlp-5fold-8ens',
           'neural_context':'experiments/realmlp-context-5fold-8ens',
           'previous_best':'outputs-round5'}


def auc_parts(y,p):
    positive=p[y==1];negative=p[y==0]
    sp=np.sort(positive);sn=np.sort(negative)
    vp=(np.searchsorted(sn,positive,side='left')+np.searchsorted(sn,positive,side='right'))/(2*len(sn))
    vn=1-(np.searchsorted(sp,negative,side='left')+np.searchsorted(sp,negative,side='right'))/(2*len(sp))
    return vp,vn


def paired(y, old, new):
    a,b=auc_parts(y,old);c,d=auc_parts(y,new)
    se=float(np.sqrt(np.var(c-a,ddof=1)/len(a)+np.var(d-b,ddof=1)/len(b)))
    delta=float(roc_auc_score(y,new)-roc_auc_score(y,old))
    return {'delta':delta,'fixed_prediction_se':se,'approx_95ci':[delta-1.96*se,delta+1.96*se],
            'limitation':'Fixed prediction, independent case approximation. Adaptive OOF weight selection and dependence from overlapping training folds are not corrected.'}


def pairs(y,old,new,group):
    result=[];total_pos=int((y==1).sum());total_neg=int((y==0).sum())
    for pg in sorted(group.unique()):
        positives=(y==1)&(group.to_numpy()==pg)
        for ng in sorted(group.unique()):
            negatives=(y==0)&(group.to_numpy()==ng)
            weight=positives.sum()*negatives.sum()/(total_pos*total_neg)
            def pair_auc(p):
                neg=np.sort(p[negatives]);pos=p[positives]
                return float(np.mean((np.searchsorted(neg,pos,side='left')+np.searchsorted(neg,pos,side='right'))/(2*len(neg))))
            before,after=pair_auc(old),pair_auc(new)
            result.append({'positive_group':pg,'negative_group':ng,'old_auc':before,'new_auc':after,
                           'global_auc_contribution_change':float(weight*(after-before))})
    return sorted(result,key=lambda r:r['global_auc_contribution_change'],reverse=True)


def run(output=None):
    tr,ts,sample,raw,ys=read_data(ROOT/'data');y=ys.to_numpy()
    hashes={k:sha(ROOT/'data'/k) for k in ('train.csv','test.csv','sample_submission.csv')}
    preds={};test={};metadata={}
    for name,relative in FOLDERS.items():
        folder=ROOT/relative
        if not (folder/'metrics.json').is_file():
            if output:raise ValueError(f'Experiment not complete: {name}')
            continue
        m=json.loads((folder/'metrics.json').read_text())
        if m['competition']!=COMPETITION or m['input_sha256']!=hashes:raise ValueError('Data fingerprint mismatch')
        o=pd.read_csv(folder/'oof.csv');s=pd.read_csv(folder/'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError('OOF alignment mismatch')
        validate_submission(s,sample)
        if sha(folder/'submission.csv')!=m['submission_sha256']:raise ValueError('Output fingerprint mismatch')
        preds[name]=o.prediction.to_numpy();test[name]=s.satisfaction.to_numpy();metadata[name]=m
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    def score(p):
        return {'auc':float(roc_auc_score(y,p)),
                'block_auc':[float(roc_auc_score(y[b],p[b])) for _,b in splits]}
    base=preds['previous_best'];base_test=test['previous_best'];base_score=score(base)
    comparisons={name:score(p) for name,p in preds.items()}
    matched={}
    for name,control in [('tree_conditional','tree_control'),('tree_offset','tree_control'),('neural_context','neural_control')]:
        if name in preds and control in preds:
            matched[name]={'control':control,'paired':paired(y,preds[control],preds[name]),
                           'fold_deltas':[a-b for a,b in zip(comparisons[name]['block_auc'],comparisons[control]['block_auc'])]}
    candidates={name:(preds[name],test[name]) for name in preds if name not in ('previous_best','neural_control')}
    for tree in ('tree_control','tree_conditional','tree_offset'):
        for nn in ('neural_control','neural_context'):
            if tree in preds and nn in preds:
                for scale in ('probability','logit'):
                    if scale=='probability':
                        p=.5*preds[tree]+.5*preds[nn];pt=.5*test[tree]+.5*test[nn]
                    else:
                        p=expit(.5*logit(np.clip(preds[tree],1e-6,1-1e-6))+.5*logit(np.clip(preds[nn],1e-6,1-1e-6)))
                        pt=expit(.5*logit(np.clip(test[tree],1e-6,1-1e-6))+.5*logit(np.clip(test[nn],1e-6,1-1e-6)))
                    candidates[tree+'+'+nn+'-'+scale]=(p,pt)
    scanned=[]
    for name,(p,pt) in candidates.items():
        for weight in (.1,.25,.5):
            metrics=score((1-weight)*base+weight*p)
            deltas=[a-b for a,b in zip(metrics['block_auc'],base_score['block_auc'])]
            scanned.append({'candidate':name,'weight':weight,**metrics,'fold_deltas':deltas,
                            'all_blocks_positive':bool(min(deltas)>0)})
    # Choose only among candidates that improve every fixed evaluation block.
    eligible=[r for r in scanned if r['all_blocks_positive']]
    selected=max(eligible,key=lambda r:r['auc']) if eligible else None
    # An additional check: select candidate/weight using only the other four
    # blocks' AUC changes, then score the fifth. This tests weight stability on
    # saved OOF; it is NOT nested re-training of every base estimator.
    meta_oof=base.copy();meta_choices=[]
    for f,(_,b) in enumerate(splits):
        pool=[r for r in scanned if min(d for i,d in enumerate(r['fold_deltas']) if i!=f)>0]
        choice=max(pool,key=lambda r:np.mean([d for i,d in enumerate(r['fold_deltas']) if i!=f])) if pool else None
        if choice:
            p=candidates[choice['candidate']][0];w=choice['weight']
            meta_oof[b]=(1-w)*base[b]+w*p[b]
            meta_choices.append({'held_out_block':f+1,'candidate':choice['candidate'],'weight':w,
                                 'held_out_delta':choice['fold_deltas'][f]})
        else:
            meta_choices.append({'held_out_block':f+1,'candidate':'previous_best','weight':0.,'held_out_delta':0.})
    result={'competition':COMPETITION,'folds':5,'seed':42,'input_sha256':hashes,
            'individual':comparisons,'matched_comparisons':matched,
            'baseline':base_score,'selected_blend':selected,
            'selection_holdout_check':{**score(meta_oof),'choices':meta_choices,
                                      'paired_against_base':paired(y,base,meta_oof)},
            'coarse_scan':sorted(scanned,key=lambda r:r['auc'],reverse=True),
            'source_submission_sha256':{k:v['submission_sha256'] for k,v in metadata.items()},
            'validation_note':'New individual models share exact five outer folds. Historical base has 3/5-fold members. Candidate and weight choices reuse OOF, not nested CV. All-block-positive filtering does not remove selection bias; official score is the external check.'}
    if selected:
        candidate,ct=candidates[selected['candidate']];w=selected['weight']
        chosen=(1-w)*base+w*candidate;pt=(1-w)*base_test+w*ct
        result['paired_against_previous_best']=paired(y,base,chosen)
        result['prediction_correlation']=float(np.corrcoef(base,chosen)[0,1])
        result['mean_absolute_prediction_change']=float(np.mean(abs(base-chosen)))
        result['pair_changes']={c:pairs(y,base,chosen,tr[c]) for c in ('Class','Type of Travel')}
        if output:
            out=output.resolve();out.mkdir(exist_ok=True)
            sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
            pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':chosen}).to_csv(out/'oof.csv',index=False)
            result['model_description']='Conditional LightGBM and RealMLP logit blend'
            result['external_sha256']=metadata['previous_best']['external_sha256']
            result['submission_sha256']=sha(out/'submission.csv')
            (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    analyses=ROOT/'analyses';analyses.mkdir(exist_ok=True)
    (analyses/'context-fivefold-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('coarse_scan','pair_changes')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);run(p.parse_args().output)
