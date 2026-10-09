"""Four fixed candidates for the sequential wide-CatBoost experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT,sha
from evaluate_bagging_round import FOLDERS
from evaluate_context import paired,pairs
from train import read_data,COMPETITION,validate_submission


def run():
    tr,ts,sample,raw,ys=read_data(ROOT/'data');y=ys.to_numpy()
    hashes={n:sha(ROOT/'data'/n) for n in ('train.csv','test.csv','sample_submission.csv')}
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    paths={n:FOLDERS[n] for n in ('compact','route','cat_old','tuned','plain42','context42','lgb_context','baseline','cat_conditional')}
    paths['wide_cat']='experiments/cat-wide-conditional-5fold'
    predictions={};metadata={}
    for name,relative in paths.items():
        folder=ROOT/relative;m=json.loads((folder/'metrics.json').read_text())
        if m['competition']!=COMPETITION or m['input_sha256']!=hashes:raise ValueError(f'Input mismatch: {name}')
        o=pd.read_csv(folder/'oof.csv');s=pd.read_csv(folder/'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError(f'OOF alignment mismatch: {name}')
        validate_submission(s,sample)
        if sha(folder/'submission.csv')!=m['submission_sha256']:raise ValueError(f'Output fingerprint mismatch: {name}')
        p=o.prediction.to_numpy();pt=s.satisfaction.to_numpy()
        if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid OOF probabilities')
        if name in ('cat_conditional','wide_cat'):
            for f,(_,b) in enumerate(folds,1):
                with np.load(folder/f'fold-{f}.npz') as saved:
                    if not np.array_equal(saved['val_idx'],b) or not np.allclose(saved['val_pred'],p[b],atol=1e-12,rtol=0):
                        raise ValueError('Cached fivefold predictions mismatch')
        predictions[name]=(p,pt);metadata[name]=m;del o,s
    def score(p):return {'auc':float(roc_auc_score(y,p)),'block_auc':[float(roc_auc_score(y[b],p[b])) for _,b in folds]}
    def mix(a,b,w):return tuple((1-w)*a[i]+w*b[i] for i in range(2))
    def mean(*members):return tuple(np.mean([m[i] for m in members],axis=0) for i in range(2))
    def logit_pair(a,b):return tuple(expit(.5*logit(np.clip(a[i],1e-6,1-1e-6))+.5*logit(np.clip(b[i],1e-6,1-1e-6))) for i in range(2))
    def blend_with_cat(cat):
        tree=mean(predictions['compact'],predictions['route'],cat)
        p5=mix(logit_pair(tree,predictions['plain42']),predictions['tuned'],.1)
        return mix(p5,logit_pair(predictions['lgb_context'],predictions['context42']),.5)
    reconstructed=blend_with_cat(predictions['cat_old'])
    errors=[float(np.max(abs(reconstructed[i]-predictions['baseline'][i]))) for i in range(2)]
    if max(errors)>1e-12:raise ValueError('Baseline fusion reconstruction mismatch')
    base,bt=predictions['baseline'];baseline=score(base)
    wide_score=score(predictions['wide_cat'][0]);control_score=score(predictions['cat_conditional'][0])
    candidates={'replace_old_cat':blend_with_cat(predictions['wide_cat'])}
    for w in (.1,.25,.5):candidates[f'wide_cat_weight{w}']=mix(predictions['baseline'],predictions['wide_cat'],w)
    comparisons=[]
    for name,(p,pt) in candidates.items():
        s=score(p);d=[a-b for a,b in zip(s['block_auc'],baseline['block_auc'])]
        comparisons.append({'candidate':name,**s,'delta':s['auc']-baseline['auc'],'fold_deltas':d,
                            'all_blocks_positive':bool(min(d)>0),
                            'correlation_with_baseline':float(np.corrcoef(base,p)[0,1]),
                            'mean_absolute_prediction_change':float(np.mean(abs(base-p)))})
    eligible=[r for r in comparisons if r['all_blocks_positive'] and r['delta']>0]
    selected=max(eligible,key=lambda r:r['auc']) if eligible else None
    meta=base.copy();choices=[]
    for f,(_,b) in enumerate(folds):
        pool=[r for r in comparisons if min(d for i,d in enumerate(r['fold_deltas']) if i!=f)>0]
        c=max(pool,key=lambda r:np.mean([d for i,d in enumerate(r['fold_deltas']) if i!=f])) if pool else None
        if c:meta[b]=candidates[c['candidate']][0][b]
        choices.append({'held_out_block':f+1,'candidate':c['candidate'] if c else 'baseline','held_out_delta':c['fold_deltas'][f] if c else 0.})
    result={'competition':COMPETITION,'folds':5,'seed':42,'model_description':'Wide conditional CatBoost: '+(selected['candidate'] if selected else 'no accepted candidate'),
            'input_sha256':hashes,'external_sha256':metadata['baseline']['external_sha256'],
            'script_sha256':sha(Path(__file__)),'baseline_reconstruction_max_difference':errors,
            'baseline':baseline,'wide_model':wide_score,'matched_control':control_score,
            'matched_feature_comparison':{'paired':paired(y,predictions['cat_conditional'][0],predictions['wide_cat'][0]),
                                          'fold_deltas':[a-b for a,b in zip(wide_score['block_auc'],control_score['block_auc'])]},
            'coarse_scan':sorted(comparisons,key=lambda r:r['auc'],reverse=True),'selected_blend':selected,
            'selection_holdout_check':{**score(meta),'choices':choices,'paired_against_base':paired(y,base,meta)},
            'source_submission_sha256':{n:m['submission_sha256'] for n,m in metadata.items()},
            'previous_experiment_analysis_sha256':sha(ROOT/'analyses'/'bagging-all-comparison.json'),
            'validation_note':'Sequential exploratory experiment after four prior fivefold experiments. Four fixed wide-CatBoost candidates: replace historical CatBoost member keeping all fusion weights, or add 10/25/50 percent probability. Every row excluded from its own estimator fitting; early stopping uses validation labels. Historical base includes three/fivefold members; OOF candidate selection and leave-one-block checks are not nested independent CV. Team membership confirmation required before submission.'}
    if selected:
        p,pt=candidates[selected['candidate']]
        result['paired_against_previous_best']=paired(y,base,p)
        result['pair_changes']={n:pairs(y,base,p,tr[n]) for n in ('Class','Type of Travel')}
        out=ROOT/'outputs-round7';out.mkdir(exist_ok=True)
        sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample)
        sub.to_csv(out/'submission.csv',index=False)
        pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':p}).to_csv(out/'oof.csv',index=False)
        result['submission_sha256']=sha(out/'submission.csv')
        (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'analyses'/'wide-cat-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('wide_model','matched_feature_comparison','baseline','selected_blend','coarse_scan','selection_holdout_check')},indent=2),flush=True)


if __name__=='__main__':run()
