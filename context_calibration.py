"""Fold-cross-fitted, regularized context intercepts on our own OOF blend.

This checks whether conditional prevalence correction improves cross-context
ranking. Calibration fitting uses other blocks only, but saved base-model OOF
training overlaps; this is OOF stacking, not fully nested base-model retraining.
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
from train import read_data, validate_submission, COMPETITION


def fit_offsets(z,y,groups,k,penalty=100.):
    offsets=np.zeros(k)
    for _ in range(40):
        p=expit(z+offsets[groups])
        gradient=np.bincount(groups,weights=y-p,minlength=k)-penalty*offsets
        curvature=np.bincount(groups,weights=p*(1-p),minlength=k)+penalty
        step=np.clip(gradient/curvature,-1.,1.)
        offsets+=step
        if np.max(abs(step))<1e-9:
            return offsets
    raise ValueError('Regularized intercept fit did not converge')


def run(base_folder,output):
    tr,ts,sample,raw,ys=read_data(ROOT/'data');y=ys.to_numpy()
    base_folder=base_folder.resolve()
    bm=json.loads((base_folder/'metrics.json').read_text())
    hashes={k:sha(ROOT/'data'/k) for k in ('train.csv','test.csv','sample_submission.csv')}
    if bm['input_sha256']!=hashes or bm['competition']!=COMPETITION:
        raise ValueError('Calibration input data differs')
    o=pd.read_csv(base_folder/'oof.csv');sub=pd.read_csv(base_folder/'submission.csv')
    if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError('Calibration OOF misaligned')
    validate_submission(sub,sample)
    if sha(base_folder/'submission.csv')!=bm['submission_sha256']:raise ValueError('Calibration base fingerprint changed')
    full=pd.concat([tr[raw],ts[raw]],ignore_index=True)
    contexts=full['Class'].astype(str)+'|'+full['Type of Travel'].astype(str)+'|'+full['Customer Type'].astype(str)
    groups,levels=pd.factorize(contexts,sort=False)
    n=len(tr);k=len(levels);z=logit(np.clip(o.prediction.to_numpy(),1e-6,1-1e-6))
    zt=logit(np.clip(sub.satisfaction.to_numpy(),1e-6,1-1e-6))
    delta=np.full(n,np.nan);fold_fits=[]
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    for f,(a,b) in enumerate(splits,1):
        offsets=fit_offsets(z[a],y[a],groups[a],k)
        if f==1:
            changed=y.copy();changed[b]=1-changed[b]
            np.testing.assert_array_equal(offsets,fit_offsets(z[a],changed[a],groups[a],k))
        delta[b]=offsets[groups[b]]
        fold_fits.append({'fold':f,'offsets':dict(zip(levels.tolist(),offsets.tolist()))})
    if not np.isfinite(delta).all():raise ValueError('Incomplete calibration OOF')
    def score(p):
        return {'auc':float(roc_auc_score(y,p)),
                'block_auc':[float(roc_auc_score(y[b],p[b])) for _,b in splits]}
    baseline=score(expit(z));compared=[]
    for weight in (0.,.25,.5,1.):
        m=score(expit(z+weight*delta))
        changes=[a-b for a,b in zip(m['block_auc'],baseline['block_auc'])]
        compared.append({'weight':weight,**m,'fold_deltas':changes,'all_blocks_positive':bool(min(changes)>0)})
    pool=[r for r in compared if r['all_blocks_positive']]
    chosen=max(pool,key=lambda r:r['auc']) if pool else compared[0]
    full_fit=fit_offsets(z,y,groups[:n],k)
    result={'competition':COMPETITION,'folds':5,'seed':42,
            'model_description':'Context intercept calibration of matched-model blend',
            'base_folder':str(base_folder.relative_to(ROOT)),'baseline':baseline,'selected_blend':chosen,
            'comparisons':compared,'penalty':100.,'fold_fits':fold_fits,
            'full_fit_offsets':dict(zip(levels.tolist(),full_fit.tolist())),
            'validation_label_perturbation':'PASS','input_sha256':hashes,
            'external_sha256':bm['external_sha256'],'base_submission_sha256':bm['submission_sha256'],
            'script_sha256':sha(Path(__file__)),
            'validation_note':'Context intercepts fit without the corresponding held-out block labels. Uses previously selected OOF base predictions, whose underlying training folds overlap. This is OOF stacking, not nested CV. Four weights reuse OOF; official Kaggle score is external verification.'}
    out=output.resolve();out.mkdir(exist_ok=True)
    prediction=expit(zt+chosen['weight']*full_fit[groups[n:]])
    sub.satisfaction=prediction;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':expit(z+chosen['weight']*delta)}).to_csv(out/'oof.csv',index=False)
    result['submission_sha256']=sha(out/'submission.csv')
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('fold_fits','full_fit_offsets')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=ROOT/'outputs-round6')
    p.add_argument('--output',type=Path,default=ROOT/'outputs-round6-calibration');a=p.parse_args();run(a.base,a.output)
