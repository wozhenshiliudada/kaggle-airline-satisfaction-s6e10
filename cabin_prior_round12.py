"""Cabin-wise train-prior calibration, excluding outer validation labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from auxiliary_round import config_file,hashes
from context_round import ROOT,sha
from round12_models import screen,report
from train import read_data,validate_submission,COMPETITION


def run(first):
    tr,ts,sample,raw,ys=read_data(ROOT/'data');y=ys.to_numpy();ih=hashes()
    source=ROOT/'experiments/round12-cabin-5fold';out=ROOT/'experiments/round12-cabin_prior-5fold'
    cfg=json.loads((source/'configuration.json').read_text(encoding='utf-8'))
    if cfg['input_sha256']!=ih or sha(ROOT/'round12_models.py')!=cfg['script_sha256']:
        raise ValueError('Source configuration changed')
    config_file(out,dict(script_sha256=sha(Path(__file__)),input_sha256=ih,external_sha256=cfg['external_sha256'],
        auxiliary_sha256=cfg['auxiliary_sha256'],coherence_sha256=None,mode='cabin_prior',folds=5,seed=42,
        source_configuration_sha256=sha(source/'configuration.json'),
        source_code_sha256={'round12_models.py':sha(ROOT/'round12_models.py')},
        rule='Per-cabin mean unlabeled test probability equals outer training prevalence via logit intercept',
        transductive=True,held_out_labels_used=False))
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,ys))
    oof=np.full(len(tr),np.nan);test=np.zeros(len(ts));records=[]
    for f,(a,b) in enumerate(folds,1):
        path=source/f'fold-{f}.npz'
        with np.load(path) as z:
            if not np.array_equal(z['val_idx'],b):raise ValueError('Source fold mismatch')
            pv,pt=z['val_pred'],z['test_pred']
        nv,nt=pv.copy(),pt.copy();groups=[]
        changed=y.copy();changed[b]=1-changed[b]
        for cabin in sorted(tr['Class'].unique()):
            ma=tr['Class'].iloc[a].eq(cabin).to_numpy();mb=tr['Class'].iloc[b].eq(cabin).to_numpy();mt=ts['Class'].eq(cabin).to_numpy()
            prior=float(y[a][ma].mean())
            if float(changed[a][ma].mean())!=prior:raise ValueError('Held-out labels affect prior')
            lt=logit(np.clip(pt[mt],1e-7,1-1e-7));lv=logit(np.clip(pv[mb],1e-7,1-1e-7))
            offset=float(brentq(lambda v:float(expit(lt+v).mean())-prior,-6,6))
            nv[mb]=expit(lv+offset);nt[mt]=expit(lt+offset)
            if abs(nt[mt].mean()-prior)>1e-10:raise ValueError('Prior alignment failed')
            before=float(roc_auc_score(y[b][mb],pv[mb]));after=float(roc_auc_score(y[b][mb],nv[mb]))
            if abs(before-after)>1e-9:raise ValueError('Within-cabin ordering changed')
            groups.append(dict(cabin=cabin,prior=prior,raw_test_mean=float(pt[mt].mean()),
                normalized_test_mean=float(nt[mt].mean()),logit_intercept=offset,within_cabin_auc=before))
        if not all(np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in (nv,nt)):
            raise ValueError('Invalid calibrated probability')
        np.savez_compressed(out/f'fold-{f}.npz',val_idx=b,val_pred=nv,test_pred=nt)
        score=dict(fold=f,auc=float(roc_auc_score(y[b],nv)),raw_auc=float(roc_auc_score(y[b],pv)),
            groups=groups,source_checkpoint_sha256=sha(path),validation_label_perturbation='PASS')
        (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
        oof[b]=nv;test+=nt/5;records.append(score);report(score)
        if first:screen(out,'cabin_prior',b,nv,ys);return
    sub=sample.copy();sub.satisfaction=test;validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=oof)).to_csv(out/'oof.csv',index=False)
    m=dict(competition=COMPETITION,folds=5,seed=42,mode='cabin_prior',input_sha256=ih,
        external_sha256=cfg['external_sha256'],submission_sha256=sha(out/'submission.csv'),
        selected_blend=dict(auc=float(roc_auc_score(y,oof))),fold_scores=records,
        validation_note='Per-cabin test covariates and outer train labels determine shifts, no validation labels. Transductive; assumes stable conditional class priors. Sequential hypothesis and reused OOF selection are not nested validation.')
    (out/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8');report(m)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--first-fold',action='store_true')
    run(parser.parse_args().first_fold)
