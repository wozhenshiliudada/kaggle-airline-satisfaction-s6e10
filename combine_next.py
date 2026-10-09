"""Compare coarse declared blends against the previous officially scored best."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from train import read_data,COMPETITION,validate_submission

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    tr,ts,sample,raw,y=read_data(ROOT/'data')
    hashes={n:sha(ROOT/'data'/n) for n in ('train.csv','test.csv','sample_submission.csv')}
    def load(folder):
        metrics=json.loads((folder/'metrics.json').read_text())
        if metrics['competition']!=COMPETITION or metrics['input_sha256']!=hashes:
            raise ValueError('Official inputs differ')
        o=pd.read_csv(folder/'oof.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError('OOF alignment differs')
        sub=pd.read_csv(folder/'submission.csv');validate_submission(sub,sample)
        if metrics['submission_sha256']!=sha(folder/'submission.csv'):raise ValueError('Candidate fingerprint differs')
        return o.prediction.to_numpy(),sub.satisfaction.to_numpy(),metrics
    base_folder=args.base.resolve()
    base,base_test,base_metrics=load(base_folder)
    folder=args.candidate.resolve();candidate,ct,cm=load(folder)
    replace_member=args.replace_member or ('compact' if args.replace_compact else None)
    if replace_member:
        old,ot,om=load(ROOT/'experiments'/replace_member)
        candidate=base+(candidate-old)/6
        ct=base_test+(ct-ot)/6
    splits=list(StratifiedKFold(3,shuffle=True,random_state=42).split(tr,y))
    def score(p):
        return {'auc':float(roc_auc_score(y,p)),
                'block_auc':[float(roc_auc_score(y.iloc[b],p[b])) for a,b in splits]}
    compared=[]
    for w in (0.,.1,.25,.5,.75,1.):
        compared.append(dict(candidate_weight=w,**score((1-w)*base+w*candidate)))
    chosen=max(compared,key=lambda s:s['auc']);weight=chosen['candidate_weight']
    baseline=compared[0]
    result={'competition':COMPETITION,'folds':3,'seed':42,
            'model_description':args.description or 'Improved blend on RealMLP5fold base',
            'candidate':str(folder.relative_to(ROOT)), 'candidate_training_folds':cm['folds'],
            'base_folder':str(base_folder.relative_to(ROOT)),
            'replace_member':replace_member,'selected_blend':chosen,
            'comparisons':compared,'baseline_auc':baseline['auc'],
            'block_improvements':[a-b for a,b in zip(chosen['block_auc'],baseline['block_auc'])],
            'correlation':float(np.corrcoef(base,candidate)[0,1]),
            'input_sha256':hashes,'external_sha256':base_metrics['external_sha256'],
            'base_submission_sha256':base_metrics['submission_sha256'],
            'candidate_submission_sha256':cm['submission_sha256'],
            'validation_note':'Same OOF used to select six coarse weights; not nested CV. Neural/tree training folds may differ; each row excluded from its own member fitting. Report blocks use original three tree folds. Kaggle public score is the external check.'}
    analyses=ROOT/'analyses';analyses.mkdir(exist_ok=True)
    (analyses/(folder.name+('-replace' if replace_member else '')+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.output:
        out=args.output.resolve();out.mkdir(exist_ok=True)
        sub=sample.copy();sub.satisfaction=(1-weight)*base_test+weight*ct
        validate_submission(sub,sample);sub.to_csv(out/'submission.csv',index=False)
        pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':(1-weight)*base+weight*candidate}).to_csv(out/'oof.csv',index=False)
        result['submission_sha256']=sha(out/'submission.csv')
        (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('candidate',type=Path)
    p.add_argument('--replace-compact',action='store_true')
    p.add_argument('--replace-member',choices=['compact','route-auc','cat'])
    p.add_argument('--output',type=Path)
    p.add_argument('--description')
    p.add_argument('--base',type=Path,default=ROOT/'outputs-round4')
    run(p.parse_args())
