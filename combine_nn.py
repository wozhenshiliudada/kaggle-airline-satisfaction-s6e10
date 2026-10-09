"""Declared small blend grid between a fixed tree baseline and RealMLP."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from train import read_data,validate_submission,COMPETITION

ROOT=Path(__file__).resolve().parent


def run(nn_folder,output):
    tr,ts,sample,features,y=read_data(ROOT/'data')
    hashes={name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
            for name in ('train.csv','test.csv','sample_submission.csv')}
    members={};tests={};reports={}
    paths={m:ROOT/'experiments'/m for m in ['compact','route-auc','cat']}
    paths['realmlp']=nn_folder.resolve()
    for name,folder in paths.items():
        metrics=json.loads((folder/'metrics.json').read_text())
        if metrics['competition']!=COMPETITION or metrics['input_sha256']!=hashes:
            raise ValueError('Member input source differs from official competition files')
        a=pd.read_csv(folder/'oof.csv');b=pd.read_csv(folder/'test_predictions.csv')
        if not a.id.equals(tr.id) or not b.id.equals(ts.id):raise ValueError('Member IDs not aligned')
        if not np.array_equal(a.satisfaction,y):raise ValueError('Member labels differ')
        members[name]=a.prediction.to_numpy();tests[name]=b.prediction.to_numpy();reports[name]=metrics
        if not np.isfinite(members[name]).all() or not np.isfinite(tests[name]).all():
            raise ValueError('Member predictions not finite')
    tree=np.mean([members[n] for n in ['compact','route-auc','cat']],axis=0)
    tree_test=np.mean([tests[n] for n in ['compact','route-auc','cat']],axis=0)
    folds=list(StratifiedKFold(3,shuffle=True,random_state=42).split(tr,y))
    scores=[]
    for weight in (0.,.1,.25,.5,.75,1.):
        p=(1-weight)*tree+weight*members['realmlp']
        scores.append({'neural_weight':weight,'auc':float(roc_auc_score(y,p)),
                       'fold_auc':[float(roc_auc_score(y.iloc[b],p[b])) for a,b in folds]})
    chosen=max(scores,key=lambda item:item['auc']);w=chosen['neural_weight']
    prediction=(1-w)*tree_test+w*tests['realmlp']
    oof=(1-w)*tree+w*members['realmlp']
    sub=sample.copy();sub.satisfaction=sub.id.map(pd.Series(prediction,index=ts.id))
    validate_submission(sub,sample)
    output.mkdir(exist_ok=True)
    sub.to_csv(output/'submission.csv',index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(output/'oof.csv',index=False)
    report={'competition':COMPETITION,'folds':3,'seed':42,
            'tree_training_folds':3,'neural_training_folds':reports['realmlp']['folds'],
            'oof_evaluation_blocks':3,
            'model_description':f"RealMLP-{reports['realmlp']['folds']}fold plus tree blend",
            'selected_blend':chosen,'blend_comparisons':scores,
            'member_auc':{n:float(roc_auc_score(y,p)) for n,p in members.items()},
            'tree_baseline_auc':float(roc_auc_score(y,tree)),
            'tree_neural_correlation':float(np.corrcoef(tree,members['realmlp'])[0,1]),
            'input_sha256':hashes,'external_sha256':reports['compact']['external_sha256'],
            'member_folders':{n:str(p.relative_to(ROOT)) for n,p in paths.items()},
            'submission_sha256':hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest(),
            'validation_note':'Fixed equal-weight tree baseline; neural weight chosen from six OOF candidates. OOF evaluation blocks are the tree folds; neural models may use different folds, with each row excluded from its own neural fit. This is not nested validation. Selection on the same OOF can bias its score; official Kaggle score is the external check.'}
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--nn',type=Path,default=ROOT/'experiments'/'realmlp-3fold-4ens')
    p.add_argument('--output',type=Path,default=ROOT/'outputs-round3')
    args=p.parse_args();run(args.nn,args.output)
