"""Independent neural seeds, keeping all old folds/features/hyperparameters.

The new seed is 142 instead of 42. Inner target encoding remains seed42 so this
is a test of training variance, not a change of validation or feature fitting.
"""
import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from nn_round import ROOT, prepare_frames, fold_frames, predict_chunks, report
from context_round import context_keys, context_blocks, sha
from train import read_data, validate_submission, COMPETITION


def make_frames(full, keys, ckeys, a, b, n, y):
    frames = fold_frames(full, keys, a, b, n, y)
    if ckeys is not None:
        blocks = context_blocks(ckeys, a, b, n, y)
        for frame, block in zip(frames, blocks):
            frame[block.columns] = block
    return frames


def run(features):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    if not torch.cuda.is_available():
        raise RuntimeError('Verified CUDA GPU required')
    torch.set_num_threads(4)
    start=time.time(); seed=142
    out=ROOT/'experiments'/f'realmlp-{features}-seed{seed}-5fold-8ens'
    out.mkdir(exist_ok=True)
    reference=ROOT/'experiments'/('realmlp-5fold-8ens' if features=='plain' else 'realmlp-context-5fold-8ens')
    old_config=json.loads((reference/'configuration.json').read_text())
    params=dict(old_config['params'], random_state=seed, verbosity=0)
    tr,ts,sample,raw,y=read_data(ROOT/'data')
    hashes={k:sha(ROOT/'data'/k) for k in ('train.csv','test.csv','sample_submission.csv')}
    if hashes!=old_config['input_sha256'] or old_config['folds']!=5 or old_config['seed']!=42:
        raise ValueError('Reference input/folds changed')
    config={'script_sha256':sha(Path(__file__)), 'raw_feature_source_sha256':sha(ROOT/'nn_round.py'),
            'context_feature_source_sha256':sha(ROOT/'context_round.py'),
            'reference_configuration_sha256':sha(reference/'configuration.json'),
            'params':params,'input_sha256':hashes,'features':features,'folds':5,
            'split_seed':42,'model_seed':seed,'target_encoding_seed':42}
    cp=out/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Changed neural seed experiment')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    full,keys=prepare_frames(tr,ts,raw)
    ckeys=context_keys(full) if features=='context' else None
    cats=list(full.select_dtypes(include='category').columns)
    oof=np.full(len(tr),np.nan);pred=np.zeros(len(ts));scores=[]
    for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y),1):
        saved_reference=np.load(reference/f'fold-{f}.npz')
        if not np.array_equal(saved_reference['val_idx'],b):raise ValueError('Reference fold differs')
        del saved_reference
        path=out/f'fold-{f}.npz'
        if path.exists():
            saved=np.load(path)
            if not np.array_equal(saved['val_idx'],b):raise ValueError('Cached fold differs')
            pv,pt=saved['val_pred'],saved['test_pred'];score=json.loads((out/f'fold-{f}.json').read_text())
        else:
            frames=make_frames(full,keys,ckeys,a,b,len(tr),y)
            if f==1:
                changed=y.copy();changed.iloc[b]=1-changed.iloc[b]
                check=make_frames(full,keys,ckeys,a,b,len(tr),changed)
                for original,perturbed in zip(frames,check):
                    pd.testing.assert_frame_equal(original,perturbed,check_exact=True)
                del check
                report({'features':features,'validation_label_perturbation':'PASS'})
            xa,xv,xt=frames
            model=RealMLP_TD_Classifier(**dict(params,random_state=seed+f))
            report({'features':features,'model_seed':seed+f,'fold':f,'columns':len(xa.columns),'stage':'training'})
            model.fit(xa,y.iloc[a],X_val=xv,y_val=y.iloc[b],cat_col_names=cats)
            trainer=model.alg_interface_.min_trainer_kwargs
            if str(model.alg_interface_.device)!='cuda:0' or trainer['accelerator']!='gpu' or trainer['devices']!=[0]:
                raise RuntimeError('Trainer is not using expected GPU')
            pv=predict_chunks(model,xv);pt=predict_chunks(model,xt)
            if not np.isfinite(pv).all() or not np.isfinite(pt).all():raise ValueError('Nonfinite neural probabilities')
            score={'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),
                   'model_seed':seed+f,'seconds_elapsed':round(time.time()-start,2)}
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            del frames,xa,xv,xt,model;gc.collect();torch.cuda.empty_cache()
        oof[b]=pv;pred+=pt/5;scores.append(score);report(dict(features=features,**score))
    if not np.isfinite(oof).all():raise ValueError('Incomplete neural OOF')
    sub=sample.copy();sub.satisfaction=pred;validate_submission(sub,sample)
    sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(out/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':pred}).to_csv(out/'test_predictions.csv',index=False)
    metrics={'competition':COMPETITION,'model_description':f'RealMLP {features} independent seed142',
             'folds':5,'seed':42,'model_seed':seed,'input_sha256':hashes,
             'fold_scores':scores,'selected_blend':{'auc':float(roc_auc_score(y,oof))},
             'submission_sha256':sha(out/'submission.csv'),'seconds':round(time.time()-start,2),
             'license':'Apache-2.0','versions':{p:importlib.metadata.version(p) for p in ('pytabkit','torch','pytorch-lightning')},
             'validation_note':'Same exact five folds and inner encodings as corresponding seed42 control; only neural RNG seed differs. Each row excluded from its own model fitting. Three fixed epochs, eight members per fold.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');report(metrics)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('features',choices=['plain','context']);run(p.parse_args().features)
