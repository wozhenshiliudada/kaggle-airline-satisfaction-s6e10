"""Independent next-round transfer and TabM experiments; old caches preserved."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from nn_round import prepare_frames, fold_frames, predict_chunks, report
from train import read_data, validate_submission, COMPETITION

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
    if not torch.cuda.is_available():
        raise RuntimeError('Verified CUDA GPU required')
    torch.set_num_threads(4)
    start = time.time()
    out = ROOT/'experiments'/f'{args.variant}-{args.folds}fold-{args.ensemble}ens-{args.epochs}epochs'
    out.mkdir(parents=True, exist_ok=True)
    tr, ts, sample, raw, y = read_data(ROOT/'data')
    full, keys = prepare_frames(tr, ts, raw)
    hashes = {n:sha(ROOT/'data'/n) for n in ('train.csv','test.csv','sample_submission.csv')}
    source = None
    if args.variant == 'transfer':
        cache = joblib.load(ROOT/'experiments'/'base.joblib')
        if not cache['tr'].id.equals(tr.id) or not cache['ts'].id.equals(ts.id):
            raise ValueError('External feature cache IDs differ')
        expected = json.loads((ROOT/'experiments'/'compact'/'metrics.json').read_text())
        source = sha(ROOT/'data-original'/'data.csv')
        if expected['input_sha256'] != hashes or expected['external_sha256'] != source:
            raise ValueError('External feature source fingerprint differs')
        for c in ('original_lgb_probability','original_cat_probability'):
            full[c] = cache['base'][c].to_numpy(dtype='float32')
        del cache
    if args.variant == 'tabm':
        # Avoid a large one-hot matrix for continuous/high-cardinality twins.
        full = full.drop(columns=['category_'+c for c in ('Age','Flight Distance',
                          'Departure Delay in Minutes','Arrival Delay in Minutes')])
        params = dict(device='cuda:0', random_state=42, n_threads=4, verbosity=1,
                      n_cv=1, n_refit=0, arch_type='tabm-mini', tabm_k=16,
                      num_emb_type='pwl',num_emb_n_bins=32,d_embedding=8,
                      d_block=256,n_blocks=2,dropout=.1,batch_size=1024,
                      lr=.002,weight_decay=0.,n_epochs=args.epochs,patience=5,
                      compile_model=False,allow_amp=True,val_metric_name='1-auc_ovr',
                      tfms=['quantile_tabr'])
        model_class = TabM_D_Classifier
    else:
        params = json.loads((ROOT/'experiments'/'realmlp-3fold-4ens'/'configuration.json').read_text())['params']
        params.update(n_ens=args.ensemble,n_epochs=args.epochs)
        if args.variant == 'refine':
            params.update(lr=.03,ls_eps=.005)
        model_class = RealMLP_TD_Classifier
    config = dict(script_sha256=sha(Path(__file__)),feature_code_sha256=sha(ROOT/'nn_round.py'),
                  input_sha256=hashes,external_sha256=source,params=params,
                  folds=args.folds,seed=42,variant=args.variant)
    if source:
        config['external_cache_sha256'] = sha(ROOT/'experiments'/'base.joblib')
    path = out/'configuration.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Experiment configuration changed; use fresh directory')
    path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    cats = list(full.select_dtypes(include='category').columns)
    oof = np.full(len(tr),np.nan)
    pt_all = np.zeros(len(ts));scores=[]
    splits = list(StratifiedKFold(args.folds,shuffle=True,random_state=42).split(tr,y))
    for f,(a,b) in enumerate(splits,1):
        path = out/f'fold-{f}.npz'
        if path.exists():
            saved=np.load(path)
            if not np.array_equal(saved['val_idx'],b):raise ValueError('Cached split differs')
            pv,pt=saved['val_pred'],saved['test_pred']
            score=json.loads((out/f'fold-{f}.json').read_text())
            report({'fold':f,'restored':True})
        else:
            xa,xv,xt=fold_frames(full,keys,a,b,len(tr),y)
            if f == 1:
                altered=y.copy();altered.iloc[b]=1-altered.iloc[b]
                check=fold_frames(full,keys,a,b,len(tr),altered)
                for original,changed in zip((xa,xv,xt),check):
                    pd.testing.assert_frame_equal(original,changed,check_exact=True)
                del check
                report({'validation_label_perturbation':'PASS'})
            local=dict(params,random_state=42+f)
            model=model_class(**local)
            report({'variant':args.variant,'fold':f,'stage':'training',
                    'features':len(xa.columns),'params':local})
            model.fit(xa,y.iloc[a],X_val=xv,y_val=y.iloc[b],cat_col_names=cats)
            if args.variant == 'tabm':
                interface=model.alg_interface_.sub_split_interfaces[0]
                if str(interface.device_) != 'cuda:0':raise ValueError('Unexpected TabM device')
            else:
                if str(model.alg_interface_.device) != 'cuda:0':raise ValueError('Unexpected RealMLP device')
            pv=predict_chunks(model,xv);pt=predict_chunks(model,xt)
            if not np.isfinite(pv).all() or not np.isfinite(pt).all():
                raise ValueError('Nonfinite predictions')
            score={'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),
                   'elapsed_seconds':round(time.time()-start,2)}
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            del model,xa,xv,xt
            gc.collect();torch.cuda.empty_cache()
        oof[b]=pv;pt_all+=pt/args.folds;scores.append(score);report(score)
        if args.first_fold:
            baseline=pd.read_csv(ROOT/'outputs-round4'/'oof.csv').prediction.to_numpy()[b]
            report({'first_fold_baseline_auc':float(roc_auc_score(y.iloc[b],baseline)),
                    'candidate_auc':score['auc'],
                    'blend_auc':{str(w):float(roc_auc_score(y.iloc[b],(1-w)*baseline+w*pv))
                                 for w in (.1,.25,.5)}})
            return
    if not np.isfinite(oof).all():raise ValueError('OOF incomplete')
    sub=sample.copy();sub.satisfaction=pt_all;validate_submission(sub,sample)
    path=out/'submission.csv';sub.to_csv(path,index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(out/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':pt_all}).to_csv(out/'test_predictions.csv',index=False)
    metrics=dict(competition=COMPETITION,folds=args.folds,seed=42,variant=args.variant,
                 model_description=args.variant,selected_blend={'auc':float(roc_auc_score(y,oof))},
                 input_sha256=hashes,submission_sha256=sha(path),fold_scores=scores,
                 seconds=round(time.time()-start,2),license='Apache-2.0',
                 versions={p:importlib.metadata.version(p) for p in ('pytabkit','torch')})
    if source:metrics['external_sha256']=source
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    report(metrics)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('variant',choices=['transfer','refine','tabm'])
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--ensemble',type=int,default=4)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--first-fold',action='store_true')
    run(p.parse_args())
