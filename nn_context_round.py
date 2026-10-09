"""RealMLP matched five-fold, eight-member comparison with context features."""
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


def run():
    import torch
    from pytabkit import RealMLP_TD_Classifier
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    torch.set_num_threads(4)
    start = time.time()
    out = ROOT / 'experiments' / 'realmlp-context-5fold-8ens'
    out.mkdir(exist_ok=True)
    tr, ts, sample, raw, y = read_data(ROOT / 'data')
    full, keys = prepare_frames(tr, ts, raw)
    ckeys = context_keys(full)
    # Preserve the complete prior recipe; only context-derived inputs differ.
    params = dict(random_state=42, device='cuda:0', n_threads=4, verbosity=0,
                  n_cv=1, n_refit=0, n_ens=8, n_epochs=3,
                  batch_size=256, predict_batch_size=1024, use_early_stopping=False,
                  val_metric_name='1-auc_ovr', lr=.053, wd=.015, sq_mom=.988,
                  lr_sched='flat_anneal', wd_sched='cos_log_15', first_layer_lr_factor=.25,
                  embedding_size=5, max_one_hot_cat_size=18, hidden_sizes=[512,256,128],
                  act='silu', p_drop=.05, p_drop_sched='invsqrtp1e-3',
                  plr_hidden_1=16, plr_hidden_2=8, plr_act_name='gelu', plr_lr_factor=.1151,
                  plr_sigma=2.33, ls_eps=.01, ls_eps_sched='sqrt_cos', add_front_scale=False,
                  bias_init_mode='neg-uniform-dynamic-2',
                  tfms=['one_hot','median_center','robust_scale','smooth_clip','embedding','l2_normalize'])
    old_config = json.loads((ROOT / 'experiments' / 'realmlp-5fold-8ens' / 'configuration.json').read_text())
    comparable = {k:v for k,v in params.items() if k != 'verbosity'}
    prior = {k:v for k,v in old_config['params'].items() if k != 'verbosity'}
    if comparable != prior:
        raise ValueError('Neural configuration differs from five-fold control')
    hashes = {k:sha(ROOT/'data'/k) for k in ('train.csv','test.csv','sample_submission.csv')}
    if hashes != old_config['input_sha256']:
        raise ValueError('Neural input differs from prior control')
    config = {'script_sha256':sha(Path(__file__)), 'base_features_sha256':sha(ROOT/'nn_round.py'),
              'context_features_sha256':sha(ROOT/'context_round.py'), 'input_sha256':hashes,
              'params':params, 'folds':5, 'seed':42}
    cp = out / 'configuration.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Changed cached neural configuration')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    cats = list(full.select_dtypes(include='category').columns)
    pred = np.zeros(len(ts)); oof = np.full(len(tr),np.nan); scores=[]
    for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y),1):
        path = out / f'fold-{f}.npz'
        old_fold = np.load(ROOT/'experiments'/'realmlp-5fold-8ens'/f'fold-{f}.npz')
        if not np.array_equal(old_fold['val_idx'], b):
            raise ValueError('Neural fold does not match saved control')
        del old_fold
        if path.exists():
            saved=np.load(path)
            if not np.array_equal(saved['val_idx'], b):raise ValueError('Cached fold differs')
            pv,pt=saved['val_pred'],saved['test_pred']
            score=json.loads((out/f'fold-{f}.json').read_text())
        else:
            frames=fold_frames(full,keys,a,b,len(tr),y)
            blocks=context_blocks(ckeys,a,b,len(tr),y)
            if f == 1:
                alternate=y.copy();alternate.iloc[b]=1-alternate.iloc[b]
                check=context_blocks(ckeys,a,b,len(tr),alternate)
                for actual,perturbed in zip(blocks,check):
                    pd.testing.assert_frame_equal(actual,perturbed,check_exact=True)
                del check
                report({'neural_context_validation_label_perturbation':'PASS'})
            for frame,block in zip(frames,blocks):
                frame[block.columns]=block
            del blocks
            xa,xv,xt=frames
            model=RealMLP_TD_Classifier(**dict(params,random_state=42+f))
            report({'model':'RealMLP context','fold':f,'features':len(xa.columns),'stage':'training'})
            model.fit(xa,y.iloc[a],X_val=xv,y_val=y.iloc[b],cat_col_names=cats)
            trainer=model.alg_interface_.min_trainer_kwargs
            if str(model.alg_interface_.device)!='cuda:0' or trainer['accelerator']!='gpu' or trainer['devices']!=[0]:
                raise RuntimeError('Unexpected neural training device')
            pv=predict_chunks(model,xv);pt=predict_chunks(model,xt)
            score={'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),
                   'seconds_elapsed':round(time.time()-start,2),'features':len(xa.columns)}
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (out/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            del model,frames,xa,xv,xt;gc.collect();torch.cuda.empty_cache()
        oof[b]=pv;pred+=pt/5;scores.append(score);report(score)
    if not np.isfinite(oof).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite neural outputs')
    sub=sample.copy();sub.satisfaction=pred;validate_submission(sub,sample)
    sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(out/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':pred}).to_csv(out/'test_predictions.csv',index=False)
    metrics={'competition':COMPETITION,'folds':5,'seed':42,
             'model_description':'Apache2 RealMLP context fivefold eight ensembles',
             'selected_blend':{'auc':float(roc_auc_score(y,oof))},'fold_scores':scores,
             'input_sha256':hashes,'submission_sha256':sha(out/'submission.csv'),
             'seconds':round(time.time()-start,2),'license':'Apache-2.0',
             'versions':{p:importlib.metadata.version(p) for p in ('pytabkit','torch','pytorch-lightning')},
             'validation_note':'Same five folds, seeds, ensemble size and three fixed epochs as saved neural control. Five inner folds for context encodings; no validation labels in features.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');report(metrics)


if __name__=='__main__':run()
