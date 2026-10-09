"""Train Apache-2.0 PyTabKit RealMLP locally, with outer-fold-safe encoding.

Public hyperparameter ideas: yekenot/ps-s6-e10-realmlp-pytabkit.
No notebook code or third-party prediction files are executed or loaded.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from train import read_data, validate_submission, COMPETITION

ROOT=Path(__file__).resolve().parent
CATS=['Gender','Customer Type','Type of Travel','Class']


def report(value):
    print(json.dumps(value),flush=True)


def prepare_frames(tr,ts,raw):
    full=pd.concat([tr[raw],ts[raw]],ignore_index=True).copy()
    for c in CATS:
        full[c]=full[c].astype('string').fillna('__MISSING__').astype('category')
    nums=[c for c in raw if c not in CATS]
    for c in nums:
        full[c]=full[c].fillna(0).astype('float32')
        full['category_'+c]=pd.Categorical(full[c])
    for c in CATS+['Leg room service']:
        full['count_'+c]=full[c].map(full[c].value_counts()).astype('float32')
    keys={}
    combos=[['Class','Type of Travel','Gender'],['Class','Type of Travel','Age'],
            ['Flight Distance','Departure/Arrival time convenient']]
    for i,cols in enumerate(combos):
        value=full[cols[0]].astype(str)
        for c in cols[1:]: value=value+'|'+full[c].astype(str)
        keys['combo_'+str(i)]=pd.factorize(value,sort=False)[0].astype('int32')
        if i==0:full['category_combo']=pd.Categorical(value)
    return full,pd.DataFrame(keys)


def fold_frames(full,keys,a,b,n,y):
    encoder=TargetEncoder(target_type='binary',smooth='auto',
                          cv=StratifiedKFold(5,shuffle=True,random_state=42))
    blocks=[encoder.fit_transform(keys.iloc[a],y.iloc[a]),
            encoder.transform(keys.iloc[b]),encoder.transform(keys.iloc[n:])]
    frames=[full.iloc[a].reset_index(drop=True).copy(),full.iloc[b].reset_index(drop=True).copy(),
            full.iloc[n:].reset_index(drop=True).copy()]
    columns=['target_'+c for c in keys]
    return [pd.concat([frame,pd.DataFrame(block.astype('float32'),columns=columns)],axis=1)
            for frame,block in zip(frames,blocks)]


def predict_chunks(model,frame):
    return np.concatenate([model.predict_proba(frame.iloc[start:start+32768])[:,1]
                           for start in range(0,len(frame),32768)])


def run(args):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    if not torch.cuda.is_available():raise RuntimeError('RealMLP experiment requires the verified NVIDIA GPU.')
    torch.set_num_threads(4)
    started=time.time()
    output=ROOT/'experiments'/f'realmlp-{args.folds}fold-{args.ensemble}ens'
    output.mkdir(parents=True,exist_ok=True)
    tr,ts,sample,raw,y=read_data(ROOT/'data')
    if args.smoke:
        tr=tr.sample(2400,random_state=42).reset_index(drop=True)
        y=tr.satisfaction.astype('int8'); ts=ts.head(600).copy(); sample=sample.head(600).copy()
        output=ROOT/'experiments'/'realmlp-smoke';output.mkdir(exist_ok=True)
    full,keys=prepare_frames(tr,ts,raw)
    cats=list(full.select_dtypes(include='category').columns)
    params=dict(random_state=42,device='cuda:0',n_threads=4,verbosity=1,
                n_cv=1,n_refit=0,n_ens=args.ensemble,n_epochs=1 if args.smoke else 3,
                batch_size=256,predict_batch_size=1024,use_early_stopping=False,
                val_metric_name='1-auc_ovr',lr=.053,wd=.015,sq_mom=.988,
                lr_sched='flat_anneal',wd_sched='cos_log_15',first_layer_lr_factor=.25,
                embedding_size=5,max_one_hot_cat_size=18,hidden_sizes=[512,256,128],
                act='silu',p_drop=.05,p_drop_sched='invsqrtp1e-3',
                plr_hidden_1=16,plr_hidden_2=8,plr_act_name='gelu',plr_lr_factor=.1151,
                plr_sigma=2.33,ls_eps=.01,ls_eps_sched='sqrt_cos',add_front_scale=False,
                bias_init_mode='neg-uniform-dynamic-2',
                tfms=['one_hot','median_center','robust_scale','smooth_clip','embedding','l2_normalize'])
    config={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_sha256':{name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                             for name in ('train.csv','test.csv','sample_submission.csv')},
            'params':params,'folds':args.folds,'seed':42,'smoke':args.smoke}
    config_path=output/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Configuration changed; preserve previous output by using a new experiment folder.')
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    report({'mode':'RealMLP','rows':len(tr),'features':len(full.columns)+3,
            'categories':len(cats),'device':torch.cuda.get_device_name(0),'params':params})
    splits=list(StratifiedKFold(args.folds,shuffle=True,random_state=42).split(tr,y))
    oof=np.full(len(tr),np.nan);prediction=np.zeros(len(ts));scores=[]
    for f,(a,b) in enumerate(splits,1):
        path=output/f'fold-{f}.npz'
        if path.exists():
            saved=np.load(path)
            if not np.array_equal(saved['val_idx'],b):raise ValueError('Cached validation IDs differ')
            oof[b]=saved['val_pred'];prediction+=saved['test_pred']/args.folds
            scores.append(json.loads((output/f'fold-{f}.json').read_text()))
            report({'fold':f,'restored':True})
        else:
            xa,xv,xt=fold_frames(full,keys,a,b,len(tr),y)
            if f==1:
                altered=y.copy();altered.iloc[b]=1-altered.iloc[b]
                check=fold_frames(full,keys,a,b,len(tr),altered)
                for first,second in zip((xa,xv,xt),check):pd.testing.assert_frame_equal(first,second)
                del check
                report({'validation_label_perturbation':'PASS'})
            params['random_state']=42+f
            model=RealMLP_TD_Classifier(**params)
            report({'fold':f,'stage':'training','rows':len(a),'seconds':round(time.time()-started,1)})
            model.fit(xa,y.iloc[a],X_val=xv,y_val=y.iloc[b],cat_col_names=cats)
            if str(model.alg_interface_.device)!='cuda:0':raise RuntimeError('Unexpected training device')
            # Lightning tears its module down onto CPU after fit; inspect the
            # actual Trainer configuration rather than post-fit tensor placement.
            trainer_config=model.alg_interface_.min_trainer_kwargs
            if trainer_config['accelerator']!='gpu' or trainer_config['devices']!=[0]:
                raise RuntimeError('Trainer did not use the expected GPU')
            pv=predict_chunks(model,xv);pt=predict_chunks(model,xt)
            if not np.isfinite(pv).all() or not np.isfinite(pt).all():raise ValueError('Nonfinite model predictions')
            score={'fold':f,'auc':float(roc_auc_score(y.iloc[b],pv)),
                   'seconds_elapsed':round(time.time()-started,2),
                   'reported_device':str(getattr(model.alg_interface_,'device','cuda:0 requested'))}
            np.savez_compressed(path,val_idx=b,val_pred=pv,test_pred=pt)
            (output/f'fold-{f}.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            oof[b]=pv;prediction+=pt/args.folds;scores.append(score)
            report(score)
            del model,xa,xv,xt
            gc.collect();torch.cuda.empty_cache()
        if args.first_fold and f==1:
            report({'stage':'first-fold-screen-complete','auc':scores[0]['auc']})
            return
    if not np.isfinite(oof).all():raise ValueError('OOF predictions incomplete')
    submission=sample.copy();submission.satisfaction=submission.id.map(pd.Series(prediction,index=ts.id))
    validate_submission(submission,sample)
    path=output/'submission.csv';submission.to_csv(path,index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':oof}).to_csv(output/'oof.csv',index=False)
    pd.DataFrame({'id':ts.id,'prediction':prediction}).to_csv(output/'test_predictions.csv',index=False)
    metrics={'competition':'sampled_smoke' if args.smoke else COMPETITION,
             'model_description':'Apache2 RealMLP numeric categories','folds':args.folds,'seed':42,
             'fold_scores':scores,'selected_blend':{'auc':float(roc_auc_score(y,oof))},
             'input_sha256':config['input_sha256'],
             'submission_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'seconds':round(time.time()-started,2),'license':'Apache-2.0',
             'versions':{p:importlib.metadata.version(p) for p in ('pytabkit','torch','pytorch-lightning')},
             'note':'Outer validation labels excluded from target features. Inner cross-fitting for train encoding.'}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    report(metrics)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--ensemble',type=int,default=4)
    parser.add_argument('--first-fold',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
