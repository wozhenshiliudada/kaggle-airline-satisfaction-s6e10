"""Prospective outer-fold pilot for actual TabICL weight fine-tuning.

Inner checkpoint selection uses a fixed 10k context and disjoint 10k validation.
Gradient training sees every remaining outer-training row each completed epoch.
Outer predictions use fixed 50k/200k contexts, not all training rows. Historical
outer labels have been researched before; this is adaptive validation.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import time
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabicl import TabICLClassifier, FinetunedTabICLClassifier
from round18_joint import ROOT, sha, save, load, idxsha
from train import read_data

OUT=ROOT/'experiments/round23-finetune'
ANALYSIS=ROOT/'analyses/round23-finetune'
WEIGHTS=ROOT/'references/round18-foundation/tabicl-classifier-v2-20260212.ckpt'


def msg(**x): print(json.dumps(x),flush=True)


def state_sha(state):
    h=hashlib.sha256()
    for name,v in sorted(state.items()):
        h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def infer_config():
    return {n:dict(offload='cpu' if n=='COL_CONFIG' else 'auto',safety_factor=.65,
        cpu_safety_factor=.4,max_pinned_memory_mb=128.,use_async=False)
        for n in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']}


def freeze():
    if (OUT/'protocol.json').exists(): raise ValueError('Already frozen')
    p=dict(round=23,created_at_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
        input_sha256={n:sha(ROOT/'data'/n) for n in ['train.csv','test.csv','sample_submission.csv']},
        protected_sha256={n:sha(ROOT/n) for n in ['best-submission.json','last-submission.json',
            'team-request.json','outputs-round13/submission.csv','outputs-round13/oof.csv']},
        pretrained_sha256=sha(WEIGHTS),dependency_audit_sha256=sha(ROOT/'references/round23-finetune/download-audit.json'),
        versions={n:md.version(n) for n in ['torch','tabicl','transformers','numpy','scikit-learn']},
        split=dict(folds=5,seed=42,inner_validation_rows=10000,inner_seed=2026100623),
        inner_context_rows=10000,outer_context_rows=[50000,200000],context_seed=2026100624,
        train_params=dict(epochs=4,learning_rate=1e-5,weight_decay=.01,grad_clip=1.,amp=True,
            n_estimators_finetune=1,n_estimators_validation=1,n_estimators_inference=1,
            max_data_size=2048,finetune_ctx_query_ratio=.2,norm_methods=['none'],
            early_stopping=True,patience=2,min_delta=1e-5,time_limit=900.,save_interval=1000,
            random_state=23,device='cuda:0',verbose=False,allow_auto_download=False),
        outer_params=dict(n_estimators=1,norm_methods=['none'],random_state=0,device='cuda:0',
            kv_cache='repr',batch_size=1,use_amp=True,use_fa3=False,n_jobs=4,offload_mode='cpu'),
        candidate_logit_weights=[.05,.10,.20],pilot_continue_delta=1e-5,
        full_gate=dict(mean_delta=1e-5,all_five_fold_deltas_positive=True),
        limitations='Historical native folds reused. No pristine holdout. Inner validation selects '
          'weights without outer labels; outer contexts fixed before metrics. Time limit is '
          'checked after epochs, not a hard interrupt. WDDM memory can exceed physical VRAM. '
          'Single-fold gains cannot authorize a submission. No round16 H1/H2 rescoring.',
        license=dict(source='https://huggingface.co/jingang/TabICL',weights='BSD-3-Clause',
                     tabpfn35_downloaded=False),
        source='https://github.com/soda-inria/tabicl')
    save(OUT/'protocol.json',p);msg(stage='frozen',sha256=sha(OUT/'protocol.json'))


def check():
    p=load(OUT/'protocol.json')
    if sha(__file__)!=p['script_sha256']:raise ValueError('Frozen source changed')
    if sha(WEIGHTS)!=p['pretrained_sha256']:raise ValueError('Weights changed')
    if sha(ROOT/'references/round23-finetune/download-audit.json')!=p['dependency_audit_sha256']:
        raise ValueError('Dependency audit changed')
    for n,h in p['protected_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Protected file changed: '+n)
    for n,h in p['input_sha256'].items():
        if sha(ROOT/'data'/n)!=h:raise ValueError('Input changed: '+n)
    return p


def data(fold,p):
    tr,ts,_,raw,y=read_data(ROOT/'data');y=y.to_numpy()
    X=pd.concat([tr[raw],ts[raw]],ignore_index=True)
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].astype('category').cat.codes
    X=X.fillna(-1).to_numpy(dtype='float32')[:len(tr)]
    a,b=list(StratifiedKFold(p['split']['folds'],shuffle=True,random_state=p['split']['seed']).split(X,y))[fold-1]
    g,v=train_test_split(a,test_size=p['split']['inner_validation_rows'],stratify=y[a],random_state=p['split']['inner_seed'])
    c,_=train_test_split(g,train_size=p['inner_context_rows'],stratify=y[g],random_state=p['context_seed'])
    large,_=train_test_split(a,train_size=200000,stratify=y[a],random_state=p['context_seed'])
    small,_=train_test_split(large,train_size=50000,stratify=y[large],random_state=p['context_seed'])
    if set(b)&set(g) or set(b)&set(v) or set(g)&set(v):raise ValueError('Split overlap')
    folder=OUT/f'fold-{fold}';folder.mkdir(parents=True,exist_ok=True)
    manifest=dict(train_idx_sha256=idxsha(a),gradient_idx_sha256=idxsha(g),inner_val_idx_sha256=idxsha(v),
        inner_context_idx_sha256=idxsha(c),outer_val_idx_sha256=idxsha(b),
        context_idx_sha256={str(k):idxsha(z) for k,z in [(50000,small),(200000,large)]},
        gradient_rows=len(g),inner_validation_rows=len(v),outer_train_rows=len(a),outer_validation_rows=len(b))
    path=folder/'split.json'
    if path.exists() and load(path)!=manifest:raise ValueError('Split changed')
    save(path,manifest)
    del tr,ts;gc.collect()
    return X,y,a,b,g,v,c,{50000:small,200000:large}


class FixedContextFinetune(FinetunedTabICLClassifier):
    def _load_pretrained(self,device):
        super()._load_pretrained(device)
        self.initial_state_sha_=state_sha(self.model_.state_dict());self.validation_events_=[];self.batch_count_=0
        self.started_=time.monotonic()

    def _compute_batch_loss(self,batch,model):
        loss=super()._compute_batch_loss(batch,model);self.batch_count_+=1
        if self.batch_count_%50==0:
            msg(stage='gradient',batches=self.batch_count_,loss=float(loss.detach()),
                seconds=time.monotonic()-self.started_,available_ram_gb=psutil.virtual_memory().available/1e9)
        return loss

    def _run_validation(self,inner,X_train,y_train,X_val,y_val):
        result=super()._run_validation(inner,X_train[self.inner_context_local_],y_train[self.inner_context_local_],X_val,y_val)
        if not np.isfinite(result.primary):raise ValueError('Invalid inner validation metric')
        event=dict(batch_count=self.batch_count_,auc=float(result.primary),seconds=time.monotonic()-self.started_,
                   state_sha256=state_sha(self.model_.state_dict()))
        self.validation_events_.append(event);msg(stage='inner_validation',**event)
        return result


def train(fold,probe=False):
    p=check();X,y,a,b,g,v,c,contexts=data(fold,p);torch.set_num_threads(4)
    folder=OUT/('probe' if probe else f'fold-{fold}')
    ckpt=folder/('weights' if not probe else 'checkpoint')
    if (folder/'training.json').exists() or (ckpt/'best.ckpt').exists():raise ValueError('Training already attempted')
    params=dict(p['train_params'])
    if probe:
        g=g[:2048];v=v[:128];c=g
        params.update(epochs=1,early_stopping=False,use_lr_scheduler=False)
    model=FixedContextFinetune(**params,model_path=str(WEIGHTS),extra_classifier_kwargs=dict(
        kv_cache=False,use_amp=True,use_fa3=False,n_jobs=4,batch_size=1,offload_mode='cpu',inference_config=infer_config()))
    pos={int(z):i for i,z in enumerate(g)};model.inner_context_local_=np.array([pos[int(z)] for z in c])
    torch.cuda.reset_peak_memory_stats();t=time.monotonic();msg(stage='probe_started' if probe else 'fine_tune_started',gradient_rows=len(g))
    try:
        model.fit(X[g],y[g],X_val=X[v],y_val=y[v],output_dir=ckpt)
        final=state_sha(model.model_.state_dict())
        result=dict(successful=True,probe=probe,seconds=time.monotonic()-t,gradient_rows=len(g),inner_validation_rows=len(v),
            initial_state_sha256=model.initial_state_sha_,final_state_sha256=final,
            final_weights_changed=final!=model.initial_state_sha_,batch_count=model.batch_count_,
            validation_events=model.validation_events_,best_inner_auc=float(model._best_metric_),checkpoint_sha256=sha(ckpt/'best.ckpt'),
            peak_allocated_vram_gb=torch.cuda.max_memory_allocated()/1e9,peak_reserved_vram_gb=torch.cuda.max_memory_reserved()/1e9,
            outer_labels_used_for_training_or_checkpoint_selection=False)
    except (RuntimeError,ValueError) as e:
        result=dict(successful=False,probe=probe,seconds=time.monotonic()-t,error=type(e).__name__+': '+str(e)[:1500])
    save(folder/'training.json',result);msg(stage='training_complete',**result)
    if not result['successful']:raise SystemExit(1)


def predict(fold,mode,rows):
    p=check();X,y,a,b,g,v,c,contexts=data(fold,p);torch.set_num_threads(4)
    folder=OUT/f'fold-{fold}';path=folder/f'{mode}-{rows}.npz'
    if path.exists():raise ValueError('Prediction exists')
    weights=WEIGHTS if mode=='pretrained' else folder/'weights/best.ckpt'
    if mode=='finetuned':
        r=load(folder/'training.json')
        if not r['successful'] or not r['final_weights_changed']:raise ValueError('No selected fine-tuned weights')
    model=TabICLClassifier(**p['outer_params'],model_path=str(weights),allow_auto_download=False,inference_config=infer_config())
    torch.cuda.reset_peak_memory_stats();t=time.monotonic();msg(stage='outer_fit_started',fold=fold,mode=mode,context_rows=rows)
    try:
        model.fit(X[contexts[rows]],y[contexts[rows]]);fit_seconds=time.monotonic()-t
        msg(stage='outer_query_started',mode=mode,context_rows=rows,fit_seconds=fit_seconds,query_rows=len(b))
        pred=model.predict_proba(X[b])[:,1]
        if not np.isfinite(pred).all() or ((pred<0)|(pred>1)).any():raise ValueError('Invalid predictions')
        np.savez_compressed(path,train_idx=contexts[rows],val_idx=b,val_pred=pred)
        result=dict(successful=True,fold=fold,mode=mode,context_rows=rows,query_rows=len(b),fit_seconds=fit_seconds,
            seconds=time.monotonic()-t,prediction_sha256=sha(path),model_sha256=sha(weights),
            peak_allocated_vram_gb=torch.cuda.max_memory_allocated()/1e9,peak_reserved_vram_gb=torch.cuda.max_memory_reserved()/1e9)
    except (RuntimeError,ValueError) as e:
        result=dict(successful=False,fold=fold,mode=mode,context_rows=rows,seconds=time.monotonic()-t,error=type(e).__name__+': '+str(e)[:1000])
    save(folder/f'{mode}-{rows}-resources.json',result);msg(stage='outer_prediction_complete',**result)
    if not result['successful']:raise SystemExit(1)


def score(fold):
    p=check();tr,_,_,_,y=read_data(ROOT/'data');y=y.to_numpy()
    baseline=pd.read_csv(ROOT/'outputs-round13/oof.csv')
    if not baseline.id.equals(tr.id):raise ValueError('Baseline ID alignment changed')
    if not np.array_equal(baseline.satisfaction.to_numpy(),y):raise ValueError('Baseline labels changed')
    base=baseline.prediction.to_numpy();folder=OUT/f'fold-{fold}'
    results=[]
    for mode in ['pretrained','finetuned']:
        for rows in p['outer_context_rows']:
            path=folder/f'{mode}-{rows}.npz'
            if not path.exists():continue
            r=load(folder/f'{mode}-{rows}-resources.json')
            if sha(path)!=r['prediction_sha256']:raise ValueError('Prediction changed')
            with np.load(path) as z:b=z['val_idx'];pred=z['val_pred']
            if idxsha(b)!=load(folder/'split.json')['outer_val_idx_sha256']:raise ValueError('Query split changed')
            old=float(roc_auc_score(y[b],base[b]));zbase=logit(np.clip(base[b],1e-6,1-1e-6));zc=logit(np.clip(pred,1e-6,1-1e-6))
            blends=[]
            for w in p['candidate_logit_weights']:
                auc=float(roc_auc_score(y[b],expit((1-w)*zbase+w*zc)))
                blends.append(dict(weight=w,auc=auc,delta=auc-old))
            results.append(dict(mode=mode,context_rows=rows,carrier_auc=float(roc_auc_score(y[b],pred)),baseline_auc=old,blends=blends))
    eligible=[dict(context_rows=r['context_rows'],**m) for r in results if r['mode']=='finetuned' for m in r['blends'] if m['delta']>=p['pilot_continue_delta']]
    result=dict(fold=fold,results=results,eligible_pilot_variants=eligible,continue_full_five_folds=bool(eligible),submission_authorized_by_pilot=False,
        limitation=p['limitations'],protected_state_unchanged=True)
    save(ANALYSIS/f'fold-{fold}-scores.json',result);msg(stage='scored',**result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','probe','train','predict','score'])
    ap.add_argument('--fold',type=int,default=1);ap.add_argument('--mode',choices=['pretrained','finetuned']);ap.add_argument('--rows',type=int,choices=[50000,200000]);args=ap.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='probe':train(args.fold,True)
    elif args.stage=='train':train(args.fold)
    elif args.stage=='predict':predict(args.fold,args.mode,args.rows)
    else:score(args.fold)
