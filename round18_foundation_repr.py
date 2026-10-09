"""Exact full-context fallback: cache row representations, one validation batch.
Avoid persistent 12-layer K/V tensors. Same pretrained network, labels, raw
inputs and seed as the pilot. No H1/H2 values used or scored.
"""
from datetime import datetime,timezone
from pathlib import Path
import time
import gc
import json
import importlib.metadata
import numpy as np
import pandas as pd
import torch
import psutil
from tabicl import TabICLClassifier
from round18_joint import ROOT,sha,save,load,idxsha
import round18_foundation_pilot as pilot

OUT=ROOT/'experiments/round18-foundation-repr'
ANALYSIS=ROOT/'analyses/round18-foundation-repr'

def msg(**x):print(json.dumps(x),flush=True)

def run():
    torch.set_num_threads(4)
    tr=pd.read_csv(ROOT/'data/train.csv');ts=pd.read_csv(ROOT/'data/test.csv')
    raw=[c for c in tr if c not in ['id','satisfaction']]
    X=pd.concat([tr[raw],ts[raw]],ignore_index=True)
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].astype('category').cat.codes
    X=X.fillna(-1).to_numpy(dtype='float32')[:len(tr)]
    source=load(ROOT/'experiments/round16-locked/protocol.json')
    with np.load(ROOT/source['split_path']) as z:dev,fd=z['dev_idx'],z['dev_fold_id']
    a,b=dev[fd!=0],dev[fd==0]
    y=np.full(len(tr),np.nan);y[dev]=tr.satisfaction.iloc[dev]
    del tr,ts;gc.collect()
    OUT.mkdir(exist_ok=True);disk=OUT/'offload';disk.mkdir(exist_ok=True)
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
        input_sha256=source['input_sha256'],checkpoint_sha256=sha(pilot.WEIGHTS),
        package_version=importlib.metadata.version('tabicl'),raw_columns=raw,
        train_idx_sha256=idxsha(a),val_idx_sha256=idxsha(b),full_training_context_rows=len(a),
        full_validation_rows=len(b),query_mode='one-shot entire development fold',
        params=dict(n_estimators=1,norm_methods=['none'],random_state=0,device='cuda:0',
                    kv_cache='repr',batch_size=1,use_amp=True,use_fa3=False,n_jobs=4),
        pilot_logit_weights=[.05,.15,.30],wall_budget_seconds=900,
        validation_note='Full-context single development fold. H1/H2 masked and not accessed. Positive pilot cannot authorize upload; full aligned retraining and excluded validation required.')
    if (OUT/'protocol.json').exists():raise ValueError('Already started; do not rescore')
    save(OUT/'protocol.json',p)
    def make(cache):
        return TabICLClassifier(**{**p['params'],'kv_cache':cache},model_path=pilot.WEIGHTS,allow_auto_download=False,
            offload_mode='disk',disk_offload_dir=str(disk),
            inference_config={name:dict(offload='disk' if name=='COL_CONFIG' else 'auto',
                 disk_offload_dir=str(disk),max_pinned_memory_mb=512.,cpu_safety_factor=.5,
                 safety_factor=.65,use_async=False) for name in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']})
    probes=[]
    for cache in [True,'repr']:
        m=make(cache);m.fit(X[a[:5000]],y[a[:5000]].astype('int8'));probes.append(m.predict_proba(X[b[:128]])[:,1])
        del m;gc.collect();torch.cuda.empty_cache()
    difference=float(np.max(np.abs(probes[0]-probes[1])))
    save(ANALYSIS/'cache-equivalence.json',dict(max_probability_difference=difference,checked_training_rows=5000,checked_query_rows=128,label_metric_computed=False))
    msg(stage='cache_equivalence',max_probability_difference=difference)
    model=make('repr');torch.cuda.reset_peak_memory_stats();t=time.time()
    msg(stage='full_repr_fit',training_rows=len(a),available_ram_gb=psutil.virtual_memory().available/1e9)
    try:
        model.fit(X[a],y[a].astype('int8'));fit_seconds=time.time()-t
        msg(stage='full_repr_query_started',fit_seconds=fit_seconds,query_rows=len(b))
        pred=model.predict_proba(X[b])[:,1]
        if not np.isfinite(pred).all() or ((pred<0)|(pred>1)).any():raise ValueError('Invalid probability')
        np.savez_compressed(OUT/'full-context-fold-1.npz',train_idx=a,val_idx=b,val_pred=pred)
        result=dict(successful=True,full_context=True,training_rows=len(a),validation_rows=len(b),fit_seconds=fit_seconds,total_seconds=time.time()-t,
                    peak_allocated_vram_gb=torch.cuda.max_memory_allocated()/1e9,peak_reserved_vram_gb=torch.cuda.max_memory_reserved()/1e9,
                    full_context_checkpoint_sha256=sha(OUT/'full-context-fold-1.npz'),hold1_scored=False,hold2_scored=False)
        pilot.OUT=OUT;pilot.ANALYSIS=ANALYSIS
        pilot.score(pred,b,y,ROOT/'experiments/round16-locked/baseline_current/fold-1.npz',p)
    except (torch.OutOfMemoryError,RuntimeError,ValueError) as e:
        result=dict(successful=False,full_context=True,error=type(e).__name__+': '+str(e)[:700],seconds=time.time()-t,hold1_scored=False,hold2_scored=False)
    save(ANALYSIS/'resources.json',result);msg(stage='full_repr_complete',**result)

if __name__=='__main__':run()
