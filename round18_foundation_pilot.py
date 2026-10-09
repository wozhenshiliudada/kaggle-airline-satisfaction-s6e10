"""Full-context TabICL feasibility, without accessing H1/H2 labels.

Small stages test resources only. If a full development-fold fit is feasible,
evaluate that entire development validation fold, with three frozen weights.
This is a one-fold pilot; it cannot authorize a submission.
"""
from pathlib import Path
import json
import time
import gc
import traceback
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.special import expit, logit
from tabicl import TabICLClassifier
from round18_joint import ROOT, sha, save, load, idxsha

OUT = ROOT / 'experiments/round18-foundation-pilot'
ANALYSIS = ROOT / 'analyses/round18-foundation-pilot'
WEIGHTS = ROOT / 'references/round18-foundation/tabicl-classifier-v2-20260212.ckpt'


def msg(**x):print(json.dumps(x, ensure_ascii=False), flush=True)


def run():
    torch.set_num_threads(4)
    tr = pd.read_csv(ROOT / 'data/train.csv'); ts = pd.read_csv(ROOT / 'data/test.csv')
    raw = [c for c in tr if c not in ['id','satisfaction']]
    X = pd.concat([tr[raw], ts[raw]], ignore_index=True)
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):X[c] = X[c].astype('category').cat.codes
    X = X.fillna(-1).to_numpy(dtype='float32')[:len(tr)]
    protocol = load(ROOT / 'experiments/round16-locked/protocol.json')
    with np.load(ROOT / protocol['split_path']) as z:
        dev, fd = z['dev_idx'], z['dev_fold_id']
    a, b = dev[fd != 0], dev[fd == 0]
    y = np.full(len(tr), np.nan); y[dev] = tr.satisfaction.iloc[dev]
    del tr, ts; gc.collect()
    expected = load(ROOT / 'references/round18-foundation/download-audit.json')['checkpoint']['sha256']
    if sha(WEIGHTS) != expected:raise ValueError('Weight fingerprint changed')
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        model='TabICL v2', package_version='2.2.0', checkpoint_sha256=expected,
        input_sha256=protocol['input_sha256'], script_sha256=sha(__file__),
        development_train_sha256=idxsha(a), development_validation_sha256=idxsha(b),
        stages=[5000, 50000, len(a)], resource_only_query_rows=512, full_context_query_batch=1024,
        pilot_logit_weights=[0.05,0.15,0.30], train_context_subsampling_in_full_stage=False,
        stage_fit_budget_seconds=600, projected_full_fold_budget_seconds=1800,
        params=dict(n_estimators=1, norm_methods=['none'], random_state=0, device='cuda:0',
           kv_cache=True, batch_size=1, use_amp=True, use_fa3=False, n_jobs=4),
        offload='disk for column embeddings; other stages auto CPU/disk offload',
        validation_note='One development-fold pilot only. No H1/H2 labels used. No pristine holdout or fully nested CV. Positive pilot does not authorize submission.')
    manifest = OUT / 'protocol.json'
    if manifest.exists():raise ValueError('Pilot already started; inspect receipt instead of rerunning')
    save(manifest, frozen)
    base_fold = ROOT / 'experiments/round16-locked/baseline_current/fold-1.npz'
    with np.load(base_fold) as z:
        # Baseline fold packets store the combined prediction for v, excluded hold and test.
        keys = z.files
    msg(stage='pilot_prepared', training_rows=len(a), validation_rows=len(b), native_baseline_keys=keys)
    results = []
    for rows in frozen['stages']:
        is_full = rows == len(a)
        selected = a if is_full else train_test_split(a, train_size=rows, stratify=y[a], random_state=20261006)[0]
        directory = OUT / f'offload-{rows}'; directory.mkdir(exist_ok=True)
        model = TabICLClassifier(**frozen['params'], model_path=WEIGHTS, allow_auto_download=False,
            offload_mode='disk', disk_offload_dir=str(directory),
            inference_config={name:dict(offload='disk' if name=='COL_CONFIG' else 'auto',
                disk_offload_dir=str(directory), max_pinned_memory_mb=512., cpu_safety_factor=.5,
                safety_factor=.65, use_async=False) for name in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']})
        torch.cuda.reset_peak_memory_stats(); t = time.time()
        msg(stage='context_fit_started', context_rows=rows, available_ram_gb=psutil.virtual_memory().available/1e9)
        try:
            model.fit(X[selected], y[selected].astype('int8'))
            fit_seconds = time.time()-t
            if not np.array_equal(model.classes_, [0,1]):raise ValueError('Wrong classifier classes')
            query_start = time.time(); probe = model.predict_proba(X[b[:512]])[:,1]
            query_seconds = time.time()-query_start
            if not np.isfinite(probe).all() or ((probe<0)|(probe>1)).any():raise ValueError('Invalid pilot prediction')
            projected = fit_seconds+query_seconds*len(b)/512.
            result = dict(context_rows=rows, fit_seconds=fit_seconds, query_512_seconds=query_seconds,
                projected_full_fold_seconds=projected, peak_allocated_vram_gb=torch.cuda.max_memory_allocated()/1e9,
                peak_reserved_vram_gb=torch.cuda.max_memory_reserved()/1e9, classes=model.classes_.tolist(),
                full_context=is_full, successful=True, available_ram_gb=psutil.virtual_memory().available/1e9)
            results.append(result); save(ANALYSIS / 'resources.json', dict(stages=results, completed=False))
            msg(stage='resource_measurement', **result)
            if is_full and fit_seconds<=frozen['stage_fit_budget_seconds'] and projected<=frozen['projected_full_fold_budget_seconds']:
                pred = np.full(len(b), np.nan); pred[:512] = probe
                for offset in range(512,len(b),1024):
                    pred[offset:offset+1024] = model.predict_proba(X[b[offset:offset+1024]])[:,1]
                    if offset//1024 % 16 == 0:msg(stage='full_context_query', predicted_rows=min(offset+1024,len(b)), total_rows=len(b))
                np.savez_compressed(OUT / 'full-context-fold-1.npz', val_idx=b, val_pred=pred, train_idx=selected)
                score(pred, b, y, base_fold, frozen)
            elif is_full:
                result['full_scoring_skipped_reason']='Estimated fold runtime exceeds prospectively fixed resource budget.'
            del model; gc.collect(); torch.cuda.empty_cache()
        except (torch.OutOfMemoryError, RuntimeError, ValueError) as error:
            results.append(dict(context_rows=rows, full_context=is_full, successful=False, error=type(error).__name__+': '+str(error)[:700], seconds=time.time()-t))
            msg(stage='resource_limit', **results[-1]); del model; gc.collect(); torch.cuda.empty_cache(); break
    save(ANALYSIS / 'resources.json', dict(stages=results, completed=True, protocol_sha256=sha(manifest),
        hold1_scored=False, hold2_scored=False, submission_authorized=False))


def score(pred, b, y, baseline_file, frozen):
    from round16_evaluate import paired
    with np.load(baseline_file) as z:
        if not np.array_equal(z['val_idx'], b):raise ValueError('Baseline order')
        base=z['val_pred']
    candidates = []
    for w in frozen['pilot_logit_weights']:
        candidate=expit((1-w)*logit(np.clip(base,1e-6,1-1e-6))+w*logit(np.clip(pred,1e-6,1-1e-6)))
        candidates.append(dict(weight=w, **paired(y[b].astype('int8'),base,candidate)))
    result=dict(baseline_auc=float(roc_auc_score(y[b],base)), carrier_auc=float(roc_auc_score(y[b],pred)),
        candidates=candidates, validation_note=frozen['validation_note'], hold1_scored=False, hold2_scored=False)
    save(ANALYSIS / 'incremental.json', result); msg(stage='full_context_incremental', **result)


if __name__=='__main__':run()
