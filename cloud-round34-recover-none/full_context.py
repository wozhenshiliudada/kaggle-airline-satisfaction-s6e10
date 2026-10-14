"""Offline recovery of the missing none arm; no AUC or competition submission.

GPU 0 reruns the same two-member none arm. Completed power predictions remain frozen on host.
Outer labels are never used for fitting or parameter/checkpoint selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path('/kaggle/working')
TRAIN_SHA = '2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36'
TEST_SHA = 'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6'
WEIGHT_SHA = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
FEATURE_SHA = '4fb0a0706b2099e15e07b26aee03fe9c616326f603a4ad96a8fa4ec24b3664e5'
ARMS = [('none', 31), ('power', 131)]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def save(name, record):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(dict(file=name, **record)), flush=True)


def locate(name, expected=None):
    files = list(Path('/kaggle/input').rglob(name))
    if expected is not None:
        files = [p for p in files if sha(p) == expected]
    if len(files) != 1:
        raise ValueError('Input not uniquely matched: ' + name)
    return files[0]


def indices_sha(indices):
    import numpy as np
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def worker(index):
    import gc
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    from sklearn.model_selection import StratifiedKFold
    from tabicl import TabICLClassifier
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not allocated')
    tr = pd.read_csv(locate('train.csv', TRAIN_SHA))
    ts = pd.read_csv(locate('test.csv', TEST_SHA))
    raw = [c for c in tr if c not in ['id', 'satisfaction']]
    full = pd.concat([tr[raw], ts[raw]], ignore_index=True)
    with np.load(locate('cloud_covariates.npz', FEATURE_SHA), allow_pickle=False) as packet:
        if not np.array_equal(packet['train_id'], tr.id) or not np.array_equal(packet['test_id'], ts.id):
            raise ValueError('Feature row ID alignment changed')
        extra = packet['features']
        names = packet['columns'].tolist()
        if extra.shape != (len(full), 17):
            raise ValueError('Auxiliary feature shape changed')
        for i, name in enumerate(names):
            full[name] = extra[:, i]
    full['Flight Distance category'] = full['Flight Distance'].astype('int64').astype(str)
    full['Flight Distance coarse category'] = (full['Flight Distance'] // 10).astype('int64').astype(str)
    for name in full:
        if not pd.api.types.is_numeric_dtype(full[name]):
            full[name] = full[name].fillna('__missing__').astype('category')
        else:
            full[name] = full[name].fillna(-1).astype('float32')
    if full.shape[1] != 40:
        raise ValueError('Expected forty covariates')
    y = tr.satisfaction.to_numpy(dtype='int8')
    a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
    weights = locate('tabicl-classifier-v2-20260212.ckpt', WEIGHT_SHA)
    norm, seed = ARMS[index]
    disk = Path('/tmp') / ('s6e10-r34-offload-' + str(index))
    disk.mkdir(exist_ok=True)
    config = {name: dict(offload='disk' if name == 'COL_CONFIG' else 'auto',
        disk_offload_dir=str(disk), safety_factor=.60, cpu_safety_factor=.25,
        max_pinned_memory_mb=128., use_async=False)
        for name in ['COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG']}
    params = dict(n_estimators=2, norm_methods=[norm], random_state=seed,
        device='cuda:0', kv_cache='repr', batch_size=1, use_amp=True,
        use_fa3=False, n_jobs=2, model_path=str(weights), allow_auto_download=False,
        offload_mode='disk', disk_offload_dir=str(disk), inference_config=config)
    descriptor = dict(worker=index, assigned_physical_gpu=index,
        gpu_name=torch.cuda.get_device_name(0),
        total_vram_gb=torch.cuda.get_device_properties(0).total_memory / 1e9,
        norm=norm, seed=seed, members=2, feature_columns=list(full),
        training_context_rows=len(a), validation_rows=len(b),
        training_idx_sha256=indices_sha(a), validation_idx_sha256=indices_sha(b),
        feature_packet_sha256=FEATURE_SHA, checkpoint_sha256=WEIGHT_SHA,
        outer_labels_used_for_fit=False, full_context_subsampled=False,
        available_ram_gb=psutil.virtual_memory().available / 1e9)
    save(f'worker-{index}-protocol.json', descriptor)
    # Resource-only warm-up; neither AUC nor weights are chosen from its query.
    small = TabICLClassifier(**params)
    small.fit(full.iloc[a[:5000]], y[a[:5000]])
    probe = small.predict_proba(full.iloc[b[:128]])[:, 1]
    if not np.isfinite(probe).all():
        raise ValueError('Warm-up predictions invalid')
    del small, probe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = TabICLClassifier(**params)
    save(f'worker-{index}-progress.json', dict(stage='full_context_fit', **descriptor))
    model.fit(full.iloc[a], y[a])
    fit_seconds = time.monotonic() - started
    save(f'worker-{index}-progress.json', dict(stage='full_validation_query',
        worker=index, fit_seconds=fit_seconds, query_rows=len(b)))
    prediction = model.predict_proba(full.iloc[b])[:, 1]
    if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
        raise ValueError('Invalid full-context prediction')
    path = OUT / f'worker-{index}-predictions.npz'
    np.savez_compressed(path, train_idx=a, val_idx=b, val_pred=prediction)
    save(f'worker-{index}-result.json', dict(successful=True, **descriptor,
        fit_seconds=fit_seconds, total_seconds=time.monotonic() - started,
        peak_allocated_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
        peak_reserved_vram_gb=torch.cuda.max_memory_reserved() / 1e9,
        prediction_sha256=sha(path), script_sha256=sha(__file__),
        auc_computed=False, submission_generated=False))


def parent():
    import torch
    import psutil
    import shutil
    hardware = dict(torch_version=torch.__version__,
        cuda_devices=[dict(index=i, name=torch.cuda.get_device_name(i),
            total_vram_gb=torch.cuda.get_device_properties(i).total_memory / 1e9)
            for i in range(torch.cuda.device_count())],
        total_ram_gb=psutil.virtual_memory().total / 1e9,
        scratch_free_gb=shutil.disk_usage('/tmp').free / 1e9)
    save('hardware.json', hardware)
    if torch.cuda.device_count() != 2:
        raise RuntimeError('This frozen pilot requires two independent CUDA devices')
    wheels = [locate(name) for name in ['tabicl-2.2.0-py3-none-any.whl',
        'einops-0.8.1-py3-none-any.whl', 'huggingface_hub-0.34.4-py3-none-any.whl']]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index',
                    '--no-deps', *map(str, wheels)], check=True)
    processes, logs = [], []
    started = time.monotonic()
    for i in range(1):
        log = open(OUT / f'worker-{i}.log', 'w')
        logs.append(log)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), HF_HUB_OFFLINE='1',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        processes.append(subprocess.Popen([sys.executable, __file__, '--worker', str(i)],
            stdout=log, stderr=subprocess.STDOUT, env=env))
    try:
        while any(p.poll() is None for p in processes):
            if time.monotonic() - started > 5100:
                for p in processes:
                    if p.poll() is None:
                        p.terminate()
                for p in processes:
                    try:
                        p.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        p.kill()
                save('budget-stop.json', dict(successful=False, reason='5100 second recovery worker budget',
                    wall_seconds=time.monotonic() - started))
                break
            time.sleep(5)
    finally:
        for log in logs:
            log.close()
    results = [json.loads((OUT / f'worker-{i}-result.json').read_text())
               if (OUT / f'worker-{i}-result.json').exists() else dict(successful=False)
               for i in range(1)]
    save('run-result.json', dict(successful=all(r['successful'] for r in results),
        worker_exit_codes=[p.poll() for p in processes], workers=results,
        total_seconds=time.monotonic() - started, script_sha256=sha(__file__),
        total_ensemble_members=2, gpu_memory_pooled=False, submission_generated=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    try:
        parent() if args.worker is None else worker(args.worker)
    except Exception as error:
        name = 'parent-error.json' if args.worker is None else f'worker-{args.worker}-error.json'
        save(name, dict(successful=False, error=type(error).__name__ + ': ' + str(error)[:1600]))
        raise
