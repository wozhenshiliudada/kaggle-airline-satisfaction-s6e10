"""Private Kaggle full-context resource probe; no submission or peer predictions.

Prepared for a larger GPU if needed. Native fold1 of sklearn SKF(5,42), raw21.
The completed local raw TabICL pilot was negative; do not launch automatically
as a score-improvement experiment without a new input/mechanism hypothesis.
"""
import subprocess
import sys
import json
import hashlib
import time
from pathlib import Path

subprocess.run([sys.executable,'-m','pip','install','--quiet','tabicl==2.2.0'],check=True)
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download
from tabicl import TabICLClassifier

INPUT_SHA={'train.csv':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36',
           'test.csv':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6'}
MODEL_SHA='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

out=Path('/kaggle/working');out.mkdir(exist_ok=True)
folders=[f.parent for f in Path('/kaggle/input').rglob('train.csv') if sha(f)==INPUT_SHA['train.csv']]
if len(folders)!=1:raise ValueError('Official competition input not uniquely identified by fingerprint')
data=folders[0]
for name,expected in INPUT_SHA.items():
    if sha(data/name)!=expected:raise ValueError('Competition input changed')
if not torch.cuda.is_available():raise RuntimeError('CUDA GPU resource required')
tr=pd.read_csv(data/'train.csv');ts=pd.read_csv(data/'test.csv');y=tr.satisfaction.to_numpy(dtype='int8')
raw=[c for c in tr if c not in ['id','satisfaction']]
X=pd.concat([tr[raw],ts[raw]],ignore_index=True)
for c in X:
    if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].astype('category').cat.codes
X=X.fillna(-1).to_numpy(dtype='float32')[:len(tr)]
a,b=next(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
weights=hf_hub_download('jingang/TabICL','tabicl-classifier-v2-20260212.ckpt',
                        revision='4dcd344ece2c00be9e831fdd35bed57b5ad83e19')
if sha(weights)!=MODEL_SHA:raise ValueError('Pretrained model fingerprint mismatch')
cfg=dict(input_sha256=INPUT_SHA,model_sha256=MODEL_SHA,script_sha256=sha(__file__),
         training_rows=len(a),validation_rows=len(b),native_folds=5,split_seed=42,fold=1,
         device=torch.cuda.get_device_name(0),params=dict(n_estimators=1,norm_methods=['none'],
            random_state=0,device='cuda:0',kv_cache='repr',batch_size=1,use_amp=True,use_fa3=False,n_jobs=4))
(out/'configuration.json').write_text(json.dumps(cfg,indent=2))
disk=out/'offload';disk.mkdir(exist_ok=True)
model=TabICLClassifier(**cfg['params'],model_path=weights,allow_auto_download=False,
        offload_mode='disk',disk_offload_dir=str(disk),
        inference_config={name:dict(offload='disk' if name=='COL_CONFIG' else 'auto',
            disk_offload_dir=str(disk),max_pinned_memory_mb=512.,cpu_safety_factor=.5,
            safety_factor=.65,use_async=False) for name in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']})
torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();t=time.time()
model.fit(X[a],y[a]);fit_seconds=time.time()-t
print(json.dumps({'fit_seconds':fit_seconds,'training_rows':len(a)}),flush=True)
pred=model.predict_proba(X[b])[:,1]
if not np.array_equal(model.classes_,[0,1]) or not np.isfinite(pred).all():raise ValueError('Invalid actual binary predictions')
np.savez_compressed(out/'fold-1.npz',train_idx=a,val_idx=b,val_pred=pred,train_id=tr.id.to_numpy())
report=dict(**cfg,auc=float(roc_auc_score(y[b],pred)),fit_seconds=fit_seconds,total_seconds=time.time()-t,
        peak_allocated_vram_gb=torch.cuda.max_memory_allocated()/1e9,checkpoint_sha256=sha(out/'fold-1.npz'),
        validation_note='First native fold only, outer validation not in context; adaptation and comparison to old OOF remain historical. No test CSV or submission generated.')
(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
