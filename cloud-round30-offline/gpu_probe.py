"""Round30 private GPU/TabICL access probe. No competition submission generated."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT=Path('/kaggle/working')
OUT.mkdir(exist_ok=True)


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data),flush=True)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    import torch
    hardware=dict(cuda_available=torch.cuda.is_available(),torch_version=torch.__version__,
        devices=[dict(index=i,name=torch.cuda.get_device_name(i),
            total_memory_gb=torch.cuda.get_device_properties(i).total_memory/1e9)
            for i in range(torch.cuda.device_count())])
    save('hardware.json',hardware)
    if not hardware['cuda_available']:raise RuntimeError('No actual CUDA resource allocated')
    import psutil
    save('system-memory.json',dict(total_ram_gb=psutil.virtual_memory().total/1e9,available_ram_gb=psutil.virtual_memory().available/1e9))
    torch.set_num_threads(4)
    z=torch.rand((1024,1024),device='cuda:0');value=float((z@z).mean());torch.cuda.synchronize();del z
    if value<=0:raise RuntimeError('CUDA computation failed')
    wheels=[next(Path('/kaggle/input').rglob(name)) for name in ['tabicl-2.2.0-py3-none-any.whl','einops-0.8.1-py3-none-any.whl','huggingface_hub-0.34.4-py3-none-any.whl']]
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-index','--no-deps',*map(str,wheels)],check=True)
    import numpy as np
    import pandas as pd
    import psutil
    from tabicl import TabICLClassifier
    expected='2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36'
    files=[f for f in Path('/kaggle/input').rglob('train.csv') if sha(f)==expected]
    if len(files)!=1:raise ValueError('Competition train fingerprint not uniquely matched')
    frame=pd.read_csv(files[0]).iloc[:2304];raw=[c for c in frame if c not in ['id','satisfaction']]
    X=frame[raw].copy()
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].astype('category').cat.codes
    X=X.fillna(-1).to_numpy(dtype='float32');y=frame.satisfaction.to_numpy(dtype='int8')
    weights=next(Path('/kaggle/input').rglob('tabicl-classifier-v2-20260212.ckpt'))
    if sha(weights)!='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0':
        raise ValueError('Pretrained checkpoint fingerprint mismatch')
    model=TabICLClassifier(n_estimators=1,norm_methods=['none'],random_state=0,device='cuda:0',
        use_amp=True,use_fa3=False,kv_cache=False,batch_size=1,n_jobs=4,model_path=weights,allow_auto_download=False)
    t=time.monotonic();model.fit(X[:2048],y[:2048]);pred=model.predict_proba(X[2048:])[:,1]
    if not np.isfinite(pred).all() or ((pred<0)|(pred>1)).any():raise ValueError('Invalid actual model inference')
    save('probe-result.json',dict(successful=True,hardware=hardware,actual_cuda_computation=True,offline=True,
        actual_TabICL_prediction_rows=len(pred),model_seconds=time.monotonic()-t,
        checkpoint_sha256=sha(weights),competition_input_sha256=expected,
        script_sha256=sha(__file__),auc_computed=False,submission_generated=False))


try:
    main()
except Exception as e:
    save('probe-error.json',dict(successful=False,error=type(e).__name__+': '+str(e)[:1200]))
    raise
