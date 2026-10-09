"""Factorial reproduction of selected INTERNAL validation, not an upload gate."""
import gc
import time
import numpy as np
import torch
from tabicl import TabICLClassifier
from sklearn.metrics import roc_auc_score
import round25_finetune_routes as source
from round18_joint import ROOT,load,save,sha,idxsha


def main():
    p=source.check();X,y,a,b,g,v,c,contexts=source.data(1,p);torch.set_num_threads(4)
    folder=ROOT/'experiments/round26-consistency';folder.mkdir(parents=True,exist_ok=True)
    weights=source.OUT/'fold-1/weights/best.ckpt';r=load(source.OUT/'fold-1/training.json')
    protocol=dict(script_sha256=sha(__file__),source_protocol_sha256=sha(source.OUT/'protocol.json'),
        weights_sha256=sha(weights),context_idx_sha256=idxsha(c),inner_val_idx_sha256=idxsha(v),
        contexts=len(c),query_rows=len(v),factorial=[dict(seed=s,cache=k) for s in [23,0] for k in [False,'repr']],
        purpose='Reproduce selected inner validation and isolate cache and random permutation differences. No outer AUC or submission decision.',
        selected_inner_auc=r['best_inner_auc'])
    if (folder/'protocol.json').exists():raise ValueError('Diagnostic already started')
    save(folder/'protocol.json',protocol);results=[]
    for variant in protocol['factorial']:
        params=dict(p['outer_params'],random_state=variant['seed'],kv_cache=variant['cache'])
        model=TabICLClassifier(**params,model_path=str(weights),allow_auto_download=False,inference_config=source.infer_config())
        t=time.monotonic();model.fit(X[c],y[c]);pred=model.predict_proba(X[v])[:,1]
        score=float(roc_auc_score(y[v],pred));results.append(dict(**variant,auc=score,seconds=time.monotonic()-t,
            delta_vs_selected_inner=score-r['best_inner_auc']))
        np.savez_compressed(folder/f"seed-{variant['seed']}-cache-{variant['cache']}.npz",context_idx=c,val_idx=v,val_pred=pred)
        source.msg(stage='internal_consistency',**results[-1]);del model;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'analyses/round26-consistency/results.json',dict(protocol_sha256=sha(folder/'protocol.json'),
        results=results,outer_auc_computed=False,upload_authorized_by_diagnostic=False))


if __name__=='__main__':main()
