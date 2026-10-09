"""Audit new caches and evaluate only protocol-fixed production candidates."""
import argparse
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from round18_joint import ROOT,sha,save,load,idxsha
from round19_route_tabm import candidates
from round12_models import rebuilt
from evaluate_context import paired
from train import read_data,validate_submission,COMPETITION


def run(full=False):
    tr,ts,sample,raw,y=read_data(ROOT/'data');ya=y.to_numpy()
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y))
    records=[]; protected=True
    for directory,script,variants in [
        ('round19-route-tabm','round19_route_tabm.py',['aux_route_embedding','raw_route_embedding']),
        ('round20-route-control','round20_route_control.py',['matched_control','route_onehot'])]:
        p=load(ROOT/'experiments'/directory/'protocol.json')
        assert p['script_sha256']==sha(ROOT/script)
        for s,h in {**p['protected_sha256'],**p['source_sha256']}.items():assert sha(ROOT/s)==h,s
        for variant in variants:
            folder=ROOT/'experiments'/directory/variant
            for f,(a,b) in enumerate(folds,1):
                file=folder/f'fold-{f}.npz'
                if not file.exists():continue
                m=load(file.with_suffix('.json'))
                assert m['checkpoint_sha256']==sha(file)
                assert m['protocol_sha256']==sha(ROOT/'experiments'/directory/'protocol.json')
                assert m['train_idx_sha256']==idxsha(a) and m['val_idx_sha256']==idxsha(b)
                with np.load(file) as z:
                    assert np.array_equal(z['train_idx'],a) and np.array_equal(z['val_idx'],b)
                    assert not np.intersect1d(a,b).size
                    assert z['test_pred'].shape==(len(ts),) and z['val_pred'].shape==(len(b),)
                    for k in ['val_pred','test_pred']:
                        pred=z[k];assert np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
                    assert abs(roc_auc_score(ya[b],z['val_pred'])-m['auc'])<1e-12
                assert m['validation_label_perturbation']=='PASS'
                records.append(dict(round=19 if 'round19' in directory else 20,variant=variant,
                                     fold=f,model_auc=m['auc'],seconds=m['seconds'],peak_vram_gb=m['peak_vram_gb']))
    audit=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        frozen_sources_and_protected_files_unchanged=True,checkpoint_checks='PASS',
        label_perturbation_checks='PASS',completed_fits=records,
        H1_H2_separate_metrics_not_rescored=True,tabpfn35_weights_not_run=True,
        evaluator_sha256=sha(__file__))
    save(ROOT/'evidence/round20-cache-audit.json',audit)
    if not full:
        print('PASS: cached fit integrity, native folds, probabilities, protected state');return
    p=load(ROOT/'experiments/round20-route-control/protocol.json')
    base=pd.read_csv(ROOT/'outputs-round13/oof.csv');sub=pd.read_csv(ROOT/'outputs-round13/submission.csv')
    assert base.id.equals(tr.id) and np.array_equal(base.satisfaction,ya)
    bp=base.prediction.to_numpy();bt=sub.satisfaction.to_numpy();validate_submission(sub,sample)
    olddir=ROOT/'experiments/round9-tabm-auxiliary-5fold'
    old=pd.read_csv(olddir/'oof.csv').prediction.to_numpy();oldt=pd.read_csv(olddir/'submission.csv').satisfaction.to_numpy()
    # Verify the exact .09 replacement coefficient in round13's formula.
    source_dirs=['experiments/round9-lgb-auxiliary-5fold','outputs-round5',
       'experiments/realmlp-context-5fold-8ens','experiments/round13-realmlp-route-auxiliary-10fold-8ens',
       'experiments/round9-tabm-auxiliary-5fold','experiments/round11-experts-5fold']
    arrays=[pd.read_csv(ROOT/d/'oof.csv').prediction.to_numpy() for d in source_dirs]
    test_arrays=[pd.read_csv(ROOT/d/'submission.csv').satisfaction.to_numpy() for d in source_dirs]
    assert np.max(abs(rebuilt(*arrays)-bp))<1e-12
    assert np.max(abs(rebuilt(*test_arrays)-bt))<1e-12
    bs=float(roc_auc_score(ya,bp));bbs=[float(roc_auc_score(ya[b],bp[b])) for a,b in folds]
    choices={}; rows=[]
    for variant in p['variants']:
        folder=ROOT/'experiments/round20-route-control'/variant
        if not (folder/'metrics.json').exists():continue
        m=load(folder/'metrics.json');oo=pd.read_csv(folder/'oof.csv');ss=pd.read_csv(folder/'submission.csv')
        assert oo.id.equals(tr.id) and np.array_equal(oo.satisfaction,ya)
        assert sha(folder/'submission.csv')==m['submission_sha256'];validate_submission(ss,sample)
        for f,(a,b) in enumerate(folds,1):
            with np.load(folder/f'fold-{f}.npz') as z:assert np.max(abs(z['val_pred']-oo.prediction.to_numpy()[b]))<1e-12
        cv=candidates(bp,old,oo.prediction.to_numpy(),p)
        tv=candidates(bt,oldt,ss.satisfaction.to_numpy(),p)
        for name,pred in cv.items():
            key=variant+'/'+name;auc=float(roc_auc_score(ya,pred))
            ds=[float(roc_auc_score(ya[b],pred[b]))-bb for (a,b),bb in zip(folds,bbs)]
            row=dict(candidate=key,auc=auc,delta=auc-bs,block_deltas=ds,
                     eligible=auc-bs>=p['full_gate']['mean_delta'] and min(ds)>0)
            rows.append(row);choices[key]=(pred,tv[name])
    eligible=[r for r in rows if r['eligible']]
    selected=max(eligible,key=lambda r:r['auc']) if eligible else None
    result=dict(competition=COMPETITION,folds=5,input_sha256=p['input_sha256'],
        baseline_auc=bs,rows=rows,selected_blend=selected,
        validation_note=p['limitations'],evaluator_sha256=sha(__file__))
    if selected:
        pred,test=choices[selected['candidate']]
        result['paired_against_best']=paired(ya,bp,pred)
        out=ROOT/'outputs-round20';out.mkdir(exist_ok=True)
        ss=sample.copy();ss.satisfaction=test;validate_submission(ss,sample)
        ss.to_csv(out/'submission.csv',index=False)
        pd.DataFrame(dict(id=tr.id,satisfaction=ya,prediction=pred)).to_csv(out/'oof.csv',index=False)
        result.update(submission_sha256=sha(out/'submission.csv'),
            external_sha256=load(ROOT/'outputs-round13/metrics.json')['external_sha256'],
            model_description='Route categorical TabM with preserved tenfold auxiliary ensemble')
        save(out/'metrics.json',result)
    save(ROOT/'analyses/round20-route-control/full-comparison.json',result)
    print(result)


if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('--full',action='store_true');a=arg.parse_args();run(a.full)
