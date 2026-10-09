"""Prospectively split and retrain development models for rounds16/17."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from auxiliary_round import hashes
from context_round import ROOT, sha
from round16_baseline import BaselineFactory
from round16_carriers import CarrierFactory, MODES
from train import read_data, COMPETITION

SPLIT_FOLDER = ROOT/'experiments/round16-locked'
CODE = ['round16_run.py','round16_baseline.py','round16_carriers.py','round16_evaluate.py',
        'improve.py','context_round.py','nn_round.py','auxiliary_neural_round.py',
        'round12_models.py','auxiliary_round.py']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def frozen(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if load(path)!=value:
            raise ValueError('Frozen configuration differs: '+str(path))
    else:
        path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def report(value):
    print(json.dumps(value,ensure_ascii=False),flush=True)


def relative(path):
    return str(path.relative_to(ROOT)).replace('\\','/')


def setup(round_number):
    tr,ts,sample,raw,y=read_data(ROOT/'data')
    SPLIT_FOLDER.mkdir(parents=True,exist_ok=True)
    split_path=SPLIT_FOLDER/'split.npz'
    if not split_path.exists():
        dev,held=train_test_split(np.arange(len(tr)),test_size=.2,stratify=y,random_state=20261006)
        h1,h2=train_test_split(held,test_size=.5,stratify=y.iloc[held],random_state=20261007)
        dev,h1,h2=[np.sort(v).astype('int64') for v in (dev,h1,h2)]
        fid=np.empty(len(dev),dtype='int8')
        for f,(_,v) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261006).split(dev,y.iloc[dev])):
            fid[v]=f
        np.savez_compressed(split_path,dev_idx=dev,hold1_idx=h1,hold2_idx=h2,dev_fold_id=fid,
            train_id=tr.id.to_numpy(),test_id=sample.id.to_numpy())
    with np.load(split_path,allow_pickle=False) as z:
        dev,h1,h2,fid=[z[k].copy() for k in ('dev_idx','hold1_idx','hold2_idx','dev_fold_id')]
        assert np.array_equal(z['train_id'],tr.id) and np.array_equal(z['test_id'],sample.id)
    hold=np.concatenate([h1,h2])
    assert np.array_equal(np.sort(np.concatenate([dev,hold])),np.arange(len(tr)))
    assert len(np.unique(hold))==len(hold)
    mask=y.astype('float64').copy();mask.iloc[hold]=np.nan
    assert mask.iloc[hold].isna().all() and mask.iloc[dev].notna().all()
    epochs=4 if round_number==16 else 8
    out=ROOT/f'experiments/round{round_number}-locked'
    codes={p:sha(ROOT/p) for p in CODE}
    model_paths={'baseline_current':'experiments/round16-locked/baseline_current',
        **{m:f'experiments/round{round_number}-locked/carrier-{m}' for m in MODES}}
    protected={p:sha(ROOT/p) for p in ['best-submission.json','outputs-round13/submission.csv','last-submission.json']}
    p=dict(competition=COMPETITION,round=round_number,folds=3,epochs=epochs,
        split_path=relative(split_path),split_sha256=sha(split_path),input_sha256=hashes(),
        external_sha256=sha(ROOT/'data-original/data.csv'),source_code_sha256=codes,
        protected_file_sha256=protected,baseline_members=['baseline_current'],candidate_members=list(MODES),
        models={k:{'path':v} for k,v in model_paths.items()},
        active_holdout='hold1' if round_number==16 else 'hold2',dev_min_delta=1e-5,hold_min_delta=1e-5,
        stack_parameters=dict(C=1.,max_iter=1000,solver='lbfgs',random_state=42),
        analysis_dir=f'analyses/round{round_number}-locked',output_dir=f'outputs-round{round_number}-locked',
        auxiliary_transductive=True,holdout_labels_masked_in_training_factory=True,
        fallback_rule='If round16 development, H1, full-training compatibility or official score is not better, round17 uses the same three inputs at eight epochs and prospectively reserved H2.',
        production_rule='Transfer development coefficients unchanged to preserved full-training baseline plus only the selected newly trained five-fold carrier. Historical OOF is an additional compatibility gate, not independent selection.',
        production_gate=dict(global_delta=1e-5,all_five_historical_evaluation_blocks_positive=True),
        note='All labels were used in earlier research; excluded retraining sets are not pristine. Development meta cross-fit is not fully nested. Production transfer changes native training coverage.')
    protocol=out/'protocol.json'
    # Once uploaded, last-submission may change. Do not silently regenerate an old manifest.
    if protocol.exists():
        prior=load(protocol)
        for key in ('protected_file_sha256',):
            p[key]=prior[key]
    frozen(protocol,p)
    for name,folder in model_paths.items():
        cfg=dict(competition=COMPETITION,folds=3,input_sha256=p['input_sha256'],
            external_sha256=p['external_sha256'],split_sha256=p['split_sha256'],
            source_code_sha256=codes,model=name,
            holdout_excluded_from_supervised_training_and_early_stopping=True,
            label_mask='Both H1 and H2 supplied to factories as NaN; development labels only.',
            training_protocol='Three supervised development folds, explicit development validation for checkpoint selection.',
            epochs=3 if name=='baseline_current' else epochs)
        cfg_path=ROOT/folder/'configuration.json'
        if name=='baseline_current' and round_number==17:
            old=load(cfg_path)
            for k in ('competition','folds','input_sha256','external_sha256','split_sha256','source_code_sha256'):
                assert old[k]==cfg[k]
        else:
            frozen(cfg_path,cfg)
    return tr,ts,sample,raw,y,mask,dev,hold,fid,protocol,p


def summarize(folder,dev,hold,fid,test_id):
    oof=np.full(len(dev),np.nan);hp=np.zeros(len(hold));tp=np.zeros(len(test_id))
    for f in range(3):
        with np.load(folder/f'fold-{f+1}.npz',allow_pickle=False) as z:
            v=fid==f;a=fid!=f
            assert np.array_equal(z['train_idx'],dev[a]) and np.array_equal(z['val_idx'],dev[v])
            assert np.array_equal(z['hold_idx'],hold) and np.array_equal(z['test_id'],test_id)
            oof[v]=z['val_pred'];hp+=z['hold_pred']/3;tp+=z['test_pred']/3
    assert all(np.isfinite(p).all() and p.min()>=0 and p.max()<=1 for p in (oof,hp,tp))
    np.savez_compressed(folder/'summary.npz',dev_idx=dev,hold_idx=hold,test_id=test_id,
        dev_oof=oof,hold_pred=hp,test_pred=tp)
    report(dict(stage='summary_saved',folder=relative(folder),rows=len(oof)))


def run(args):
    tr,ts,sample,raw,y,masked,dev,hold,fid,protocol,p=setup(args.round)
    for name,digest in p['source_code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    report(dict(stage='protocol_ready',round=args.round,epochs=p['epochs'],development=len(dev),
        h1=len(hold)//2,h2=len(hold)-len(hold)//2,holdout_labels_masked=True,manifest=relative(protocol)))
    if args.model=='init':
        return
    if args.model=='baseline':
        if args.round!=16:raise ValueError('Reuse the exact stage16 retrained baseline in round17')
        factory=BaselineFactory(tr,ts,raw,masked)
        folder=ROOT/p['models']['baseline_current']['path']
        frozen(folder/'source-audit.json',factory.source_info)
        for f in range(3):
            a=dev[fid!=f];v=dev[fid==f];b=np.concatenate([v,hold]);native=folder/f'fold-{f+1}.npz'
            if native.exists():
                report(dict(stage='baseline_fold_restored',fold=f+1));continue
            fitted=factory.fit_predict(a,b,42+f+1,validation_idx=v,
                check_label_invariance=(f==0),cache_dir=folder/f'components-fold-{f+1}')
            bp,pt,info=fitted['baseline_current']
            np.savez_compressed(native,train_idx=a,val_idx=v,val_pred=bp[:len(v)],
                hold_idx=hold,hold_pred=bp[len(v):],test_id=sample.id.to_numpy(),test_pred=pt)
            detail=dict(fold=f+1,components={k:r[2] for k,r in fitted.items()},
                baseline_formula=info,checkpoint_sha256=sha(native),supervised_holdout_labels_available=False)
            (folder/f'fold-{f+1}.json').write_text(json.dumps(detail,indent=2,ensure_ascii=False),encoding='utf-8')
            del fitted;gc.collect()
        summarize(folder,dev,hold,fid,sample.id.to_numpy())
    else:
        factory=CarrierFactory(tr,ts,raw,masked,epochs=p['epochs'])
        folder=ROOT/p['models'][args.model]['path']
        frozen(folder/'source-audit.json',factory.audit)
        for f in range(3):
            a=dev[fid!=f];v=dev[fid==f];native=folder/f'fold-{f+1}.npz'
            if native.exists():
                report(dict(stage='carrier_fold_restored',mode=args.model,fold=f+1));continue
            vp,hp,tp,info=factory.fit_predict(args.model,a,v,hold,42+f+1)
            assert info['held_y_values_not_available']
            np.savez_compressed(native,train_idx=a,val_idx=v,val_pred=vp,
                hold_idx=hold,hold_pred=hp,test_id=sample.id.to_numpy(),test_pred=tp)
            info.update(fold=f+1,checkpoint_sha256=sha(native))
            (folder/f'fold-{f+1}.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
        summarize(folder,dev,hold,fid,sample.id.to_numpy())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--round',type=int,choices=[16,17],default=16)
    parser.add_argument('model',choices=['init','baseline',*MODES])
    run(parser.parse_args())
