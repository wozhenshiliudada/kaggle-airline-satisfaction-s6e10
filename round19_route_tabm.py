"""Independent route-category TabM; no new pretrained weights or peer predictions.

Only this process replaces PyTabKit's Model constructor. Installed source and
all historical models remain unchanged. High-cardinality categories receive
trainable embeddings, avoiding a large one-hot first layer. Historical OOF
screening and early stopping are adaptive, not an untouched holdout estimate.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import json
import time
import importlib.metadata as md
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pytabkit import TabM_D_Classifier
from pytabkit.models.alg_interfaces import tabm_interface
from auxiliary_round import AUXDIR, hashes
from auxiliary_neural_round import TEACHERDIR, route_keys, frames
from nn_round import prepare_frames
from improve import CACHE
from train import read_data, COMPETITION, validate_submission
from round18_joint import ROOT, sha, save, load, idxsha

OUT = ROOT/'experiments/round19-route-tabm'
ANALYSIS = ROOT/'analyses/round19-route-tabm'
OLD = ROOT/'experiments/round9-tabm-auxiliary-5fold'
OriginalModel = tabm_interface.Model


class RouteCategories(nn.Module):
    def __init__(self, cards):
        super().__init__()
        self.cards = cards
        self.embeddings = nn.ModuleDict()
        for i, c in enumerate(cards):
            if c > 18:
                emb = nn.Embedding(c+1, 8, padding_idx=c)
                nn.init.normal_(emb.weight, std=8**-.5)
                with torch.no_grad(): emb.weight[c].zero_()
                self.embeddings[str(i)] = emb

    def forward(self, x):
        return torch.cat([self.embeddings[str(i)](x[..., i]) if c > 18 else
            nn.functional.one_hot(x[..., i], c+1)[..., :-1].float()
            for i, c in enumerate(self.cards)], dim=-1)


class RouteEmbeddingTabM(OriginalModel):
    def __init__(self, *args, cat_cardinalities, **kwargs):
        super().__init__(*args, cat_cardinalities=[8 if c > 18 else c
                                                  for c in cat_cardinalities], **kwargs)
        if cat_cardinalities: self.cat_module = RouteCategories(cat_cardinalities)


def msg(**x): print(json.dumps(x), flush=True)


def freeze():
    if (OUT/'protocol.json').exists(): raise ValueError('Already frozen')
    params = dict(load(OLD/'configuration.json')['params'], batch_size=512,
                  verbosity=1, n_epochs=40, patience=6)
    p = dict(round=19, created_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__), input_sha256=hashes(),
        protected_sha256={s:sha(ROOT/s) for s in ['best-submission.json',
            'last-submission.json', 'team-request.json', 'outputs-round13/submission.csv',
            'outputs-round13/oof.csv']},
        source_sha256={str(s.relative_to(ROOT)).replace('\\','/'):sha(s) for s in
            [AUXDIR/'features.pkl', TEACHERDIR/'probability.npy', CACHE,
             OLD/'configuration.json', OLD/'oof.csv', ROOT/'nn_round.py',
             ROOT/'auxiliary_neural_round.py', ROOT/'round12_models.py']},
        installed_model_sha256=sha(ROOT/'.venv/Lib/site-packages/pytabkit/models/nn_models/tabm.py'),
        versions={s:md.version(s) for s in ['pytabkit','torch','numpy','scikit-learn']},
        variants=['aux_route_embedding', 'raw_route_embedding'],
        params=params, category_embedding=dict(threshold=18, dimension=8,
            init_std=8**-.5, unseen_category='zero vector'),
        route_features=['categorical exact Flight Distance', 'categorical floor(Flight Distance/10)'],
        split=dict(folds=5, seed=42), candidate_weights=[.05, .10, .20],
        replacement_weight=.09, first_fold_continue_delta=3e-6,
        full_gate=dict(mean_delta=1e-5, all_five_blocks_positive=True),
        reference='Exact current round13 prediction retaining its ten-fold auxiliary network',
        limitations='Historical native five-fold OOF and early-stopping labels are reused. '
          'No pristine holdout remains. Pilot gains cannot establish public-score gains. '
          'No repeated round16 H1/H2 scoring. This adapts route-category ideas to TabM; '
          'it does not reproduce TabPFN architecture or its reported performance.',
        tabpfn35=dict(weights_downloaded=False, weights_run=False,
                     reason='User confirmed personal study; professional eligibility not established'),
        source_idea='https://www.kaggle.com/code/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160')
    save(OUT/'protocol.json', p)
    msg(stage='frozen', protocol_sha256=sha(OUT/'protocol.json'))


def check():
    p=load(OUT/'protocol.json')
    if p['script_sha256'] != sha(__file__): raise ValueError('Frozen source changed')
    for s,h in {**p['protected_sha256'], **p['source_sha256']}.items():
        if sha(ROOT/s) != h: raise ValueError('Changed source or protected file: '+s)
    for s,h in p['input_sha256'].items():
        if sha(ROOT/'data'/s) != h: raise ValueError('Input changed')
    return p


def prepare(tr, ts, raw, a, b, y, variant):
    if variant == 'raw_route_embedding':
        full = pd.concat([tr[raw], ts[raw]], ignore_index=True)
        for c in full:
            if not pd.api.types.is_numeric_dtype(full[c]): full[c]=pd.Categorical(full[c].fillna('__NA__'))
            else: full[c]=full[c].fillna(0).astype('float32')
        full['category_Flight Distance']=pd.Categorical(full['Flight Distance'])
        full['category_fd10']=pd.Categorical(np.floor(full['Flight Distance']/10))
        return tuple(full.iloc[idx].reset_index(drop=True).copy()
                     for idx in [a,b,np.arange(len(tr),len(full))])
    full, keys = prepare_frames(tr, ts, raw)
    rkeys = route_keys(full)
    data=joblib.load(CACHE)
    if not data['tr'].id.equals(tr.id) or not data['ts'].id.equals(ts.id): raise ValueError('Cache rows changed')
    for c in ['original_lgb_probability','original_cat_probability']:
        full[c]=logit(data['base'][c].clip(1e-6,1-1e-6)).astype('float32')
    del data
    full['original_xgb_logit']=logit(np.clip(np.load(TEACHERDIR/'probability.npy'),1e-6,1-1e-6)).astype('float32')
    for c in ['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']:
        full['route_count_'+c]=full[c].map(full[c].value_counts()).astype('float32')
    aux=pd.read_pickle(AUXDIR/'features.pkl')
    if len(aux)!=len(full) or not aux.index.equals(full.index): raise ValueError('Auxiliary rows changed')
    full[aux.columns]=aux
    del aux
    full['category_fd10']=pd.Categorical(np.floor(full['Flight Distance']/10))
    prepared=frames(full, raw, keys, rkeys, a, b, len(tr), y)
    # Restore exact distance only; keep all other round9 feature choices.
    drop=['category_'+c for c in ['Age','Departure Delay in Minutes','Arrival Delay in Minutes']]
    return tuple(f.drop(columns=drop) for f in prepared)


def candidates(base, old, pred, p):
    result={f'blend_{w}':(1-w)*base+w*pred for w in p['candidate_weights']}
    result['replace_tabm']=base+p['replacement_weight']*(pred-old)
    return result


def score(variant, b, pv, y, p):
    base=pd.read_csv(ROOT/'outputs-round13/oof.csv', usecols=['prediction']).prediction.to_numpy()[b]
    old=pd.read_csv(OLD/'oof.csv', usecols=['prediction']).prediction.to_numpy()[b]
    bs=float(roc_auc_score(y.iloc[b],base))
    rows=[dict(candidate=n,auc=float(roc_auc_score(y.iloc[b],pr)))
          for n,pr in candidates(base,old,pv,p).items()]
    for r in rows:r['delta']=r['auc']-bs
    result=dict(scope='adaptive native five-fold block 1 pilot', variant=variant,
        baseline_auc=bs, previous_tabm_auc=float(roc_auc_score(y.iloc[b],old)),
        model_auc=float(roc_auc_score(y.iloc[b],pv)), candidates=rows,
        continue_full=max(r['delta'] for r in rows)>=p['first_fold_continue_delta'],
        H1_rescored=False,H2_rescored=False)
    save(ANALYSIS/f'{variant}-pilot.json',result);msg(stage='pilot_result',**result)


def run(variant, first_fold=True):
    p=check(); torch.set_num_threads(4); tabm_interface.Model=RouteEmbeddingTabM
    tr,ts,sample,raw,y=read_data(ROOT/'data')
    folder=OUT/variant; folder.mkdir(parents=True,exist_ok=True)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y))
    oof=np.full(len(tr),np.nan); pt=np.zeros(len(ts)); records=[]
    for f,(a,b) in enumerate(folds,1):
        if first_fold and f>1:break
        target=folder/f'fold-{f}.npz'; meta=target.with_suffix('.json')
        expected=dict(train_idx_sha256=idxsha(a), val_idx_sha256=idxsha(b),
                      script_sha256=p['script_sha256'],protocol_sha256=sha(OUT/'protocol.json'))
        if target.exists():
            m=load(meta)
            if any(m[k]!=v for k,v in expected.items()) or m['checkpoint_sha256']!=sha(target): raise ValueError('Cached fold mismatch')
            with np.load(target) as z:
                if not np.array_equal(z['val_idx'],b):raise ValueError('Cached indices changed')
                pv,tt=z['val_pred'],z['test_pred']
        else:
            t=time.time(); msg(stage='prepare',variant=variant,fold=f,training_rows=len(a),validation_rows=len(b))
            prepared=prepare(tr,ts,raw,a,b,y,variant)
            altered=y.copy();altered.iloc[b]=1-altered.iloc[b]
            other=prepare(tr,ts,raw,a,b,altered,variant)
            for actual,changed in zip(prepared,other):pd.testing.assert_frame_equal(actual,changed,check_exact=True)
            del other,altered;gc.collect()
            xa,xv,xt=prepared; cats=list(xa.select_dtypes(include='category').columns)
            if set(xa).intersection(['id','satisfaction']):raise ValueError('Forbidden feature')
            msg(stage='training',variant=variant,fold=f,features=len(xa.columns),
                label_perturbation='PASS',category_cardinalities={c:len(xa[c].cat.categories) for c in cats})
            model=TabM_D_Classifier(**dict(p['params'],random_state=42+f))
            torch.cuda.reset_peak_memory_stats()
            model.fit(xa,y.iloc[a],X_val=xv,y_val=y.iloc[b],cat_col_names=cats)
            interface=model.alg_interface_.sub_split_interfaces[0]
            if str(interface.device_)!='cuda:0':raise ValueError('GPU mismatch')
            if not isinstance(interface.model_.cat_module,RouteCategories):raise ValueError('Embedding patch inactive')
            def predict(frame):return np.concatenate([model.predict_proba(frame.iloc[i:i+4096])[:,1] for i in range(0,len(frame),4096)])
            pv,tt=predict(xv),predict(xt)
            for pred in [pv,tt]:
                if not np.isfinite(pred).all() or pred.min()<0 or pred.max()>1:raise ValueError('Invalid probabilities')
            np.savez_compressed(target,train_idx=a,val_idx=b,val_pred=pv,test_pred=tt)
            m=dict(**expected,fold=f,auc=float(roc_auc_score(y.iloc[b],pv)),
                validation_label_perturbation='PASS',seconds=time.time()-t,
                checkpoint_sha256=sha(target),peak_vram_gb=torch.cuda.max_memory_allocated()/1e9)
            save(meta,m); msg(stage='trained',variant=variant,**m)
            del model,interface,prepared,xa,xv,xt;gc.collect();torch.cuda.empty_cache()
        records.append(m);oof[b]=pv;pt+=tt/5
        if f==1:score(variant,b,pv,y,p)
    if first_fold:return
    if not np.isfinite(oof).all():raise ValueError('Incomplete OOF')
    sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample)
    sub.to_csv(folder/'submission.csv',index=False)
    pd.DataFrame(dict(id=tr.id,satisfaction=y,prediction=oof)).to_csv(folder/'oof.csv',index=False)
    save(folder/'metrics.json',dict(competition=COMPETITION,folds=5,input_sha256=p['input_sha256'],
         selected_blend=dict(auc=float(roc_auc_score(y,oof))),submission_sha256=sha(folder/'submission.csv'),fold_scores=records))


if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('action', choices=['freeze','pilot','full'])
    arg.add_argument('--variant', choices=['aux_route_embedding','raw_route_embedding'],default='aux_route_embedding')
    a=arg.parse_args()
    if a.action=='freeze':freeze()
    else:run(a.variant,a.action=='pilot')
