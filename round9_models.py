"""Fresh auxiliary-feature tree and route-aware TabM trials; old files untouched.

All target-derived features fit outer training rows only. The previously built
auxiliary features use unlabeled train/test covariates (transductive learning).
TabM uses the installed Apache-2.0 PyTabKit implementation, trained from scratch.
"""
import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from auxiliary_round import AUXDIR, config_file, hashes
from auxiliary_neural_round import TEACHERDIR, route_keys, frames
from context_round import ROOT, sha, context_keys, context_blocks, add_context
from improve import CACHE, make_keys, encoded_fold
from nn_round import prepare_frames, predict_chunks, report
from train import read_data, COMPETITION, validate_submission


def run(args):
    started = time.time()
    tr, ts, sample, raw, y = read_data(ROOT / 'data')
    input_hashes = hashes()
    auxiliary_meta = json.loads((AUXDIR / 'metrics.json').read_text(encoding='utf-8'))
    teacher_meta = json.loads((TEACHERDIR / 'metrics.json').read_text(encoding='utf-8'))
    reference = ROOT / 'experiments' / 'realmlp-5fold-8ens'
    tree_ref = json.loads((ROOT / 'experiments/context-conditional-5fold/metrics.json').read_text(encoding='utf-8'))
    if any(h != input_hashes for h in (auxiliary_meta['configuration']['input_sha256'],
            teacher_meta['configuration']['input_sha256'], tree_ref['input_sha256'])):
        raise ValueError('Input fingerprint mismatch')
    if sha(AUXDIR / 'features.pkl') != auxiliary_meta['features_sha256']:
        raise ValueError('Auxiliary fingerprint mismatch')
    if sha(ROOT / 'data-original/data.csv') != tree_ref['external_sha256']:
        raise ValueError('Original data fingerprint mismatch')
    aux = pd.read_pickle(AUXDIR / 'features.pkl')
    if len(aux) != len(tr) + len(ts) or not aux.index.equals(pd.RangeIndex(len(aux))):
        raise ValueError('Auxiliary row alignment mismatch')
    if set(aux).intersection({'id', 'satisfaction'}):
        raise ValueError('Forbidden auxiliary feature')
    data = joblib.load(CACHE)
    if not data['tr'].id.equals(tr.id) or not data['ts'].id.equals(ts.id):
        raise ValueError('Base cache row mismatch')
    pd.testing.assert_series_equal(data['y'], y)
    if args.model == 'lgb':
        full = data['base'].copy()
        full[aux.columns] = aux
        keys = make_keys(pd.concat([tr[raw], ts[raw]], ignore_index=True), raw, True)
        ckeys = context_keys(pd.concat([tr[raw], ts[raw]], ignore_index=True))
        params = dict(objective='binary', metric='auc', learning_rate=.025,
            num_threads=8, lambda_l2=5, lambda_l1=.05, min_data_in_leaf=20,
            num_leaves=63, max_depth=8, feature_fraction=.85, bagging_fraction=.9,
            bagging_freq=1, verbosity=-1, deterministic=True, force_col_wise=True,
            boost_from_average=True)
        def prepare(a, b, target):
            result = encoded_fold(full, keys, a, b, len(tr), target, True)
            add_context(result, context_blocks(ckeys, a, b, len(tr), target), ckeys, a, b, len(tr))
            return result
    else:
        import torch
        from pytabkit import TabM_D_Classifier
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.set_num_threads(4)
        full, keys = prepare_frames(tr, ts, raw)
        rkeys = route_keys(full)
        for c in ('original_lgb_probability', 'original_cat_probability'):
            full[c] = logit(data['base'][c].clip(1e-6, 1 - 1e-6)).astype('float32')
        if sha(TEACHERDIR / 'probability.npy') != teacher_meta['probability_sha256']:
            raise ValueError('Teacher fingerprint mismatch')
        full['original_xgb_logit'] = logit(np.clip(np.load(TEACHERDIR / 'probability.npy'), 1e-6, 1-1e-6)).astype('float32')
        for c in ('Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'):
            full['route_count_' + c] = full[c].map(full[c].value_counts()).astype('float32')
        full[aux.columns] = aux
        params = dict(device='cuda:0', random_state=42, n_threads=4, verbosity=1,
            n_cv=1, n_refit=0, arch_type='tabm-mini', tabm_k=16, num_emb_type='pwl',
            num_emb_n_bins=32, d_embedding=8, d_block=256, n_blocks=2, dropout=.1,
            batch_size=1024, lr=.002, weight_decay=0., n_epochs=40, patience=6,
            compile_model=False, allow_amp=True, val_metric_name='1-auc_ovr',
            share_training_batches=True, tfms=['quantile_tabr'])
        def prepare(a, b, target):
            result = frames(full, raw, keys, rkeys, a, b, len(tr), target)
            # TabM one-hot encodes categories; avoid thousands of one-hot columns.
            drop = ['category_' + c for c in ('Age', 'Flight Distance',
                'Departure Delay in Minutes', 'Arrival Delay in Minutes')]
            return tuple(frame.drop(columns=drop) for frame in result)
    del data, aux
    out = ROOT / 'experiments' / ('round9-' + args.model + '-auxiliary-5fold')
    config = dict(script_sha256=sha(Path(__file__)), input_sha256=input_hashes,
        auxiliary_features_sha256=auxiliary_meta['features_sha256'], base_cache_sha256=sha(CACHE),
        source_code_sha256={p: sha(ROOT / p) for p in ('improve.py', 'context_round.py',
            'nn_round.py', 'auxiliary_neural_round.py')},
        external_sha256=tree_ref['external_sha256'], params=params, folds=5, split_seed=42,
        model_seed=42, model=args.model, transductive_auxiliary=True,
        original_teacher_sha256=teacher_meta['probability_sha256'] if args.model == 'tabm' else None)
    config_file(out, config)
    oof = np.full(len(tr), np.nan)
    test = np.zeros(len(ts))
    scores = []
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y), 1):
        with np.load(reference / f'fold-{f}.npz') as saved:
            if not np.array_equal(saved['val_idx'], b):
                raise ValueError('Outer fold mismatch')
        path = out / f'fold-{f}.npz'
        if path.exists():
            with np.load(path) as saved:
                if not np.array_equal(saved['val_idx'], b):
                    raise ValueError('Cached fold mismatch')
                pv, pt = saved['val_pred'], saved['test_pred']
            score = json.loads((out / f'fold-{f}.json').read_text(encoding='utf-8'))
        else:
            xa, xv, xt = prepare(a, b, y)
            if f == 1:
                changed = y.copy()
                changed.iloc[b] = 1 - changed.iloc[b]
                perturbed = prepare(a, b, changed)
                for actual, alternate in zip((xa, xv, xt), perturbed):
                    pd.testing.assert_frame_equal(actual, alternate, check_exact=True)
                del perturbed
                report(dict(model=args.model, validation_label_perturbation='PASS'))
            if any(c in xa for c in ('id', 'satisfaction')):
                raise ValueError('Forbidden supervised feature')
            report(dict(stage='training', model=args.model, fold=f, features=len(xa.columns)))
            if args.model == 'lgb':
                fit = lgb.Dataset(xa, y.iloc[a])
                val = lgb.Dataset(xv, y.iloc[b], reference=fit)
                model = lgb.train(dict(params, seed=42+f), fit, num_boost_round=3500,
                    valid_sets=[val], valid_names=['val'], callbacks=[
                        lgb.early_stopping(180, first_metric_only=True, verbose=False),
                        lgb.log_evaluation(500)])
                pv, pt = model.predict(xv), model.predict(xt)
                extra = dict(iterations=model.best_iteration)
                pd.DataFrame(dict(feature=xa.columns, gain=model.feature_importance('gain'))).to_csv(out / f'importance-{f}.csv', index=False)
                del fit, val
            else:
                model = TabM_D_Classifier(**dict(params, random_state=42+f))
                cats = list(xa.select_dtypes(include='category').columns)
                model.fit(xa, y.iloc[a], X_val=xv, y_val=y.iloc[b], cat_col_names=cats)
                interface = model.alg_interface_.sub_split_interfaces[0]
                if str(interface.device_) != 'cuda:0':
                    raise RuntimeError('TabM GPU mismatch')
                pv, pt = predict_chunks(model, xv), predict_chunks(model, xt)
                extra = dict(device=str(interface.device_))
            if not all(np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 for p in (pv, pt)):
                raise ValueError('Invalid model probability')
            score = dict(fold=f, auc=float(roc_auc_score(y.iloc[b], pv)), features=len(xa.columns),
                elapsed_seconds=round(time.time()-started, 2), **extra)
            np.savez_compressed(path, val_idx=b, val_pred=pv, test_pred=pt)
            (out / f'fold-{f}.json').write_text(json.dumps(score, indent=2), encoding='utf-8')
            del model, xa, xv, xt
            gc.collect()
            if args.model == 'tabm':
                torch.cuda.empty_cache()
        oof[b] = pv
        test += pt / 5
        scores.append(score)
        report(dict(model=args.model, **score))
        if args.first_fold:
            base = pd.read_csv(ROOT / 'outputs-round8/oof.csv').prediction.to_numpy()[b]
            result = dict(scope='first_fold_only', model=args.model, baseline_auc=float(roc_auc_score(y.iloc[b], base)),
                model_auc=score['auc'], correlation=float(np.corrcoef(base, pv)[0, 1]),
                blend_auc={str(w): float(roc_auc_score(y.iloc[b], (1-w)*base+w*pv)) for w in (.1, .25, .5)})
            (out / 'first-fold-screen.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            report(result)
            return
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete OOF')
    sub = sample.copy()
    sub.satisfaction = test
    validate_submission(sub, sample)
    sub.to_csv(out / 'submission.csv', index=False)
    pd.DataFrame(dict(id=tr.id, satisfaction=y, prediction=oof)).to_csv(out / 'oof.csv', index=False)
    metadata = dict(competition=COMPETITION, model_description='Round9 '+args.model+' auxiliary',
        folds=5, seed=42, params=params, input_sha256=input_hashes, external_sha256=tree_ref['external_sha256'],
        submission_sha256=sha(out / 'submission.csv'), selected_blend=dict(auc=float(roc_auc_score(y, oof))),
        fold_scores=scores, seconds=round(time.time()-started, 2),
        versions={p: importlib.metadata.version(p) for p in ('lightgbm', 'pytabkit', 'torch')},
        validation_note='Five-fold supervised OOF, inner cross-fit target features. Reused auxiliary reconstruction uses only unlabeled covariates but is transductive. Validation for early stopping, excluded from feature fitting.')
    (out / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report(metadata)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model', choices=['lgb', 'tabm'])
    p.add_argument('--first-fold', action='store_true')
    run(p.parse_args())
