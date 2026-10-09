"""Matched five-fold control, context features, and original-data logit residuals.

Every mode uses the same rows, folds, inner target-encoding splits and tree
parameters. Only the context mode adds features; only offset changes initial
logits. The offset is from original-data models, never competition OOF labels.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from improve import ROOT, CACHE, make_keys, encoded_fold, emit, CATS
from train import read_data, COMPETITION, validate_submission

SERVICES = ['Inflight wifi service', 'Departure/Arrival time convenient',
            'Ease of Online booking', 'Gate location', 'Food and drink',
            'Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Inflight service', 'Cleanliness']
PAIRS = [('Inflight wifi service', 'Online boarding'),
         ('Inflight wifi service', 'Ease of Online booking'),
         ('Seat comfort', 'Inflight entertainment'),
         ('On-board service', 'Leg room service'),
         ('Inflight entertainment', 'Online boarding'),
         ('Food and drink', 'Cleanliness')]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def context_keys(full):
    ctx = full['Class'].astype(str) + '|' + full['Type of Travel'].astype(str) + '|' + full['Customer Type'].astype(str)
    values = {'context': ctx}
    for c in SERVICES:
        values['context__' + c] = ctx + '|' + full[c].astype(str)
    for c, d in PAIRS:
        values['context__' + c + '__' + d] = ctx + '|' + full[c].astype(str) + '|' + full[d].astype(str)
    values['context__age5'] = ctx + '|' + (full['Age'] // 5).astype(str)
    values['context__fd500'] = ctx + '|' + (full['Flight Distance'] // 500).astype(str)
    return pd.DataFrame({c: pd.factorize(v, sort=False)[0].astype('int32') for c, v in values.items()})


def context_blocks(keys, a, b, n, y):
    # Same inner partition for each context prior and conditional probability.
    inner = StratifiedKFold(5, shuffle=True, random_state=42)
    enc = TargetEncoder(target_type='binary', smooth=100., cv=inner)
    blocks = [enc.fit_transform(keys.iloc[a], y.iloc[a]),
              enc.transform(keys.iloc[b]), enc.transform(keys.iloc[n:])]
    result = []
    for block in blocks:
        margin = logit(np.clip(block, 1e-5, 1 - 1e-5))
        relative = margin[:, 1:] - margin[:, :1]
        names = ['conditional_probability_' + c for c in keys.columns]
        effects = ['context_logodds_effect_' + c for c in keys.columns[1:]]
        result.append(pd.DataFrame(np.concatenate([block, relative], axis=1).astype('float32'), columns=names + effects))
    return result


def add_context(frames, blocks, keys, a, b, n):
    for frame, block, idx in zip(frames, blocks, (a, b, np.arange(n, len(keys)))):
        frame[block.columns] = block
        frame['context_category'] = pd.Categorical(keys.iloc[idx]['context'].to_numpy(), categories=np.unique(keys['context']))
        s = frame[SERVICES]
        additions = {'service_mean': s.mean(axis=1), 'service_min': s.min(axis=1),
                     'service_zero_count': (s == 0).sum(axis=1),
                     'service_low_count': (s <= 2).sum(axis=1),
                     'service_high_count': (s >= 4).sum(axis=1),
                     'service_std': s.std(axis=1)}
        for c, values in additions.items():
            frame[c] = values.astype('float32')
        for c in ('Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment'):
            frame['relative_service_' + c] = (frame[c] - frame['service_mean']).astype('float32')


def offset_semantics_smoke():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(1500, 4)), columns=['a', 'b', 'c', 'd'])
    offset = 1.2 * x.a.to_numpy()
    target = rng.binomial(1, expit(offset + .7 * x.b.to_numpy()))
    fit = lgb.Dataset(x.iloc[:1000], target[:1000], init_score=offset[:1000])
    val = lgb.Dataset(x.iloc[1000:], target[1000:], init_score=offset[1000:], reference=fit)
    model = lgb.train(dict(objective='binary', metric='auc', num_leaves=7,
                           min_data_in_leaf=30, verbosity=-1, num_threads=2,
                           boost_from_average=False), fit, num_boost_round=12,
                      valid_sets=[val], valid_names=['val'])
    manual = roc_auc_score(target[1000:], expit(offset[1000:] + model.predict(x.iloc[1000:], raw_score=True)))
    recorded = model.best_score['val']['auc']
    if abs(manual - recorded) > 1e-10:
        raise ValueError(f'Offset evaluation/prediction mismatch: {manual} != {recorded}')
    return {'check': 'PASS', 'manual_auc': manual, 'native_auc': recorded}


def run(mode):
    start = time.time()
    out = ROOT / 'experiments' / ('context-' + mode + '-5fold')
    out.mkdir(exist_ok=True)
    data = joblib.load(CACHE)
    base, tr, ts, sample, raw, y = (data[k] for k in ('base', 'tr', 'ts', 'sample', 'raw', 'y'))
    hashes = {k: sha(ROOT / 'data' / k) for k in ('train.csv', 'test.csv', 'sample_submission.csv')}
    reference = json.loads((ROOT / 'experiments' / 'route-auc' / 'metrics.json').read_text())
    if hashes != reference['input_sha256'] or sha(ROOT / 'data-original' / 'data.csv') != reference['external_sha256']:
        raise ValueError('Input data fingerprints changed')
    official = read_data(ROOT / 'data')
    if not tr.id.equals(official[0].id) or not ts.id.equals(official[1].id):
        raise ValueError('Cache ID alignment changed')
    pd.testing.assert_series_equal(y, official[4])
    del official
    params = dict(objective='binary', metric='auc', learning_rate=.025,
                  num_threads=8, lambda_l2=5, lambda_l1=.05,
                  min_data_in_leaf=20, num_leaves=63, max_depth=8,
                  feature_fraction=.85, bagging_fraction=.9, bagging_freq=1,
                  verbosity=-1, deterministic=True, force_col_wise=True,
                  boost_from_average=(mode != 'offset'))
    config = {'script_sha256': sha(Path(__file__)), 'feature_script_sha256': sha(ROOT / 'improve.py'),
              'cache_sha256': sha(CACHE), 'params': params, 'mode': mode, 'folds': 5,
              'seed': 42, 'rounds': 3500, 'early_stopping': 180,
              'input_sha256': hashes, 'external_sha256': reference['external_sha256']}
    cp = out / 'configuration.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Experiment configuration changed; use fresh folder')
    cp.write_text(json.dumps(config, indent=2), encoding='utf-8')
    full = pd.concat([tr[raw], ts[raw]], ignore_index=True)
    keys = make_keys(full, raw, True)
    ckeys = context_keys(full) if mode == 'conditional' else None
    n = len(tr)
    offset = .5 * logit(np.clip(base.original_lgb_probability.to_numpy(), 1e-5, 1 - 1e-5)) + .5 * logit(np.clip(base.original_cat_probability.to_numpy(), 1e-5, 1 - 1e-5))
    if mode == 'offset':
        smoke = offset_semantics_smoke()
        (out / 'offset-semantics.json').write_text(json.dumps(smoke, indent=2), encoding='utf-8')
        emit(smoke)
    oof = np.full(n, np.nan)
    pred = np.zeros(len(ts))
    scores = []
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y), 1):
        record = out / f'fold-{f}.npz'
        if record.exists():
            saved = np.load(record)
            if not np.array_equal(saved['val_idx'], b):
                raise ValueError('Saved split differs')
            pv, pt = saved['val_pred'], saved['test_pred']
            score = json.loads((out / f'fold-{f}.json').read_text())
        else:
            xa, xv, xt = encoded_fold(base, keys, a, b, n, y, True)
            if mode == 'conditional':
                blocks = context_blocks(ckeys, a, b, n, y)
                if f == 1:
                    alternate = y.copy()
                    alternate.iloc[b] = 1 - alternate.iloc[b]
                    check = context_blocks(ckeys, a, b, n, alternate)
                    for actual, perturbed in zip(blocks, check):
                        pd.testing.assert_frame_equal(actual, perturbed, check_exact=True)
                    del check
                    emit({'context_validation_label_perturbation': 'PASS'})
                add_context((xa, xv, xt), blocks, ckeys, a, b, n)
                del blocks
            fit = lgb.Dataset(xa, y.iloc[a], init_score=offset[a] if mode == 'offset' else None)
            val = lgb.Dataset(xv, y.iloc[b], reference=fit, init_score=offset[b] if mode == 'offset' else None)
            emit({'mode': mode, 'fold': f, 'features': len(xa.columns), 'stage': 'training'})
            model = lgb.train(dict(params, seed=42 + f), fit, num_boost_round=3500,
                              valid_sets=[val], valid_names=['val'],
                              callbacks=[lgb.early_stopping(180, first_metric_only=True, verbose=False), lgb.log_evaluation(1000)])
            pv = expit(model.predict(xv, raw_score=True) + (offset[b] if mode == 'offset' else 0))
            pt = expit(model.predict(xt, raw_score=True) + (offset[n:] if mode == 'offset' else 0))
            auc = float(roc_auc_score(y.iloc[b], pv))
            if abs(auc - model.best_score['val']['auc']) > 1e-8:
                raise ValueError('Prediction AUC differs from native evaluation')
            score = {'fold': f, 'auc': auc, 'iterations': int(model.best_iteration),
                     'features': len(xa.columns), 'elapsed_seconds': round(time.time() - start, 2)}
            np.savez_compressed(record, val_idx=b, val_pred=pv, test_pred=pt)
            (out / f'fold-{f}.json').write_text(json.dumps(score, indent=2), encoding='utf-8')
            pd.DataFrame({'feature': xa.columns, 'gain': model.feature_importance('gain')}).to_csv(out / f'importance-{f}.csv', index=False)
            del model, fit, val, xa, xv, xt
            gc.collect()
        oof[b] = pv
        pred += pt / 5
        scores.append(score)
        emit(dict(mode=mode, **score))
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete OOF')
    sub = sample.copy()
    sub.satisfaction = pred
    validate_submission(sub, sample)
    path = out / 'submission.csv'
    sub.to_csv(path, index=False)
    pd.DataFrame({'id': tr.id, 'satisfaction': y, 'prediction': oof}).to_csv(out / 'oof.csv', index=False)
    pd.DataFrame({'id': ts.id, 'prediction': pred}).to_csv(out / 'test_predictions.csv', index=False)
    metrics = {'competition': COMPETITION, 'folds': 5, 'seed': 42, 'mode': mode,
               'model_description': 'Matched LightGBM ' + mode + ' 5fold',
               'selected_blend': {'auc': float(roc_auc_score(y, oof))}, 'fold_scores': scores,
               'input_sha256': hashes, 'external_sha256': reference['external_sha256'],
               'original_rows_used': data['original_rows'], 'original_matches_removed': data['original_removed'],
               'submission_sha256': sha(path), 'seconds': round(time.time() - start, 2),
               'validation_note': 'Identical five outer folds (seed42), 5 inner-fold target encoding. Offset uses only external original-data models. Validation used for early stopping, never feature fitting.'}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    emit(metrics)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['control', 'conditional', 'offset'])
    run(p.parse_args().mode)
