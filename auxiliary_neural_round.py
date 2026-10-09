"""Matched route-aware neural control versus feature reconstruction extension.

Public ideas are credited in SOURCES.md, independently implemented here.
Same neural parameters, folds and supervised encodings between the two modes.
Route profiles fit only to outer training covariates. Original-data matches are
removed. Auxiliary reconstructions use unlabeled train/test covariates only.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from auxiliary_round import AUXDIR, config_file, hashes
from context_round import ROOT, sha
from improve import CATS, CACHE
from nn_round import prepare_frames, fold_frames, predict_chunks, report
from train import read_data, encode_target, COMPETITION, validate_submission

TEACHERDIR = ROOT / 'experiments' / 'original-xgb-600rounds'


def teacher():
    import xgboost as xgb
    tr, ts, sample, raw, y = read_data(ROOT / 'data')
    full = pd.concat([tr[raw], ts[raw]], ignore_index=True)
    original = pd.read_csv(ROOT / 'data-original' / 'data.csv')
    overlaps = pd.util.hash_pandas_object(original[raw], index=False).isin(pd.util.hash_pandas_object(full, index=False))
    original = original.loc[~overlaps].reset_index(drop=True)
    params = dict(n_estimators=600, learning_rate=.05, max_depth=8, subsample=.8,
                  colsample_bytree=.5, tree_method='hist', device='cuda', random_state=0, n_jobs=4)
    config = dict(script_sha256=sha(Path(__file__)), input_sha256=hashes(),
                  original_sha256=sha(ROOT / 'data-original' / 'data.csv'),
                  overlaps_removed=int(overlaps.sum()), original_rows=len(original),
                  params=params, xgboost_version=xgb.__version__)
    config_file(TEACHERDIR, config)
    path = TEACHERDIR / 'probability.npy'
    if path.exists():
        metadata = json.loads((TEACHERDIR / 'metrics.json').read_text(encoding='utf-8'))
        if metadata['probability_sha256'] != sha(path):
            raise ValueError('Original teacher fingerprint changed')
        return
    both = pd.concat([full, original[raw]], ignore_index=True)
    for c in CATS:
        both[c] = pd.Categorical(both[c]).codes
    model = xgb.XGBClassifier(**params)
    report(dict(stage='original_xgb_training', rows=len(original), overlaps_removed=int(overlaps.sum())))
    model.fit(both.iloc[len(full):], encode_target(original.satisfaction))
    if json.loads(model.get_booster().save_config())['learner']['generic_param']['device'] != 'cuda:0':
        raise RuntimeError('Teacher GPU mismatch')
    probability = model.predict_proba(both.iloc[:len(full)])[:, 1].astype('float32')
    np.save(path, probability)
    metadata = dict(configuration=config, probability_sha256=sha(path),
                    descriptive_auc_against_competition_labels=float(roc_auc_score(y, probability[:len(tr)])))
    (TEACHERDIR / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report(metadata)


def route_keys(full):
    combinations = [('Flight Distance',), ('Age',), ('Flight Distance','Age'),
                    ('Flight Distance','Class'), ('Flight Distance','Type of Travel'),
                    ('Age','Class','Type of Travel','Customer Type')]
    result = {}
    for i, cols in enumerate(combinations):
        parts = full[cols[0]].astype(str)
        for c in cols[1:]:
            parts = parts + '|' + full[c].astype(str)
        result['route_key_' + str(i)] = pd.factorize(parts, sort=False)[0].astype('int32')
    return pd.DataFrame(result)


def frames(full, raw, keys, rkeys, a, b, n, y):
    result = fold_frames(full, keys, a, b, n, y)
    encoder = TargetEncoder(target_type='binary', smooth=20., cv=StratifiedKFold(5, shuffle=True, random_state=42))
    blocks = [encoder.fit_transform(rkeys.iloc[a], y.iloc[a]),
              encoder.transform(rkeys.iloc[b]), encoder.transform(rkeys.iloc[n:])]
    fit = full.iloc[a]
    measures = [c for c in raw if c not in CATS and c != 'Flight Distance']
    stats = fit.groupby('Flight Distance', observed=True)[measures].mean()
    counts = fit['Flight Distance'].value_counts()
    proportions = {}
    for c in CATS:
        for value in full[c].cat.categories:
            proportions[c, str(value)] = (fit[c] == value).groupby(fit['Flight Distance']).mean()
    for frame, block in zip(result, blocks):
        for i, c in enumerate(rkeys):
            frame['route_target_' + c] = block[:, i].astype('float32')
        additions = {'route_profile_count': frame['Flight Distance'].map(counts).fillna(0).astype('float32')}
        for c in measures:
            additions['route_profile_mean_' + c] = frame['Flight Distance'].map(stats[c]).fillna(fit[c].mean()).astype('float32')
        for (c, value), series in proportions.items():
            additions['route_fraction_' + c + '_' + value] = frame['Flight Distance'].map(series).fillna((fit[c].astype(str) == value).mean()).astype('float32')
        frame[list(additions)] = pd.DataFrame(additions, index=frame.index)
    return result


def run(mode):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    start = time.time()
    tr, ts, sample, raw, y = read_data(ROOT / 'data')
    reference = ROOT / 'experiments' / 'realmlp-5fold-8ens'
    old = json.loads((reference / 'configuration.json').read_text(encoding='utf-8'))
    aux_meta = json.loads((AUXDIR / 'metrics.json').read_text(encoding='utf-8'))
    teacher_meta = json.loads((TEACHERDIR / 'metrics.json').read_text(encoding='utf-8'))
    if any(d != hashes() for d in (old['input_sha256'], aux_meta['configuration']['input_sha256'], teacher_meta['configuration']['input_sha256'])):
        raise ValueError('Input fingerprint mismatch')
    if sha(AUXDIR / 'features.pkl') != aux_meta['features_sha256'] or sha(TEACHERDIR / 'probability.npy') != teacher_meta['probability_sha256']:
        raise ValueError('Feature fingerprint mismatch')
    full, keys = prepare_frames(tr, ts, raw)
    rkeys = route_keys(full)
    base_data = joblib.load(CACHE)
    for c in ['original_lgb_probability', 'original_cat_probability']:
        full[c] = logit(base_data['base'][c].clip(1e-6, 1-1e-6)).astype('float32')
    del base_data
    full['original_xgb_logit'] = logit(np.clip(np.load(TEACHERDIR / 'probability.npy'), 1e-6, 1-1e-6)).astype('float32')
    for c in ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
        full['route_count_' + c] = full[c].map(full[c].value_counts()).astype('float32')
    if mode == 'all':
        aux = pd.read_pickle(AUXDIR / 'features.pkl')
        if len(aux) != len(full):
            raise ValueError('Auxiliary row alignment mismatch')
        full[aux.columns] = aux
        del aux
    cats = list(full.select_dtypes(include='category').columns)
    params = dict(old['params'], verbosity=0)
    out = ROOT / 'experiments' / f'realmlp-route-auxiliary-{mode}-5fold-8ens'
    config_file(out, dict(script_sha256=sha(Path(__file__)), raw_feature_source_sha256=sha(ROOT / 'nn_round.py'),
                params=params, input_sha256=hashes(), base_cache_sha256=sha(CACHE),
                auxiliary_features_sha256=aux_meta['features_sha256'], original_teacher_sha256=teacher_meta['probability_sha256'],
                reference_configuration_sha256=sha(reference / 'configuration.json'),
                mode=mode, split_seed=42, model_seed=42, folds=5))
    oof = np.full(len(tr), np.nan); prediction = np.zeros(len(ts)); scores = []
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y), 1):
        with np.load(reference / f'fold-{f}.npz') as saved:
            if not np.array_equal(saved['val_idx'], b):
                raise ValueError('Reference split mismatch')
        path = out / f'fold-{f}.npz'
        if path.exists():
            with np.load(path) as saved:
                if not np.array_equal(saved['val_idx'], b):
                    raise ValueError('Cached split mismatch')
                pv, pt = saved['val_pred'], saved['test_pred']
            score = json.loads((out / f'fold-{f}.json').read_text(encoding='utf-8'))
        else:
            prepared = frames(full, raw, keys, rkeys, a, b, len(tr), y)
            if f == 1:
                changed = y.copy(); changed.iloc[b] = 1-changed.iloc[b]
                check = frames(full, raw, keys, rkeys, a, b, len(tr), changed)
                for actual, perturbed in zip(prepared, check):
                    pd.testing.assert_frame_equal(actual, perturbed, check_exact=True)
                del check
                report(dict(mode=mode, validation_label_perturbation='PASS'))
            xa, xv, xt = prepared
            model = RealMLP_TD_Classifier(**dict(params, random_state=42+f))
            report(dict(stage='route_neural_training', mode=mode, fold=f, features=len(xa.columns)))
            model.fit(xa, y.iloc[a], X_val=xv, y_val=y.iloc[b], cat_col_names=cats)
            trainer = model.alg_interface_.min_trainer_kwargs
            if str(model.alg_interface_.device) != 'cuda:0' or trainer['accelerator'] != 'gpu' or trainer['devices'] != [0]:
                raise RuntimeError('GPU trainer mismatch')
            pv, pt = predict_chunks(model, xv), predict_chunks(model, xt)
            score = dict(fold=f, auc=float(roc_auc_score(y.iloc[b], pv)), features=len(xa.columns),
                         elapsed_seconds=round(time.time()-start, 2))
            np.savez_compressed(path, val_idx=b, val_pred=pv, test_pred=pt)
            (out / f'fold-{f}.json').write_text(json.dumps(score, indent=2), encoding='utf-8')
            del prepared, xa, xv, xt, model
            gc.collect(); torch.cuda.empty_cache()
        oof[b] = pv; prediction += pt / 5; scores.append(score)
        report(dict(mode=mode, **score))
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete neural OOF')
    sub = sample.copy(); sub.satisfaction = prediction
    validate_submission(sub, sample); sub.to_csv(out / 'submission.csv', index=False)
    pd.DataFrame(dict(id=tr.id, satisfaction=y, prediction=oof)).to_csv(out / 'oof.csv', index=False)
    metrics = dict(competition=COMPETITION, mode=mode, folds=5, seed=42,
                   input_sha256=hashes(), fold_scores=scores, selected_blend=dict(auc=float(roc_auc_score(y, oof))),
                   submission_sha256=sha(out / 'submission.csv'), seconds=round(time.time()-start, 2),
                   external_sha256=teacher_meta['configuration']['original_sha256'],
                   validation_note='Matched control/all route neural trials. Original data removes competition exact matches. Route profiles fit outer training rows; target encodings cross-fit inner folds. Auxiliary models only use unlabeled covariates, transductive validation.')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    report(metrics)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['teacher','control','all'])
    mode = p.parse_args().mode
    teacher() if mode == 'teacher' else run(mode)
