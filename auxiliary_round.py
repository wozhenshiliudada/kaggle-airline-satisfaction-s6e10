"""Independently implement feature reconstruction, then matched RealMLP trials.

Idea source: goodpjw2008 S6E10 Auxiliary-Task Features / More Aux Carriers.
No third-party prediction files or notebook execution. Auxiliary models see
only raw feature values (including unlabeled test covariates), never satisfaction.
Every auxiliary row is predicted by a model excluding that row from fitting.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from context_round import ROOT, SERVICES, sha
from nn_round import prepare_frames, fold_frames, predict_chunks, report
from train import read_data, COMPETITION, validate_submission

CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
AUXDIR = ROOT / 'experiments' / 'auxiliary-reconstruction-300rounds'
SOURCE = 'https://www.kaggle.com/code/goodpjw2008/s6e10-more-aux-carriers-lb-0-96152'


def config_file(folder, config):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / 'configuration.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8')) != config:
        raise ValueError('Changed experiment configuration; use a new directory')
    p.write_text(json.dumps(config, indent=2), encoding='utf-8')


def hashes():
    return {c: sha(ROOT / 'data' / c) for c in ('train.csv', 'test.csv', 'sample_submission.csv')}


def build():
    import xgboost as xgb
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    start = time.time()
    # Build deliberately never loads satisfaction into its working frame.
    tr = pd.read_csv(ROOT / 'data' / 'train.csv', usecols=lambda c: c not in ('id', 'satisfaction'))
    ts = pd.read_csv(ROOT / 'data' / 'test.csv', usecols=lambda c: c != 'id')
    full = pd.concat([tr, ts], ignore_index=True)
    del tr, ts
    for c in CATS:
        full[c] = pd.Categorical(full[c]).codes.astype('float32')
    full = full.astype('float32')
    full['Arrival Delay in Minutes'] = full['Arrival Delay in Minutes'].fillna(-1)
    params = dict(n_estimators=300, learning_rate=.1, max_depth=6,
                  subsample=.8, colsample_bytree=.7, tree_method='hist',
                  device='cuda', random_state=0, n_jobs=4)
    config = dict(script_sha256=sha(Path(__file__)), input_sha256=hashes(),
                  xgboost_version=xgb.__version__, params=params, auxiliary_folds=5,
                  auxiliary_seed=0, transductive=True, label_used=False,
                  idea_source=SOURCE)
    config_file(AUXDIR, config)
    splits = list(KFold(5, shuffle=True, random_state=0).split(full))
    features = pd.DataFrame(index=full.index)
    reports = []
    for c in SERVICES + ['Class', 'Type of Travel', 'Customer Type']:
        values = np.sort(full[c].unique())
        target = np.searchsorted(values, full[c].to_numpy())
        k = len(values)
        p = np.zeros((len(full), k), dtype='float32')
        name = c.replace('/', '_').replace(' ', '_')
        others = [v for v in full.columns if v != c]
        for f, (a, b) in enumerate(splits, 1):
            path = AUXDIR / f'{name}-fold-{f}.npz'
            if path.exists():
                with np.load(path) as saved:
                    if not np.array_equal(saved['val_idx'], b):
                        raise ValueError('Auxiliary row exclusion mismatch')
                    p[b] = saved['probability']
            else:
                report(dict(stage='auxiliary_training', column=c, fold=f,
                            classes=k, elapsed_seconds=round(time.time()-start, 1)))
                model = xgb.XGBClassifier(**params, objective='multi:softprob' if k > 2 else 'binary:logistic')
                model.fit(full.iloc[a][others], target[a])
                if json.loads(model.get_booster().save_config())['learner']['generic_param']['device'] != 'cuda:0':
                    raise RuntimeError('XGBoost did not use GPU')
                predictions = model.predict_proba(full.iloc[b][others])
                if predictions.shape != (len(b), k) or not np.isfinite(predictions).all():
                    raise ValueError('Invalid auxiliary prediction')
                p[b] = predictions
                np.savez_compressed(path, val_idx=b, probability=predictions)
                del model, predictions
                gc.collect()
            report(dict(stage='auxiliary_fold_complete', column=c, fold=f,
                        elapsed_seconds=round(time.time()-start, 1)))
        if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise ValueError('Incomplete auxiliary predictions')
        features['aux_probability_' + c] = p[np.arange(len(full)), target]
        if c in SERVICES:
            features['aux_expected_' + c] = p @ values
        record = dict(column=c, classes=k, accuracy=float((p.argmax(1) == target).mean()),
                      mean_observed_probability=float(p[np.arange(len(full)), target].mean()))
        reports.append(record)
        report(record)
    pc = [c for c in features if c.startswith('aux_probability_')]
    features['aux_log_likelihood'] = np.log(features[pc].clip(lower=1e-6)).sum(axis=1)
    features.to_pickle(AUXDIR / 'features.pkl')
    result = dict(configuration=config, shape=list(features.shape),
                  features_sha256=sha(AUXDIR / 'features.pkl'), column_scores=reports,
                  elapsed_seconds=round(time.time()-start, 2),
                  validation_note='Raw 21 feature columns only; satisfaction never loaded. Five-fold row exclusion over train+test, fixed 300 rounds without target early stopping.')
    (AUXDIR / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report(result)


def run(mode):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    start = time.time()
    tr, ts, sample, raw, y = read_data(ROOT / 'data')
    aux_meta = json.loads((AUXDIR / 'metrics.json').read_text(encoding='utf-8'))
    if aux_meta['configuration']['input_sha256'] != hashes() or aux_meta['features_sha256'] != sha(AUXDIR / 'features.pkl'):
        raise ValueError('Auxiliary inputs changed')
    aux = pd.read_pickle(AUXDIR / 'features.pkl')
    if mode == 'expected':
        aux = aux[[c for c in aux if c.startswith('aux_expected_')]]
    if len(aux) != len(tr) + len(ts):
        raise ValueError('Auxiliary row alignment mismatch')
    ref = ROOT / 'experiments' / 'realmlp-5fold-8ens'
    old = json.loads((ref / 'configuration.json').read_text(encoding='utf-8'))
    if old['input_sha256'] != hashes():
        raise ValueError('Reference inputs changed')
    params = dict(old['params'], verbosity=0)
    out = ROOT / 'experiments' / f'realmlp-auxiliary-{mode}-5fold-8ens'
    config_file(out, dict(script_sha256=sha(Path(__file__)), params=params,
                input_sha256=hashes(), auxiliary_features_sha256=aux_meta['features_sha256'],
                raw_feature_source_sha256=sha(ROOT / 'nn_round.py'),
                reference_configuration_sha256=sha(ref / 'configuration.json'),
                mode=mode, split_seed=42, model_seed=42, folds=5))
    full, keys = prepare_frames(tr, ts, raw)
    full[aux.columns] = aux
    cats = list(full.select_dtypes(include='category').columns)
    oof = np.full(len(tr), np.nan)
    prediction = np.zeros(len(ts))
    scores = []
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y), 1):
        with np.load(ref / f'fold-{f}.npz') as saved:
            if not np.array_equal(saved['val_idx'], b):
                raise ValueError('Control split mismatch')
        path = out / f'fold-{f}.npz'
        if path.exists():
            with np.load(path) as saved:
                if not np.array_equal(saved['val_idx'], b):
                    raise ValueError('Cached split mismatch')
                pv, pt = saved['val_pred'], saved['test_pred']
            score = json.loads((out / f'fold-{f}.json').read_text(encoding='utf-8'))
        else:
            frames = fold_frames(full, keys, a, b, len(tr), y)
            if f == 1:
                changed = y.copy(); changed.iloc[b] = 1 - changed.iloc[b]
                check = fold_frames(full, keys, a, b, len(tr), changed)
                for original, altered in zip(frames, check):
                    pd.testing.assert_frame_equal(original, altered, check_exact=True)
                del check
                report(dict(mode=mode, validation_label_perturbation='PASS'))
            xa, xv, xt = frames
            model = RealMLP_TD_Classifier(**dict(params, random_state=42+f))
            report(dict(stage='neural_training', mode=mode, fold=f, columns=len(xa.columns)))
            model.fit(xa, y.iloc[a], X_val=xv, y_val=y.iloc[b], cat_col_names=cats)
            trainer = model.alg_interface_.min_trainer_kwargs
            if str(model.alg_interface_.device) != 'cuda:0' or trainer['accelerator'] != 'gpu' or trainer['devices'] != [0]:
                raise RuntimeError('GPU trainer mismatch')
            pv, pt = predict_chunks(model, xv), predict_chunks(model, xt)
            score = dict(fold=f, auc=float(roc_auc_score(y.iloc[b], pv)),
                         features=len(xa.columns), elapsed_seconds=round(time.time()-start, 2))
            np.savez_compressed(path, val_idx=b, val_pred=pv, test_pred=pt)
            (out / f'fold-{f}.json').write_text(json.dumps(score, indent=2), encoding='utf-8')
            del frames, xa, xv, xt, model
            gc.collect(); torch.cuda.empty_cache()
        oof[b] = pv; prediction += pt / 5; scores.append(score)
        report(dict(mode=mode, **score))
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete neural OOF')
    sub = sample.copy(); sub.satisfaction = prediction
    validate_submission(sub, sample)
    sub.to_csv(out / 'submission.csv', index=False)
    pd.DataFrame(dict(id=tr.id, satisfaction=y, prediction=oof)).to_csv(out / 'oof.csv', index=False)
    metrics = dict(competition=COMPETITION, mode=mode, folds=5, seed=42,
                   input_sha256=hashes(), fold_scores=scores,
                   selected_blend=dict(auc=float(roc_auc_score(y, oof))),
                   submission_sha256=sha(out / 'submission.csv'),
                   seconds=round(time.time()-start, 2), idea_source=SOURCE,
                   validation_note='Matched five outer folds, same seed/parameters as plain RealMLP. Auxiliary features use unlabeled train/test covariates; satisfaction excluded. This is transductive validation.')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    report(metrics)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['build', 'expected', 'all'])
    mode = p.parse_args().mode
    build() if mode == 'build' else run(mode)
