"""Prospective repeat of native-coverage ranking fusion.

This is adaptive meta validation on historical OOF, not nested validation.
The same input library and fitting procedure produce crossfit and test scores;
final coefficients are refit rather than transferred from a different baseline.
Round22's original result and acceptance rule are never modified.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata as md
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import round22_preserved as native
from round18_joint import ROOT, sha, save, load
from train import COMPETITION, validate_submission

OUT = ROOT / 'experiments/round28-native-fusion'
ANALYSIS = ROOT / 'analyses/round28-native-fusion'
FINAL = ROOT / 'outputs-round28'


def msg(**value):
    native.msg(**value)


def freeze():
    if (OUT / 'protocol.json').exists():
        raise ValueError('Protocol already frozen')
    prior = load(native.OUT / 'protocol.json')
    # Sources retain their original 3/5/10 fold coverage. No replacement with
    # lower-coverage approximations of the strongest tenfold member occurs.
    p = dict(round=28, created_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__),
        dependencies={s: sha(ROOT / s) for s in ['round22_preserved.py',
            'round18_joint.py', 'train.py',
            'experiments/round22-preserved/protocol.json']},
        protected_sha256=prior['protected_sha256'],
        input_sha256=prior['input_sha256'], members=prior['members'],
        source_sha256=prior['native_source_sha256'],
        optimization=dict(prior['optimization'], pair_seed=2026101028,
                          pair_count=2000000, l2_to_best=0.0001),
        meta_cv=dict(folds=5, seeds=[20261010, 20261011, 20261012]),
        gates=dict(mean_delta=0.00001, each_repeat_positive=True,
                   minimum_positive_blocks=9, worst_block_floor=-0.00004),
        max_official_submissions=1,
        production_rule='Refit identical simplex algorithm on the identical '
            'native OOF library; apply to the corresponding audited test '
            'predictions. Keep best prediction at least 50 percent.',
        limitations='All labels and input models have been studied before. '
            'New meta seeds are sensitivity checks, not new unseen data. '
            'Base fits overlap across meta folds; this is not nested CV. '
            'Original round22 outcome remains rejected under its old rule.',
        versions={n: md.version(n) for n in ['numpy', 'scipy', 'scikit-learn']})
    save(OUT / 'protocol.json', p)
    msg(stage='frozen', round=28, protocol_sha256=sha(OUT / 'protocol.json'))


def check():
    p = load(OUT / 'protocol.json')
    if sha(__file__) != p['script_sha256']:
        raise ValueError('Frozen source changed')
    for name, expected in {**p['dependencies'], **p['protected_sha256']}.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Source or protected state changed: ' + name)
    for name, expected in p['input_sha256'].items():
        if sha(ROOT / 'data' / name) != expected:
            raise ValueError('Input changed')
    return p


def run():
    p = check()
    if (ANALYSIS / 'crossfit.json').exists():
        raise ValueError('Already evaluated')
    old = load(native.OUT / 'protocol.json')
    native.OUT = OUT  # Keep the fresh audit outside historical round22 outputs.
    tr, sample, y, P, T = native.library(old)
    if list(p['members']) != list(old['members']):
        raise ValueError('Input order changed')
    X = logit(np.clip(P, 1e-6, 1 - 1e-6))
    base = P[:, 0]
    baseline = float(roc_auc_score(y, base))
    predictions, repeats = [], []
    started = time.monotonic()
    with threadpool_limits(limits=4):
        for repeat, seed in enumerate(p['meta_cv']['seeds']):
            pred = np.full(len(y), np.nan)
            deltas, weights = [], []
            splits = StratifiedKFold(5, shuffle=True, random_state=seed)
            for fold, (a, b) in enumerate(splits.split(X, y), 1):
                pair_seed = p['optimization']['pair_seed'] + repeat * 100 + fold
                w = native.fit(X[a], y[a], p['optimization'], pair_seed)
                pred[b] = expit(X[b] @ w)
                delta = float(roc_auc_score(y[b], pred[b]) -
                              roc_auc_score(y[b], base[b]))
                deltas.append(delta)
                weights.append(w.tolist())
                msg(stage='meta_fold', repeat=repeat + 1, fold=fold,
                    delta=delta, best_weight=float(w[0]),
                    seconds=time.monotonic() - started)
            auc = float(roc_auc_score(y, pred))
            repeats.append(dict(seed=seed, auc=auc, delta=auc - baseline,
                                block_deltas=deltas, weights=weights))
            predictions.append(pred)
    all_deltas = [d for r in repeats for d in r['block_deltas']]
    mean_delta = float(np.mean([r['delta'] for r in repeats]))
    positive = sum(d > 0 for d in all_deltas)
    worst = min(all_deltas)
    oof = np.mean(predictions, axis=0)
    averaged_delta = float(roc_auc_score(y, oof) - baseline)
    gate = p['gates']
    eligible = (mean_delta >= gate['mean_delta'] and
        averaged_delta >= gate['mean_delta'] and
        min(r['delta'] for r in repeats) > 0 and
        positive >= gate['minimum_positive_blocks'] and
        worst >= gate['worst_block_floor'])
    result = dict(competition=COMPETITION, input_sha256=p['input_sha256'],
        folds=5, native_member_folds='preserved mixed native folds',
        baseline_auc=baseline, mean_delta=mean_delta,
        averaged_crossfit_delta=averaged_delta, positive_blocks=positive,
        worst_block_delta=worst, repeats=repeats, eligible=bool(eligible),
        gates=gate, protocol_sha256=sha(OUT / 'protocol.json'),
        validation_note=p['limitations'],
        seconds=time.monotonic() - started)
    np.savez_compressed(OUT / 'crossfit-predictions.npz',
                        predictions=np.asarray(predictions))
    if eligible:
        with threadpool_limits(limits=4):
            weights = native.fit(X, y, p['optimization'],
                                 p['optimization']['pair_seed'] + 999)
        test = expit(logit(np.clip(T, 1e-6, 1 - 1e-6)) @ weights)
        FINAL.mkdir(exist_ok=True)
        sub = sample.copy()
        sub.satisfaction = test
        validate_submission(sub, sample)
        sub.to_csv(FINAL / 'submission.csv', index=False)
        pd.DataFrame(dict(id=tr.id, satisfaction=y, prediction=oof)).to_csv(
            FINAL / 'oof.csv', index=False)
        refit_oof = expit(X @ weights)
        result.update(final_weights=weights.tolist(),
            full_refit_oof_auc=float(roc_auc_score(y, refit_oof)),
            full_refit_oof_auc_is_not_crossfit=True,
            selected_blend=dict(auc=float(roc_auc_score(y, oof)),
                                candidate='native-preserved ranking simplex'),
            source_submission_sha256={n: sha(ROOT / d / 'submission.csv')
                                       for n, d in p['members'].items()},
            external_sha256=load(ROOT / 'outputs-round13/metrics.json')['external_sha256'],
            submission_sha256=sha(FINAL / 'submission.csv'),
            model_description='Native-coverage preserved ranking fusion R28')
        save(FINAL / 'metrics.json', result)
    save(ANALYSIS / 'crossfit.json', result)
    msg(stage='complete', eligible=bool(eligible), mean_delta=mean_delta,
        averaged_delta=averaged_delta, positive_blocks=positive,
        worst_block_delta=worst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run'])
    args = parser.parse_args()
    freeze() if args.stage == 'freeze' else run()
