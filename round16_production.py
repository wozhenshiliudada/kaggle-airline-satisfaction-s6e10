"""Transfer one locked development carrier to the protected full-training blend.

Only a carrier that already passed the declared development and active
holdout gates is trained here. Its two logit coefficients and intercept are
copied unchanged. Five native production folds use all official training
rows. The resulting historical OOF comparison is a compatibility gate,
not an independent holdout estimate or a new weight search. This module
never uploads anything.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from round16_carriers import CarrierFactory, MODES
from round16_evaluate import (read_inputs, paired, TRANSFER_NOTE,
                              check_probability, path, rel, sha)
from round14_common import sources, BASE_SHA
from train import COMPETITION, read_data, validate_submission

ROOT = Path(__file__).resolve().parent
CLIP = [1e-6, 1 - 1e-6]
VALIDATION_NOTE = (
    'One development-selected carrier, coefficients and intercept transferred unchanged. '
    'The freshly retrained structure passed its designated active holdout; H2 was '
    'prospectively reserved for the round17 fallback, whose eight-epoch budget was '
    'declared before H1 was scored. Production trains on all official training labels '
    'with native fivefold seed42, including the rows formerly excluded as H1/H2. '
    'Each production native validation set may select the best RealMLP epoch even '
    'though use_early_stopping=False. Historical mixed 3/5/10-fold OOF has participated '
    'in earlier adaptive research. Global historical delta >=1e-5 and all five '
    'historical evaluation blocks positive are a compatibility gate only; there is '
    'no carrier reselection or coefficient fitting here. Auxiliary reconstruction '
    'and categorical vocabularies are unlabeled transductive. ' + TRANSFER_NOTE)


def load(value):
    return json.loads(path(value).read_text(encoding='utf-8'))


def save(value, obj):
    output = path(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def frozen(value, obj):
    value = path(value)
    value.parent.mkdir(parents=True, exist_ok=True)
    if value.exists():
        if load(value) != obj:
            raise ValueError('Production configuration changed; retain the old experiment: ' + rel(value))
    else:
        save(value, obj)


def report(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def fixed_stack(baseline, carrier, stack):
    """The exact fixed transformation used by the development LogisticRegression."""
    baseline = np.asarray(baseline, dtype=np.float64)
    carrier = np.asarray(carrier, dtype=np.float64)
    if baseline.shape != carrier.shape or baseline.ndim != 1:
        raise ValueError('Fixed logit stack inputs must be aligned one-dimensional arrays.')
    coefficients = np.asarray(stack['coefficients'], dtype=np.float64)
    intercept = float(stack['intercept'])
    if coefficients.shape != (2,) or not np.isfinite(coefficients).all() or not np.isfinite(intercept):
        raise ValueError('Invalid locked two-member logistic coefficients.')
    matrix = np.column_stack([logit(np.clip(baseline, *CLIP)),
                              logit(np.clip(carrier, *CLIP))])
    return expit(matrix @ coefficients + intercept)


def locked_context(round_number):
    manifest = ROOT / f'experiments/round{round_number}-locked/protocol.json'
    protocol, fingerprints, tr, sample, dev, hold, fid, yd, pred, active = read_inputs(manifest)
    if protocol['round'] != round_number or protocol['epochs'] != (4 if round_number == 16 else 8):
        raise ValueError('Round/epoch budget does not match its prospectively declared protocol.')
    if protocol['active_holdout'] != ('hold1' if round_number == 16 else 'hold2'):
        raise ValueError('Wrong predeclared active holdout.')
    if protocol['production_gate'] != dict(global_delta=1e-5,
                                          all_five_historical_evaluation_blocks_positive=True):
        raise ValueError('Historical compatibility gate changed.')
    analysis = path(protocol['analysis_dir'])
    lock_path = analysis / 'development-selection.json'
    final_path = analysis / 'final-comparison.json'
    lock, final = load(lock_path), load(final_path)
    if lock['source_fingerprints'] != fingerprints or final['source_fingerprints'] != fingerprints:
        raise ValueError('Frozen development or holdout sources changed.')
    if lock['protocol_path'] != rel(manifest) or final['development_selection_sha256'] != sha(lock_path):
        raise ValueError('Development selection lock changed.')
    if final['script_sha256'] != sha(ROOT / 'round16_evaluate.py'):
        raise ValueError('Held-out evaluation implementation changed.')
    if final['input_sha256'] != protocol['input_sha256'] or final['external_sha256'] != protocol['external_sha256']:
        raise ValueError('Locked comparison data fingerprints changed.')
    selected = lock['selected_candidate']
    if (selected is None or not selected['eligible'] or selected['delta'] < 1e-5
            or min(selected['block_deltas']) <= 0):
        raise ValueError('There is no development-eligible locked carrier.')
    mode = selected['candidate']
    if mode not in MODES or final['selected_development_candidate'] != selected:
        raise ValueError('Holdout scored a different development-selected carrier.')
    if (not final['eligible_for_production_transfer'] or not final['heldout']['passed']
            or final['heldout']['min_delta'] != 1e-5 or final['heldout']['delta'] < 1e-5
            or final['selected_blend']['candidate'] != mode):
        raise ValueError('The locked carrier did not pass its declared active holdout gate.')
    if final['deployment_stack'] != lock['deployment_stack']:
        raise ValueError('Deployment coefficients changed after development selection.')
    stack = lock['deployment_stack'][mode]
    if stack['members'] != ['baseline_current', mode]:
        raise ValueError('Locked stack does not contain the preserved baseline plus exactly one carrier.')
    predicted = path(lock['locked_prediction_path'])
    if (sha(predicted) != lock['locked_prediction_sha256']
            or final['locked_prediction_sha256'] != lock['locked_prediction_sha256']):
        raise ValueError('Frozen stacked predictions changed.')
    errors = {}
    # Check coefficient semantics against the actual frozen predictions without
    # opening/scoring any held-out labels or changing the selected coefficients.
    with np.load(predicted, allow_pickle=False) as z:
        for role in ('hold', 'test'):
            rebuilt = fixed_stack(pred['baseline_current'][role], pred[mode][role], stack)
            frozen_prediction = z[mode + '_' + role]
            errors[role] = float(np.max(np.abs(rebuilt - frozen_prediction)))
            if errors[role] > 1e-12:
                raise ValueError('Frozen logit stack formula does not match its locked ' + role + ' predictions.')
    for value in (lock_path, final_path, predicted):
        fingerprints[rel(value)] = sha(value)
    locked = dict(
        protocol_path=rel(manifest), development_selection_path=rel(lock_path),
        development_selection_sha256=sha(lock_path), final_comparison_path=rel(final_path),
        final_comparison_sha256=sha(final_path), locked_prediction_path=rel(predicted),
        locked_prediction_sha256=sha(predicted), active_holdout=protocol['active_holdout'],
        development_delta=selected['delta'], holdout_delta=final['heldout']['delta'],
        holdout_passed=True, eligible_for_production_transfer=True,
        coefficient_reconstruction_max_difference=errors,
        holdout_approximation=final['heldout']['paired_fixed_prediction_approximation'],
        production_transfer_limitation=TRANSFER_NOTE)
    del tr, sample, dev, hold, fid, yd, pred, active
    gc.collect()
    return protocol, mode, stack, locked, fingerprints


def native_cache(folder, fold, a, b, tr, sample):
    checkpoint = folder / f'fold-{fold}.npz'
    detail = load(folder / f'fold-{fold}.json')
    if detail['checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('Native cached fold checkpoint changed.')
    with np.load(checkpoint, allow_pickle=False) as z:
        expected = dict(train_idx=a, val_idx=b, val_id=tr.id.iloc[b].to_numpy(),
                        test_id=sample.id.to_numpy(), hold_idx=np.empty(0, dtype=np.int64))
        for key, value in expected.items():
            if not np.array_equal(z[key], value):
                raise ValueError(f'Native fold {fold} index/ID changed: {key}')
        pv = check_probability(z['val_pred'], len(b), 'native validation').copy()
        pt = check_probability(z['test_pred'], len(sample), 'native test').copy()
        if z['hold_pred'].shape != (0,):
            raise ValueError('Production fold must have no independent holdout predictions.')
    return pv, pt, detail


def run(args):
    started = time.time()
    protocol, mode, stack, locked, fingerprints = locked_context(args.round)
    analysis = ROOT / f'analyses/round{args.round}-production-comparison.json'
    if analysis.exists():
        raise ValueError('Production compatibility comparison was already recorded.')
    tr, ts, sample, raw, ys = read_data(ROOT / 'data')
    y = ys.to_numpy()
    preds, old_metrics, old_fingerprints, reconstruction = sources(tr, sample, ys)
    baseline, baseline_test = preds['baseline']
    fingerprints.update(old_fingerprints)
    if old_metrics['baseline']['submission_sha256'] != BASE_SHA:
        raise ValueError('Protected round13 baseline CSV changed.')
    if protocol['protected_file_sha256']['outputs-round13/submission.csv'] != BASE_SHA:
        raise ValueError('The development experiment was not based on the protected round13 baseline.')
    fingerprints['outputs-round13/metrics.json'] = sha(ROOT / 'outputs-round13/metrics.json')
    split = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, ys))
    # Retain the original fivefold partition and prove row identity against
    # its historical native predictions and aligned source OOF table.
    old_folder = ROOT / 'experiments/round9-lgb-auxiliary-5fold'
    for f, (_, b) in enumerate(split, 1):
        checkpoint = old_folder / f'fold-{f}.npz'
        with np.load(checkpoint, allow_pickle=False) as z:
            if not np.array_equal(z['val_idx'], b):
                raise ValueError('Production native validation partition differs from historical fivefold blocks.')
        fingerprints[rel(checkpoint)] = sha(checkpoint)
    factory = CarrierFactory(tr, ts, raw, ys, epochs=protocol['epochs'])
    folder = ROOT / f'experiments/round{args.round}-carrier-{mode}-5fold'
    code = dict(protocol['source_code_sha256'])
    code.update({name: sha(ROOT / name) for name in ('round16_production.py', 'round14_common.py')})
    for name, digest in code.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen production dependency changed: ' + name)
    config = dict(
        competition=COMPETITION, round=args.round, folds=5, seed=42, mode=mode,
        epochs=protocol['epochs'], input_sha256=protocol['input_sha256'],
        external_sha256=protocol['external_sha256'], script_sha256=sha(Path(__file__)),
        source_code_sha256=code, params=factory.params(42),
        source_audit=factory.audit, locked_validation=locked, frozen_stack=stack,
        production_coefficient_lock=dict(candidate=mode, members=stack['members'],
            coefficients=stack['coefficients'], intercept=stack['intercept'], clip=CLIP),
        protected_baseline_oof_sha256=sha(ROOT / 'outputs-round13/oof.csv'),
        protected_baseline_submission_sha256=BASE_SHA,
        native_validation_best_epoch_selection=True,
        holdout_excluded_from_supervised_training_and_early_stopping=False,
        full_training_coverage=True, validation_note=VALIDATION_NOTE)
    frozen(folder / 'configuration.json', config)
    frozen(folder / 'source-audit.json', factory.audit)
    report(dict(stage='production_protocol_ready', round=args.round, mode=mode,
                epochs=protocol['epochs'], fixed_stack=stack, historical_gate_only=True,
                locked_active_holdout=locked['active_holdout']))
    oof = np.full(len(tr), np.nan)
    test = np.zeros(len(sample), dtype=np.float64)
    coverage = np.zeros(len(tr), dtype=np.int8)
    details = []
    for f, (a, b) in enumerate(split, 1):
        checkpoint = folder / f'fold-{f}.npz'
        if checkpoint.exists():
            pv, pt, info = native_cache(folder, f, a, b, tr, sample)
            if (info['source_audit'] != factory.audit or info['params'] != factory.params(42 + f)
                    or info['actual_device'] != 'cuda:0' or not info['actual_binary_classifier']):
                raise ValueError('Cached carrier configuration or actual training proof changed.')
            report(dict(stage='production_native_fold_restored', fold=f, mode=mode))
        else:
            pv, ph, pt, info = factory.fit_predict(mode, a, b, [], 42 + f)
            if ph.shape != (0,) or info['holdout_rows'] != 0:
                raise ValueError('Production folds must use empty independent holdout.')
            np.savez_compressed(checkpoint, train_idx=a, val_idx=b, val_id=tr.id.iloc[b].to_numpy(),
                val_pred=pv, hold_idx=np.empty(0, dtype=np.int64), hold_pred=ph,
                test_id=sample.id.to_numpy(), test_pred=pt)
            info.update(fold=f, auc=float(roc_auc_score(y[b], pv)),
                checkpoint_sha256=sha(checkpoint),
                validation_role='Conventional production native OOF validation may select best epoch; not an independent locked holdout')
            save(folder / f'fold-{f}.json', info)
        oof[b] = pv
        coverage[b] += 1
        test += pt / 5
        details.append(dict(fold=f, auc=float(roc_auc_score(y[b], pv)),
            features=info['feature_count'], best_mean_val_epochs=info['best_mean_val_epochs'],
            fit_seconds=info['fit_seconds']))
        if args.first_fold:
            if f != 1:
                raise ValueError('First-fold mode cannot run later folds.')
            save(folder / 'first-fold-training.json', dict(round=args.round, mode=mode,
                fixed_stack=stack, completed_native_folds=1, checkpoint_sha256=sha(checkpoint),
                no_screening_or_weight_selection=True,
                note='A training/shape checkpoint only; no compatibility or candidate gate is evaluated on this one fold.'))
            report(dict(stage='production_first_fold_only_complete', mode=mode,
                        no_screening_or_weight_selection=True))
            return
    if not np.all(coverage == 1):
        raise ValueError('Production OOF rows were not each predicted exactly once.')
    oof = check_probability(oof, len(tr), 'full native OOF')
    test = check_probability(test, len(sample), 'native fivefold test average')
    carrier_sub = sample.copy()
    carrier_sub.satisfaction = test
    validate_submission(carrier_sub, sample)
    carrier_sub.to_csv(folder / 'submission.csv', index=False)
    pd.DataFrame(dict(id=tr.id, satisfaction=y, prediction=oof)).to_csv(folder / 'oof.csv', index=False)
    pd.DataFrame(dict(id=sample.id, prediction=test)).to_csv(folder / 'test_predictions.csv', index=False)
    native_metrics = dict(competition=COMPETITION, round=args.round, mode=mode, folds=5,
        seed=42, epochs=protocol['epochs'], input_sha256=protocol['input_sha256'],
        external_sha256=protocol['external_sha256'], selected_blend=dict(auc=float(roc_auc_score(y, oof))),
        submission_sha256=sha(folder / 'submission.csv'), oof_sha256=sha(folder / 'oof.csv'),
        fold_scores=details, oof_rows=len(tr), oof_coverage_count=1,
        configuration_sha256=sha(folder / 'configuration.json'),
        script_sha256=sha(Path(__file__)), source_code_sha256=code,
        auxiliary_features_sha256=factory.audit['auxiliary_features_sha256'],
        seconds=round(time.time() - started, 3), validation_note=VALIDATION_NOTE)
    save(folder / 'metrics.json', native_metrics)
    for name in ('configuration.json', 'source-audit.json', 'metrics.json', 'oof.csv',
                 'submission.csv', 'test_predictions.csv'):
        fingerprints[rel(folder / name)] = sha(folder / name)
    for f in range(1, 6):
        for extension in ('npz', 'json'):
            fingerprints[rel(folder / f'fold-{f}.{extension}')] = sha(folder / f'fold-{f}.{extension}')
    combined = check_probability(fixed_stack(baseline, oof, stack), len(tr), 'fixed production OOF')
    combined_test = check_probability(fixed_stack(baseline_test, test, stack), len(sample), 'fixed production test')
    def score(p):
        return dict(auc=float(roc_auc_score(y, p)),
                    block_auc=[float(roc_auc_score(y[b], p[b])) for _, b in split])
    before, after = score(baseline), score(combined)
    delta = after['auc'] - before['auc']
    blocks = [new - old for new, old in zip(after['block_auc'], before['block_auc'])]
    passed = bool(delta >= 1e-5 and min(blocks) > 0)
    selected = dict(candidate=mode + '_locked_logit', auc=after['auc'], delta=delta,
                    fold_deltas=blocks, all_blocks_positive=min(blocks) > 0) if passed else None
    result = dict(competition=COMPETITION, round=args.round, folds=5, native_folds=5,
        evaluation_blocks=5, historical_native_folds=[3, 5, 10], seed=42,
        epochs=protocol['epochs'], carrier_mode=mode,
        input_sha256=protocol['input_sha256'], external_sha256=protocol['external_sha256'],
        source_status=dict(carrier='completed_five_folds'), native_model_path=rel(folder),
        native_carrier_folder=rel(folder), locked_protocol_path=locked['protocol_path'],
        locked_development_path=locked['development_selection_path'],
        locked_final_path=locked['final_comparison_path'], frozen_stack=stack,
        production_coefficient_lock=config['production_coefficient_lock'],
        locked_validation=locked, baseline=before, baseline_reconstruction_max_difference=reconstruction,
        protected_baseline_oof_sha256=config['protected_baseline_oof_sha256'],
        protected_baseline_submission_sha256=BASE_SHA,
        full_oof=dict(rows=len(tr), coverage_count=1, baseline_auc=before['auc'],
                      selected_auc=after['auc'], delta=delta, block_auc=after['block_auc'], block_deltas=blocks),
        compatibility_gate=dict(min_delta=1e-5, delta=delta,
                                all_five_blocks_positive=min(blocks) > 0, passed=passed),
        native_carrier_auc=native_metrics['selected_blend']['auc'], selected_blend=selected,
        eligible_for_submission=passed, uploaded=False,
        paired_against_previous_best=paired(y, baseline, combined),
        source_fingerprints=fingerprints, script_sha256=sha(Path(__file__)),
        feature_code_sha256=code,
        model_description=f'Raw auxiliary RealMLP {mode} {protocol["epochs"]}epochs fixed development logit transfer',
        seconds=round(time.time() - started, 3), validation_note=VALIDATION_NOTE)
    # Retain the same one fixed candidate even on failure for reproducibility,
    # but failed candidates never acquire a formal outputs-roundN directory.
    candidate_sub = sample.copy()
    candidate_sub.satisfaction = combined_test
    validate_submission(candidate_sub, sample)
    candidate_oof = pd.DataFrame(dict(id=tr.id, satisfaction=y, prediction=combined))
    candidate_sub.to_csv(folder / 'production-submission.csv', index=False)
    candidate_oof.to_csv(folder / 'production-oof.csv', index=False)
    result['production_candidate_sha256'] = sha(folder / 'production-submission.csv')
    result['production_oof_sha256'] = sha(folder / 'production-oof.csv')
    result['production_candidate_path'] = rel(folder / 'production-submission.csv')
    result['production_oof_path'] = rel(folder / 'production-oof.csv')
    if passed:
        output = ROOT / f'outputs-round{args.round}'
        if any((output / name).exists() for name in ('submission.csv', 'oof.csv', 'metrics.json')):
            raise ValueError('Formal production output already exists; do not replace it.')
        output.mkdir(parents=True, exist_ok=True)
        candidate_sub.to_csv(output / 'submission.csv', index=False)
        candidate_oof.to_csv(output / 'oof.csv', index=False)
        result['submission_sha256'] = sha(output / 'submission.csv')
        result['oof_sha256'] = sha(output / 'oof.csv')
        if (result['submission_sha256'] != result['production_candidate_sha256']
                or result['oof_sha256'] != result['production_oof_sha256']):
            raise ValueError('Formal CSV bytes differ from the evaluated fixed candidate.')
        save(output / 'metrics.json', result)
    save(analysis, result)
    report({k: result[k] for k in ('round', 'carrier_mode', 'native_carrier_auc', 'baseline',
                                 'full_oof', 'compatibility_gate', 'selected_blend', 'eligible_for_submission')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, choices=(16, 17), required=True)
    parser.add_argument('--first-fold', action='store_true')
    args = parser.parse_args()
    run(args)
