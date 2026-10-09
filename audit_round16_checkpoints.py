"""Audit saved rounds16/17 predictions without reading H1/H2 label values.

Run after native training and before development selection. This independent
file imports no frozen model module and never trains or evaluates label metrics.
Only development rows' satisfaction values are parsed from train.csv.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parent
PRIMITIVES = ('compact', 'route_auc', 'route_tuned', 'cat', 'plain_nn',
              'lgb', 'context', 'aux', 'tabm', 'travel')
REFERENCE = {
    'compact': 'experiments/compact/configuration.json',
    'route_auc': 'experiments/route-auc/configuration.json',
    'route_tuned': 'experiments/route-tuned-3fold/configuration.json',
    'cat': 'experiments/cat/configuration.json',
    'plain_nn': 'experiments/realmlp-5fold-8ens/configuration.json',
    'lgb': 'experiments/round9-lgb-auxiliary-5fold/configuration.json',
    'context': 'experiments/realmlp-context-5fold-8ens/configuration.json',
    'aux': 'experiments/round13-realmlp-route-auxiliary-10fold-8ens/configuration.json',
    'tabm': 'experiments/round9-tabm-auxiliary-5fold/configuration.json',
    'travel': 'experiments/round11-experts-5fold/configuration.json',
}
RECIPE = {
    'compact': 'improve.py', 'cat': 'improve.py', 'route_auc': 'route_auc_round.py',
    'route_tuned': 'route_tuned.py', 'plain_nn': 'nn_round.py', 'context': 'nn_context_round.py',
    'lgb': 'round9_models.py', 'tabm': 'round9_models.py', 'aux': 'round13_tenfold.py',
    'travel': 'divergent_round.py',
}


def path(name):
    result = Path(name)
    result = (ROOT / result if not result.is_absolute() else result).resolve()
    result.relative_to(ROOT.resolve())
    return result


def rel(name):
    return str(path(name).relative_to(ROOT)).replace('\\', '/')


def sha(name):
    result = hashlib.sha256()
    with path(name).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load(name):
    return json.loads(path(name).read_text(encoding='utf-8'))


def idx_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def target_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='int8').tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, name):
    require(np.array_equal(actual, expected), 'Row/index mismatch: ' + name)


def close(actual, expected, name):
    require(np.asarray(actual).shape == np.asarray(expected).shape,
            'Prediction shape mismatch: ' + name)
    difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    require(difference <= 1e-12, 'Prediction reconstruction differs: ' + name)
    return difference


def probability(value, length, name):
    # Preserve stored float32/float64 arithmetic when reconstructing the exact
    # historical recipe; casting neural primitives changes scalar products.
    result = np.asarray(value)
    require(result.shape == (length,) and np.isfinite(result).all()
            and (result >= 0).all() and (result <= 1).all(), 'Invalid probabilities: ' + name)
    return result


def formula(pred):
    tree = (pred['compact'] + pred['route_auc'] + pred['cat']) / 3.
    calibrated = expit(.5 * logit(np.clip(tree, 1e-6, 1-1e-6))
                       + .5 * logit(np.clip(pred['plain_nn'], 1e-6, 1-1e-6)))
    old5 = .9 * calibrated + .1 * pred['route_tuned']
    ctx = expit(.5 * logit(np.clip(pred['lgb'], 1e-6, 1-1e-6))
                + .5 * logit(np.clip(pred['context'], 1e-6, 1-1e-6)))
    old9 = .9 * (.75 * (.5 * old5 + .5 * ctx) + .25 * pred['aux']) + .1 * pred['tabm']
    return .9 * old9 + .1 * pred['travel']


def parameters(kind, cfg, seed):
    if kind == 'compact':
        return dict(n_estimators=cfg['iterations'], learning_rate=.025, n_jobs=8,
            objective='binary', reg_lambda=1, reg_alpha=0, random_state=seed,
            min_child_samples=100, num_leaves=127, max_depth=-1, colsample_bytree=.5,
            subsample=.8, subsample_freq=1, verbosity=-1, deterministic=True, force_col_wise=True)
    if kind == 'route_auc':
        return dict(n_estimators=cfg['iterations'], learning_rate=.025, n_jobs=8,
            objective='binary', reg_lambda=5, reg_alpha=.05, random_state=seed,
            min_child_samples=20, num_leaves=63, max_depth=8, colsample_bytree=.85,
            subsample=.9, subsample_freq=1, verbosity=-1, deterministic=True, force_col_wise=True, metric='auc')
    if kind == 'route_tuned':
        return dict(cfg['params'], random_state=seed)
    if kind == 'cat':
        return dict(iterations=cfg['iterations'], learning_rate=.05, depth=7,
            l2_leaf_reg=3, border_count=254, task_type='GPU', eval_metric='AUC',
            random_seed=seed, verbose=False, allow_writing_files=False, thread_count=8)
    if kind in ('lgb', 'travel'):
        return dict(cfg['params'], seed=seed)
    return dict(cfg['params'], random_state=seed, verbosity=0)


def main(round_number):
    manifest = f'experiments/round{round_number}-locked/protocol.json'
    protocol = load(manifest)
    fingerprints = {}

    def record(name, expected=None):
        actual = sha(name)
        if expected is not None:
            require(actual == expected, 'Saved/source SHA mismatch: ' + str(name))
        fingerprints[rel(name)] = actual
        return actual

    def read(name):
        record(name)
        return load(name)

    record(manifest)
    require(protocol['competition'] == 'playground-series-s6e10' and protocol['round'] == round_number
            and protocol['folds'] == 3 and protocol['epochs'] == (4 if round_number == 16 else 8),
            'Unexpected round/development protocol')
    require(protocol['active_holdout'] == ('hold1' if round_number == 16 else 'hold2'), 'Active holdout changed')
    require(protocol['holdout_labels_masked_in_training_factory'] is True, 'Masked-label declaration missing')
    require(protocol['baseline_members'] == ['baseline_current']
            and protocol['candidate_members'] == ['raw', 'raw13', 'raw30'], 'Frozen model list changed')
    for name, expected in protocol['source_code_sha256'].items():
        record(name, expected)
    for name, expected in protocol['input_sha256'].items():
        record('data/' + name, expected)
    record('data-original/data.csv', protocol['external_sha256'])
    for name, expected in protocol['protected_file_sha256'].items():
        record(name, expected)
    record(protocol['split_path'], protocol['split_sha256'])
    train_id = pd.read_csv(path('data/train.csv'), usecols=['id']).id.to_numpy()
    test_id = pd.read_csv(path('data/sample_submission.csv'), usecols=['id']).id.to_numpy()
    with np.load(path(protocol['split_path']), allow_pickle=False) as z:
        dev, h1, h2, fid = [z[k].copy() for k in ('dev_idx', 'hold1_idx', 'hold2_idx', 'dev_fold_id')]
        same(z['train_id'], train_id, 'split train IDs')
        same(z['test_id'], test_id, 'split test IDs')
    hold = np.concatenate([h1, h2])
    require(np.array_equal(np.sort(np.concatenate([dev, hold])), np.arange(len(train_id))), 'Split does not partition input')
    require(len(np.unique(hold)) == len(hold) and len(np.unique(dev)) == len(dev), 'Duplicate split positions')
    require(fid.shape == (len(dev),) and set(np.unique(fid)) == {0, 1, 2}, 'Bad development fold IDs')
    # csv row zero is the header; skip both holdout sets before pandas parses
    # satisfaction. The label table contains development rows only.
    dev_set = set(dev.tolist())
    labels = pd.read_csv(path('data/train.csv'), usecols=['id', 'satisfaction'],
                         skiprows=lambda line: line > 0 and line - 1 not in dev_set)
    same(labels.id.to_numpy(), train_id[dev], 'development label IDs')
    require(set(labels.satisfaction.unique()) == {0, 1}, 'Nonbinary development targets')
    ydev = pd.Series(labels.satisfaction.to_numpy(dtype='int8'), index=dev)
    del labels, dev_set

    base = protocol['models']['baseline_current']['path']
    expected_base = 'experiments/round16-locked/baseline_current'
    require(rel(base) == expected_base, 'Round17 must reuse the exact freshly trained round16 baseline')
    base_source = read(base + '/source-audit.json')
    require(base_source['input_sha256'] == protocol['input_sha256']
            and base_source['script_sha256'] == protocol['source_code_sha256']['round16_baseline.py']
            and base_source['competition_oof_reused'] is False
            and base_source['auxiliary_transductive'] is True, 'Baseline source/OOF provenance mismatch')
    for key in ('frozen_covariate_sha256', 'feature_code_sha256'):
        for source, expected in base_source[key].items():
            record(source, expected)
            if source in protocol['source_code_sha256']:
                require(expected == protocol['source_code_sha256'][source], 'Baseline feature/manifest source mismatch')
    references = {}
    for kind, source in REFERENCE.items():
        record(source, base_source['reference_configuration_sha256'][kind])
        references[kind] = load(source)
        record(RECIPE[kind], references[kind]['script_sha256'])
    aux_meta = read('experiments/auxiliary-reconstruction-300rounds/metrics.json')
    teacher_meta = read('experiments/original-xgb-600rounds/metrics.json')
    require(aux_meta['configuration']['input_sha256'] == protocol['input_sha256']
            and aux_meta['configuration']['label_used'] is False
            and aux_meta['configuration']['transductive'] is True, 'Auxiliary target-use/source mismatch')
    record('experiments/auxiliary-reconstruction-300rounds/features.pkl', aux_meta['features_sha256'])
    record('experiments/original-xgb-600rounds/probability.npy', teacher_meta['probability_sha256'])
    require(teacher_meta['configuration']['input_sha256'] == protocol['input_sha256']
            and teacher_meta['configuration']['original_sha256'] == protocol['external_sha256']
            and teacher_meta['configuration']['overlaps_removed'] == base_source['external_exact_feature_matches_removed'],
            'Original teacher source mismatch')

    def native(folder, fold, a, v):
        packet_path = f'{folder}/fold-{fold}.npz'
        meta = read(f'{folder}/fold-{fold}.json')
        record(packet_path, meta['checkpoint_sha256'])
        with np.load(path(packet_path), allow_pickle=False) as z:
            same(z['train_idx'], a, folder + ' train')
            same(z['val_idx'], v, folder + ' development val')
            same(z['hold_idx'], hold, folder + ' excluded holdouts')
            same(z['test_id'], test_id, folder + ' test IDs')
            predictions = (probability(z['val_pred'], len(v), folder + ' val'),
                           probability(z['hold_pred'], len(hold), folder + ' hold'),
                           probability(z['test_pred'], len(test_id), folder + ' test'))
        return meta, predictions

    report_models, fit_count = {}, 0
    for name, model in protocol['models'].items():
        folder = model['path']
        cfg = read(folder + '/configuration.json')
        require(cfg['competition'] == protocol['competition'] and cfg['folds'] == 3
                and cfg['model'] == name and cfg['epochs'] == (3 if name == 'baseline_current' else protocol['epochs'])
                and cfg['input_sha256'] == protocol['input_sha256']
                and cfg['external_sha256'] == protocol['external_sha256']
                and cfg['split_sha256'] == protocol['split_sha256']
                and cfg['source_code_sha256'] == protocol['source_code_sha256']
                and cfg['holdout_excluded_from_supervised_training_and_early_stopping'] is True,
                'Native model configuration differs from frozen manifest: ' + name)
        summary_path = folder + '/summary.npz'
        record(summary_path)
        with np.load(path(summary_path), allow_pickle=False) as z:
            same(z['dev_idx'], dev, name + ' summary development')
            same(z['hold_idx'], hold, name + ' summary holdout')
            same(z['test_id'], test_id, name + ' summary test')
            summary = (probability(z['dev_oof'], len(dev), name + ' development OOF'),
                       probability(z['hold_pred'], len(hold), name + ' holdout average'),
                       probability(z['test_pred'], len(test_id), name + ' test average'))
        coverage = np.zeros(len(dev), dtype='int8')
        hold_average, test_average = np.zeros(len(hold)), np.zeros(len(test_id))
        diffs, folds = [], []
        carrier_source = None if name == 'baseline_current' else read(folder + '/source-audit.json')
        if carrier_source is not None:
            require(carrier_source['implementation_sha256'] == protocol['source_code_sha256']['round16_carriers.py']
                    and carrier_source['input_sha256'] == protocol['input_sha256'], 'Carrier source manifest mismatch')
            for key in ('target_encoding', 'route_profiles', 'counts', 'original_teacher_features',
                        'third_party_predictions', 'auxiliary_target_used'):
                require(carrier_source[key] is False, 'Forbidden carrier feature/provenance: ' + key)
            require(carrier_source['transductive'] is True, 'Missing transductive declaration')
            record(REFERENCE['plain_nn'], carrier_source['reference_configuration_sha256'])
            record('experiments/auxiliary-reconstruction-300rounds/metrics.json', carrier_source['auxiliary_meta_sha256'])
            record('experiments/auxiliary-reconstruction-300rounds/features.pkl', carrier_source['auxiliary_features_sha256'])
            record('auxiliary_round.py', carrier_source['auxiliary_builder_sha256'])
            record('references/round8/s6e10-more-aux-carriers-lb-0-96152/s6e10-more-aux-carriers-lb-0-96152.ipynb',
                   carrier_source['public_method_source_sha256'])
        for fold in (1, 2, 3):
            val_pos = fid == fold - 1
            a, v = dev[~val_pos], dev[val_pos]
            meta, predictions = native(folder, fold, a, v)
            require(meta['fold'] == fold, 'Native saved fold number mismatch')
            coverage[val_pos] += 1
            diffs.append(close(predictions[0], summary[0][val_pos], name + ' development coverage'))
            hold_average += predictions[1] / 3
            test_average += predictions[2] / 3
            if name == 'baseline_current':
                require(meta['supervised_holdout_labels_available'] is False, 'Baseline declares supervised holdout labels')
                require(meta['baseline_formula'] == dict(formula='historical round13 weights',
                        members=['old5', 'contextual_pair', 'aux', 'tabm', 'travel'],
                        weights=[.30375, .30375, .2025, .09, .1]), 'Saved baseline aggregate formula changed')
                b = np.concatenate([v, hold])
                pb, pt = {}, {}
                for kind in PRIMITIVES:
                    component_path = f'{folder}/components-fold-{fold}/{kind}'
                    info = read(component_path + '.json')
                    record(component_path + '.npz', info['checkpoint_sha256'])
                    require(meta['components'][kind] == info, 'Primitive JSON differs from parent fold record: ' + kind)
                    require(info['component'] == kind and info['seed'] == 42 + fold
                            and info['source'] == base_source
                            and info['reference_configuration'] == references[kind]
                            and rel(info['historical_reference']) == REFERENCE[kind]
                            and info['params'] == parameters(kind, references[kind], 42 + fold),
                            'Primitive source/recipe/parameter mismatch: ' + kind)
                    require(info['train_idx_sha256'] == idx_sha(a)
                            and info['evaluation_idx_sha256'] == idx_sha(b)
                            and info['validation_idx_sha256'] == idx_sha(v)
                            and info['train_target_sha256'] == target_sha(ydev.loc[a])
                            and info['validation_target_sha256'] == target_sha(ydev.loc[v]),
                            'Primitive development index/target fingerprint mismatch: ' + kind)
                    require(info['validation_scope'] == 'explicit development validation only; holdout never supplies labels',
                            'Primitive validation scope mismatch')
                    require(info['label_perturbation'] == ('PASS' if fold == 1 else 'not_run'), 'Unexpected label-perturbation result')
                    if kind in ('plain_nn', 'context', 'aux', 'tabm'):
                        require(info['device'] == 'cuda:0', 'Baseline neural GPU mismatch')
                    if kind == 'travel':
                        require(len(info['experts']) == 2 and sum(e['training_rows'] for e in info['experts']) == len(a)
                                and sum(e['early_stop_rows'] for e in info['experts']) == len(v), 'Travel expert coverage mismatch')
                        fit_count += 2
                    else:
                        fit_count += 1
                    with np.load(path(component_path + '.npz'), allow_pickle=False) as z:
                        same(z['evaluation_idx'], b, kind + ' evaluation packet')
                        same(z['validation_idx'], v, kind + ' validation packet')
                        pb[kind] = probability(z['evaluation_pred'], len(b), kind + ' evaluation')
                        pt[kind] = probability(z['test_pred'], len(test_id), kind + ' test')
                reconstructed_b, reconstructed_t = formula(pb), formula(pt)
                diffs.append(close(reconstructed_b[:len(v)], predictions[0], 'Fresh baseline formula development'))
                diffs.append(close(reconstructed_b[len(v):], predictions[1], 'Fresh baseline formula holdout'))
                diffs.append(close(reconstructed_t, predictions[2], 'Fresh baseline formula test'))
            else:
                require(meta['mode'] == name and meta['seed'] == 42 + fold
                        and meta['source_audit'] == carrier_source, 'Carrier mode/seed/source differs from native fit')
                expected_params = dict(references['plain_nn']['params'], n_epochs=protocol['epochs'], verbosity=2,
                                       random_state=42 + fold)
                require(meta['params'] == expected_params, 'Carrier training parameters changed')
                require(meta['actual_binary_classifier'] is True and meta['actual_device'] == 'cuda:0',
                        'Carrier actual binary classification/device check missing')
                trainer = meta['trainer_configuration']
                require(trainer['accelerator'] == 'gpu' and trainer['devices'] == [0]
                        and trainer['max_epochs'] == protocol['epochs'], 'Carrier GPU Trainer configuration mismatch')
                for key in ('holdout_labels_used_in_fit',):
                    require(meta[key] is False, 'Carrier used supervised holdout labels')
                for key in ('holdout_label_values_never_indexed', 'caller_supplied_masked_label_array', 'held_y_values_not_available'):
                    require(meta[key] is True, 'Carrier masked/excluded label declaration missing')
                require(meta['index_sha256'] == dict(train=idx_sha(a), development_validation=idx_sha(v), holdout=idx_sha(hold)),
                        'Carrier index fingerprint mismatch')
                require(meta['train_rows'] == len(a) and meta['validation_rows'] == len(v)
                        and meta['holdout_rows'] == len(hold) and meta['test_rows'] == len(test_id), 'Carrier fit row counts mismatch')
                require(meta['feature_count'] == carrier_source['feature_counts'][name]
                        and len(meta['features']) == meta['feature_count']
                        and meta['categorical_columns'] == carrier_source['categorical_columns'], 'Carrier feature schema mismatch')
            folds.append(dict(fold=fold, native_checkpoint_sha256=meta['checkpoint_sha256'],
                              training_rows=len(a), development_validation_rows=len(v), excluded_holdout_rows=len(hold)))
        require(np.all(coverage == 1), 'Development OOF coverage is not exactly once')
        diffs.append(close(hold_average, summary[1], name + ' heldout three-fold average'))
        diffs.append(close(test_average, summary[2], name + ' test three-fold average'))
        report_models[name] = dict(folds=folds, max_prediction_reconstruction_difference=max(diffs),
                                   development_oof_coverage_exactly_once=True, native_three_fold_averaging=True)
    require(fit_count == 33, 'Expected 11 baseline estimator fits per fold, three folds')
    result = dict(passed=True, round=round_number, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha(Path(__file__)), protocol_path=manifest,
                  baseline_estimator_fits=fit_count, baseline_development_folds=3,
                  carrier_fits=9, models=report_models, audited_file_sha256=fingerprints,
                  label_scope='Only development CSV rows parsed for target fingerprints; H1/H2 satisfaction skipped before parsing.',
                  holdout_labels_loaded=False, holdout_metrics_computed=False, development_metrics_computed=False,
                  classification_proof='Carrier fit metadata records actual binary check from frozen code that checks model.classes_ == [0,1].',
                  exclusion_proof='Frozen orchestrator masks both H target sets; factories read a/v labels only. Native scope/index/target/source records verified.',
                  note='Checkpoint integrity/exclusion audit only; no model-selection or generalization claim.')
    destination = path(f'evidence/round{round_number}-checkpoint-audit.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'round', 'baseline_estimator_fits', 'carrier_fits',
                                           'holdout_labels_loaded', 'holdout_metrics_computed')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, choices=[16, 17], required=True)
    main(parser.parse_args().round)
