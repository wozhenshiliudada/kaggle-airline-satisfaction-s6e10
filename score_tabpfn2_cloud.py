"""Built with PriorLabs-TabPFN. Audit and score fixed TabPFN2 GPU pilots."""
import json
import argparse
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from round18_joint import ROOT, sha, save, load, idxsha
from evaluate_context import paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=[35, 36], default=35)
    args = parser.parse_args()
    variant = 'round35-tabpfn2' if args.round == 35 else 'round36-tabpfn2-fixed'
    script = ROOT / ('cloud-round35-tabpfn2' if args.round == 35 else 'cloud-round36-tabpfn2-fixed') / 'tabpfn2_pilot.py'
    out = ROOT / 'experiments' / variant
    p = load(out / 'protocol.json')
    if sha(script) != p['script_sha256']:
        raise ValueError('Frozen source changed')
    for name, expected in p['protected_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Protected competition state changed')
    if sha(ROOT / p['baseline_oof_path']) != p['baseline_oof_sha256']:
        raise ValueError('Baseline changed')
    run_path = out / 'outputs/run-result.json'
    analysis = ROOT / 'analyses' / variant / 'results.json'
    if not run_path.exists():
        status = load(ROOT / 'evidence' / f'round{args.round}-cloud-latest-status.json')
        if status['status'] in ['RUNNING', 'QUEUED']:
            raise ValueError('Pending job is not a failure')
        save(analysis, dict(successful=False, official_status=status,
                           official_submission_attempted=False))
        print(json.dumps(dict(successful=False, status=status['status'])))
        return
    executed_source_sha = hashlib.sha256((script).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    run = load(run_path)
    if run['script_sha256'] != executed_source_sha or run['gpu_memory_pooled']:
        raise ValueError('Executed source changed')
    if not any(w.get('successful', False) for w in run['workers']):
        save(analysis, dict(successful=False, resource_result=run,
                           official_submission_attempted=False))
        print(json.dumps(dict(successful=False, worker_exit_codes=run['worker_exit_codes'])))
        return
    tr = pd.read_csv(ROOT / 'data/train.csv', usecols=['id', 'satisfaction'])
    y = tr.satisfaction.to_numpy(dtype='int8')
    a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
    selected, _ = next(StratifiedShuffleSplit(1, train_size=20000, random_state=35).split(a, y[a]))
    context = np.sort(a[selected])
    baseline = pd.read_csv(ROOT / p['baseline_oof_path'])
    if not baseline.id.equals(tr.id) or not np.array_equal(baseline.satisfaction, y):
        raise ValueError('Baseline rows changed')
    base = baseline.prediction.to_numpy()[b]
    predictions, resource = {}, []
    for i, arm in enumerate(p['arms']):
        if not run['workers'][i].get('successful', False):
            resource.append(dict(branch=arm['norm'], successful=False, reason='cloud arm did not complete'))
            continue
        record = load(out / 'outputs' / f'worker-{i}-result.json')
        packet = out / 'outputs' / f'worker-{i}-predictions.npz'
        checks = [record['script_sha256'] == executed_source_sha,
                  record['checkpoint_sha256'] == p['checkpoint_sha256'],
                  record['norm'] == arm['norm'], record['seed'] == arm['seed'],
                  record['members'] == 1, record['model_version'] == 'v2',
                  record['inference_precision'] == 'float16', record['query_batch_rows'] == 2048,
                  not record['outer_labels_used_for_fit'], record['full_context_subsampled'],
                  record['training_context_rows'] == 20000,
                  record['validation_rows'] == len(b), record['outer_training_rows'] == len(a),
                  record['training_idx_sha256'] == idxsha(context),
                  record['outer_training_idx_sha256'] == idxsha(a),
                  record['validation_idx_sha256'] == idxsha(b),
                  sha(packet) == record['prediction_sha256'],
                  len(record['feature_columns']) == arm['feature_columns'],
                  record['feature_packet_sha256'] == (p['feature_packet_sha256'] if i == 1 else None)]
        if not all(checks):
            raise ValueError('Cloud member provenance or parameters changed')
        with np.load(packet, allow_pickle=False) as z:
            if not np.array_equal(z['train_idx'], context) or not np.array_equal(z['outer_train_idx'], a) or not np.array_equal(z['val_idx'], b):
                raise ValueError('Cloud context/split order changed')
            pred = z['val_pred']
        if pred.shape != (len(b),) or not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
            raise ValueError('Invalid prediction')
        predictions[arm['norm']] = pred
        resource.append(dict(branch=arm['norm'], fit_seconds=record['fit_seconds'],
            total_seconds=record['total_seconds'], peak_allocated_vram_gb=record['peak_allocated_vram_gb'],
            peak_reserved_vram_gb=record['peak_reserved_vram_gb']))
    candidates = dict(predictions)
    if len(predictions) == 2:
        candidates['equal_probability_mean'] = np.mean(list(predictions.values()), axis=0)
    if any(name not in p['candidates'] for name in candidates):
        raise ValueError('Undeclared candidate')
    base_auc = float(roc_auc_score(y[b], base))
    zbase = logit(np.clip(base, 1e-6, 1 - 1e-6))
    metrics = []
    for name, pred in candidates.items():
        znew = logit(np.clip(pred, 1e-6, 1 - 1e-6))
        blends = []
        for weight in p['blend_weights']:
            blend = expit((1 - weight) * zbase + weight * znew)
            auc = float(roc_auc_score(y[b], blend))
            blends.append(dict(weight=weight, auc=auc, delta=auc - base_auc,
                               paired_fixed_prediction=paired(y[b], base, blend)))
        metrics.append(dict(candidate=name, auc=float(roc_auc_score(y[b], pred)), blends=blends,
                            continue_full_folds=any(m['delta'] >= p['pilot_continue_delta'] for m in blends)))
    result = dict(round=args.round, successful=True, cloud_all_branches_completed=run['successful'],
        unavailable_candidates=[name for name in p['candidates'] if name not in candidates], model_version='TabPFN2.0', attribution=p['attribution'],
        training_context_rows=len(context), outer_training_rows=len(a), validation_rows=len(b),
        full_context_subsampled=True, baseline_auc=base_auc, candidates=metrics, resources=resource,
        continue_full_folds=any(m['continue_full_folds'] for m in metrics),
        cloud_total_seconds=run['total_seconds'], protocol_sha256=sha(out / 'protocol.json'),
        official_submission_attempted=False, limitation=p['limitations'])
    save(analysis, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
