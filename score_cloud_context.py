"""Audit downloaded full-context GPU outputs and score the frozen pilot."""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from round18_joint import ROOT, sha, save, load, idxsha
from evaluate_context import paired

VARIANTS = {31: 'round31-cloud-full-context', 32: 'round32-cloud-full-raw', 34: 'round34-cloud-recovery'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=[31, 32, 34], default=32)
    args = parser.parse_args()
    variant = VARIANTS[args.round]
    OUT = ROOT / 'experiments' / variant
    analysis = ROOT / 'analyses' / variant / 'results.json'
    p = load(OUT / 'protocol.json')
    script_folders = {31: 'cloud-round31-full-context', 32: 'cloud-round32-full-raw', 34: 'cloud-round34-recover-none'}
    script = ROOT / script_folders[args.round] / 'full_context.py'
    if sha(script) != p['script_sha256']:
        raise ValueError('Frozen cloud source changed')
    for name, expected in p['protected_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Protected competition state changed')
    if sha(ROOT / p['baseline_oof_path']) != p['baseline_oof_sha256']:
        raise ValueError('Baseline changed')
    run_path = OUT / 'outputs/run-result.json'
    if not run_path.exists():
        status = load(ROOT / 'evidence' / f'round{args.round}-cloud-latest-status.json')
        if status['status'] in ['RUNNING', 'QUEUED']:
            raise ValueError('Job has not ended; missing output is not failure evidence')
        save(analysis, dict(round=args.round, successful=False,
            official_status=status, completed_model_result_present=False,
            official_submission_generated=False))
        print(json.dumps(dict(round=args.round, successful=False, status=status['status'])))
        return
    executed_source_sha = hashlib.sha256(script.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    run = load(run_path)
    if run['script_sha256'] != executed_source_sha or run['gpu_memory_pooled']:
        raise ValueError('Executed source or GPU strategy changed')
    if not run['successful']:
        save(analysis,
             dict(round=args.round, successful=False, resource_result=run,
                  official_submission_generated=False))
        print(json.dumps(dict(successful=False, worker_exit_codes=run['worker_exit_codes'])))
        return
    tr = pd.read_csv(ROOT / 'data/train.csv', usecols=['id', 'satisfaction'])
    y = tr.satisfaction.to_numpy(dtype='int8')
    a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
    baseline = pd.read_csv(ROOT / p['baseline_oof_path'])
    if not baseline.id.equals(tr.id) or not np.array_equal(baseline.satisfaction, y):
        raise ValueError('Baseline row alignment changed')
    base = baseline.prediction.to_numpy()[b]
    predictions, records = [], []
    for index, arm in enumerate(p['arms']):
        folder = ROOT / arm['outputs_relative'] if 'outputs_relative' in arm else OUT / 'outputs'
        record = load(folder / f'worker-{index}-result.json')
        packet = folder / f'worker-{index}-predictions.npz'
        if sha(packet) != record['prediction_sha256']:
            raise ValueError('Cloud prediction bytes changed')
        if record.get('feature_packet_sha256') != p['feature_packet_sha256']:
            raise ValueError('Feature packet changed')
        if record['checkpoint_sha256'] != 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0':
            raise ValueError('Model checkpoint changed')
        if 'frozen_prediction_sha256' in arm and sha(packet) != arm['frozen_prediction_sha256']:
            raise ValueError('Previously completed prediction changed')
        if 'frozen_record_sha256' in arm and sha(folder / f'worker-{index}-result.json') != arm['frozen_record_sha256']:
            raise ValueError('Previously completed arm receipt changed')
        expected_arm_source_sha = arm.get('executed_script_sha256', p['script_sha256'])
        if expected_arm_source_sha == p['script_sha256']:
            expected_arm_source_sha = executed_source_sha
        if record['script_sha256'] != expected_arm_source_sha or record['norm'] != arm['norm']:
            raise ValueError('Cloud inference parameters changed')
        if record['members'] != arm['n_estimators'] or record['seed'] != arm['seed']:
            raise ValueError('Cloud member configuration changed')
        if record['full_context_subsampled'] or record['outer_labels_used_for_fit']:
            raise ValueError('Context/label scope changed')
        if record['training_context_rows'] != len(a) or record['validation_rows'] != len(b):
            raise ValueError('Cloud row counts changed')
        if record['training_idx_sha256'] != idxsha(a) or record['validation_idx_sha256'] != idxsha(b):
            raise ValueError('Cloud split fingerprint changed')
        with np.load(packet, allow_pickle=False) as z:
            if not np.array_equal(z['train_idx'], a) or not np.array_equal(z['val_idx'], b):
                raise ValueError('Cloud index order changed')
            prediction = z['val_pred']
        if prediction.shape != (len(b),):
            raise ValueError('Prediction shape changed')
        if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
            raise ValueError('Invalid cloud probabilities')
        predictions.append(prediction)
        records.append(dict(norm=arm['norm'], single_arm_auc=float(roc_auc_score(y[b], prediction)),
            peak_allocated_vram_gb=record['peak_allocated_vram_gb'],
            peak_reserved_vram_gb=record['peak_reserved_vram_gb'],
            fit_seconds=record['fit_seconds'], total_seconds=record['total_seconds'],
            candidate_eligible_on_its_own=False))
    # The intended four-member candidate is the equal average of the two
    # independently run two-member arms. Individual scores are diagnostic.
    prediction = np.mean(predictions, axis=0)
    base_auc = float(roc_auc_score(y[b], base))
    zbase = logit(np.clip(base, 1e-6, 1 - 1e-6))
    znew = logit(np.clip(prediction, 1e-6, 1 - 1e-6))
    blends = []
    for weight in p['blend_weights']:
        blend = expit((1 - weight) * zbase + weight * znew)
        auc = float(roc_auc_score(y[b], blend))
        blends.append(dict(weight=weight, auc=auc, delta=auc - base_auc,
                           paired_fixed_prediction=paired(y[b], base, blend)))
    eligible = [m for m in blends if m['delta'] >= p['pilot_continue_delta']]
    result = dict(round=args.round, successful=True, cloud_full_context_completed=True,
        training_context_rows=len(a), validation_rows=len(b), total_members=4,
        baseline_auc=base_auc, combined_model_auc=float(roc_auc_score(y[b], prediction)),
        arms=records, probability_diversity_max_difference=float(np.max(abs(predictions[0] - predictions[1]))),
        blends=blends, continue_full_folds=bool(eligible), eligible_weights=eligible,
        cloud_total_seconds=run['total_seconds'],
        protocol_sha256=sha(OUT / 'protocol.json'),
        official_submission_generated=False, official_submission_attempted=False,
        limitation=p['limitations'])
    np.savez_compressed(OUT / 'combined-pilot.npz', val_idx=b, val_pred=prediction)
    save(analysis, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
