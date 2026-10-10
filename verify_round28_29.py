"""Reconstruct candidates from audited fitted packets before any upload."""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from round18_joint import ROOT, sha, save, load, idxsha
from evaluate_context import paired
from train import read_data, validate_submission


def equal(left, right, name, tolerance=1e-12):
    if np.shape(left) != np.shape(right):
        raise ValueError('Shape mismatch: ' + name)
    error = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    if error > tolerance:
        raise ValueError('Reconstruction failed: ' + name + ': ' + str(error))
    return error


def simplex(w):
    w = np.asarray(w)
    if w[0] < .5 - 1e-7 or w.min() < -1e-7 or abs(w.sum() - 1) > 1e-7:
        raise ValueError('Preservation/simplex constraints failed')
    return w


def verify28():
    import round28_native_fusion as run
    import round22_preserved as native
    p = run.check()
    old = load(ROOT / 'experiments/round22-preserved/protocol.json')
    native.OUT = run.OUT
    tr, sample, y, P, T = native.library(old)
    X = logit(np.clip(P, 1e-6, 1 - 1e-6))
    m = load(run.FINAL / 'metrics.json')
    r = load(run.ANALYSIS / 'crossfit.json')
    if not m['eligible'] or not r['eligible']:
        raise ValueError('Frozen gate rejected the candidate')
    if m['protocol_sha256'] != sha(run.OUT / 'protocol.json'):
        raise ValueError('Protocol changed')
    with np.load(run.OUT / 'crossfit-predictions.npz') as z:
        packets = z['predictions']
    all_deltas, repeat_deltas = [], []
    for repeat, record in enumerate(r['repeats']):
        if record['seed'] != p['meta_cv']['seeds'][repeat]:
            raise ValueError('Meta split changed')
        reconstructed = np.full(len(y), np.nan)
        splits = StratifiedKFold(5, shuffle=True, random_state=record['seed'])
        for fold, (a, b) in enumerate(splits.split(X, y)):
            w = simplex(record['weights'][fold])
            reconstructed[b] = expit(X[b] @ w)
            delta = float(roc_auc_score(y[b], reconstructed[b]) - roc_auc_score(y[b], P[b, 0]))
            equal(delta, record['block_deltas'][fold], 'meta block AUC')
            all_deltas.append(delta)
        equal(reconstructed, packets[repeat], 'crossfit prediction')
        delta = float(roc_auc_score(y, reconstructed) - roc_auc_score(y, P[:, 0]))
        equal(delta, record['delta'], 'repeat delta')
        repeat_deltas.append(delta)
    gate = p['gates']
    if not (np.mean(repeat_deltas) >= gate['mean_delta'] and min(repeat_deltas) > 0
            and sum(d > 0 for d in all_deltas) >= gate['minimum_positive_blocks']
            and min(all_deltas) >= gate['worst_block_floor']):
        raise ValueError('Gate reconstruction failed')
    expected_oof = packets.mean(axis=0)
    averaged_delta = float(roc_auc_score(y, expected_oof) - roc_auc_score(y, P[:, 0]))
    if averaged_delta < gate['mean_delta']:
        raise ValueError('Averaged crossfit prediction fails frozen gate')
    expected_test = expit(logit(np.clip(T, 1e-6, 1 - 1e-6)) @ simplex(m['final_weights']))
    return finalize(run.FINAL, p, tr, sample, y, P[:, 0],
        expected_oof, expected_test, dict(native_audit_sha256=sha(run.OUT / 'native-audit.json'),
        mean_repeat_delta=float(np.mean(repeat_deltas)),
        worst_block_delta=min(all_deltas), positive_blocks=sum(d > 0 for d in all_deltas),
        final_weight_fit_scope='all historical native OOF; same library and fitting algorithm',
        validation_note=p['limitations']))


def verify29():
    import round29_matched_context as run
    from round12_models import rebuilt
    p = run.check()
    tr, ts, sample, raw, ys = read_data(ROOT / 'data')
    y = ys.to_numpy()
    splits = list(StratifiedKFold(10, shuffle=True, random_state=42).split(tr, ys))
    new, audit = {}, []
    for member in p['new_members']:
        oof, test = np.full(len(tr), np.nan), np.zeros(len(ts))
        for fold, (a, b) in enumerate(splits, 1):
            g, inner = train_test_split(a, test_size=p['split']['inner_rows'],
                stratify=ys.iloc[a], random_state=p['split']['inner_seed'] + fold)
            evaluation = np.concatenate([b, inner])
            folder = run.OUT / f'fold-{fold}'
            info = load(folder / (member + '.json'))
            packet = folder / (member + '.npz')
            if sha(packet) != info['checkpoint_sha256']:
                raise ValueError('Fresh fitted packet changed')
            if (info['train_idx_sha256'] != idxsha(g) or
                info['evaluation_idx_sha256'] != idxsha(evaluation) or
                info['validation_idx_sha256'] != idxsha(inner) or
                info['seed'] != p['split']['model_seed'] + fold):
                raise ValueError('Training/validation scope changed')
            if set(g) & set(b) or set(inner) & set(b):
                raise ValueError('Outer label scope overlaps training/selection')
            if fold == 1 and info['label_perturbation'] != 'PASS':
                raise ValueError('Missing target-feature invariance check')
            with np.load(packet) as z:
                if not np.array_equal(z['evaluation_idx'], evaluation):
                    raise ValueError('Evaluation index ordering changed')
                if not np.array_equal(z['validation_idx'], inner):
                    raise ValueError('Selection indices changed')
                oof[b] = z['evaluation_pred'][:len(b)]
                test += z['test_pred'] / 10
            audit.append(dict(member=member, fold=fold, gradient_rows=len(g),
                outer_rows=len(b), inner_rows=len(inner),
                checkpoint_sha256=info['checkpoint_sha256'],
                outer_labels_used_for_selection=False,
                feature_label_invariance=info['label_perturbation']))
        saved_oof = pd.read_csv(run.OUT / member / 'oof.csv')
        saved_test = pd.read_csv(run.OUT / member / 'submission.csv')
        equal(oof, saved_oof.prediction.to_numpy(), 'fresh native OOF')
        equal(test, saved_test.satisfaction.to_numpy(), 'fresh tenfold test average')
        new[member] = (oof, test)
    save(run.OUT / 'verified-training-scope.json', audit)
    r = load(run.ANALYSIS / 'comparison.json')
    preds = {}
    for name, directory in run.SOURCES.items():
        oof = pd.read_csv(ROOT / directory / 'oof.csv')
        sub = pd.read_csv(ROOT / directory / 'submission.csv')
        if not oof.id.equals(tr.id) or not np.array_equal(oof.satisfaction, y):
            raise ValueError('Old member OOF alignment changed')
        validate_submission(sub, sample)
        preds[name] = (oof.prediction.to_numpy(), sub.satisfaction.to_numpy())
    replacement = tuple(rebuilt(new['lgb'][side], preds['old5'][side],
        new['context'][side], preds['aux'][side], preds['tabm'][side],
        preds['travel'][side]) for side in (0, 1))
    reconstructed_candidates = {}
    reconstructed_rows = []
    for row in r['candidates']:
        fraction = row['fraction']
        if fraction not in p['replacement_fractions']:
            raise ValueError('Unplanned candidate')
        pred = (1 - fraction) * preds['best'][0] + fraction * replacement[0]
        reconstructed_candidates[row['candidate']] = pred
        delta = float(roc_auc_score(y, pred) - roc_auc_score(y, preds['best'][0]))
        equal(delta, row['delta'], 'replacement delta')
        deltas = [float(roc_auc_score(y[b], pred[b]) -
                        roc_auc_score(y[b], preds['best'][0][b])) for _, b in splits]
        equal(deltas, row['block_deltas'], 'replacement block deltas')
        reconstructed_rows.append(dict(candidate=row['candidate'], delta=delta,
            positive_blocks=sum(d > 0 for d in deltas), worst_block_delta=min(deltas)))
    chosen = np.full(len(y), np.nan)
    for a, b in splits:
        scores = {'baseline': float(roc_auc_score(y[a], preds['best'][0][a]))}
        scores.update({n: float(roc_auc_score(y[a], v[a]))
                       for n, v in reconstructed_candidates.items()})
        pick = max(scores, key=scores.get)
        chosen[b] = (preds['best'][0] if pick == 'baseline'
                     else reconstructed_candidates[pick])[b]
    leave_delta = float(roc_auc_score(y, chosen) - roc_auc_score(y, preds['best'][0]))
    equal(leave_delta, r['leave_block_selection_delta'], 'leave-block selection delta')
    gate = p['gates']
    accepted = [v for v in reconstructed_rows if v['delta'] >= gate['full_oof_delta']
        and v['positive_blocks'] >= gate['minimum_positive_blocks']
        and v['worst_block_delta'] >= gate['worst_block_floor']
        and leave_delta >= gate['leave_block_selection_delta_floor']]
    if bool(accepted) != bool(r['eligible']):
        raise ValueError('Frozen acceptance decision does not reproduce')
    if not r['eligible']:
        result = dict(passed=True, eligible=False, complete_fresh_fits=len(audit),
            submission_created=False, training_scope_audit_sha256=sha(run.OUT / 'verified-training-scope.json'),
            protocol_sha256=sha(run.OUT / 'protocol.json'), validation_note=p['limitations'])
        save(ROOT / 'evidence/round29-final-validation.json', result)
        return result
    selected = r['selected_blend']
    if selected['candidate'] not in [v['candidate'] for v in accepted]:
        raise ValueError('Selected candidate failed frozen gate')
    fraction = selected['fraction']
    expected_oof, expected_test = tuple((1 - fraction) * preds['best'][side] +
                                      fraction * replacement[side] for side in (0, 1))
    return finalize(run.FINAL, p, tr, sample, y, preds['best'][0],
        expected_oof, expected_test, dict(complete_fresh_fits=len(audit),
        training_scope_audit_sha256=sha(run.OUT / 'verified-training-scope.json'),
        preserved_auxiliary_member=run.SOURCES['aux'], validation_note=p['limitations']))


def finalize(folder, protocol, tr, sample, y, base, expected_oof, expected_test, details):
    m = load(folder / 'metrics.json')
    sub = pd.read_csv(folder / 'submission.csv')
    oof = pd.read_csv(folder / 'oof.csv')
    validate_submission(sub, sample)
    if not oof.id.equals(tr.id) or not np.array_equal(oof.satisfaction, y):
        raise ValueError('Candidate OOF alignment changed')
    if m['input_sha256'] != protocol['input_sha256']:
        raise ValueError('Candidate input provenance changed')
    if sha(folder / 'submission.csv') != m['submission_sha256']:
        raise ValueError('Candidate fingerprint changed')
    auc = float(roc_auc_score(y, oof.prediction))
    equal(auc, m['selected_blend']['auc'], 'candidate AUC', tolerance=2e-11)
    oof_error = equal(oof.prediction.to_numpy(), expected_oof, 'candidate OOF reconstruction')
    test_error = equal(sub.satisfaction.to_numpy(), expected_test, 'candidate test reconstruction')
    result = dict(passed=True, eligible=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        submission_sha256=m['submission_sha256'], oof_sha256=sha(folder / 'oof.csv'),
        metrics_sha256=sha(folder / 'metrics.json'), protocol_sha256=m['protocol_sha256'],
        submission_rows=len(sub), train_rows=len(oof),
        oof_reconstruction_max_error=oof_error, test_reconstruction_max_error=test_error,
        paired_fixed_prediction_diagnostic=paired(y, base, oof.prediction.to_numpy()),
        verifier_sha256=sha(__file__), **details)
    save(folder / 'validation.json', result)
    save(ROOT / 'evidence' / (folder.name.replace('outputs-', '') + '-final-validation.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('round', type=int, choices=[28, 29])
    args = parser.parse_args()
    result = verify28() if args.round == 28 else verify29()
    from round22_preserved import msg
    msg(stage='verified', round=args.round, **result)
