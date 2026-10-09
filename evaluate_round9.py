"""Fixed coarse candidates against round8, with aligned five-block diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT, sha
from auxiliary_round import hashes
from evaluate_context import paired
from train import read_data, COMPETITION, validate_submission


def run(stage):
    tr, ts, sample, raw, ys = read_data(ROOT / 'data')
    y = ys.to_numpy()
    input_hashes = hashes()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, ys))
    paths = dict(baseline='outputs-round8', old6='outputs-round6', old5='outputs-round5',
        old_lgb='experiments/context-conditional-5fold', nn_context='experiments/realmlp-context-5fold-8ens',
        nn_aux='experiments/realmlp-route-auxiliary-all-5fold-8ens',
        lgb='experiments/round9-lgb-auxiliary-5fold')
    if stage == 'all':
        paths['tabm'] = 'experiments/round9-tabm-auxiliary-5fold'
    predictions, meta = {}, {}
    for name, relative in paths.items():
        folder = ROOT / relative
        m = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        if m['competition'] != COMPETITION or m['input_sha256'] != input_hashes:
            raise ValueError('Input mismatch: ' + name)
        o = pd.read_csv(folder / 'oof.csv')
        s = pd.read_csv(folder / 'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction, y):
            raise ValueError('OOF alignment mismatch: ' + name)
        validate_submission(s, sample)
        if sha(folder / 'submission.csv') != m['submission_sha256']:
            raise ValueError('Prediction fingerprint mismatch: ' + name)
        p = o.prediction.to_numpy()
        if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise ValueError('Invalid OOF probability')
        if name in ('lgb', 'tabm', 'old_lgb', 'nn_context', 'nn_aux'):
            for f, (_, b) in enumerate(folds, 1):
                with np.load(folder / f'fold-{f}.npz') as saved:
                    if not np.array_equal(saved['val_idx'], b) or not np.allclose(saved['val_pred'], p[b], rtol=0, atol=1e-12):
                        raise ValueError('Fold cache mismatch: ' + name)
        predictions[name] = (p, s.satisfaction.to_numpy())
        meta[name] = m
    def mix(a, b, w):
        return tuple((1-w)*a[i] + w*b[i] for i in range(2))
    def logmean(a, b):
        return tuple(expit(.5*logit(np.clip(a[i], 1e-6, 1-1e-6)) + .5*logit(np.clip(b[i], 1e-6, 1-1e-6))) for i in range(2))
    def replace(member):
        context = logmean(member, predictions['nn_context'])
        r6 = mix(predictions['old5'], context, .5)
        return mix(r6, predictions['nn_aux'], .25)
    def score(p):
        return dict(auc=float(roc_auc_score(y, p)), block_auc=[float(roc_auc_score(y[b], p[b])) for _, b in folds])
    baseline = predictions['baseline']
    reconstructed = replace(predictions['old_lgb'])
    errors = [float(np.max(np.abs(reconstructed[i]-baseline[i]))) for i in range(2)]
    if max(errors) > 1e-12:
        raise ValueError('Current official best formula differs')
    bs = score(baseline[0])
    candidates = {'lgb_replace': replace(predictions['lgb'])}
    for name in ('lgb', 'tabm'):
        if name in predictions:
            for w in (.1, .25, .5):
                candidates[f'{name}_weight{w}'] = mix(baseline, predictions[name], w)
    if stage == 'all':
        for w in (.1, .25):
            candidates[f'lgb_replace_tabm{w}'] = mix(candidates['lgb_replace'], predictions['tabm'], w)
    comparison = []
    for name, (p, pt) in candidates.items():
        s = score(p)
        deltas = [a-b for a, b in zip(s['block_auc'], bs['block_auc'])]
        comparison.append(dict(candidate=name, **s, delta=s['auc']-bs['auc'], fold_deltas=deltas,
            all_blocks_positive=bool(min(deltas)>0), correlation=float(np.corrcoef(p, baseline[0])[0, 1])))
    eligible = [r for r in comparison if r['all_blocks_positive'] and r['delta'] > 0]
    selected = max(eligible, key=lambda r: r['auc']) if eligible else None
    chosen_oof = baseline[0].copy()
    leave_block = []
    for f, (_, b) in enumerate(folds):
        a = np.concatenate([folds[i][1] for i in range(5) if i != f])
        other_best = None
        best_auc = float(roc_auc_score(y[a], baseline[0][a]))
        for row in comparison:
            if min(d for i, d in enumerate(row['fold_deltas']) if i != f) <= 0:
                continue
            p = candidates[row['candidate']][0]
            auc = float(roc_auc_score(y[a], p[a]))
            if auc > best_auc:
                best_auc, other_best = auc, row['candidate']
        choice = baseline[0] if other_best is None else candidates[other_best][0]
        chosen_oof[b] = choice[b]
        leave_block.append(dict(held_out_block=f+1, selected=other_best or 'baseline',
            auc=float(roc_auc_score(y[b], choice[b])), delta=float(roc_auc_score(y[b], choice[b])-bs['block_auc'][f])))
    label = 'Round9 auxiliary LightGBM replacement'
    if selected and 'tabm' in selected['candidate']:
        label = 'Round9 auxiliary LightGBM and route TabM' if selected['candidate'].startswith('lgb_replace_') else 'Round9 route auxiliary TabM fusion'
    result = dict(competition=COMPETITION, stage=stage, folds=5, seed=42,
        input_sha256=input_hashes, external_sha256=meta['old_lgb']['external_sha256'],
        model_description=label, script_sha256=sha(Path(__file__)),
        baseline=bs, baseline_reconstruction_max_difference=errors,
        model_scores={k: score(predictions[k][0]) for k in ('lgb', 'tabm') if k in predictions},
        matched_lgb_auxiliary=paired(y, predictions['old_lgb'][0], predictions['lgb'][0]),
        coarse_scan=sorted(comparison, key=lambda r: r['auc'], reverse=True), selected_blend=selected,
        selection_stability=dict(auc=float(roc_auc_score(y, chosen_oof)), held_out_blocks=leave_block),
        source_submission_sha256={k: m['submission_sha256'] for k, m in meta.items()},
        validation_note='At most nine candidates declared before training. Full supervised OOF; reused auxiliary covariates are transductive. Historical blend mixes three/five-fold members. Leave-block selection does not retrain base models and is not nested CV; repeated exploration introduces selection bias.')
    if selected:
        p, pt = candidates[selected['candidate']]
        result['paired_against_previous_best'] = paired(y, baseline[0], p)
        out = ROOT / 'outputs-round9'
        out.mkdir(exist_ok=True)
        sub = sample.copy()
        sub.satisfaction = pt
        validate_submission(sub, sample)
        sub.to_csv(out / 'submission.csv', index=False)
        pd.DataFrame(dict(id=tr.id, satisfaction=y, prediction=p)).to_csv(out / 'oof.csv', index=False)
        result['submission_sha256'] = sha(out / 'submission.csv')
        (out / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (ROOT / 'analyses' / ('round9-' + stage + '-comparison.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['lgb', 'all'], default='lgb')
    run(p.parse_args().stage)
