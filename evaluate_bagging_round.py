"""Fixed-weight seed pooling and matched CatBoost comparisons.

The seed candidates change only which neural RNG predictions are averaged.
All existing fusion weights stay fixed. CatBoost candidates use three coarse
probability weights against round6, rather than a continuous weight search.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from context_round import ROOT, sha
from evaluate_context import paired, pairs
from train import COMPETITION, read_data, validate_submission

FOLDERS = {
    'compact': 'experiments/compact',
    'route': 'experiments/route-auc',
    'cat_old': 'experiments/cat',
    'tuned': 'experiments/route-tuned-3fold',
    'plain42': 'experiments/realmlp-5fold-8ens',
    'context42': 'experiments/realmlp-context-5fold-8ens',
    'lgb_context': 'experiments/context-conditional-5fold',
    'round5': 'outputs-round5',
    'baseline': 'outputs-round6',
    'plain142': 'experiments/realmlp-plain-seed142-5fold-8ens',
    'context142': 'experiments/realmlp-context-seed142-5fold-8ens',
    'cat_control': 'experiments/cat-control-5fold',
    'cat_conditional': 'experiments/cat-conditional-5fold',
}


def run(stage, output):
    tr, ts, sample, raw, ys = read_data(ROOT/'data'); y = ys.to_numpy()
    hashes = {n: sha(ROOT/'data'/n) for n in ('train.csv', 'test.csv', 'sample_submission.csv')}
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, ys))
    predictions = {}; metadata = {}
    for name, relative in FOLDERS.items():
        if stage == 'plain' and name in ('context142', 'cat_control', 'cat_conditional'):
            continue
        if stage == 'seeds' and name in ('cat_control', 'cat_conditional'):
            continue
        folder = ROOT/relative
        m = json.loads((folder/'metrics.json').read_text())
        if m['competition'] != COMPETITION or m['input_sha256'] != hashes:
            raise ValueError(f'Input fingerprints differ: {name}')
        o = pd.read_csv(folder/'oof.csv'); s = pd.read_csv(folder/'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction, y):
            raise ValueError(f'OOF alignment differs: {name}')
        validate_submission(s, sample)
        if sha(folder/'submission.csv') != m['submission_sha256']:
            raise ValueError(f'Output fingerprint differs: {name}')
        p, pt = o.prediction.to_numpy(), s.satisfaction.to_numpy()
        if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise ValueError(f'Invalid OOF probabilities: {name}')
        if name in ('plain42', 'context42', 'plain142', 'context142', 'cat_control', 'cat_conditional'):
            for f, (_, b) in enumerate(splits, 1):
                with np.load(folder/f'fold-{f}.npz') as saved:
                    if not np.array_equal(saved['val_idx'], b):
                        raise ValueError(f'Fold mismatch: {name}, {f}')
                    if not np.allclose(saved['val_pred'], p[b], rtol=0, atol=1e-12):
                        raise ValueError(f'Fold predictions mismatch: {name}, {f}')
        predictions[name] = (p, pt); metadata[name] = m
        del o, s

    def score(p):
        return {'auc': float(roc_auc_score(y, p)),
                'block_auc': [float(roc_auc_score(y[b], p[b])) for _, b in splits]}

    def mean(*members):
        return tuple(np.mean([member[i] for member in members], axis=0) for i in range(2))

    def mix(a, b, w):
        return tuple((1-w)*a[i] + w*b[i] for i in range(2))

    def logit_pair(a, b):
        return tuple(expit(.5*logit(np.clip(a[i], 1e-6, 1-1e-6)) +
                           .5*logit(np.clip(b[i], 1e-6, 1-1e-6))) for i in range(2))

    tree = mean(*(predictions[n] for n in ('compact', 'route', 'cat_old')))

    def round5(plain, tree_member=tree):
        return mix(logit_pair(tree_member, plain), predictions['tuned'], .1)

    def round6(plain, context, tree_member=tree):
        return mix(round5(plain, tree_member), logit_pair(predictions['lgb_context'], context), .5)

    old5 = round5(predictions['plain42'])
    old6 = round6(predictions['plain42'], predictions['context42'])
    reconstruction = {}
    for name, rebuilt in (('round5', old5), ('baseline', old6)):
        errors = [float(np.max(abs(rebuilt[i]-predictions[name][i]))) for i in range(2)]
        if max(errors) > 1e-12:
            raise ValueError(f'Fusion reconstruction differs: {name}, {errors}')
        reconstruction[name] = {'max_oof_difference': errors[0], 'max_test_difference': errors[1]}

    base, base_test = predictions['baseline']; base_score = score(base)
    plain_pool = mean(predictions['plain42'], predictions['plain142'])
    pooled = {'plain_pool': plain_pool}
    candidates = {'pool_plain_only': round6(plain_pool, predictions['context42'])}
    if stage != 'plain':
        context_pool = mean(predictions['context42'], predictions['context142'])
        pooled['context_pool'] = context_pool
        candidates['pool_context_only'] = round6(predictions['plain42'], context_pool)
        candidates['pool_both'] = round6(plain_pool, context_pool)
    if stage == 'all':
        for name in ('cat_control', 'cat_conditional'):
            replacement_tree = mean(predictions['compact'], predictions['route'], predictions[name])
            candidates[f'replace_{name}'] = round6(predictions['plain42'], predictions['context42'], replacement_tree)
            for w in (.1, .25, .5):
                candidates[f'{name}-weight{w}'] = mix(predictions['baseline'], predictions[name], w)

    individual = {name: score(p[0]) for name, p in predictions.items()}
    individual.update({name: score(p[0]) for name, p in pooled.items()})
    matched = {}
    for name, control in (('plain142', 'plain42'), ('context142', 'context42'),
                          ('plain_pool', 'plain42'), ('context_pool', 'context42'),
                          ('cat_conditional', 'cat_control')):
        lookup = {**predictions, **pooled}
        if name in lookup and control in lookup:
            matched[name] = {'control': control,
                             'paired': paired(y, lookup[control][0], lookup[name][0]),
                             'fold_deltas': [a-b for a,b in zip(individual[name]['block_auc'],
                                                              individual[control]['block_auc'])]}
    scanned = []
    for name, (p, _) in candidates.items():
        result = score(p)
        deltas = [a-b for a,b in zip(result['block_auc'], base_score['block_auc'])]
        scanned.append({'candidate': name, **result, 'delta': result['auc']-base_score['auc'],
                        'fold_deltas': deltas, 'all_blocks_positive': bool(min(deltas)>0),
                        'correlation_with_baseline': float(np.corrcoef(base,p)[0,1]),
                        'mean_absolute_prediction_change': float(np.mean(abs(base-p)))})
    eligible = [r for r in scanned if r['all_blocks_positive'] and r['delta']>0]
    selected = max(eligible, key=lambda r:r['auc']) if eligible else None
    meta_oof = base.copy(); choices = []
    for f, (_, b) in enumerate(splits):
        pool = [r for r in scanned if min(d for i,d in enumerate(r['fold_deltas']) if i != f)>0]
        choice = max(pool, key=lambda r:np.mean([d for i,d in enumerate(r['fold_deltas']) if i != f])) if pool else None
        if choice:
            meta_oof[b] = candidates[choice['candidate']][0][b]
            choices.append({'held_out_block': f+1, 'candidate': choice['candidate'],
                            'held_out_delta': choice['fold_deltas'][f]})
        else:
            choices.append({'held_out_block': f+1, 'candidate': 'baseline', 'held_out_delta': 0.})
    result = {'competition': COMPETITION, 'folds': 5, 'seed': 42, 'stage': stage,
              'input_sha256': hashes, 'external_sha256': metadata['baseline']['external_sha256'],
              'script_sha256': sha(Path(__file__)), 'fusion_reconstruction': reconstruction,
              'baseline': base_score, 'individual': individual, 'matched_comparisons': matched,
              'neural_seed_correlations': {name:float(np.corrcoef(predictions[name][0],predictions[control][0])[0,1])
                                           for name,control in (('plain142','plain42'),('context142','context42'))
                                           if name in predictions},
              'coarse_scan': sorted(scanned, key=lambda r:r['auc'], reverse=True),
              'selected_blend': selected,
              'selection_holdout_check': {**score(meta_oof), 'choices': choices,
                                          'paired_against_base': paired(y, base, meta_oof)},
              'source_submission_sha256': {name:m['submission_sha256'] for name,m in metadata.items()},
              'validation_note': 'All new individual models share exact five outer folds, each row excluded from member fitting. Historical baseline includes three/five-fold members. Eleven fixed candidates at most: three seed pools preserving existing fusion weights, two replacements of the historical CatBoost member preserving fusion weights, six coarse CatBoost probability mixtures. The two CatBoost replacements were added after the seed-only analysis and first control fold, before any complete fivefold CatBoost result; this is sequential adaptive experimentation, not a preregistered independent test. Selection and leave-one-block checks reuse saved OOF; neither is nested independent CV. Fixed-prediction intervals do not correct selection bias or fold dependence. Team membership must be confirmed before official submission.'}
    if selected:
        p, pt = candidates[selected['candidate']]
        result['paired_against_previous_best'] = paired(y, base, p)
        result['prediction_correlation'] = float(np.corrcoef(base, p)[0,1])
        result['mean_absolute_prediction_change'] = float(np.mean(abs(base-p)))
        result['pair_changes'] = {name:pairs(y,base,p,tr[name]) for name in ('Class', 'Type of Travel')}
        if output:
            out = output.resolve(); out.mkdir(exist_ok=True)
            sub = sample.copy(); sub.satisfaction = pt; validate_submission(sub,sample)
            sub.to_csv(out/'submission.csv', index=False)
            pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':p}).to_csv(out/'oof.csv', index=False)
            result['model_description'] = 'Independent-seed neural bagging and CatBoost comparison: '+selected['candidate']
            result['submission_sha256'] = sha(out/'submission.csv')
            (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    analyses = ROOT/'analyses'; analyses.mkdir(exist_ok=True)
    (analyses/f'bagging-{stage}-comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('baseline', 'selected_blend', 'coarse_scan', 'matched_comparisons', 'selection_holdout_check')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['plain','seeds','all'], default='all')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); run(args.stage, args.output)
