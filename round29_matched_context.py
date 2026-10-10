"""Matched tenfold conditional pair with inner-only checkpoint selection.

The round13 auxiliary tenfold member and all other old components are retained.
Only the conditional tree/neural pair is replaced or half-replaced. Identical
formulas and fitted component packets create OOF and test predictions. This
remains adaptive validation; historical labels cannot become pristine again.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata as md
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from round16_baseline import BaselineFactory
from round18_joint import ROOT, sha, save, load, idxsha
from round12_models import rebuilt
from train import COMPETITION, read_data, validate_submission

OUT = ROOT / 'experiments/round29-matched-context'
ANALYSIS = ROOT / 'analyses/round29-matched-context'
FINAL = ROOT / 'outputs-round29'
SOURCES = dict(best='outputs-round13', old5='outputs-round5',
    lgb='experiments/round9-lgb-auxiliary-5fold',
    context='experiments/realmlp-context-5fold-8ens',
    aux='experiments/round13-realmlp-route-auxiliary-10fold-8ens',
    tabm='experiments/round9-tabm-auxiliary-5fold',
    travel='experiments/round11-experts-5fold')


def msg(**value):
    from round22_preserved import msg as emit
    emit(**value)


def freeze():
    if (OUT / 'protocol.json').exists():
        raise ValueError('Already frozen')
    from round16_baseline import REFERENCES, REFERENCE_SCRIPTS
    from improve import CACHE
    from auxiliary_round import AUXDIR
    from auxiliary_neural_round import TEACHERDIR
    dependencies = {'round16_baseline.py', 'round12_models.py',
        'round18_joint.py', 'train.py', 'auxiliary_round.py',
        'auxiliary_neural_round.py', 'nn_round.py', 'context_round.py',
        *REFERENCES.values(), *REFERENCE_SCRIPTS.values()}
    dependencies.update(str(f.relative_to(ROOT)).replace('\\', '/') for f in
        [CACHE, AUXDIR / 'features.pkl', AUXDIR / 'metrics.json',
         TEACHERDIR / 'probability.npy', TEACHERDIR / 'metrics.json',
         ROOT / 'data-original/data.csv'])
    for directory in SOURCES.values():
        dependencies.update(directory + '/' + name for name in
                            ['oof.csv', 'submission.csv', 'metrics.json'])
    p = dict(round=29, created_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__),
        source_sha256={n: sha(ROOT / n) for n in sorted(dependencies)},
        input_sha256=load(ROOT / 'outputs-round13/metrics.json')['input_sha256'],
        protected_sha256={n: sha(ROOT / n) for n in
            ['best-submission.json', 'last-submission.json', 'team-request.json',
             'outputs-round13/submission.csv', 'outputs-round13/oof.csv']},
        split=dict(folds=10, seed=42, inner_rows=10000,
                   inner_seed=2026101029, model_seed=2026101029),
        new_members=['lgb', 'context'], preserve_auxiliary_member=SOURCES['aux'],
        replacement_fractions=[0.5, 1.0],
        gates=dict(full_oof_delta=0.00001, minimum_positive_blocks=6,
                   worst_block_floor=-0.00005,
                   leave_block_selection_delta_floor=0.000005),
        max_official_submissions=1,
        production_rule='Every new fold has the same source recipe and '
            'inner-only validation. Average the ten corresponding test '
            'packets. Use the exact same rebuilt formula and replacement '
            'fraction for OOF and test. No copied coefficients.',
        limitations='The best anchor retains historical mixed folds. '
            'Only the two new members share ten native folds; the whole '
            'ensemble is not unified tenfold CV. Historical labels have '
            'already been studied. Leave-block selection does not retrain '
            'all base models and is not independent nested validation.',
        versions={n: md.version(n) for n in ['torch', 'pytabkit', 'lightgbm',
                                            'numpy', 'scikit-learn']})
    save(OUT / 'protocol.json', p)
    msg(stage='frozen', round=29, protocol_sha256=sha(OUT / 'protocol.json'))


def check():
    p = load(OUT / 'protocol.json')
    if sha(__file__) != p['script_sha256']:
        raise ValueError('Frozen source changed')
    for name, expected in {**p['source_sha256'], **p['protected_sha256']}.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Source or protected artifact changed: ' + name)
    for name, expected in p['input_sha256'].items():
        if sha(ROOT / 'data' / name) != expected:
            raise ValueError('Input changed')
    return p


def train():
    p = check()
    tr, ts, sample, raw, y = read_data(ROOT / 'data')
    factory = BaselineFactory(tr, ts, raw, y)
    save(OUT / 'factory-source.json', factory.source_info)
    started = time.monotonic()
    all_splits = list(StratifiedKFold(10, shuffle=True, random_state=42).split(tr, y))
    for fold, (a, b) in enumerate(all_splits, 1):
        g, inner = train_test_split(a, test_size=p['split']['inner_rows'],
            stratify=y.iloc[a], random_state=p['split']['inner_seed'] + fold)
        evaluation = np.concatenate([b, inner])
        packet_dir = OUT / f'fold-{fold}'
        packet_dir.mkdir(exist_ok=True)
        split = dict(fold=fold, outer_train_idx_sha256=idxsha(a),
            gradient_train_idx_sha256=idxsha(g),
            outer_val_idx_sha256=idxsha(b), inner_val_idx_sha256=idxsha(inner),
            train_rows=len(g), outer_rows=len(b), inner_rows=len(inner),
            outer_labels_used_for_stopping=False)
        split_path = packet_dir / 'split.json'
        if split_path.exists() and load(split_path) != split:
            raise ValueError('Split changed')
        save(split_path, split)
        for member in p['new_members']:
            pb, pt, info = factory.fit_component(member, g, evaluation,
                seed=p['split']['model_seed'] + fold, validation_idx=inner,
                check_label_invariance=(fold == 1), cache_dir=packet_dir)
            path = packet_dir / (member + '.npz')
            msg(stage='fresh_component_complete', member=member, fold=fold,
                train_rows=len(g), outer_rows=len(b), inner_rows=len(inner),
                seconds=time.monotonic() - started,
                checkpoint_sha256=sha(path), outer_labels_used_for_stopping=False)
        gc.collect()
    for member in p['new_members']:
        oof = np.full(len(tr), np.nan)
        test = np.zeros(len(ts))
        audit = []
        for fold, (_, b) in enumerate(all_splits, 1):
            folder = OUT / f'fold-{fold}'
            info = load(folder / (member + '.json'))
            if sha(folder / (member + '.npz')) != info['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            with np.load(folder / (member + '.npz')) as z:
                if not np.array_equal(z['evaluation_idx'][:len(b)], b):
                    raise ValueError('Outer index order changed')
                if np.intersect1d(b, z['validation_idx']).size:
                    raise ValueError('Outer labels leaked into stopping')
                oof[b] = z['evaluation_pred'][:len(b)]
                test += z['test_pred'] / 10
            audit.append(dict(fold=fold, checkpoint_sha256=info['checkpoint_sha256'],
                              outer_auc=float(roc_auc_score(y.iloc[b], oof[b]))))
        if not np.isfinite(oof).all():
            raise ValueError('Incomplete OOF')
        folder = OUT / member
        folder.mkdir(exist_ok=True)
        sub = sample.copy()
        sub.satisfaction = test
        validate_submission(sub, sample)
        sub.to_csv(folder / 'submission.csv', index=False)
        pd.DataFrame(dict(id=tr.id, satisfaction=y, prediction=oof)).to_csv(
            folder / 'oof.csv', index=False)
        save(folder / 'metrics.json', dict(competition=COMPETITION, folds=10,
            input_sha256=p['input_sha256'], selected_blend=dict(auc=float(roc_auc_score(y, oof))),
            submission_sha256=sha(folder / 'submission.csv'),
            oof_sha256=sha(folder / 'oof.csv'), fold_scores=audit,
            early_stopping_scope='10k inner validation only'))
    msg(stage='all_training_complete', seconds=time.monotonic() - started)


def evaluate():
    p = check()
    if (ANALYSIS / 'comparison.json').exists():
        raise ValueError('Already evaluated')
    tr, ts, sample, raw, ys = read_data(ROOT / 'data')
    y = ys.to_numpy()
    preds = {}
    for name, directory in {**SOURCES,
        'new_lgb': str((OUT / 'lgb').relative_to(ROOT)),
        'new_context': str((OUT / 'context').relative_to(ROOT))}.items():
        folder = ROOT / directory
        m = load(folder / 'metrics.json')
        if m['input_sha256'] != p['input_sha256']:
            raise ValueError('Source inputs differ')
        if sha(folder / 'submission.csv') != m['submission_sha256']:
            raise ValueError('Submission fingerprint changed')
        o = pd.read_csv(folder / 'oof.csv')
        s = pd.read_csv(folder / 'submission.csv')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction, y):
            raise ValueError('OOF alignment changed')
        validate_submission(s, sample)
        preds[name] = (o.prediction.to_numpy(), s.satisfaction.to_numpy())
    def reconstruct(tree, neural):
        return tuple(rebuilt(preds[tree][side], preds['old5'][side],
            preds[neural][side], preds['aux'][side], preds['tabm'][side],
            preds['travel'][side]) for side in (0, 1))
    old = reconstruct('lgb', 'context')
    if max(float(np.max(abs(old[i] - preds['best'][i]))) for i in (0, 1)) > 1e-12:
        raise ValueError('Best reconstruction mismatch')
    replacement = reconstruct('new_lgb', 'new_context')
    base = preds['best'][0]
    baseline = float(roc_auc_score(y, base))
    splits = list(StratifiedKFold(10, shuffle=True, random_state=42).split(tr, y))
    candidates, rows = {}, []
    for fraction in p['replacement_fractions']:
        name = f'replace_context_fraction_{fraction}'
        pv, pt = tuple((1 - fraction) * preds['best'][i] + fraction * replacement[i]
                       for i in (0, 1))
        candidates[name] = (pv, pt)
        auc = float(roc_auc_score(y, pv))
        deltas = [float(roc_auc_score(y[b], pv[b]) - roc_auc_score(y[b], base[b]))
                  for _, b in splits]
        rows.append(dict(candidate=name, fraction=fraction, auc=auc,
            delta=auc - baseline, block_deltas=deltas,
            positive_blocks=sum(d > 0 for d in deltas), worst_block_delta=min(deltas)))
    # Select only among predeclared candidates using nine blocks; score the
    # omitted block. This checks selection sensitivity, not independent fits.
    chosen = np.full(len(y), np.nan)
    leave = []
    for fold, (a, b) in enumerate(splits, 1):
        scores = {'baseline': float(roc_auc_score(y[a], base[a]))}
        scores.update({n: float(roc_auc_score(y[a], v[0][a])) for n, v in candidates.items()})
        pick = max(scores, key=scores.get)
        pv = base if pick == 'baseline' else candidates[pick][0]
        chosen[b] = pv[b]
        leave.append(dict(fold=fold, selected=pick,
            delta=float(roc_auc_score(y[b], pv[b]) - roc_auc_score(y[b], base[b]))))
    leave_delta = float(roc_auc_score(y, chosen) - baseline)
    gate = p['gates']
    eligible = [r for r in rows if r['delta'] >= gate['full_oof_delta'] and
        r['positive_blocks'] >= gate['minimum_positive_blocks'] and
        r['worst_block_delta'] >= gate['worst_block_floor'] and
        leave_delta >= gate['leave_block_selection_delta_floor']]
    selected = max(eligible, key=lambda r: r['auc']) if eligible else None
    result = dict(competition=COMPETITION, folds=10, input_sha256=p['input_sha256'],
        baseline_auc=baseline, candidates=rows, selected_blend=selected,
        leave_block_selection_delta=leave_delta, leave_block_selection=leave,
        eligible=selected is not None, gates=gate,
        protocol_sha256=sha(OUT / 'protocol.json'), validation_note=p['limitations'],
        preserved_auxiliary_submission_sha256=sha(ROOT / SOURCES['aux'] / 'submission.csv'),
        external_sha256=load(ROOT / 'outputs-round13/metrics.json')['external_sha256'],
        model_description='Inner-validated tenfold conditional pair preserving R13 auxiliary')
    if selected:
        FINAL.mkdir(exist_ok=True)
        pv, pt = candidates[selected['candidate']]
        sub = sample.copy()
        sub.satisfaction = pt
        validate_submission(sub, sample)
        sub.to_csv(FINAL / 'submission.csv', index=False)
        pd.DataFrame(dict(id=tr.id, satisfaction=y, prediction=pv)).to_csv(
            FINAL / 'oof.csv', index=False)
        result['submission_sha256'] = sha(FINAL / 'submission.csv')
        save(FINAL / 'metrics.json', result)
    save(ANALYSIS / 'comparison.json', result)
    msg(stage='evaluated', eligible=selected is not None, candidates=rows,
        leave_block_selection_delta=leave_delta, selected=selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'train', 'evaluate'])
    args = parser.parse_args()
    {'freeze': freeze, 'train': train, 'evaluate': evaluate}[args.stage]()
