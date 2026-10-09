"""Prospectively frozen eight-member fusion; H1 used only after dev selection.

Historical five-fold production OOF is an adaptive compatibility check. Meta CV
on overlapping base fits is not nested CV. No third-party predictions are used.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/round18-joint'
ANALYSIS = ROOT / 'analyses/round18-joint'
MEMBERS = {
    'plain_nn': 'realmlp-5fold-8ens',
    'context': 'realmlp-context-5fold-8ens',
    'aux': 'realmlp-route-auxiliary-all-5fold-8ens',
    'lgb': 'round9-lgb-auxiliary-5fold',
    'tabm': 'round9-tabm-auxiliary-5fold',
    'travel': 'round11-experts-5fold',
    'compact_auc': 'compact-auc-5fold',
    'cat_wide': 'cat-wide-conditional-5fold',
}
METHODS = ('logistic', 'positive_logloss', 'positive_pairwise')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''): h.update(b)
    return h.hexdigest()


def save(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def load(p): return json.loads(Path(p).read_text('utf-8'))
def msg(**data): print(json.dumps(data, ensure_ascii=False), flush=True)
def idxsha(x): return hashlib.sha256(np.asarray(x, dtype='<i8').tobytes()).hexdigest()


def freeze():
    if (OUT / 'protocol.json').exists(): raise ValueError('Protocol already frozen')
    p = load(ROOT / 'experiments/round16-locked/protocol.json')
    protected = {k: sha(ROOT / k) for k in ['best-submission.json', 'last-submission.json',
                    'outputs-round13/submission.csv', 'team-request.json']}
    native = {}
    for n, d in MEMBERS.items():
        folder = ROOT / 'experiments' / d
        native[n] = {str(f.relative_to(ROOT)).replace('\\', '/'): sha(f)
                     for f in [folder / 'configuration.json', folder / 'metrics.json',
                               folder / 'oof.csv', folder / 'submission.csv',
                               *sorted(folder.glob('fold-*.npz'))]}
    cfg = dict(round=18, created_at_utc=datetime.now(timezone.utc).isoformat(),
       script_sha256=sha(__file__), members=MEMBERS, methods=list(METHODS),
       input_sha256=p['input_sha256'], split_path=p['split_path'], split_sha256=p['split_sha256'],
       source_protocol_sha256=sha(ROOT / 'experiments/round16-locked/protocol.json'),
       protected_sha256=protected, native_production_sha256=native,
       meta_cv=dict(folds=3, seeds=[42, 142, 242]),
       positive_optimization=dict(l2=1e-4, max_iter=160, weight_bounds=[0., 3.],
            pair_count=2000000, pair_seed=20261006),
       gates=dict(development_mean_delta=1e-5, each_repeat_delta_positive=True,
            worst_development_block_delta=-5e-5, hold1_delta=1e-5,
            production_mean_delta=1e-5, each_production_repeat_delta_positive=True,
            worst_production_block_delta=-5e-5),
       holdout_rule='Select one method by development mean AUC; H1 accessed once only if eligible. H2 not scored.',
       production_rule='Use the same eight inputs and fitting algorithm; refit combiner on all native five-fold OOF. Never copy dev coefficients to a different baseline.',
       limitations='All labels used in earlier research; reused dev fits and meta CV are adaptive and not fully nested. H1 newly excluded from these fits, not pristine data. Production fits have five rather than three folds and greater training coverage.',
       baseline='Exact preserved round13 ensemble retrained on excluded development for H1; preserved full OOF for compatibility.')
    save(OUT / 'protocol.json', cfg); msg(stage='frozen', protocol_sha256=sha(OUT / 'protocol.json'))


def check():
    p = load(OUT / 'protocol.json')
    if p['script_sha256'] != sha(__file__):
        repair = load(ROOT / 'evidence/round18-reader-repair.json')
        if (repair['previous_script_sha256'] != p['script_sha256'] or
            repair['repaired_script_sha256'] != sha(__file__) or
            repair['protocol_sha256'] != sha(OUT / 'protocol.json') or
            not repair['no_label_metrics_computed_before_repair']):
            raise ValueError('Frozen script changed without preregistered reader repair')
    for name, h in p['protected_sha256'].items():
        if sha(ROOT / name) != h: raise ValueError('Protected state changed: ' + name)
    for name, h in p['input_sha256'].items():
        if sha(ROOT / 'data' / name) != h: raise ValueError('Input changed')
    if sha(ROOT / p['split_path']) != p['split_sha256']: raise ValueError('Split changed')
    return p


def train_new():
    from round16_baseline import BaselineFactory
    from cat_wide_round import make_frames
    from round12_models import digest
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    import torch
    p = check()
    tr = pd.read_csv(ROOT / 'data/train.csv'); ts = pd.read_csv(ROOT / 'data/test.csv')
    with np.load(ROOT / p['split_path']) as z:
        dev, hold = z['dev_idx'], np.r_[z['hold1_idx'], z['hold2_idx']]
        folds = z['dev_fold_id']
    target = np.full(len(tr), np.nan); target[dev] = tr.satisfaction.iloc[dev]
    raw = [c for c in tr if c not in ('id', 'satisfaction')]
    tr = tr.drop(columns='satisfaction')
    factory = BaselineFactory(tr, ts, raw, target)
    for fold in range(3):
        a, v = dev[folds != fold], dev[folds == fold]
        b = np.r_[v, hold]
        for member in ['compact_auc', 'cat_wide']:
            f = OUT / f'{member}-fold-{fold+1}.npz'; meta = f.with_suffix('.json')
            recipe = ROOT / 'experiments' / MEMBERS[member] / 'configuration.json'
            config = load(recipe)
            expected = dict(member=member, fold=fold+1, train_idx_sha256=idxsha(a),
                            validation_idx_sha256=idxsha(v), evaluation_idx_sha256=idxsha(b),
                            train_target_sha256=hashlib.sha256(target[a].astype('int8').tobytes()).hexdigest(),
                            validation_target_sha256=hashlib.sha256(target[v].astype('int8').tobytes()).hexdigest(),
                            recipe_sha256=sha(recipe), script_sha256=sha(__file__),
                            protocol_sha256=sha(OUT / 'protocol.json'))
            if f.exists() or meta.exists():
                m = load(meta)
                if any(m[k] != value for k, value in expected.items()) or m['checkpoint_sha256'] != sha(f):
                    raise ValueError('New component checkpoint changed')
                msg(stage='restored', member=member, fold=fold+1); continue
            t = time.time()
            if member == 'compact_auc':
                frames, fingerprints = factory.prepare('compact', a, b, True)
                params = dict(config['params'], random_state=43+fold)
                model = lgb.LGBMClassifier(**params)
                msg(stage='training_new_component', member=member, fold=fold+1, features=len(frames[0].columns), training_rows=len(a))
                model.fit(frames[0], target[a], eval_set=[(frames[1].iloc[:len(v)], target[v])],
                          eval_metric='auc', callbacks=[lgb.early_stopping(180, first_metric_only=True, verbose=False)])
                cats = None
            else:
                masked = pd.Series(np.nan, index=pd.RangeIndex(len(tr)))
                masked.iloc[a] = target[a]
                frames = make_frames(factory.base, factory.wide_keys, factory.ckeys, a, b, len(tr), masked)
                fingerprints = [digest(x) for x in frames]
                del frames; gc.collect()
                changed = masked.copy(); other = np.setdiff1d(np.arange(len(tr)), a)
                changed.iloc[other] = np.arange(len(other)) % 2
                probe = make_frames(factory.base, factory.wide_keys, factory.ckeys, a, b, len(tr), changed)
                if [digest(x) for x in probe] != fingerprints: raise ValueError('Evaluation labels influenced cat features')
                del probe, changed; gc.collect()
                frames = make_frames(factory.base, factory.wide_keys, factory.ckeys, a, b, len(tr), masked)
                cats = list(frames[0].select_dtypes(include='category'))
                for x in frames:
                    for c in cats: x[c] = x[c].cat.codes.astype('int32')
                    x.fillna(-999, inplace=True)
                params = dict(config['params'], random_seed=43+fold, verbose=False)
                model = CatBoostClassifier(**params)
                msg(stage='training_new_component', member=member, fold=fold+1, features=len(frames[0].columns), training_rows=len(a))
                model.fit(frames[0], target[a], cat_features=cats,
                          eval_set=(frames[1].iloc[:len(v)], target[v]), early_stopping_rounds=200)
            msg(stage='predicting_new_component', member=member, fold=fold+1)
            pb, pt = model.predict_proba(frames[1])[:, 1], model.predict_proba(frames[2])[:, 1]
            for x, n in [(pb, len(b)), (pt, len(ts))]: probability(x, n)
            np.savez_compressed(f, evaluation_idx=b, validation_idx=v, evaluation_pred=pb, test_pred=pt)
            info = dict(**expected, params=params, feature_fingerprints=fingerprints,
                        label_perturbation='PASS', holdout_labels_used=False,
                        supervised_scope='Only a/v in development; H1/H2 masked and excluded from early stopping.',
                        checkpoint_sha256=sha(f), seconds=round(time.time()-t, 2),
                        iterations=int(model.best_iteration_) if member == 'compact_auc' else int(model.get_best_iteration()+1))
            save(meta, info); msg(stage='component_done', member=member, fold=fold+1, seconds=info['seconds'])
            del frames, model; gc.collect(); torch.cuda.empty_cache()


def probability(x, n):
    if x.shape != (n,) or not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError('Bad probabilities')


def dev_library(p):
    with np.load(ROOT / p['split_path']) as z:
        dev, hold = z['dev_idx'], np.r_[z['hold1_idx'], z['hold2_idx']]
        fold_id = z['dev_fold_id']
    X, H = np.full((len(dev), len(MEMBERS)), np.nan), np.zeros((len(hold), len(MEMBERS)))
    audit = {}
    base = ROOT / 'experiments/round16-locked/baseline_current'
    for col, name in enumerate(MEMBERS):
        for fold in range(3):
            a, v = dev[fold_id != fold], dev[fold_id == fold]
            native = base / f'components-fold-{fold+1}' / f'{name}.npz' if name in list(MEMBERS)[:6] else OUT / f'{name}-fold-{fold+1}.npz'
            m = load(native.with_suffix('.json'))
            expected_label_check = 'PASS' if fold == 0 or name in ['compact_auc', 'cat_wide'] else 'not_run'
            if m['checkpoint_sha256'] != sha(native) or m['label_perturbation'] != expected_label_check:
                raise ValueError('Checkpoint integrity: '+name)
            if m['train_idx_sha256'] != idxsha(a): raise ValueError('Checkpoint training rows')
            with np.load(native, allow_pickle=False) as z:
                if not np.array_equal(z['validation_idx'], v) or not np.array_equal(z['evaluation_idx'], np.r_[v, hold]):
                    raise ValueError('Checkpoint evaluation rows')
                pred = z['evaluation_pred']; probability(pred, len(v)+len(hold))
                X[fold_id == fold, col] = pred[:len(v)]; H[:, col] += pred[len(v):] / 3.
            audit[str(native.relative_to(ROOT)).replace('\\', '/')] = sha(native)
    if not np.isfinite(X).all(): raise ValueError('Missing development prediction')
    with np.load(base / 'summary.npz') as z: control, hc = z['dev_oof'], z['hold_pred']
    np.savez_compressed(OUT / 'library.npz', dev_idx=dev, hold_idx=hold, X=X, H=H, control=control, hold_control=hc)
    save(OUT / 'library-audit.json', dict(native_sha256=audit, holdout_scored=False,
        note='H1/H2 exclusion verified by prior independent audit and new index/target/perturbation checks.'))
    return dev, X, H, control, hc


def fit(method, x, y, p):
    # Every development/production fit follows this identical procedure.
    q = p['positive_optimization']; x = np.asarray(x, dtype=np.float64)
    if method == 'logistic':
        clf = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, random_state=42).fit(x, y)
        if np.max(clf.n_iter_) >= 1000: raise ValueError('LR did not converge')
        return clf.coef_[0], float(clf.intercept_[0])
    if method == 'positive_pairwise':
        rng = np.random.default_rng(q['pair_seed'])
        xp, xn = x[y == 1], x[y == 0]
        d = xp[rng.integers(len(xp), size=q['pair_count'])] - xn[rng.integers(len(xn), size=q['pair_count'])]
        def objective(w):
            margin = d @ w
            return (float(np.logaddexp(0., -margin).mean() + q['l2'] * np.dot(w, w)),
                    -d.T @ expit(-margin) / len(d) + 2 * q['l2'] * w)
        r = minimize(objective, np.full(x.shape[1], 1./x.shape[1]), jac=True, method='L-BFGS-B',
                     bounds=[tuple(q['weight_bounds'])]*x.shape[1], options={'maxiter':q['max_iter'], 'ftol':1e-11})
        if not r.success: raise ValueError('Pairwise optimizer did not converge: '+r.message)
        return r.x, 0.
    def objective(w):
        s = x @ w[:-1] + w[-1]
        error = expit(s) - y
        return (float((np.logaddexp(0., s)-y*s).mean()+q['l2']*np.dot(w[:-1], w[:-1])),
                np.r_[x.T @ error / len(y)+2*q['l2']*w[:-1], error.mean()])
    r = minimize(objective, np.r_[np.full(x.shape[1], 1./x.shape[1]), 0.], jac=True, method='L-BFGS-B',
                 bounds=[tuple(q['weight_bounds'])]*x.shape[1]+[(None, None)], options={'maxiter':q['max_iter'], 'ftol':1e-11})
    if not r.success: raise ValueError('Logloss optimizer did not converge: '+r.message)
    return r.x[:-1], float(r.x[-1])


def crossfit(method, X, y, control, p, folds):
    repeats = []
    for seed in p['meta_cv']['seeds']:
        pred = np.full(len(y), np.nan); weights = []; blocks = []
        for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
            w, intercept = fit(method, X[a], y[a], p)
            pred[b] = expit(X[b] @ w + intercept)
            weights.append(dict(coefficients=w.tolist(), intercept=intercept))
            blocks.append(float(roc_auc_score(y[b], pred[b])-roc_auc_score(y[b], control[b])))
        delta = float(roc_auc_score(y, pred)-roc_auc_score(y, control))
        repeats.append(dict(seed=seed, auc=float(roc_auc_score(y, pred)), delta=delta, block_deltas=blocks, weights=weights))
        msg(stage='meta_repeat', method=method, seed=seed, delta=delta, worst_block_delta=min(blocks))
    return repeats


def develop():
    p = check()
    if (ANALYSIS / 'development.json').exists(): raise ValueError('Development already evaluated')
    dev, probabilities, H, control, hc = dev_library(p)
    y = pd.read_csv(ROOT / 'data/train.csv', usecols=['satisfaction']).satisfaction.iloc[dev].to_numpy(dtype='int8')
    X = logit(np.clip(probabilities, 1e-6, 1-1e-6))
    scores = []
    for method in METHODS:
        r = crossfit(method, X, y, control, p, 3)
        mean = float(np.mean([k['delta'] for k in r])); worst = min(min(k['block_deltas']) for k in r)
        eligible = mean >= p['gates']['development_mean_delta'] and min(k['delta'] for k in r) > 0 and worst >= p['gates']['worst_development_block_delta']
        scores.append(dict(method=method, mean_delta=mean, worst_block_delta=worst, eligible=eligible, repeats=r))
    qualified = [s for s in scores if s['eligible']]
    selected = max(qualified, key=lambda s:s['mean_delta'])['method'] if qualified else None
    result = dict(control_auc=float(roc_auc_score(y, control)), candidates=scores, selected=selected,
                  protocol_sha256=sha(OUT / 'protocol.json'), hold1_scored=False, hold2_scored=False)
    if selected:
        w, intercept = fit(selected, X, y, p)
        result['selected_fit'] = dict(coefficients=w.tolist(), intercept=intercept)
        np.save(OUT / 'selected-hold-pred.npy', expit(logit(np.clip(H, 1e-6, 1-1e-6)) @ w + intercept))
    save(ANALYSIS / 'development.json', result); msg(stage='development_complete', selected=selected)


def held():
    from round16_evaluate import paired
    p = check(); d = load(ANALYSIS / 'development.json')
    if not d['selected']: raise ValueError('No development selection')
    receipt = ANALYSIS / 'hold1-access.json'
    if receipt.exists(): raise ValueError('H1 access already recorded; do not repeat')
    save(receipt, dict(started_at_utc=datetime.now(timezone.utc).isoformat(), selected=d['selected'],
                      protocol_sha256=sha(OUT / 'protocol.json'), development_sha256=sha(ANALYSIS / 'development.json')))
    with np.load(ROOT / p['split_path']) as z: h1 = z['hold1_idx']
    with np.load(OUT / 'library.npz') as z:
        if not np.array_equal(z['hold_idx'][:len(h1)], h1): raise ValueError('Held order')
        old = z['hold_control'][:len(h1)]
    new = np.load(OUT / 'selected-hold-pred.npy')[:len(h1)]
    y = pd.read_csv(ROOT / 'data/train.csv', usecols=['satisfaction']).satisfaction.iloc[h1].to_numpy(dtype='int8')
    result = paired(y, old, new)
    result.update(method=d['selected'], baseline_auc=float(roc_auc_score(y, old)),
                  candidate_auc=float(roc_auc_score(y, new)), eligible=result['delta']>=p['gates']['hold1_delta'],
                  hold1_accessed_once=True, hold2_scored=False, label_scope=p['limitations'])
    save(ANALYSIS / 'hold1.json', result); msg(stage='hold1_complete', **result)


def production():
    p = check(); held_result = load(ANALYSIS / 'hold1.json')
    if not held_result['eligible']: raise ValueError('Holdout failed')
    if (ANALYSIS / 'production.json').exists(): raise ValueError('Production already evaluated')
    tr = pd.read_csv(ROOT / 'data/train.csv'); sample = pd.read_csv(ROOT / 'data/sample_submission.csv')
    y = tr.satisfaction.to_numpy(dtype='int8')
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
    P, T = np.full((len(tr), len(MEMBERS)), np.nan), np.zeros((len(sample), len(MEMBERS)))
    audit = {}
    for col, (name, directory) in enumerate(MEMBERS.items()):
        folder = ROOT / 'experiments' / directory
        for file, h in p['native_production_sha256'][name].items():
            if sha(ROOT / file) != h: raise ValueError('Native production file changed: '+file)
        c, metrics = load(folder / 'configuration.json'), load(folder / 'metrics.json')
        if c['folds'] != 5 or c.get('seed', c.get('split_seed')) != 42: raise ValueError('Different five-fold recipe')
        if metrics['input_sha256'] != p['input_sha256']: raise ValueError('Native input mismatch')
        for fold, (_, b) in enumerate(splits, 1):
            with np.load(folder / f'fold-{fold}.npz', allow_pickle=False) as z:
                if not np.array_equal(b, z['val_idx']): raise ValueError('Different native folds')
                probability(z['val_pred'], len(b)); probability(z['test_pred'], len(sample))
                P[b, col] = z['val_pred']; T[:, col] += z['test_pred']/5.
        saved = pd.read_csv(folder / 'oof.csv'); sub = pd.read_csv(folder / 'submission.csv')
        if not np.array_equal(saved.id, tr.id) or not np.array_equal(sub.id, sample.id): raise ValueError('Native ID mismatch')
        error = max(float(np.max(np.abs(saved.prediction-P[:, col]))), float(np.max(np.abs(sub.satisfaction-T[:, col]))))
        if error > 1e-7: raise ValueError('Native OOF/submission reconstruction mismatch')
        audit[name] = dict(folds_aligned=True, reconstruction_max_difference=error)
    old_frame = pd.read_csv(ROOT / 'outputs-round13/oof.csv')
    if not np.array_equal(old_frame.id, tr.id): raise ValueError('Baseline IDs')
    old = old_frame.prediction.to_numpy()
    X, Xt = logit(np.clip(P, 1e-6, 1-1e-6)), logit(np.clip(T, 1e-6, 1-1e-6))
    method = held_result['method']; repeats = crossfit(method, X, y, old, p, 5)
    mean = float(np.mean([r['delta'] for r in repeats])); worst = min(min(r['block_deltas']) for r in repeats)
    eligible = mean >= p['gates']['production_mean_delta'] and min(r['delta'] for r in repeats) > 0 and worst >= p['gates']['worst_production_block_delta']
    w, intercept = fit(method, X, y, p)
    test = expit(Xt @ w + intercept)
    sub = sample.copy(); sub.satisfaction = test
    from train import validate_submission
    validate_submission(sub, sample)
    sub.to_csv(OUT / 'production-submission.csv', index=False)
    result = dict(method=method, native_audit=audit, baseline_auc=float(roc_auc_score(y, old)),
                  mean_delta=mean, worst_block_delta=worst, repeats=repeats, eligible=eligible,
                  final_refit=dict(coefficients=w.tolist(), intercept=intercept),
                  submission_sha256=sha(OUT / 'production-submission.csv'),
                  validation_note='Adaptive historical OOF compatibility, not independent or fully nested CV. Same fitting procedure, coefficients refit for native fivefold inputs.')
    save(ANALYSIS / 'production.json', result)
    if eligible:
        out = ROOT / 'outputs-round18'; out.mkdir(exist_ok=False)
        (out / 'submission.csv').write_bytes((OUT / 'production-submission.csv').read_bytes())
        save(out / 'metrics.json', dict(competition='playground-series-s6e10', input_sha256=p['input_sha256'],
              submission_sha256=result['submission_sha256'], selected_blend={'auc':float(np.mean([r['auc'] for r in repeats]))},
              validation_note=result['validation_note'], method=method, source='analyses/round18-joint/production.json'))
    msg(stage='production_complete', mean_delta=mean, worst_block_delta=worst, eligible=eligible)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage', choices=['freeze','train','dev','held','production'])
    args = cli.parse_args()
    with threadpool_limits(limits=4):
        {'freeze':freeze,'train':train_new,'dev':develop,'held':held,'production':production}[args.stage]()
