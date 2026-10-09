"""Retrain the complete round13 ensemble on a new development split.

No saved competition OOF or submission prediction is loaded. Frozen covariate
features, unlabeled auxiliary reconstruction, and original-data teacher outputs
are allowed after fingerprint checks. Every competition-target feature is fit
only on ``a``. ``b`` may contain both development validation and locked holdout;
only the explicit ``validation_idx`` subset supplies model validation labels.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from auxiliary_round import AUXDIR, hashes
from auxiliary_neural_round import TEACHERDIR, frames as route_frames, route_keys
from context_round import ROOT, add_context, context_blocks, context_keys, sha
from improve import CACHE, CATS, encoded_fold, make_keys
from nn_round import fold_frames, predict_chunks, prepare_frames, report
from round12_models import digest, rebuilt


warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

COMPONENTS = ('compact', 'route_auc', 'route_tuned', 'cat', 'plain_nn',
              'lgb', 'context', 'aux', 'tabm', 'travel')
REFERENCES = {
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
REFERENCE_SCRIPTS = {
    'compact': 'improve.py', 'cat': 'improve.py', 'route_auc': 'route_auc_round.py',
    'route_tuned': 'route_tuned.py', 'plain_nn': 'nn_round.py', 'context': 'nn_context_round.py',
    'lgb': 'round9_models.py', 'tabm': 'round9_models.py', 'aux': 'round13_tenfold.py',
    'travel': 'divergent_round.py',
}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def index_sha(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Not JSON serializable: ' + type(value).__name__)


def combine_components(predictions):
    """The exact historical blend formula, applied to freshly fit members."""
    result = {}
    for side in (0, 1):
        p = {name: value[side] for name, value in predictions.items()}
        tree = (p['compact'] + p['route_auc'] + p['cat']) / 3.
        calibrated = expit(.5 * logit(np.clip(tree, 1e-6, 1-1e-6))
                           + .5 * logit(np.clip(p['plain_nn'], 1e-6, 1-1e-6)))
        old5 = .9 * calibrated + .1 * p['route_tuned']
        contextual_pair = expit(.5 * logit(np.clip(p['lgb'], 1e-6, 1-1e-6))
                                + .5 * logit(np.clip(p['context'], 1e-6, 1-1e-6)))
        baseline = rebuilt(p['lgb'], old5, p['context'], p['aux'], p['tabm'], p['travel'])
        result[side] = (old5, contextual_pair, baseline)
    return {
        'old5': (result[0][0], result[1][0], {'formula': 'historical round5 weights'}),
        'contextual_pair': (result[0][1], result[1][1], {'formula': 'equal logit auxiliary tree/context neural'}),
        'baseline_current': (result[0][2], result[1][2], {
            'formula': 'historical round13 weights', 'members': ['old5', 'contextual_pair', 'aux', 'tabm', 'travel'],
            'weights': [.30375, .30375, .2025, .09, .1]}),
    }


class BaselineFactory:
    """Feature factory and sequential estimator trainer; constructor never fits.

    ``y`` may have NaN at locked holdout positions. Constructor validates source
    rows and feature fingerprints without consulting cached competition labels.
    ``fit_component`` supports per-component checkpointing by the caller.
    """

    def __init__(self, tr, ts, raw, y):
        self.tr, self.ts, self.raw = tr, ts, list(raw)
        self.n = len(tr)
        self.y = pd.Series(np.asarray(y), index=pd.RangeIndex(self.n))
        if len(self.y) != self.n:
            raise ValueError('Training target length mismatch')
        self.configs = {k: load(ROOT / v) for k, v in REFERENCES.items()}
        ih = hashes()
        am = load(AUXDIR / 'metrics.json')
        tm = load(TEACHERDIR / 'metrics.json')
        if am['configuration']['label_used'] is not False:
            raise ValueError('Auxiliary reconstruction used satisfaction labels')
        if any(value != ih for value in (am['configuration']['input_sha256'],
                                        tm['configuration']['input_sha256'])):
            raise ValueError('Feature cache input mismatch')
        expected_cache = self.configs['lgb']['base_cache_sha256']
        expected_aux = am['features_sha256']
        expected_teacher = tm['probability_sha256']
        expected_external = tm['configuration']['original_sha256']
        actual_sources = {
            str(CACHE.relative_to(ROOT)): sha(CACHE),
            str((AUXDIR / 'features.pkl').relative_to(ROOT)): sha(AUXDIR / 'features.pkl'),
            str((TEACHERDIR / 'probability.npy').relative_to(ROOT)): sha(TEACHERDIR / 'probability.npy'),
            'data-original/data.csv': sha(ROOT / 'data-original/data.csv'),
        }
        if list(actual_sources.values()) != [expected_cache, expected_aux, expected_teacher, expected_external]:
            raise ValueError('Frozen feature or original-data source fingerprint changed')
        for kind, cfg in self.configs.items():
            if cfg['script_sha256'] != sha(ROOT / REFERENCE_SCRIPTS[kind]):
                raise ValueError('Historical training recipe code changed: ' + kind)
            if 'input_sha256' in cfg and cfg['input_sha256'] != ih:
                raise ValueError('Reference inputs changed: ' + kind)
            if cfg.get('cache_sha256', cfg.get('base_cache_sha256', expected_cache)) != expected_cache:
                raise ValueError('Reference cache changed: ' + kind)
            for key in ('external_sha256',):
                if key in cfg and cfg[key] != expected_external:
                    raise ValueError('Reference external source changed: ' + kind)
            for source, expected in cfg.get('source_code_sha256', {}).items():
                if sha(ROOT / source) != expected:
                    raise ValueError('Frozen feature code changed: ' + source)
        # Original compact/CatBoost runs stored their recipe in improve.py.
        if self.configs['compact']['script_sha256'] != sha(ROOT / 'improve.py'):
            raise ValueError('Original compact feature/training code changed')
        if self.configs['cat']['script_sha256'] != sha(ROOT / 'improve.py'):
            raise ValueError('Original CatBoost feature/training code changed')
        if self.configs['travel']['params'] != self.configs['lgb']['params']:
            raise ValueError('Historical travel/global tree parameter mismatch')
        if self.configs['aux']['params'] != {**self.configs['plain_nn']['params'], 'verbosity': 0}:
            raise ValueError('Historical route neural/control parameter mismatch')
        if self.configs['context']['params'] != {**self.configs['plain_nn']['params'], 'verbosity': 0}:
            raise ValueError('Historical context neural/control parameter mismatch')
        data = joblib.load(CACHE)
        if not data['tr'].id.equals(tr.id) or not data['ts'].id.equals(ts.id):
            raise ValueError('Frozen covariate cache row mismatch')
        pd.testing.assert_frame_equal(data['tr'][self.raw], tr[self.raw], check_exact=True)
        pd.testing.assert_frame_equal(data['ts'][self.raw], ts[self.raw], check_exact=True)
        self.base = data['base']
        # This packet contains old y for its historical provenance. It is never
        # used as a label source, nor is any OOF/checkpoint loaded.
        del data
        self.auxiliary = pd.read_pickle(AUXDIR / 'features.pkl')
        expected_index = pd.RangeIndex(len(tr) + len(ts))
        if not self.base.index.equals(expected_index) or not self.auxiliary.index.equals(expected_index):
            raise ValueError('Frozen covariate row alignment mismatch')
        if set(self.base).intersection({'id', 'satisfaction'}) or set(self.auxiliary).intersection({'id', 'satisfaction'}):
            raise ValueError('Forbidden covariate feature')
        raw_full = pd.concat([tr[self.raw], ts[self.raw]], ignore_index=True)
        self.small_keys = make_keys(raw_full, self.raw, False)
        self.wide_keys = make_keys(raw_full, self.raw, True)
        self.ckeys = context_keys(raw_full)
        self.travel_gate = raw_full['Type of Travel'].to_numpy()
        del raw_full
        self.neural, self.neural_keys = prepare_frames(tr, ts, self.raw)
        self.rkeys = route_keys(self.neural)
        self.route_neural = self.neural.copy()
        for c in ('original_lgb_probability', 'original_cat_probability'):
            self.route_neural[c] = logit(self.base[c].clip(1e-6, 1-1e-6)).astype('float32')
        teacher = np.load(TEACHERDIR / 'probability.npy')
        if teacher.shape != (len(expected_index),):
            raise ValueError('Frozen teacher length mismatch')
        self.route_neural['original_xgb_logit'] = logit(np.clip(teacher, 1e-6, 1-1e-6)).astype('float32')
        for c in ('Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'):
            self.route_neural['route_count_' + c] = self.route_neural[c].map(self.route_neural[c].value_counts()).astype('float32')
        self.route_neural = pd.concat([self.route_neural, self.auxiliary], axis=1)
        self.tree_aux = pd.concat([self.base, self.auxiliary], axis=1)
        self.source_info = {
            'input_sha256': ih, 'frozen_covariate_sha256': actual_sources,
            'reference_configuration_sha256': {k: sha(ROOT / v) for k, v in REFERENCES.items()},
            'feature_code_sha256': {p: sha(ROOT / p) for p in
                ('improve.py', 'context_round.py', 'nn_round.py', 'auxiliary_neural_round.py', 'round12_models.py')},
            'script_sha256': sha(Path(__file__)),
            'external_exact_feature_matches_removed': tm['configuration']['overlaps_removed'],
            'auxiliary_transductive': True, 'competition_oof_reused': False,
            'versions': {p: importlib.metadata.version(p) for p in ('lightgbm', 'catboost', 'torch', 'pytabkit')},
        }
        gc.collect()

    def _split(self, a, b, validation_idx):
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        v = np.asarray(validation_idx, dtype=np.int64)
        if not all(x.ndim == 1 and len(x) and len(np.unique(x)) == len(x) for x in (a, b, v)):
            raise ValueError('Training/evaluation/validation indices must be nonempty unique vectors')
        if min(a.min(), b.min(), v.min()) < 0 or max(a.max(), b.max(), v.max()) >= self.n:
            raise ValueError('Training row index out of bounds')
        if np.intersect1d(a, b).size or not np.isin(v, b).all():
            raise ValueError('Training/evaluation overlap or validation outside evaluation packet')
        lookup = pd.Series(np.arange(len(b)), index=b)
        positions = lookup.loc[v].to_numpy(dtype=np.int64)
        ya, yv = self.y.iloc[a], self.y.iloc[v]
        if not all(np.isfinite(q).all() and set(np.unique(q)) == {0, 1} for q in (ya, yv)):
            raise ValueError('Only training and development validation require binary labels')
        return a, b, v, positions

    def _prepare(self, kind, a, b, target):
        if kind in ('compact', 'cat'):
            result = encoded_fold(self.base, self.small_keys, a, b, self.n, target, False)
        elif kind in ('route_auc', 'route_tuned'):
            result = encoded_fold(self.base, self.wide_keys, a, b, self.n, target, True)
        elif kind in ('lgb', 'travel'):
            result = encoded_fold(self.tree_aux, self.wide_keys, a, b, self.n, target, True)
            add_context(result, context_blocks(self.ckeys, a, b, self.n, target), self.ckeys, a, b, self.n)
        elif kind in ('plain_nn', 'context'):
            result = fold_frames(self.neural, self.neural_keys, a, b, self.n, target)
            if kind == 'context':
                blocks = context_blocks(self.ckeys, a, b, self.n, target)
                for frame, block in zip(result, blocks):
                    frame[block.columns] = block
        elif kind in ('aux', 'tabm'):
            result = route_frames(self.route_neural, self.raw, self.neural_keys, self.rkeys, a, b, self.n, target)
            if kind == 'tabm':
                drop = ['category_' + c for c in ('Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes')]
                result = tuple(frame.drop(columns=drop) for frame in result)
        else:
            raise ValueError('Unknown baseline component: ' + kind)
        if kind == 'cat':
            for frame in result:
                for c in CATS:
                    frame[c] = frame[c].cat.codes.astype('int32')
                frame.fillna(-999, inplace=True)
        if any(set(frame).intersection({'id', 'satisfaction'}) for frame in result):
            raise ValueError('Forbidden supervised feature')
        return result

    def prepare(self, kind, a, b, check_label_invariance=True):
        """Target features never receive any evaluation labels, even temporarily."""
        target = pd.Series(np.nan, index=pd.RangeIndex(self.n))
        target.iloc[a] = self.y.iloc[a].to_numpy()
        result = self._prepare(kind, a, b, target)
        fingerprints = [digest(frame) for frame in result]
        if check_label_invariance:
            del result
            gc.collect()
            changed = target.copy()
            other = np.setdiff1d(np.arange(self.n), a)
            changed.iloc[other] = (np.arange(len(other)) % 2).astype('int8')
            check = self._prepare(kind, a, b, changed)
            if [digest(frame) for frame in check] != fingerprints:
                raise ValueError('Evaluation labels changed feature preparation: ' + kind)
            del check, changed
            gc.collect()
            result = self._prepare(kind, a, b, target)
            if [digest(frame) for frame in result] != fingerprints:
                raise ValueError('Nondeterministic feature preparation: ' + kind)
        return result, fingerprints

    def parameters(self, kind, seed):
        if kind == 'compact':
            return dict(n_estimators=self.configs[kind]['iterations'], learning_rate=.025, n_jobs=8,
                        objective='binary', reg_lambda=1, reg_alpha=0, random_state=seed,
                        min_child_samples=100, num_leaves=127, max_depth=-1, colsample_bytree=.5,
                        subsample=.8, subsample_freq=1, verbosity=-1, deterministic=True, force_col_wise=True)
        if kind == 'route_auc':
            return dict(n_estimators=self.configs[kind]['iterations'], learning_rate=.025, n_jobs=8,
                        objective='binary', reg_lambda=5, reg_alpha=.05, random_state=seed,
                        min_child_samples=20, num_leaves=63, max_depth=8, colsample_bytree=.85,
                        subsample=.9, subsample_freq=1, verbosity=-1, deterministic=True, force_col_wise=True, metric='auc')
        if kind == 'route_tuned':
            return dict(self.configs[kind]['params'], random_state=seed)
        if kind == 'cat':
            return dict(iterations=self.configs[kind]['iterations'], learning_rate=.05, depth=7,
                        l2_leaf_reg=3, border_count=254, task_type='GPU', eval_metric='AUC',
                        random_seed=seed, verbose=False, allow_writing_files=False, thread_count=8)
        if kind in ('lgb', 'travel'):
            return dict(self.configs[kind]['params'], seed=seed)
        return dict(self.configs[kind]['params'], random_state=seed, verbosity=0)

    def fit_component(self, kind, a, b, seed, *, validation_idx, check_label_invariance=True, cache_dir=None):
        """Fit one fresh component; locked holdout labels are never accessed."""
        started = time.time()
        a, b, v, vp = self._split(a, b, validation_idx)
        expected = dict(component=kind, seed=int(seed), params=self.parameters(kind, int(seed)),
                        source=self.source_info, train_idx_sha256=index_sha(a),
                        evaluation_idx_sha256=index_sha(b), validation_idx_sha256=index_sha(v),
                        train_target_sha256=hashlib.sha256(self.y.iloc[a].to_numpy(dtype='int8').tobytes()).hexdigest(),
                        validation_target_sha256=hashlib.sha256(self.y.iloc[v].to_numpy(dtype='int8').tobytes()).hexdigest())
        cached = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = cache_dir / (kind + '.npz')
            metadata_path = cache_dir / (kind + '.json')
            if cached.exists() or metadata_path.exists():
                if not cached.exists() or not metadata_path.exists():
                    raise ValueError('Incomplete fresh baseline component checkpoint: ' + kind)
                info = load(metadata_path)
                if {key: info[key] for key in expected} != expected or sha(cached) != info['checkpoint_sha256']:
                    raise ValueError('Changed fresh baseline component checkpoint: ' + kind)
                if check_label_invariance and info['label_perturbation'] != 'PASS':
                    raise ValueError('Cached baseline has no label-perturbation check: ' + kind)
                with np.load(cached) as packet:
                    if not np.array_equal(packet['evaluation_idx'], b) or not np.array_equal(packet['validation_idx'], v):
                        raise ValueError('Cached fresh baseline evaluation ordering changed: ' + kind)
                    pb, pt = packet['evaluation_pred'], packet['test_pred']
                if pb.shape != (len(b),) or pt.shape != (len(self.ts),):
                    raise ValueError('Cached fresh baseline prediction shape mismatch')
                if not all(np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 for p in (pb, pt)):
                    raise ValueError('Invalid cached fresh baseline probability')
                report(dict(stage='round16_baseline_component_restored', component=kind))
                return pb, pt, info
        prepared, feature_fingerprints = self.prepare(kind, a, b, check_label_invariance)
        xa, xb, xt = prepared
        xv = xb.iloc[vp].reset_index(drop=True)
        ya, yv = self.y.iloc[a], self.y.iloc[v]
        params = self.parameters(kind, int(seed))
        report(dict(stage='round16_baseline_training', component=kind, training_rows=len(a),
                    evaluation_rows=len(b), early_stop_rows=len(v), features=len(xa.columns)))
        details = {}
        if kind in ('compact', 'route_auc', 'route_tuned'):
            model = lgb.LGBMClassifier(**params)
            model.fit(xa, ya, eval_set=[(xv, yv)], eval_metric='auc', callbacks=[
                lgb.early_stopping(180, first_metric_only=(kind != 'compact'), verbose=False)])
            pb, pt = model.predict_proba(xb)[:, 1], model.predict_proba(xt)[:, 1]
            details['iterations'] = int(model.best_iteration_)
        elif kind == 'cat':
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**params)
            model.fit(xa, ya, cat_features=CATS, eval_set=(xv, yv), early_stopping_rounds=200)
            pb, pt = model.predict_proba(xb)[:, 1], model.predict_proba(xt)[:, 1]
            details['iterations'] = int(model.get_best_iteration() + 1)
            details['device'] = 'GPU requested'
        elif kind in ('lgb', 'travel'):
            pb, pt = np.full(len(b), np.nan), np.full(len(self.ts), np.nan)
            groups = sorted(np.unique(self.travel_gate)) if kind == 'travel' else ['all']
            experts = []
            for group in groups:
                ma = np.ones(len(a), bool) if group == 'all' else self.travel_gate[a] == group
                mb = np.ones(len(b), bool) if group == 'all' else self.travel_gate[b] == group
                mv = np.ones(len(v), bool) if group == 'all' else self.travel_gate[v] == group
                mt = np.ones(len(self.ts), bool) if group == 'all' else self.travel_gate[self.n:] == group
                if len(np.unique(yv.to_numpy()[mv])) != 2:
                    raise ValueError('Development validation lacks two classes for group: ' + str(group))
                fit = lgb.Dataset(xa.loc[ma], ya.to_numpy()[ma])
                val = lgb.Dataset(xv.loc[mv], yv.to_numpy()[mv], reference=fit)
                model = lgb.train(params, fit, num_boost_round=3500, valid_sets=[val], valid_names=['dev_val'],
                    callbacks=[lgb.early_stopping(180, first_metric_only=True, verbose=False)])
                if mb.any():
                    pb[mb] = model.predict(xb.loc[mb])
                if mt.any():
                    pt[mt] = model.predict(xt.loc[mt])
                experts.append(dict(group=group, training_rows=int(ma.sum()),
                                    early_stop_rows=int(mv.sum()), iterations=int(model.best_iteration)))
                del model, fit, val
                gc.collect()
            details['experts'] = experts
            model = None
        else:
            import torch
            from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for historical neural component')
            torch.set_num_threads(4)
            model_class = TabM_D_Classifier if kind == 'tabm' else RealMLP_TD_Classifier
            model = model_class(**params)
            cats = list(xa.select_dtypes(include='category').columns)
            model.fit(xa, ya, X_val=xv, y_val=yv, cat_col_names=cats)
            if kind == 'tabm':
                device = str(model.alg_interface_.sub_split_interfaces[0].device_)
                if device != 'cuda:0':
                    raise RuntimeError('TabM device mismatch')
            else:
                device = str(model.alg_interface_.device)
                trainer = model.alg_interface_.min_trainer_kwargs
                if device != 'cuda:0' or trainer['accelerator'] != 'gpu' or trainer['devices'] != [0]:
                    raise RuntimeError('RealMLP GPU trainer mismatch')
            pb, pt = predict_chunks(model, xb), predict_chunks(model, xt)
            details['device'] = device
        if not all(np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 for p in (pb, pt)):
            raise ValueError('Invalid fresh baseline prediction')
        info = dict(**expected,
                    feature_count=len(xa.columns), feature_fingerprints=feature_fingerprints,
                    label_perturbation='PASS' if check_label_invariance else 'not_run',
                    validation_scope='explicit development validation only; holdout never supplies labels',
                    historical_reference=REFERENCES[kind], reference_configuration=self.configs[kind],
                    resource_overrides={'compact_cpu_threads': 8, 'catboost_cpu_threads': 8,
                                        'silent_progress': True},
                    training_split_note='Fresh development split replaces historical mixed CV partitions; no historical fit reused.',
                    seconds=round(time.time() - started, 2), **details)
        info = json.loads(json.dumps(info, default=native))
        if cached is not None:
            np.savez_compressed(cached, evaluation_idx=b, validation_idx=v,
                                evaluation_pred=pb, test_pred=pt)
            info['checkpoint_sha256'] = sha(cached)
            metadata_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
        del prepared, xa, xb, xt, xv, model
        gc.collect()
        if kind in ('cat', 'plain_nn', 'context', 'aux', 'tabm'):
            import torch
            torch.cuda.empty_cache()
        report(dict(stage='round16_baseline_component_done', component=kind, seconds=info['seconds']))
        return pb, pt, info

    def fit_predict(self, a, b, seed, *, validation_idx, check_label_invariance=True, cache_dir=None):
        result = {kind: self.fit_component(kind, a, b, seed, validation_idx=validation_idx,
                    check_label_invariance=check_label_invariance, cache_dir=cache_dir) for kind in COMPONENTS}
        result.update(combine_components(result))
        return result
