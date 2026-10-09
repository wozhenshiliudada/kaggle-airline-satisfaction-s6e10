"""Raw-input auxiliary RealMLP carriers, independently implemented locally.

Method inspiration is the Part 6 raw+aux branch of goodpjw2008's public
S6E10 notebook. This module imports no third-party predictions or weights.
The three fixed input modes have no target encodings, route profiles,
frequencies, original-data teachers, or target-dependent preprocessing.

An explicit development validation set selects the best epoch. The held-out
rows are supplied only to predict_proba after fitting. The pre-existing
auxiliary reconstructions and category vocabularies are transductive, using
unlabelled train/test covariates; this is documented rather than described
as an entirely inductive holdout.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
AUXDIR = ROOT / 'experiments' / 'auxiliary-reconstruction-300rounds'
REFERENCE = ROOT / 'experiments' / 'realmlp-5fold-8ens' / 'configuration.json'
NOTEBOOK = (ROOT / 'references' / 'round8' /
            's6e10-more-aux-carriers-lb-0-96152' /
            's6e10-more-aux-carriers-lb-0-96152.ipynb')
SOURCE = 'https://www.kaggle.com/code/goodpjw2008/s6e10-more-aux-carriers-lb-0-96152'
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
SERVICES = ['Inflight wifi service', 'Departure/Arrival time convenient',
            'Ease of Online booking', 'Gate location', 'Food and drink',
            'Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Cleanliness']
MODES = ('raw', 'raw13', 'raw30')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    # PyTabKit's best-epoch validation errors may be scalar Torch tensors.
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported metadata value: {type(value)!r}')


def report(value):
    print(json.dumps(jsonable(value), ensure_ascii=False), flush=True)


def idx_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


class CarrierFactory:
    """Prepare a shared input frame and fit the three fixed carrier variants.

    a/v/h are global row positions within the supplied training frame.
    prepare(mode,a,v,h) returns train, development-val, holdout and test
    frames and never reads labels. fit_predict returns (pv,ph,pt,info).
    Optional artifact_dir writes the fitted local model and its metadata.
    """

    def __init__(self, tr, ts, raw, y, epochs=4):
        if epochs not in (4, 8):
            raise ValueError('The declared epoch comparison permits only 4 or 8 epochs.')
        self.epochs = int(epochs)
        self.n = len(tr)
        self.nt = len(ts)
        self.raw = list(raw)
        self.y = np.asarray(y, dtype=np.float64).copy()
        if (self.y.shape != (self.n,) or np.isinf(self.y).any()
                or not np.isin(self.y[np.isfinite(self.y)], [0, 1]).all()):
            raise ValueError('Aligned binary labels with optional masked NaN holdout values are required.')
        self.has_masked_labels = bool(np.isnan(self.y).any())
        if len(self.raw) != 21 or len(set(self.raw)) != 21:
            raise ValueError('Expected the competition\'s 21 unique raw covariates.')
        if set(self.raw) & {'id', 'satisfaction'} or not set(CATS).issubset(self.raw):
            raise ValueError('Target/ID features or missing raw categorical columns.')
        for name, frame, size in [('train', tr, self.n), ('test', ts, self.nt)]:
            if not frame.index.equals(pd.RangeIndex(size)):
                raise ValueError(f'{name} row order must retain the original RangeIndex.')
            if 'id' not in frame or not frame.id.is_unique or frame.id.isna().any():
                raise ValueError(f'{name} IDs must be unique and nonmissing.')
        self.train_ids = tr.id.to_numpy(copy=True)
        self.test_ids = ts.id.to_numpy(copy=True)
        input_sha = {name: sha(ROOT / 'data' / name)
                     for name in ('train.csv', 'test.csv', 'sample_submission.csv')}
        reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
        meta_path = AUXDIR / 'metrics.json'
        aux_path = AUXDIR / 'features.pkl'
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        cfg = metadata['configuration']
        if reference['input_sha256'] != input_sha or cfg['input_sha256'] != input_sha:
            raise ValueError('Raw input fingerprints differ from the auxiliary/reference source.')
        if cfg.get('label_used') is not False or cfg.get('transductive') is not True:
            raise ValueError('Auxiliary cache must declare no target usage and transductive covariates.')
        if cfg.get('auxiliary_folds') != 5:
            raise ValueError('Unexpected auxiliary row-exclusion protocol.')
        if cfg['script_sha256'] != sha(ROOT / 'auxiliary_round.py'):
            raise ValueError('Auxiliary feature-building source changed.')
        if metadata['features_sha256'] != sha(aux_path):
            raise ValueError('Auxiliary feature pickle does not match its recorded SHA256.')
        self.aux = pd.read_pickle(aux_path)
        if not isinstance(self.aux, pd.DataFrame) or self.aux.shape != (self.n + self.nt, 30):
            raise ValueError('Unexpected auxiliary cache row count or 30-column shape.')
        if not self.aux.index.equals(pd.RangeIndex(self.n + self.nt)):
            raise ValueError('Auxiliary row positions are not the original concatenated RangeIndex.')
        if list(metadata['shape']) != list(self.aux.shape) or not self.aux.columns.is_unique:
            raise ValueError('Auxiliary shape metadata or column uniqueness mismatch.')
        self.expected_cols = ['aux_expected_' + c for c in SERVICES]
        probability_cols = ['aux_probability_' + c for c in SERVICES + CATS[1:]]
        expected_aux = set(self.expected_cols + probability_cols + ['aux_log_likelihood'])
        if set(self.aux.columns) != expected_aux or len(self.expected_cols) != 13:
            raise ValueError('Unexpected auxiliary column schema, including any target/ID features.')
        if not np.isfinite(self.aux.to_numpy()).all():
            raise ValueError('Auxiliary features must be finite.')
        self.aux = self.aux.astype('float32')
        if sha(NOTEBOOK) != '2bd22cd99d78932374d4ae826df42926c991b04a414336228ffd19eb3f177b7c':
            raise ValueError('Read-only public method reference changed.')

        # Verify the supplied frames have not been sampled/reordered before
        # attaching a positional feature cache that has no ID column.
        for name, frame in [('train.csv', tr), ('test.csv', ts)]:
            disk = pd.read_csv(ROOT / 'data' / name, usecols=['id'] + self.raw)
            pd.testing.assert_frame_equal(frame[['id'] + self.raw], disk[['id'] + self.raw],
                                          check_dtype=False, check_exact=True)
            del disk

        self.full = pd.concat([tr[self.raw], ts[self.raw]], ignore_index=True).copy()
        numeric = [c for c in self.raw if c not in CATS]
        self.full['Arrival Delay in Minutes'] = self.full['Arrival Delay in Minutes'].fillna(0.0)
        for c in numeric:
            if not pd.api.types.is_numeric_dtype(self.full[c]):
                raise ValueError(f'Raw numeric covariate has a nonnumeric dtype: {c}')
            if not np.isfinite(self.full[c].to_numpy()).all():
                raise ValueError(f'Unexpected nonfinite values in {c}')
            # Match the reference's categorical twin: integer truncation,
            # then a string token and a categorical vocabulary.
            self.full[c + '_cat_'] = pd.Categorical(self.full[c].astype(np.int64).astype(str))
            self.full[c] = self.full[c].astype('float32')
        for c in CATS:
            if self.full[c].isna().any():
                raise ValueError(f'Unexpected missing original categorical value: {c}')
            self.full[c] = self.full[c].astype(str).astype('category')
        self.cats = CATS + [c + '_cat_' for c in numeric]
        if len(self.full.columns) != 38 or len(self.cats) != 21:
            raise ValueError('Unexpected raw numeric-twin input width.')
        self.base_params = copy.deepcopy(reference['params'])
        if (self.base_params.get('n_ens') != 8 or self.base_params.get('lr_sched') != 'flat_anneal'
                or self.base_params.get('use_early_stopping') is not False):
            raise ValueError('Reference neural parameters differ from the fixed experiment.')
        self.base_params.update(n_epochs=self.epochs, verbosity=2)
        self.audit = dict(
            implementation_sha256=sha(Path(__file__)), input_sha256=input_sha,
            reference_configuration_sha256=sha(REFERENCE),
            auxiliary_meta_sha256=sha(meta_path), auxiliary_features_sha256=sha(aux_path),
            auxiliary_builder_sha256=sha(ROOT / 'auxiliary_round.py'),
            public_method_url=SOURCE, public_method_source_sha256=sha(NOTEBOOK),
            public_method_cells_0based=[12, 14, 32],
            raw_columns=self.raw, numeric_twin_columns=[c + '_cat_' for c in numeric],
            categorical_columns=self.cats, expected_aux_columns=self.expected_cols,
            all_aux_columns=list(self.aux.columns),
            input_transform='Arrival Delay missing=0; raw numeric twin=int64->string->category; numeric=float32',
            vocabulary_source='Unlabelled competition train+test raw covariates only',
            auxiliary_source='Our five-fold excluded-row reconstruction of raw covariates over train+test; satisfaction never loaded',
            target_encoding=False, route_profiles=False, counts=False,
            original_teacher_features=False, third_party_predictions=False,
            transductive=True, auxiliary_target_used=False,
            feature_counts={'raw': 38, 'raw13': 51, 'raw30': 68},
            versions={p: importlib.metadata.version(p) for p in ('pytabkit', 'torch', 'pytorch-lightning')})

    def params(self, seed):
        return dict(copy.deepcopy(self.base_params), random_state=int(seed))

    def full_frame(self, mode):
        if mode not in MODES:
            raise ValueError(f'Undeclared raw-carrier mode: {mode}')
        if mode == 'raw':
            return self.full.copy()
        aux = self.aux[self.expected_cols] if mode == 'raw13' else self.aux
        result = pd.concat([self.full, aux], axis=1)
        if len(result.columns) != self.audit['feature_counts'][mode] or not result.columns.is_unique:
            raise ValueError('Unexpected feature count or duplicated column names.')
        return result

    def _indices(self, a, v, h):
        result = []
        for name, value in [('train', a), ('development validation', v), ('holdout', h)]:
            value = np.asarray(value)
            if value.ndim != 1 or (len(value) and not np.issubdtype(value.dtype, np.integer)):
                raise ValueError(f'{name} positions must be a one-dimensional integer array.')
            value = value.astype(np.int64, copy=False)
            if len(np.unique(value)) != len(value) or np.any(value < 0) or np.any(value >= self.n):
                raise ValueError(f'{name} positions are duplicated or out of range.')
            result.append(value)
        a, v, h = result
        if not len(a) or not len(v):
            raise ValueError('Explicit training and development validation rows are required.')
        if any(np.intersect1d(first, second).size for first, second in [(a, v), (a, h), (v, h)]):
            raise ValueError('Training, development validation and holdout positions overlap.')
        return a, v, h

    def prepare(self, mode, a, v, h):
        a, v, h = self._indices(a, v, h)
        full = self.full_frame(mode)
        frames = [full.iloc[idx].reset_index(drop=True).copy() for idx in (a, v, h)]
        frames.append(full.iloc[self.n:].reset_index(drop=True).copy())
        return tuple(frames)

    @staticmethod
    def _predict(model, frame):
        if not len(frame):
            return np.empty(0, dtype=np.float64)
        result = np.concatenate([model.predict_proba(frame.iloc[start:start + 32768])[:, 1]
                                 for start in range(0, len(frame), 32768)]).astype('float64')
        if result.shape != (len(frame),) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError('Invalid classification probabilities.')
        return result

    def fit_predict(self, mode, a, v, h, seed, artifact_dir=None):
        import torch
        from pytabkit import RealMLP_TD_Classifier

        started = time.time()
        a, v, h = self._indices(a, v, h)
        if not torch.cuda.is_available():
            raise RuntimeError('This controlled experiment requires the verified CUDA device.')
        # Read labels only for the explicitly supplied development positions.
        # Held-out labels may be NaN and are never indexed by this module.
        ya, yv = self.y[a], self.y[v]
        if (not np.isfinite(ya).all() or not np.isfinite(yv).all()
                or not np.isin(ya, [0, 1]).all() or not np.isin(yv, [0, 1]).all()
                or np.unique(ya).size != 2 or np.unique(yv).size != 2):
            raise ValueError('Training and development validation must both contain two classes.')
        ya, yv = ya.astype('int8'), yv.astype('int8')
        if artifact_dir is not None:
            artifact_dir = Path(artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            if (artifact_dir / 'model.joblib').exists() or (artifact_dir / 'info.json').exists():
                raise ValueError('Use a new carrier artifact folder; prior models must be preserved.')
        torch.set_num_threads(4)
        xa, xv, xh, xt = self.prepare(mode, a, v, h)
        params = self.params(seed)
        model = RealMLP_TD_Classifier(**params)
        report(dict(stage='round16_raw_carrier_training', mode=mode, seed=int(seed),
                    train_rows=len(a), development_validation_rows=len(v), holdout_rows=len(h),
                    features=len(xa.columns), epochs=self.epochs, ensemble=params['n_ens'],
                    holdout_labels_used_in_fit=False, device=torch.cuda.get_device_name(0)))
        # h never appears here; it cannot select a checkpoint or best epoch.
        model.fit(xa, ya, X_val=xv, y_val=yv, cat_col_names=self.cats)
        trained_seconds = time.time() - started
        interface = model.alg_interface_
        trainer = interface.min_trainer_kwargs
        if (str(interface.device) != 'cuda:0' or trainer['accelerator'] != 'gpu'
                or trainer['devices'] != [0] or trainer['max_epochs'] != self.epochs):
            raise RuntimeError('Actual RealMLP Trainer did not use the declared CUDA configuration.')
        if not interface.model.is_classification_ or not np.array_equal(model.classes_, [0, 1]):
            raise RuntimeError('Actual fitted model is not the required binary classifier.')
        pv, ph, pt = [self._predict(model, frame) for frame in (xv, xh, xt)]
        module = interface.model
        info = jsonable(dict(
            mode=mode, seed=int(seed), features=list(xa.columns), feature_count=len(xa.columns),
            categorical_columns=self.cats, params=params,
            train_rows=len(a), validation_rows=len(v), holdout_rows=len(h), test_rows=len(xt),
            index_sha256={'train': idx_sha(a), 'development_validation': idx_sha(v), 'holdout': idx_sha(h)},
            fit_seconds=round(trained_seconds, 3), elapsed_seconds=round(time.time() - started, 3),
            trainer_configuration=trainer, actual_device=str(interface.device),
            actual_binary_classifier=bool(module.is_classification_),
            fit_params=interface.fit_params,
            best_mean_val_epochs=module.best_mean_val_epochs,
            best_val_epochs=module.best_val_epochs,
            best_mean_val_errors=module.best_mean_val_errors,
            checkpoint_selection='Best epoch on the supplied development validation labels only; use_early_stopping=False still permits best-epoch checkpoint selection',
            holdout_labels_used_in_fit=False,
            holdout_label_values_never_indexed=True,
            caller_supplied_masked_label_array=self.has_masked_labels,
            held_y_values_not_available=(bool(len(h)) and self.has_masked_labels),
            held_y_mask_note='Caller masks held-out label values as NaN; this module does not inspect h label values to verify the mask',
            source_audit=self.audit))
        if artifact_dir is not None:
            import joblib
            model_path = artifact_dir / 'model.joblib'
            joblib.dump(model, model_path, compress=3)
            info['fitted_model_path'] = str(model_path.resolve())
            info['fitted_model_sha256'] = sha(model_path)
            info['elapsed_seconds'] = round(time.time() - started, 3)
            (artifact_dir / 'info.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
        report(dict(stage='round16_raw_carrier_complete', mode=mode, seed=int(seed),
                    seconds=info['elapsed_seconds'], best_mean_val_epochs=info['best_mean_val_epochs'],
                    holdout_labels_used_in_fit=False))
        del model, xa, xv, xh, xt
        gc.collect()
        torch.cuda.empty_cache()
        return pv, ph, pt, info
