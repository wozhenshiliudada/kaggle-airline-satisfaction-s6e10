"""Run the declared sequential rounds16/17 cycle up to validated upload readiness.

Requires the freshly retrained baseline's threefold summary at startup. Child
processes run one at a time; no API upload or extra holdout evaluation occurs.
Existing one-shot stage artifacts are checked and reused. A subprocess or
integrity error stops the cycle rather than being treated as model failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COMPETITION = 'playground-series-s6e10'
MODES = ('raw', 'raw13', 'raw30')
CHILDREN = ('round16_run.py', 'audit_round16_checkpoints.py',
            'round16_evaluate.py', 'round16_production.py', 'record_round16.py')


def path(name):
    result = Path(name)
    result = (ROOT / result if not result.is_absolute() else result).resolve()
    result.relative_to(ROOT.resolve())
    return result


def rel(name):
    return str(path(name).relative_to(ROOT)).replace('\\', '/')


def sha(name):
    result = hashlib.sha256()
    with path(name).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def load(name):
    return json.loads(path(name).read_text(encoding='utf-8'))


def save(name, value):
    target = path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprints(mapping):
    for name, digest in mapping.items():
        require(sha(name) == digest, 'Frozen stage artifact/source changed: ' + str(name))


def close(actual, expected, message):
    require(np.asarray(actual).shape == np.asarray(expected).shape and
            np.allclose(actual, expected, rtol=0, atol=1e-12), message)


def probability(value, length, message):
    value = np.asarray(value)
    require(value.shape == (length,) and np.isfinite(value).all()
            and (value >= 0).all() and (value <= 1).all(), message)
    return value


def baseline_complete():
    """Check saved coverage/IDs/provenance without parsing any target value."""
    folder = ROOT / 'experiments/round16-locked/baseline_current'
    require((folder / 'summary.npz').exists(), 'Fresh baseline threefold summary is not ready.')
    cfg = load(folder / 'configuration.json')
    require(cfg['competition'] == COMPETITION and cfg['folds'] == 3
            and cfg['model'] == 'baseline_current'
            and cfg['holdout_excluded_from_supervised_training_and_early_stopping'],
            'Fresh baseline threefold configuration is invalid.')
    fingerprints(cfg['source_code_sha256'])
    fingerprints({'data/' + k: v for k, v in cfg['input_sha256'].items()})
    require(sha('data-original/data.csv') == cfg['external_sha256'], 'Baseline external data changed.')
    split = ROOT / 'experiments/round16-locked/split.npz'
    require(sha(split) == cfg['split_sha256'], 'Saved baseline split changed.')
    with np.load(split, allow_pickle=False) as z:
        dev, h1, h2, fid = [z[k].copy() for k in ('dev_idx', 'hold1_idx', 'hold2_idx', 'dev_fold_id')]
        train_id, test_id = z['train_id'].copy(), z['test_id'].copy()
    hold = np.concatenate([h1, h2])
    require(np.array_equal(np.sort(np.concatenate([dev, hold])), np.arange(len(train_id)))
            and len(np.unique(hold)) == len(hold) and set(np.unique(fid)) == {0, 1, 2},
            'Saved baseline split does not retain the declared partition.')
    require(np.array_equal(train_id, pd.read_csv(ROOT / 'data/train.csv', usecols=['id']).id)
            and np.array_equal(test_id, pd.read_csv(ROOT / 'data/sample_submission.csv', usecols=['id']).id),
            'Baseline split ID sequence differs from official inputs.')
    with np.load(folder / 'summary.npz', allow_pickle=False) as z:
        require(np.array_equal(z['dev_idx'], dev) and np.array_equal(z['hold_idx'], hold)
                and np.array_equal(z['test_id'], test_id), 'Fresh baseline summary positions/IDs changed.')
        oof = probability(z['dev_oof'], len(dev), 'Incomplete fresh baseline development OOF.').copy()
        hp = probability(z['hold_pred'], len(hold), 'Incomplete baseline holdout averages.').copy()
        tp = probability(z['test_pred'], len(test_id), 'Incomplete baseline test averages.').copy()
    coverage, ha, ta = np.zeros(len(dev), dtype=np.int8), np.zeros(len(hold)), np.zeros(len(test_id))
    for f in (1, 2, 3):
        packet = folder / f'fold-{f}.npz'
        meta = load(folder / f'fold-{f}.json')
        require(meta['checkpoint_sha256'] == sha(packet) and meta['fold'] == f
                and meta['supervised_holdout_labels_available'] is False,
                'Fresh baseline native fold receipt changed.')
        b = fid == f - 1
        with np.load(packet, allow_pickle=False) as z:
            require(np.array_equal(z['train_idx'], dev[~b]) and np.array_equal(z['val_idx'], dev[b])
                    and np.array_equal(z['hold_idx'], hold) and np.array_equal(z['test_id'], test_id),
                    'Fresh baseline native fold positions/IDs changed.')
            close(probability(z['val_pred'], int(b.sum()), 'Baseline validation probability error.'),
                  oof[b], 'Fresh baseline OOF native coverage changed.')
            ha += probability(z['hold_pred'], len(hold), 'Baseline hold probability error.') / 3
            ta += probability(z['test_pred'], len(test_id), 'Baseline test probability error.') / 3
        coverage[b] += 1
    require(np.all(coverage == 1), 'Fresh baseline development coverage must be exactly once.')
    close(ha, hp, 'Fresh baseline native holdout averaging changed.')
    close(ta, tp, 'Fresh baseline native test averaging changed.')
    return dict(folds=3, development_rows=len(dev), excluded_holdout_rows=len(hold),
                summary_path=rel(folder / 'summary.npz'), summary_sha256=sha(folder / 'summary.npz'),
                holdout_labels_loaded=False, holdout_metrics_computed=False)


@contextmanager
def singleton():
    """Process-scoped OS lock prevents two cycle instances training concurrently."""
    name = ROOT / 'evidence/round16-cycle.lock'
    name.parent.mkdir(parents=True, exist_ok=True)
    with name.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Another round16/17 cycle instance is already active.') from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Cycle:
    def __init__(self, start_round):
        self.start = start_round
        self.state_path = ROOT / f'evidence/round{start_round}-cycle-state.json'
        self.events = ROOT / f'evidence/round{start_round}-cycle-events.jsonl'
        self.state = dict(competition=COMPETITION, start_round=start_round,
            status='running', completed=False, completion_scope='sequential_preparation_for_upload',
            active_round=start_round, stage='startup', prepared_upload_round=None,
            eligible_for_upload=False, uploaded=False, uploads=[], api_uploads_performed=0,
            rounds=[], stages=[], events_path=rel(self.events),
            source_artifact_sha256={}, script_sha256=sha(Path(__file__)))
        self.prior = {}
        if self.state_path.exists():
            prior = load(self.state_path)
            self.prior = prior
            self.state['stages'] = list(prior.get('stages', []))
        self.configuration = ROOT / f'evidence/round{start_round}-cycle-configuration.json'

    def event(self, stage, status, **extra):
        utc = datetime.now(timezone.utc)
        row = dict(checked_at_utc=utc.isoformat(),
                   checked_at_beijing=utc.astimezone(timezone(timedelta(hours=8))).isoformat(),
                   round=self.state['active_round'], stage=stage, status=status, **extra)
        self.events.parent.mkdir(parents=True, exist_ok=True)
        with self.events.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.state['stage'] = stage
        self.state['updated_at_utc'] = row['checked_at_utc']
        if status in ('completed', 'reused'):
            self.state['stages'].append(row)
        save(self.state_path, self.state)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def remember(self, name):
        self.state['source_artifact_sha256'][rel(name)] = sha(name)

    def child(self, round_number, stage, script, args):
        fingerprints(self.state['child_script_sha256'])
        log = ROOT / f'evidence/round{round_number}-{stage}.log'
        command = [sys.executable, '-u', str(ROOT / script), *map(str, args)]
        self.event(stage, 'started', log_path=rel(log), script=script, arguments=list(map(str, args)))
        # Append rather than erase evidence if a failed/incomplete child is resumed.
        with log.open('ab') as output:
            marker = dict(cycle_stage=stage, restarted_at_utc=datetime.now(timezone.utc).isoformat())
            output.write((json.dumps(marker) + '\n').encode('utf-8'))
            output.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        require(result.returncode == 0,
                f'Child {stage} exited {result.returncode}; inspect {rel(log)}. Integrity/runtime errors cannot trigger fallback.')
        self.event(stage, 'completed', log_path=rel(log), exit_code=result.returncode)

    def uploaded_guard(self, round_number):
        """Return before any protected pre-upload snapshot verification on resume."""
        folder = ROOT / f'outputs-round{round_number}'
        metrics_file, validation_file = folder / 'metrics.json', folder / 'validation.json'
        if not metrics_file.exists():
            return False
        metrics = load(metrics_file)
        validation = load(validation_file) if validation_file.exists() else {}
        finalized = bool(metrics.get('uploaded') or validation.get('uploaded'))
        receipt = load('last-submission.json') if path('last-submission.json').exists() else {}
        receipt_match = bool(metrics.get('submission_sha256') and
                             receipt.get('sha256') == metrics['submission_sha256'])
        if not finalized and not receipt_match:
            return False
        require(metrics['competition'] == COMPETITION and metrics['round'] == round_number,
                'Uploaded production metadata has the wrong competition/round.')
        require(sha(folder / 'submission.csv') == metrics['submission_sha256'],
                'Uploaded local prediction bytes changed.')
        if finalized:
            require(metrics.get('uploaded') is True and validation.get('uploaded') is True
                    and validation.get('passed') is True and validation.get('official_status') == 'COMPLETE'
                    and metrics.get('official_status') == 'COMPLETE', 'Partial official finalization needs root reconciliation.')
        self.state['rounds'] = list(self.prior.get('rounds', []))
        for key in ('child_script_sha256', 'cycle_configuration_path',
                    'cycle_configuration_sha256', 'baseline_ready'):
            if key in self.prior:
                self.state[key] = self.prior[key]
        self.state.update(status='upload_already_completed' if finalized else 'upload_receipt_detected',
            completed=True, completion_scope='official_upload_completed' if finalized else 'awaiting_root_official_reconciliation',
            active_round=round_number, uploaded=True, eligible_for_upload=False,
            prepared_upload_round=round_number,
            submission_ref=validation.get('submission_ref'), public_score=validation.get('public_score'),
            requires_root_upload=False, requires_root_reconciliation=not finalized)
        self.remember(metrics_file)
        if validation_file.exists():
            self.remember(validation_file)
        self.event('uploaded_resume_guard', 'completed', skipped_all_pre_upload_source_checks=True,
                   official_finalized=finalized)
        return True

    def initialize(self):
        config = dict(competition=COMPETITION, start_round=self.start,
                      script_sha256=sha(Path(__file__)),
                      child_script_sha256={name: sha(name) for name in CHILDREN},
                      no_api_upload=True, sequential_gpu_training=True)
        if self.configuration.exists():
            require(load(self.configuration) == config, 'Cycle source/configuration changed since its first run.')
        else:
            save(self.configuration, config)
        self.state.update(child_script_sha256=config['child_script_sha256'],
                          cycle_configuration_path=rel(self.configuration),
                          cycle_configuration_sha256=sha(self.configuration))
        self.remember(self.configuration)
        self.state['baseline_ready'] = baseline_complete()
        self.event('baseline_ready', 'completed', **self.state['baseline_ready'])

    def audit(self, round_number):
        audit = ROOT / f'evidence/round{round_number}-checkpoint-audit.json'
        if not audit.exists():
            self.child(round_number, 'checkpoint-audit', 'audit_round16_checkpoints.py', ['--round', round_number])
        else:
            self.event('checkpoint-audit', 'reused', artifact_path=rel(audit))
        value = load(audit)
        require(value['passed'] is True and value['round'] == round_number
                and value['script_sha256'] == sha('audit_round16_checkpoints.py')
                and value['holdout_labels_loaded'] is False and value['holdout_metrics_computed'] is False,
                'Checkpoint audit did not pass the declared no-hold-label scope.')
        fingerprints(value['audited_file_sha256'])
        self.remember(audit)

    def development(self, round_number, protocol, analysis):
        name = analysis / 'development-selection.json'
        if not name.exists():
            self.child(round_number, 'development-selection', 'round16_evaluate.py',
                       ['--manifest', f'experiments/round{round_number}-locked/protocol.json', '--stage', 'dev'])
        else:
            self.event('development-selection', 'reused', artifact_path=rel(name))
        value = load(name)
        require(value['competition'] == COMPETITION and value['stage'] == 'development_selection'
                and value['folds'] == 3 and value['holdout_labels_evaluated'] is False,
                'Invalid development selection stage.')
        require(value['protocol_path'] == f'experiments/round{round_number}-locked/protocol.json',
                'Development lock refers to a different protocol.')
        fingerprints(value['source_fingerprints'])
        require(sha(value['locked_prediction_path']) == value['locked_prediction_sha256'],
                'Frozen development predictions changed.')
        rows = value['candidates']
        require([r['candidate'] for r in rows] == list(MODES), 'Development changed the three declared inputs.')
        eligible = []
        for row in rows:
            close(row['delta'], row['auc'] - value['baseline']['auc'], 'Development reported delta is inconsistent.')
            blocks = [a - b for a, b in zip(row['block_auc'], value['baseline']['block_auc'])]
            require(len(blocks) == 3, 'Development does not have exactly three blocks.')
            close(row['block_deltas'], blocks, 'Development reported block deltas are inconsistent.')
            passed = bool(row['delta'] >= protocol['dev_min_delta'] and min(blocks) > 0)
            require(row['eligible'] is passed, 'Development eligibility flag disagrees with its actual gate.')
            if passed:
                eligible.append(row)
        selected = max(eligible, key=lambda r: r['auc'], default=None)
        require(value['selected_candidate'] == selected, 'Development did not lock the best gate-eligible declared input.')
        self.remember(name)
        return value

    def holdout(self, round_number, protocol, analysis, development):
        name = analysis / 'final-comparison.json'
        if not name.exists():
            self.child(round_number, 'locked-final-comparison', 'round16_evaluate.py',
                       ['--manifest', f'experiments/round{round_number}-locked/protocol.json', '--stage', 'finalize'])
        else:
            self.event('locked-final-comparison', 'reused', artifact_path=rel(name))
        value = load(name)
        require(value['competition'] == COMPETITION and value['stage'] == 'final_holdout_check'
                and value['development_selection_sha256'] == sha(analysis / 'development-selection.json')
                and value['selected_development_candidate'] == development['selected_candidate']
                and value['deployment_stack'] == development['deployment_stack']
                and value['source_fingerprints'] == development['source_fingerprints']
                and value['counts']['active_holdout_name'] == protocol['active_holdout'],
                'Final holdout changed the locked development choice, source or active set.')
        fingerprints(value['source_fingerprints'])
        close(value['heldout']['delta'], value['heldout']['selected_auc'] - value['heldout']['baseline_auc'],
              'Holdout reported delta is inconsistent.')
        passed = bool(value['heldout']['delta'] >= protocol['hold_min_delta'])
        require(value['heldout']['min_delta'] == protocol['hold_min_delta']
                and value['heldout']['passed'] is passed
                and value['eligible_for_production_transfer'] is passed,
                'Holdout pass flags disagree with the actual declared gate.')
        self.remember(name)
        return value

    def production(self, round_number, development, final):
        name = ROOT / f'analyses/round{round_number}-production-comparison.json'
        if not name.exists():
            self.child(round_number, 'production-training', 'round16_production.py', ['--round', round_number])
        else:
            self.event('production-training', 'reused', artifact_path=rel(name))
        value = load(name)
        mode = development['selected_candidate']['candidate']
        require(value['competition'] == COMPETITION and value['round'] == round_number
                and value['native_folds'] == 5 and value['carrier_mode'] == mode
                and value['frozen_stack'] == development['deployment_stack'][mode]
                and value['locked_validation']['development_selection_sha256'] ==
                    sha(f'analyses/round{round_number}-locked/development-selection.json')
                and value['locked_validation']['final_comparison_sha256'] ==
                    sha(f'analyses/round{round_number}-locked/final-comparison.json'),
                'Production transfer does not match the locked carrier and coefficients.')
        fingerprints(value['source_fingerprints'])
        fingerprints(value['feature_code_sha256'])
        require(value['script_sha256'] == sha('round16_production.py'), 'Production code changed after training.')
        require(sha(value['production_candidate_path']) == value['production_candidate_sha256']
                and sha(value['production_oof_path']) == value['production_oof_sha256'],
                'The retained fixed production predictions changed.')
        full = value['full_oof']
        require(full['coverage_count'] == 1 and len(full['block_deltas']) == 5,
                'Production OOF must retain full exactly-once coverage and five comparison blocks.')
        close(full['delta'], full['selected_auc'] - full['baseline_auc'], 'Production reported global delta is inconsistent.')
        passed = bool(full['delta'] >= 1e-5 and min(full['block_deltas']) > 0)
        gate = value['compatibility_gate']
        require(gate['min_delta'] == 1e-5 and gate['passed'] is passed
                and value['eligible_for_submission'] is passed
                and gate['all_five_blocks_positive'] is (min(full['block_deltas']) > 0),
                'Production compatibility flags disagree with the actual declared gate.')
        require((value['selected_blend'] is not None) is passed, 'Production selection field disagrees with the gate.')
        self.remember(name)
        return value

    def formal_validation(self, round_number, production):
        folder = ROOT / f'outputs-round{round_number}'
        name = folder / 'validation.json'
        if not name.exists():
            self.child(round_number, 'formal-validation', 'record_round16.py', ['validate', '--round', round_number])
        else:
            self.event('formal-validation', 'reused', artifact_path=rel(name))
        value = load(name)
        require(value['passed'] is True and value['uploaded'] is False and value['round'] == round_number
                and value['record_script_sha256'] == sha('record_round16.py')
                and value['metrics_pre_official_sha256'] == sha(folder / 'metrics.json')
                and value['submission_sha256'] == production['submission_sha256']
                and value['submission_sha256'] == sha(folder / 'submission.csv')
                and value['oof_sha256'] == production['oof_sha256']
                and value['oof_sha256'] == sha(folder / 'oof.csv'), 'Formal upload-readiness validation changed or failed.')
        fingerprints(value['source_fingerprints'])
        fingerprints(value['feature_code_sha256'])
        self.remember(name)
        self.remember(folder / 'metrics.json')
        return value

    def run(self):
        # This precedes initialization and all historical protected-pointer checks.
        if self.uploaded_guard(self.start):
            return
        self.initialize()
        for number in range(self.start, 18):
            self.state['active_round'] = number
            if self.uploaded_guard(number):
                return
            manifest = ROOT / f'experiments/round{number}-locked/protocol.json'
            if not manifest.exists():
                self.child(number, 'initialize-cycle', 'round16_run.py', ['--round', number, 'init'])
            protocol = load(manifest)
            require(protocol['round'] == number and protocol['competition'] == COMPETITION
                    and protocol['epochs'] == (4 if number == 16 else 8)
                    and protocol['active_holdout'] == ('hold1' if number == 16 else 'hold2')
                    and protocol['dev_min_delta'] == 1e-5 and protocol['hold_min_delta'] == 1e-5,
                    'Cycle protocol differs from the predeclared two-round schedule.')
            fingerprints(protocol['source_code_sha256'])
            fingerprints(protocol['protected_file_sha256'])
            analysis = path(protocol['analysis_dir'])
            locked_already = any((analysis / n).exists() for n in ('development-selection.json', 'final-comparison.json'))
            for mode in MODES:
                folder = path(protocol['models'][mode]['path'])
                complete = all((folder / name).exists() for name in ['summary.npz', 'source-audit.json'] +
                               [f'fold-{f}.{ext}' for f in (1, 2, 3) for ext in ('npz', 'json')])
                require(complete or not locked_already, 'A locked development source is now incomplete; refusing to retrain/replace it.')
                if complete:
                    self.event('carrier-' + mode + '-training', 'reused', artifact_path=rel(folder / 'summary.npz'))
                else:
                    self.child(number, 'carrier-' + mode + '-training', 'round16_run.py', ['--round', number, mode])
                    require((folder / 'summary.npz').exists(), 'Carrier child exited without a complete summary.')
            self.audit(number)
            development = self.development(number, protocol, analysis)
            chosen = development['selected_candidate']
            outcome = dict(round=number, stage='development_rejected', development_selected=None,
                           development_delta=None, holdout_delta=None, production_delta=None,
                           eligible_for_upload=False)
            self.state['rounds'].append(outcome)
            if chosen is None:
                self.event('development_gate', 'completed', passed=False, fallback_round=17 if number == 16 else None)
                continue
            outcome.update(development_selected=chosen['candidate'], development_delta=chosen['delta'])
            final = self.holdout(number, protocol, analysis, development)
            outcome.update(stage='locked_holdout_rejected', holdout_delta=final['heldout']['delta'])
            if not final['eligible_for_production_transfer']:
                self.event('locked_holdout_gate', 'completed', passed=False, fallback_round=17 if number == 16 else None)
                continue
            production = self.production(number, development, final)
            outcome.update(stage='production_compatibility_rejected', production_delta=production['full_oof']['delta'])
            if not production['eligible_for_submission']:
                self.event('production_compatibility_gate', 'completed', passed=False,
                           fallback_round=17 if number == 16 else None)
                continue
            validation = self.formal_validation(number, production)
            outcome.update(stage='eligible_for_upload', eligible_for_upload=True)
            self.state.update(status='eligible_for_upload', completed=True,
                completion_scope='upload_preparation_only', prepared_upload_round=number,
                eligible_for_upload=True, requires_root_upload=True,
                submission_sha256=validation['submission_sha256'],
                submission_path=f'outputs-round{number}/submission.csv')
            self.event('cycle_preparation_complete', 'completed', ready_round=number,
                       no_upload_performed=True, no_further_round_started=True)
            return
        self.state.update(status='both_rounds_not_eligible' if self.start == 16 else 'round17_not_eligible',
                          completed=True, completion_scope='declared_modeling_rounds_finished_without_upload',
                          eligible_for_upload=False, requires_root_upload=False)
        self.event('cycle_no_eligible_candidate', 'completed', no_upload_performed=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-round', type=int, choices=(16, 17), default=16)
    args = parser.parse_args()
    cycle = Cycle(args.start_round)
    owns_cycle_lock = False
    try:
        with singleton():
            owns_cycle_lock = True
            cycle.run()
    except Exception as error:
        if owns_cycle_lock:
            cycle.state.update(status='error', completed=False, eligible_for_upload=False,
                               prepared_upload_round=None, error=str(error), no_fallback_on_error=True)
            cycle.event(cycle.state.get('stage', 'startup'), 'error', error=str(error))
        else:
            print(str(error), file=sys.stderr, flush=True)
        raise
