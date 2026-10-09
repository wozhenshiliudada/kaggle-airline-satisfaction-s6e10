"""Validate the divergent candidate and reconcile its official scoring receipt."""
import argparse
from datetime import datetime, timezone
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from context_round import ROOT, sha
from auxiliary_round import hashes
from train import read_data, validate_submission, COMPETITION

OUT = ROOT / 'outputs-round11'
BASE = 'outputs-round9/submission.csv'
BASE_SHA = 'c5f0b36742ddc27e265802c5118509cbf8dbabacca171bb33cbe1d822a852e69'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def validate():
    tr, ts, sample, raw, y = read_data(ROOT / 'data')
    m = load(OUT / 'metrics.json')
    current = load(ROOT / 'best-submission.json')
    if current['path'] != BASE or current['sha256'] != BASE_SHA or sha(ROOT / BASE) != BASE_SHA:
        raise ValueError('Preserved official baseline changed')
    if m['competition'] != COMPETITION or m['input_sha256'] != hashes():
        raise ValueError('Candidate competition inputs changed')
    if sha(ROOT / 'data-original/data.csv') != m['external_sha256']:
        raise ValueError('External data changed')
    if sha(ROOT / 'evaluate_divergent.py') != m['script_sha256']:
        raise ValueError('Selection code changed')
    selected = m['selected_blend']
    if not selected or not selected['all_blocks_positive'] or selected['delta'] <= 0:
        raise ValueError('Candidate failed predeclared screening')
    if sha(OUT / 'submission.csv') != m['submission_sha256']:
        raise ValueError('Candidate fingerprint mismatch')
    s = pd.read_csv(OUT / 'submission.csv')
    validate_submission(s, sample)
    o = pd.read_csv(OUT / 'oof.csv')
    if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction, y):
        raise ValueError('OOF labels or IDs changed')
    p = o.prediction.to_numpy()
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError('Invalid OOF probabilities')
    auc = float(roc_auc_score(y, p))
    if abs(auc - selected['auc']) > 1e-12:
        raise ValueError('Saved OOF AUC mismatch')
    checkpoints = {}
    for mode in m['completed_modes']:
        folder = ROOT / 'experiments' / ('round11-' + mode + '-5fold')
        cfg = load(folder / 'configuration.json')
        if cfg['input_sha256'] != hashes() or cfg['script_sha256'] != sha(ROOT / 'divergent_round.py'):
            raise ValueError('Training configuration changed')
        for name, digest in cfg['source_code_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Training source changed: ' + name)
        if sha(ROOT / 'experiments/auxiliary-reconstruction-300rounds/features.pkl') != cfg['auxiliary_sha256']:
            raise ValueError('Auxiliary features changed')
        if cfg['uncertainty_sha256'] is not None:
            if sha(ROOT / 'experiments/round11-auxiliary-uncertainty/features.pkl') != cfg['uncertainty_sha256']:
                raise ValueError('Uncertainty features changed')
        for f in range(1, 6):
            path = folder / f'fold-{f}.npz'
            checkpoints[str(path.relative_to(ROOT))] = sha(path)
    result = dict(passed=True, uploaded=False, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        rows=len(s), columns=list(s), input_sha256=hashes(), submission_sha256=m['submission_sha256'],
        preserved_baseline_sha256=BASE_SHA, id_alignment=True, probabilities_finite=True,
        probability_min=float(s.satisfaction.min()), probability_max=float(s.satisfaction.max()),
        oof_auc=auc, oof_labels_and_ids_aligned=True, checkpoint_sha256=checkpoints,
        selection_stability=m['selection_stability'], validation_note=m['validation_note'])
    save(OUT / 'validation.json', result)
    print(json.dumps({k: result[k] for k in ('passed', 'rows', 'submission_sha256', 'oof_auc')}, indent=2))


def finalize():
    official = load(ROOT / 'kaggle-results.json')
    before = load(ROOT / 'evidence/round11-before-upload-official-results.json')
    receipt = load(ROOT / 'last-submission.json')
    m = load(OUT / 'metrics.json')
    validation = load(OUT / 'validation.json')
    if not validation['passed'] or validation['submission_sha256'] != m['submission_sha256']:
        raise ValueError('Candidate was not validated')
    if receipt['sha256'] != m['submission_sha256'] or sha(OUT / 'submission.csv') != receipt['sha256']:
        raise ValueError('Uploaded fingerprint mismatch')
    old = load(ROOT / 'best-submission.json')
    if old['path'] != BASE or old['sha256'] != BASE_SHA or sha(ROOT / BASE) != BASE_SHA:
        raise ValueError('Official baseline changed before scoring reconciliation')
    previous_refs = {row['ref'] for row in before['submissions']}
    additions = [row for row in official['submissions'] if row['ref'] not in previous_refs]
    if len(additions) != 1:
        raise ValueError('Expected exactly one new official submission')
    new = additions[0]
    if new['status'] != 'SubmissionStatus.COMPLETE' or not new['publicScore']:
        raise ValueError('Official scoring is not complete')
    m.update(official_status='COMPLETE', official_submission_ref=new['ref'],
        official_public_score=float(new['publicScore']), official_checked_at_utc=official['checked_at_utc'])
    save(OUT / 'metrics.json', m)
    receipt.update(submission_ref=new['ref'], public_score=new['publicScore'], official_status='COMPLETE',
        official_checked_at_utc=official['checked_at_utc'], official_submission=new)
    save(ROOT / 'last-submission.json', receipt)
    save(ROOT / 'evidence/round11-submission-receipt.json', receipt)
    improved = float(new['publicScore']) > float(old['public_score'])
    if improved:
        save(ROOT / 'best-submission.json', dict(competition=COMPETITION,
            path='outputs-round11/submission.csv', sha256=m['submission_sha256'],
            submission_ref=new['ref'], public_score=new['publicScore'],
            official_checked_at_utc=official['checked_at_utc'], previous_best=old))
    validation.update(uploaded=True, submission_ref=new['ref'], official_status='COMPLETE',
        public_score=new['publicScore'], new_official_best=improved)
    save(OUT / 'validation.json', validation)
    shutil.copy2(ROOT / 'kaggle-results.json', ROOT / 'evidence/round11-after-submit-official-results.json')
    print(json.dumps(dict(submission_ref=new['ref'], public_score=new['publicScore'],
        improved=improved, previous_best_score=old['public_score'],
        competition=official['competition'], checked_at_utc=official['checked_at_utc']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['validate', 'finalize'])
    args = parser.parse_args()
    validate() if args.action == 'validate' else finalize()
