"""Validate a concrete candidate and record official API results after upload."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from context_round import ROOT, sha
from auxiliary_round import hashes
from train import read_data, validate_submission, COMPETITION

OUT = ROOT / 'outputs-round9'


def validate():
    tr, ts, sample, raw, y = read_data(ROOT / 'data')
    m = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
    if m['competition'] != COMPETITION or m['input_sha256'] != hashes():
        raise ValueError('Candidate competition input mismatch')
    if not m['selected_blend']['all_blocks_positive'] or m['selected_blend']['delta'] <= 0:
        raise ValueError('Candidate did not pass predeclared selection')
    current = json.loads((ROOT / 'best-submission.json').read_text(encoding='utf-8'))
    if current['path'] != 'outputs-round8/submission.csv' or sha(ROOT / current['path']) != current['sha256']:
        raise ValueError('Preserved official baseline changed')
    if sha(OUT / 'submission.csv') != m['submission_sha256']:
        raise ValueError('Candidate fingerprint mismatch')
    s = pd.read_csv(OUT / 'submission.csv')
    validate_submission(s, sample)
    o = pd.read_csv(OUT / 'oof.csv')
    if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction, y):
        raise ValueError('OOF row alignment mismatch')
    auc = float(roc_auc_score(y, o.prediction))
    if abs(auc - m['selected_blend']['auc']) > 1e-12:
        raise ValueError('Saved OOF score mismatch')
    checkpoint_hashes = {}
    for kind in ('lgb', 'tabm'):
        folder = ROOT / 'experiments' / ('round9-' + kind + '-auxiliary-5fold')
        if (folder / 'metrics.json').exists():
            for path in sorted(folder.glob('fold-*.npz')):
                checkpoint_hashes[str(path.relative_to(ROOT))] = sha(path)
    result = dict(passed=True, uploaded=False, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        rows=len(s), columns=list(s), input_sha256=hashes(), submission_sha256=m['submission_sha256'],
        preserved_baseline_sha256=current['sha256'], id_alignment=True, probabilities_finite=True,
        probability_min=float(s.satisfaction.min()), probability_max=float(s.satisfaction.max()),
        oof_auc=auc, oof_labels_and_ids_aligned=True, checkpoint_sha256=checkpoint_hashes,
        selection_stability=m['selection_stability'], validation_note=m['validation_note'])
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'rows', 'submission_sha256', 'oof_auc')}, indent=2))


def finalize():
    official = json.loads((ROOT / 'kaggle-results.json').read_text(encoding='utf-8'))
    receipt_path = ROOT / 'last-submission.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    mpath = OUT / 'metrics.json'
    m = json.loads(mpath.read_text(encoding='utf-8'))
    if receipt['sha256'] != m['submission_sha256'] or sha(OUT / 'submission.csv') != receipt['sha256']:
        raise ValueError('Uploaded file fingerprint mismatch')
    # Before-upload snapshot makes identification independent of description reuse.
    before = json.loads((ROOT / 'evidence/round9-before-upload-official-results.json').read_text(encoding='utf-8'))
    previous_refs = {r['ref'] for r in before['submissions']}
    additions = [r for r in official['submissions'] if r['ref'] not in previous_refs]
    if len(additions) != 1:
        raise ValueError('Expected exactly one new official submission')
    new = additions[0]
    if new['status'] != 'SubmissionStatus.COMPLETE' or not new['publicScore']:
        raise ValueError('Official scoring is not complete')
    m.update(official_status='COMPLETE', official_submission_ref=new['ref'],
        official_public_score=float(new['publicScore']), official_checked_at_utc=official['checked_at_utc'])
    mpath.write_text(json.dumps(m, indent=2), encoding='utf-8')
    receipt.update(submission_ref=new['ref'], public_score=new['publicScore'], official_status='COMPLETE',
        official_checked_at_utc=official['checked_at_utc'], official_submission=new)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (ROOT / 'evidence/round9-submission-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    current_path = ROOT / 'best-submission.json'
    old = json.loads(current_path.read_text(encoding='utf-8'))
    improved = float(new['publicScore']) > float(old['public_score'])
    if improved:
        updated = dict(competition=COMPETITION, path='outputs-round9/submission.csv',
            sha256=m['submission_sha256'], submission_ref=new['ref'], public_score=new['publicScore'],
            official_checked_at_utc=official['checked_at_utc'], previous_best=old)
        current_path.write_text(json.dumps(updated, indent=2), encoding='utf-8')
    vpath = OUT / 'validation.json'
    v = json.loads(vpath.read_text(encoding='utf-8'))
    v.update(uploaded=True, submission_ref=new['ref'], official_status='COMPLETE',
        public_score=new['publicScore'], new_official_best=improved)
    vpath.write_text(json.dumps(v, indent=2), encoding='utf-8')
    shutil.copy2(ROOT / 'kaggle-results.json', ROOT / 'evidence/round9-after-submit-official-results.json')
    print(json.dumps(dict(submission=new, improved=improved, previous_best=old,
        competition=official['competition'], checked_at_utc=official['checked_at_utc']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['validate', 'finalize'])
    args = p.parse_args()
    validate() if args.action == 'validate' else finalize()
