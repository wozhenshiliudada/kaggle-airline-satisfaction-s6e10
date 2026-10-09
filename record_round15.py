"""Validate the tenfold tree candidate and reconcile its official scoring receipt."""
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

OUT = ROOT / 'outputs-round15'
BASE = 'outputs-round13/submission.csv'
BASE_SHA = 'b3451e0310cf0892ebd89cab80f400a5c0fe7fe3725ff93965e0499bf9225f93'


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
    if sha(ROOT / 'evaluate_round15.py') != m['script_sha256']:
        raise ValueError('Selection code changed')
    for name, digest in m['feature_code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Evaluation dependency changed: ' + name)
    selected = m['selected_blend']
    if not selected or not selected['all_blocks_positive'] or selected['delta'] < 1e-5:
        raise ValueError('Candidate failed predeclared screening')
    if not m['selection_stability']['all_held_blocks_positive']:
        raise ValueError('Leave-block screening failed')
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
    for name,digest in m['source_fingerprints'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Source prediction or checkpoint changed: '+name)
    if m['source_status']!={'tree':'completed_ten_folds'}:
        raise ValueError('Tenfold native model was not completed')
    folder=ROOT/'experiments/round15-lgb-10fold';cfg=load(folder/'configuration.json')
    ref=ROOT/'experiments/round9-lgb-auxiliary-5fold';rc=load(ref/'configuration.json')
    if cfg['params']!=rc['params'] or cfg['bin_rule'] is not None:
        raise ValueError('Unexpected tree parameter changes')
    if cfg['folds']!=10 or cfg['input_sha256']!=hashes() or cfg['script_sha256']!=sha(ROOT/'round15_models.py'):
        raise ValueError('Native training configuration changed')
    if cfg['source_reference_sha256']!=sha(ref/'configuration.json'):
        raise ValueError('Native reference changed')
    if cfg['base_cache_sha256']!=sha(ROOT/'experiments/base.joblib') or cfg['base_cache_sha256']!=rc['base_cache_sha256']:
        raise ValueError('Feature cache changed')
    if cfg['auxiliary_features_sha256']!=sha(ROOT/'experiments/auxiliary-reconstruction-300rounds/features.pkl'):
        raise ValueError('Auxiliary features changed')
    if cfg['external_sha256']!=sha(ROOT/'data-original/data.csv'):
        raise ValueError('Original source changed')
    for name,digest in cfg['source_code_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Training dependency changed')
    for f in range(1,11):
        score=load(folder/f'fold-{f}.json')
        if score['features']!=292 or score['flight_distance_bin_count']>255:
            raise ValueError('Native feature or bin configuration changed')
        path=folder/f'fold-{f}.npz';checkpoints[str(path.relative_to(ROOT))]=sha(path)
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
    before = load(ROOT / 'evidence/round15-before-upload-official-results.json')
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
    save(ROOT / 'evidence/round15-submission-receipt.json', receipt)
    improved = float(new['publicScore']) > float(old['public_score'])
    if improved:
        save(ROOT / 'best-submission.json', dict(competition=COMPETITION,
            path='outputs-round15/submission.csv', sha256=m['submission_sha256'],
            submission_ref=new['ref'], public_score=new['publicScore'],
            official_checked_at_utc=official['checked_at_utc'], previous_best=old))
    validation.update(uploaded=True, submission_ref=new['ref'], official_status='COMPLETE',
        public_score=new['publicScore'], new_official_best=improved)
    save(OUT / 'validation.json', validation)
    shutil.copy2(ROOT / 'kaggle-results.json', ROOT / 'evidence/round15-after-submit-official-results.json')
    print(json.dumps(dict(submission_ref=new['ref'], public_score=new['publicScore'],
        improved=improved, previous_best_score=old['public_score'],
        competition=official['competition'], checked_at_utc=official['checked_at_utc']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['validate', 'finalize'])
    args = parser.parse_args()
    validate() if args.action == 'validate' else finalize()
