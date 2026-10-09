"""Read-only reconciliation of the scored candidate and preserved best."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def main():
    before = load('evidence/round15-before-upload-official-results.json')
    after = load('evidence/round15-after-submit-official-results.json')
    receipt = load('last-submission.json')
    metric = load('outputs-round15/metrics.json')
    validation = load('outputs-round15/validation.json')
    best = load('best-submission.json')
    previous = {r['ref'] for r in before['submissions']}
    added = [r for r in after['submissions'] if r['ref'] not in previous]
    assert len(added) == 1
    row = added[0]
    assert all(r['status'] == 'SubmissionStatus.COMPLETE' for r in after['submissions'])
    assert validation['passed'] and validation['uploaded']
    assert row['ref'] == validation['submission_ref'] == receipt['submission_ref'] == metric['official_submission_ref']
    assert float(row['publicScore']) == float(receipt['public_score']) == metric['official_public_score']
    assert digest('outputs-round15/submission.csv') == metric['submission_sha256'] == receipt['sha256'] == validation['submission_sha256']
    assert metric['selected_blend']['delta'] >= 1e-5
    assert metric['selected_blend']['all_blocks_positive']
    assert metric['selection_stability']['all_held_blocks_positive']
    for name, value in metric['source_fingerprints'].items():
        assert digest(name) == value, name
    for name, value in metric['feature_code_sha256'].items():
        assert digest(name) == value, name
    assert digest('evaluate_round15.py') == metric['script_sha256']
    config = load('experiments/round15-lgb-10fold/configuration.json')
    reference = load('experiments/round9-lgb-auxiliary-5fold/configuration.json')
    assert config['folds'] == 10 and config['params'] == reference['params'] and config['bin_rule'] is None
    assert digest('round15_models.py') == config['script_sha256']
    for name, value in config['source_code_sha256'].items():
        assert digest(name) == value, name
    assert len(validation['checkpoint_sha256']) == 10
    for name, value in validation['checkpoint_sha256'].items():
        assert digest(name) == value, name
    assert not validation['new_official_best']
    assert best['path'] == 'outputs-round13/submission.csv'
    assert best['sha256'] == digest(best['path']) == 'b3451e0310cf0892ebd89cab80f400a5c0fe7fe3725ff93965e0499bf9225f93'
    assert best['public_score'] == '0.96132'
    best_row = next(r for r in after['submissions'] if r['ref'] == best['submission_ref'])
    assert best_row['publicScore'] == best['public_score']
    assert 'skipping duplicate upload' in (ROOT / 'evidence/round15-duplicate-after-finalize.log').read_text(encoding='utf-8')
    assert not (ROOT / 'outputs-round14/submission.csv').exists()
    team = load('team-request.json')
    assert team['membership_status'] == 'pending' and not team['require_membership_before_submission']
    assert team['prepared_prediction_sha256'] == metric['submission_sha256']
    result = dict(passed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        official_checked_at_utc=after['checked_at_utc'], new_official_submissions=1,
        submission_count=len(after['submissions']), all_official_complete=True,
        submission_ref=row['ref'], candidate_public_score=row['publicScore'],
        best_public_score=best['public_score'], best_path=best['path'],
        best_sha256=best['sha256'], old_best_unchanged=True,
        native_checkpoints=10, input_code_and_prediction_fingerprints_verified=True,
        duplicate_upload_skipped=True, round14_uploaded=False,
        team_membership_unchanged=True, competition=after['competition'])
    (ROOT / 'evidence/round15-final-state-validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
