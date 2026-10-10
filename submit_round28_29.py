"""Bounded official SDK upload and scoring reconciliation for R28/R29.

Credentials stay in the official SDK and credential-safe TLS transport. A saved
attempt prevents automatic mutation retries. SDK submission IDs and unique
descriptions identify receipts even if other account jobs submit concurrently.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import io

import pandas as pd

from round18_joint import ROOT, sha, save, load
from train import COMPETITION, validate_submission


def msg(**value):
    from round22_preserved import msg as emit
    emit(**value)


def api_client():
    import kaggle_tls13_transport  # Verified certificates; credentials on stdin.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def snapshot(api):
    from kagglesdk.competitions.types.competition_api_service import (
        ApiGetCompetitionRequest, ApiGetSubmissionLimitsRequest)
    with api.build_kaggle_client() as client:
        request = ApiGetCompetitionRequest()
        request.competition_name = COMPETITION
        competition = client.competitions.competition_api_client.get_competition(request).to_dict()
        request = ApiGetSubmissionLimitsRequest()
        request.competition_name = COMPETITION
        limits = client.competitions.competition_api_client.get_submission_limits(request).to_dict()
    submissions = [s.to_dict() for s in api.competition_submissions(COMPETITION)]
    return dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                competition=competition, limits=limits, submissions=submissions)


def files(round_number):
    if round_number not in (28, 29):
        raise ValueError('Unexpected round')
    folder = ROOT / f'outputs-round{round_number}'
    attempt = ROOT / 'evidence' / f'round{round_number}-official-attempt.json'
    return folder, attempt


def check_ready(round_number):
    folder, attempt = files(round_number)
    plan = load(ROOT / 'evidence/round28-29-submission-plan.json')
    # Both rounds must finish before any upload changes the protected receipts.
    for round_id in plan['eligible_rounds']:
        report = load(ROOT / 'evidence' / f'round{round_id}-final-validation.json')
        if not report['passed']:
            raise ValueError('Incomplete round validation')
    report = load(folder / 'validation.json')
    m = load(folder / 'metrics.json')
    if not report['passed'] or not report['eligible'] or not m['eligible']:
        raise ValueError('Not an eligible validated candidate')
    digest = sha(folder / 'submission.csv')
    if digest != report['submission_sha256'] or digest != m['submission_sha256']:
        raise ValueError('Prediction file changed')
    if sha(folder / 'oof.csv') != report['oof_sha256']:
        raise ValueError('OOF changed after validation')
    if sha(folder / 'metrics.json') != report['metrics_sha256']:
        raise ValueError('Candidate metadata changed before upload')
    if sha(ROOT / 'verify_round28_29.py') != report['verifier_sha256']:
        raise ValueError('Candidate verifier changed')
    if report['protocol_sha256'] != plan['before_evaluation_protocols'][str(round_number)]:
        raise ValueError('Frozen experiment differs from plan')
    for name, expected in m['input_sha256'].items():
        if sha(ROOT / 'data' / name) != expected:
            raise ValueError('Competition input changed')
    if sha(ROOT / 'data-original/data.csv') != m['external_sha256']:
        raise ValueError('External source changed')
    validate_submission(pd.read_csv(folder / 'submission.csv'),
                        pd.read_csv(ROOT / 'data/sample_submission.csv'))
    team = load(ROOT / 'team-request.json')
    if (team.get('require_membership_before_submission', True) and
            team.get('membership_status') not in ('joined', 'cancelled')):
        raise ValueError('An active user team constraint blocks submission')
    if attempt.exists():
        raise ValueError('Attempt already recorded; use refresh rather than re-upload')
    attempts = list((ROOT / 'evidence').glob('round2[89]-official-attempt.json'))
    if len(attempts) >= plan['maximum_new_submissions']:
        raise ValueError('Frozen account submission budget reached')
    for previous in attempts:
        if load(previous)['sha256'] == digest:
            raise ValueError('Identical prediction already attempted')
    return folder, attempt, m, report


def upload(round_number):
    folder, path, m, report = check_ready(round_number)
    api = api_client()
    before = snapshot(api)
    if not before['competition'].get('userHasEntered'):
        raise ValueError('Account has not entered this competition')
    if before['limits'].get('numAllowedNow', 0) < 1:
        raise ValueError('No official submission allowance remains')
    current_best = max(float(s.get('publicScore') or 0) for s in before['submissions'])
    now = datetime.now(timezone.utc)
    description = (f'R{round_number} matched-native SHA {m["submission_sha256"][:12]} '
                   f'OOF {m["selected_blend"]["auc"]:.6f} {now:%Y%m%dT%H%M%SZ}')
    state = dict(round=round_number, competition=COMPETITION,
        path=str((folder / 'submission.csv').relative_to(ROOT)).replace('\\', '/'),
        sha256=m['submission_sha256'], description=description,
        started_at_utc=now.isoformat(), status='attempt_recorded',
        account_best_before=current_best, before=before,
        validation_sha256=sha(folder / 'validation.json'),
        submitter_sha256=sha(__file__), retry_policy='Never retry uploads automatically')
    save(path, state)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            response = api.competition_submit(str(folder / 'submission.csv'), description,
                                              COMPETITION, quiet=True)
        if not response.ref:
            state.update(status='sdk_did_not_return_submission_id',
                         sdk_message=response.message)
            save(path, state)
            raise RuntimeError('No official submission ID; inspect before retrying')
        state.update(status='uploaded', submission_ref=int(response.ref),
            sdk_message=response.message, uploaded_at_utc=datetime.now(timezone.utc).isoformat())
        save(path, state)
        save(ROOT / 'last-submission.json', dict(sha256=state['sha256'],
            submission_ref=state['submission_ref'], receipt=response.message,
            description=description, round=round_number, official_status='UPLOADED'))
        msg(stage='uploaded', round=round_number, submission_ref=state['submission_ref'],
            sha256=state['sha256'], remaining_before=before['limits']['numAllowedNow'])
    except Exception as error:
        state.setdefault('submission_ref', 0)
        state['error_type'] = type(error).__name__
        save(path, state)
        raise RuntimeError('Upload needs receipt inspection: ' + type(error).__name__) from None


def refresh(round_number):
    folder, path = files(round_number)
    state = load(path)
    if sha(folder / 'submission.csv') != state['sha256']:
        raise ValueError('Attempted file changed')
    if state.get('status') == 'complete':
        msg(stage='already_reconciled', round=round_number,
            submission_ref=state['submission_ref'], public_score=state['public_score'],
            new_official_best=state['new_official_best'])
        return
    api = api_client()
    official = snapshot(api)
    matches = [s for s in official['submissions'] if
        (state.get('submission_ref') and s['ref'] == state['submission_ref']) or
        s.get('description') == state['description']]
    if len(matches) != 1:
        save(ROOT / 'evidence' / f'round{round_number}-receipt-inspection.json', official)
        raise ValueError('Could not uniquely identify the official submission; do not upload again')
    new = matches[0]
    if new.get('description') != state['description']:
        raise ValueError('Official ID and description disagree')
    status = new.get('status', '').split('.')[-1]
    state.update(submission_ref=new['ref'], official_status=status,
                 official_checked_at_utc=official['checked_at_utc'])
    save(ROOT / 'evidence' / f'round{round_number}-after-official.json', official)
    if status != 'COMPLETE' or not new.get('publicScore'):
        save(path, state)
        msg(stage='scoring', round=round_number, status=status, submission_ref=new['ref'])
        return
    score = float(new['publicScore'])
    best = load(ROOT / 'best-submission.json')
    improved = score > float(best['public_score']) and score > state['account_best_before']
    if improved:
        save(ROOT / 'best-submission.json', dict(competition=COMPETITION,
            path=state['path'], sha256=state['sha256'], submission_ref=new['ref'],
            public_score=new['publicScore'], official_checked_at_utc=official['checked_at_utc'],
            previous_best=best))
    state.update(status='complete', public_score=new['publicScore'], official_submission=new,
                 new_official_best=bool(improved))
    save(path, state)
    save(ROOT / 'evidence' / f'round{round_number}-submission-receipt.json', state)
    save(ROOT / 'last-submission.json', dict(sha256=state['sha256'],
        submission_ref=new['ref'], public_score=new['publicScore'], official_status='COMPLETE',
        description=state['description'], round=round_number,
        official_checked_at_utc=official['checked_at_utc']))
    m = load(folder / 'metrics.json')
    m.update(official_status='COMPLETE', official_submission_ref=new['ref'],
             official_public_score=score, official_checked_at_utc=official['checked_at_utc'])
    save(folder / 'metrics.json', m)
    validation = load(folder / 'validation.json')
    validation.update(uploaded=True, official_submission_ref=new['ref'],
        official_public_score=score, new_official_best=bool(improved),
        official_metrics_sha256=sha(folder / 'metrics.json'))
    save(folder / 'validation.json', validation)
    save(ROOT / 'kaggle-results.json', official)
    msg(stage='official_complete', round=round_number, public_score=score,
        previous_account_best=state['account_best_before'], new_official_best=bool(improved),
        submission_ref=new['ref'], rank=official['competition'].get('userRank'),
        teams=official['competition'].get('teamCount'),
        remaining=official['limits'].get('numAllowedNow'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['upload', 'refresh'])
    parser.add_argument('round', type=int, choices=[28, 29])
    args = parser.parse_args()
    upload(args.round) if args.stage == 'upload' else refresh(args.round)
