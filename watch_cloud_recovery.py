"""One-shot bounded watcher: collect and audit recovery, never submit."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import requests
import kaggle_output_transport
from kaggle.api.kaggle_api_extended import KaggleApi
from round18_joint import ROOT, save


def main():
    api = KaggleApi()
    api.authenticate()
    kernel = 'woshiliudada/airline-s6e10-recover-none-round34'
    started = time.monotonic()
    previous = None
    # One local watcher for this already-launched job; not a scheduled automation.
    while time.monotonic() - started < 6300:
        try:
            status = api.kernels_status(kernel).to_dict()
            record = dict(queried_at_utc=datetime.now(timezone.utc).isoformat(),
                          watcher_elapsed_seconds=time.monotonic() - started, **status)
            save(ROOT / 'evidence/round34-watch-state.json', record)
            if status['status'] != previous:
                print(json.dumps(record), flush=True)
                previous = status['status']
            if status['status'] not in ['RUNNING', 'QUEUED']:
                r = subprocess.run([sys.executable, str(ROOT / 'collect_cloud_context.py')],
                                   cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
                if r.returncode:
                    save(ROOT / 'evidence/round34-watch-result.json',
                         dict(successful=False, stage='read_only_output_collection', exit_code=r.returncode))
                    raise RuntimeError('Completed output collection needs follow-up')
                r = subprocess.run([sys.executable, str(ROOT / 'score_cloud_context.py'), '--round', '34'],
                                   cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
                save(ROOT / 'evidence/round34-watch-result.json',
                     dict(successful=r.returncode == 0, stage='audit_finished', exit_code=r.returncode,
                          competition_submission_attempted=False))
                if r.returncode:
                    raise RuntimeError('Recovery scoring needs follow-up')
                print('RECOVERY_AUDIT_READY', flush=True)
                print(r.stdout, flush=True)
                return
        except requests.ConnectionError:
            save(ROOT / 'evidence/round34-watch-state.json',
                 dict(queried_at_utc=datetime.now(timezone.utc).isoformat(),
                      status='READ_ERROR_NOT_A_JOB_FAILURE', retry_read_only=True))
        time.sleep(60)
    save(ROOT / 'evidence/round34-watch-result.json',
         dict(successful=False, stage='watcher_time_budget', competition_submission_attempted=False))
    raise RuntimeError('Watcher finished without completed output evidence')


if __name__ == '__main__':
    main()
