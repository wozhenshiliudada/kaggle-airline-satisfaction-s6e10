"""Read-only status and output retrieval for the two frozen cloud pilots."""
from pathlib import Path
import json
import hashlib
import time
import requests
from filelock import FileLock
from datetime import datetime, timezone
import kaggle_output_transport
from kaggle.api.kaggle_api_extended import KaggleApi
from round18_joint import ROOT, save

JOBS = {
    31: ('airline-s6e10-full-context-round31', 'round31-cloud-full-context'),
    32: ('airline-s6e10-full-raw-round32', 'round32-cloud-full-raw'),
    34: ('airline-s6e10-recover-none-round34', 'round34-cloud-recovery'),
    35: ('tabpfn-airline-s6e10-v2-pilot-round35', 'round35-tabpfn2'),
    36: ('tabpfn-airline-s6e10-v2-pilot-round36', 'round36-tabpfn2-fixed'),
}


def collect_all():
    api = KaggleApi()
    api.authenticate()
    snapshots = []
    for number, (slug, variant) in JOBS.items():
        kernel = 'woshiliudada/' + slug
        for attempt in range(3):
            try:
                status = api.kernels_status(kernel).to_dict()
                break
            except requests.ConnectionError:
                if attempt == 2:
                    raise
                time.sleep(2)
        record = dict(round=number, kernel=kernel,
                      queried_at_utc=datetime.now(timezone.utc).isoformat(), **status)
        if status['status'] not in ['RUNNING', 'QUEUED']:
            folder = ROOT / 'experiments' / variant / 'outputs'
            folder.mkdir(exist_ok=True)
            manifest_path = folder / 'download-manifest.json'
            complete = False
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                complete = manifest['kernel'] == kernel and all(
                    (folder / name).is_file() and
                    hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
                    for name, expected in manifest['files_sha256'].items())
            if not complete:
                files, token = api.kernels_output(kernel, str(folder),
                    file_pattern=r'(^|/)(hardware|run-result|parent-error|budget-stop|worker-[01].*)\.(json|npz|log)$',
                    quiet=True)
                if token:
                    raise RuntimeError('Output pagination requires explicit follow-up')
                save(manifest_path, dict(kernel=kernel, files_sha256={
                    str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in folder.iterdir() if p.is_file() and p != manifest_path}))
            record['downloaded_files'] = sorted(str(p.relative_to(folder)) for p in folder.iterdir() if p.is_file())
            record['run_result_present'] = (folder / 'run-result.json').exists()
        save(ROOT / 'evidence' / f'round{number}-cloud-latest-status.json', record)
        snapshots.append(record)
    print(json.dumps(snapshots, indent=2))


def main():
    with FileLock(str(ROOT / 'evidence/cloud-collector.lock'), timeout=180):
        collect_all()


if __name__ == '__main__':
    main()
