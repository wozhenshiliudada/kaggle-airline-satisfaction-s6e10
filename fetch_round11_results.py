"""Bounded retries for read-only official CLI queries during TLS interruptions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CLI = Path(sys.executable).with_name('kaggle.exe')
COMPETITION = 'playground-series-s6e10'


def request(*args):
    for attempt in range(1, 4):
        try:
            result = subprocess.run([str(CLI), *args, '--format', 'json'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
        except subprocess.TimeoutExpired:
            if attempt == 3:
                raise RuntimeError('Official read-only query timed out after three attempts')
            print(json.dumps(dict(request=args[:2], attempt=attempt, error='timeout')), flush=True)
            continue
        if result.returncode == 0:
            return json.loads(result.stdout)
        error = re.sub(r'KG(?:AT|RT)_[A-Za-z0-9_-]+', '[REDACTED]', result.stderr + result.stdout)
        print(json.dumps(dict(request=args[:2], attempt=attempt, error=error[:1800])), flush=True)
        # Authentication and policy errors should not trigger retries or login changes.
        if attempt == 3 or not any(term in error for term in ('SSLError', 'ConnectionError', 'timed out', 'Max retries exceeded')):
            raise RuntimeError('Official query failed; previous results file preserved')
        time.sleep(1)


submissions = request('competitions', 'submissions', COMPETITION)
competition = request('competitions', 'list', '--group', 'entered', '--search', COMPETITION)
report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), submissions=submissions, competition=competition)
(ROOT / 'kaggle-results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(dict(checked_at_utc=report['checked_at_utc'], competition=competition,
    submissions_count=len(submissions)), indent=2), flush=True)
