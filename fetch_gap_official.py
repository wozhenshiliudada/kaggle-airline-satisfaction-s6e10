"""Read current leaderboard and account position without submitting predictions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CLI = Path(sys.executable).with_name('kaggle.exe')
COMPETITION = 'playground-series-s6e10'


def request(*args):
    for attempt in range(1, 4):
        try:
            p = subprocess.run([str(CLI), *args, '--format', 'json'], capture_output=True,
                text=True, encoding='utf-8', errors='replace', timeout=30)
        except subprocess.TimeoutExpired:
            if attempt == 3:
                raise RuntimeError('Read-only request timed out')
            print(json.dumps({'request':args[:2], 'attempt':attempt, 'error':'timeout'}), flush=True)
            continue
        if p.returncode == 0:
            # The official leaderboard CLI prints a paging hint before JSON.
            body='\n'.join(line for line in p.stdout.splitlines() if not line.startswith('Next Page Token = '))
            return json.loads(body)
        error = re.sub(r'KG(?:AT|RT)_[A-Za-z0-9_-]+', '[REDACTED]', p.stderr+p.stdout)
        print(json.dumps({'request':args[:2], 'attempt':attempt, 'error':error[:1000]}), flush=True)
        if attempt == 3 or not any(s in error for s in ('SSLError','ConnectionError','timed out','Max retries exceeded')):
            raise RuntimeError('Read-only official query failed')


if __name__ == '__main__':
    leaderboard = request('competitions','leaderboard',COMPETITION,'--show','--page-size','200')
    competition = request('competitions','list','--group','entered','--search',COMPETITION)
    submissions = request('competitions','submissions',COMPETITION)
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), competition=competition,
        leaderboard=leaderboard, submissions=submissions)
    (ROOT/'evidence/gap-official.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(checked_at_utc=report['checked_at_utc'],competition=competition,
        leaderboard_rows=len(leaderboard),top_rows=leaderboard[:5],submission_count=len(submissions)),ensure_ascii=False,indent=2))
