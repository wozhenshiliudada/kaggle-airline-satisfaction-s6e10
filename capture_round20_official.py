"""Read-only official snapshots with verified TLS and safe UTF-8 capture."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import sys
import json
import re
import argparse
from round18_joint import ROOT,save

queries={
    'competition':['competitions','list','--group','entered','--search','playground-series-s6e10','--format','json'],
    'submissions':['competitions','submissions','playground-series-s6e10','--format','json'],
    'leaderboard':['competitions','leaderboard','playground-series-s6e10','--show','--format','json'],
}
arg=argparse.ArgumentParser();arg.add_argument('--prefix',default='round20')
prefix=arg.parse_args().prefix
if not re.fullmatch(r'round[0-9]+',prefix):raise ValueError('Invalid snapshot prefix')
results={}
for key,args in queries.items():
    attempts=[]
    for shim in ['kaggle_tls13_transport.py','kaggle_windows_transport.py']:
        r=subprocess.run([sys.executable,str(ROOT/shim),*args],cwd=ROOT,
                         capture_output=True,text=True,encoding='utf-8',errors='replace')
        attempts.append(dict(transport=shim,returncode=r.returncode))
        if r.returncode==0:
            content=re.sub(r'KG(?:AT|RT)_[A-Za-z0-9_-]+','[REDACTED]',r.stdout)
            content=re.sub(r'^Next Page Token = [^\r\n]*\r?\n','',content).lstrip()
            (ROOT/'evidence'/f'{prefix}-official-{key}.json').write_text(content,encoding='utf-8')
            try:results[key]=dict(successful=True,data=json.loads(content),attempts=attempts)
            except ValueError:results[key]=dict(successful=False,error='Expected JSON not received',attempts=attempts)
            break
    else:results[key]=dict(successful=False,error='Official read failed',attempts=attempts)
    print(json.dumps(dict(key=key,successful=results[key]['successful'],attempts=attempts)),flush=True)
save(ROOT/'evidence'/f'{prefix}-official-snapshot.json',dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),results=results))
