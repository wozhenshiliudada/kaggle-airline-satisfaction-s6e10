"""Describe where the selected predictions change ranking, without refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from evaluate_context import pairs

ROOT = Path(__file__).resolve().parent
raw = pd.read_csv(ROOT / 'data/train.csv', usecols=['id', 'satisfaction', 'Type of Travel', 'Class'])
base = pd.read_csv(ROOT / 'outputs-round9/oof.csv')
new = pd.read_csv(ROOT / 'outputs-round11/oof.csv')
if not base.id.equals(raw.id) or not new.id.equals(raw.id):
    raise ValueError('OOF row mismatch')
if not np.array_equal(base.satisfaction, new.satisfaction):
    raise ValueError('OOF labels mismatch')
y = base.satisfaction.to_numpy()
p = base.prediction.to_numpy()
q = new.prediction.to_numpy()
shift = abs(rankdata(p) - rankdata(q))
result = dict(
    selected=json.loads((ROOT / 'outputs-round11/metrics.json').read_text())['selected_blend']['candidate'],
    scope='Descriptive reused-OOF diagnostics; no fitted parameters or new blend candidates',
    probability_correlation=float(np.corrcoef(p, q)[0, 1]),
    mean_absolute_probability_change=float(np.mean(abs(p-q))),
    absolute_rank_shift_quantiles={str(v):float(np.quantile(shift,v)) for v in (.5,.9,.99)},
    travel_pair_contribution=pairs(y,p,q,raw['Type of Travel']),
    class_pair_contribution=pairs(y,p,q,raw['Class']))
groups=[]
for field in ('Type of Travel','Class'):
    for name in sorted(raw[field].unique()):
        mask=raw[field].eq(name).to_numpy()
        a=float(roc_auc_score(y[mask],p[mask]));b=float(roc_auc_score(y[mask],q[mask]))
        groups.append(dict(field=field,group=name,rows=int(mask.sum()),baseline_auc=a,candidate_auc=b,delta=b-a))
result['group_auc']=groups
(ROOT/'analyses/round11-ranking-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
