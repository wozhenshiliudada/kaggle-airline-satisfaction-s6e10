"""Compare a small, declared blend grid and a cross-fitted linear combiner."""
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from improve import ROOT
from train import read_data, validate_submission, COMPETITION

names=['compact','route','cat','xgb']
tr,ts,sample,features,y=read_data(ROOT/'data')
members=[]; test_members=[]; reports=[]
for name in names:
    folder=ROOT/'experiments'/name
    oof=pd.read_csv(folder/'oof.csv'); prediction=pd.read_csv(folder/'test_predictions.csv')
    if not oof.id.equals(tr.id) or not prediction.id.equals(ts.id): raise ValueError('Member IDs differ')
    if not np.array_equal(oof.satisfaction,y): raise ValueError('Member labels differ')
    reports.append(json.loads((folder/'metrics.json').read_text()))
    members.append(oof.prediction.to_numpy()); test_members.append(prediction.prediction.to_numpy())
oof=np.column_stack(members); test=np.column_stack(test_members)
if not np.isfinite(oof).all() or not np.isfinite(test).all(): raise ValueError('Invalid member predictions')
if any(r['input_sha256'] != reports[0]['input_sha256'] for r in reports): raise ValueError('Inputs differ')
grid=[]
for i in range(len(names)):
    w=np.zeros(len(names)); w[i]=1; grid.append(w)
for i,j in itertools.combinations(range(len(names)),2):
    for fraction in (.25,.5,.75):
        w=np.zeros(len(names)); w[i]=fraction; w[j]=1-fraction; grid.append(w)
for idx in itertools.combinations(range(len(names)),3):
    w=np.zeros(len(names)); w[list(idx)]=1/3; grid.append(w)
grid.append(np.ones(len(names))/len(names))
comparisons=[{'method':'arithmetic','weights':dict(zip(names,w.tolist())),
              'auc':float(roc_auc_score(y,oof@w))} for w in grid]
best=max(comparisons,key=lambda r:r['auc'])
w=np.array([best['weights'][n] for n in names])
chosen_oof=oof@w; chosen_test=test@w
z=logit(np.clip(oof,1e-6,1-1e-6)); zt=logit(np.clip(test,1e-6,1-1e-6))
crossfit=np.zeros(len(y))
for a,b in StratifiedKFold(3,shuffle=True,random_state=42).split(z,y):
    model=LogisticRegression(C=.1,max_iter=500).fit(z[a],y.iloc[a])
    crossfit[b]=model.predict_proba(z[b])[:,1]
stack_auc=float(roc_auc_score(y,crossfit))
if stack_auc>best['auc']:
    model=LogisticRegression(C=.1,max_iter=500).fit(z,y)
    chosen_test=model.predict_proba(zt)[:,1]; chosen_oof=crossfit
    best={'method':'crossfit_logit_logistic','auc':stack_auc,
          'coefficients':dict(zip(names,model.coef_[0].tolist())),'intercept':float(model.intercept_[0])}
output=ROOT/'outputs-round2'; output.mkdir(exist_ok=True)
sub=sample.copy(); sub.satisfaction=sub.id.map(pd.Series(chosen_test,index=ts.id))
validate_submission(sub,sample)
sub.to_csv(output/'submission.csv',index=False)
pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':chosen_oof}).to_csv(output/'oof.csv',index=False)
report={'competition':COMPETITION,'folds':3,'seed':42,
        'model_description':'Transfer features LGBM CatBoost XGBoost',
        'selected_blend':best,'member_auc':dict(zip(names,[float(roc_auc_score(y,p)) for p in members])),
        'arithmetic_comparisons':comparisons,'crossfit_combiner_auc':stack_auc,
        'input_sha256':reports[0]['input_sha256'],'external_sha256':reports[0]['external_sha256'],
        'original_matches_removed':reports[0]['original_matches_removed'],
        'submission_sha256':hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest(),
        'validation_note':'Models are outer-fold OOF. Grid selection uses the same OOF; combiner weights are cross-fitted at OOF level, not a full nested refit of base models. Model selection can still bias validation.'}
(output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='arithmetic_comparisons'}),flush=True)
