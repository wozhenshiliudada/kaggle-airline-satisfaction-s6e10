"""Save the declared 10% wide-CatBoost candidate for a future official check.

This candidate failed the conservative all-five-block gate. It is explicitly
marked exploratory and does not replace the already scored round6 baseline.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from context_round import ROOT,sha
from train import read_data,COMPETITION,validate_submission


def run():
    analysis_path=ROOT/'analyses'/'wide-cat-comparison.json'
    analysis=json.loads(analysis_path.read_text())
    chosen=next(c for c in analysis['coarse_scan'] if c['candidate']=='wide_cat_weight0.1')
    tr,ts,sample,raw,ys=read_data(ROOT/'data');y=ys.to_numpy()
    hashes={n:sha(ROOT/'data'/n) for n in ('train.csv','test.csv','sample_submission.csv')}
    if analysis['competition']!=COMPETITION or analysis['input_sha256']!=hashes:raise ValueError('Inputs changed')
    members=[]
    for name,folder in [('baseline',ROOT/'outputs-round6'),('wide_cat',ROOT/'experiments'/'cat-wide-conditional-5fold')]:
        m=json.loads((folder/'metrics.json').read_text());o=pd.read_csv(folder/'oof.csv');s=pd.read_csv(folder/'submission.csv')
        if m['input_sha256']!=hashes or sha(folder/'submission.csv')!=analysis['source_submission_sha256'][name]:
            raise ValueError('Source prediction fingerprints changed')
        if not o.id.equals(tr.id) or not np.array_equal(o.satisfaction,y):raise ValueError('OOF rows changed')
        validate_submission(s,sample);members.append((o.prediction.to_numpy(),s.satisfaction.to_numpy()))
    p=.9*members[0][0]+.1*members[1][0];pt=.9*members[0][1]+.1*members[1][1]
    if abs(roc_auc_score(y,p)-chosen['auc'])>1e-12:raise ValueError('Exploratory AUC reconstruction mismatch')
    if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid OOF probabilities')
    out=ROOT/'outputs-round7-experimental';out.mkdir(exist_ok=True)
    sub=sample.copy();sub.satisfaction=pt;validate_submission(sub,sample)
    sub.to_csv(out/'submission.csv',index=False)
    pd.DataFrame({'id':tr.id,'satisfaction':y,'prediction':p}).to_csv(out/'oof.csv',index=False)
    report=dict(analysis)
    report['conservative_filter_selected_blend']=analysis['selected_blend']
    report['selected_blend']=chosen
    report['model_description']='Exploratory 10pct wide conditional CatBoost plus round6'
    report['selection_status']='Exploratory candidate; four of five blocks improve, one declines; conservative all-block and leave-one-block checks failed. Official gain unverified.'
    report['export_script_sha256']=sha(Path(__file__))
    report['source_analysis_sha256']=sha(analysis_path)
    report['submission_sha256']=sha(out/'submission.csv')
    report['submission_status']='Not submitted; teammate membership confirmation pending.'
    report['validation_note']+=' Saved separately for an exploratory official check after team confirmation; this is not an accepted result of the conservative all-block filter. Existing official-best round6 is preserved.'
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'candidate':chosen,'submission_rows':len(sub),'submission_sha256':report['submission_sha256'],
                      'selection_status':report['selection_status'],'output':str(out)},indent=2),flush=True)


if __name__=='__main__':run()
