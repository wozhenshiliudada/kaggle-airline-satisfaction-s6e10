"""Matched original-TabM control and exact/coarse route one-hot categories.

Reuses the audited round19 I/O, target-feature checks and fold pipeline, without
editing it. Native PyTabKit architecture is restored in this process. Both
arms share batch size, initialization seed, training settings and covariates;
the only feature difference is the two distance-category columns.
"""
from datetime import datetime, timezone
import argparse
from pytabkit.models.nn_models.tabm import OneHotEncoding0d
import round19_route_tabm as previous
from round18_joint import ROOT, sha, save, load

OUT=ROOT/'experiments/round20-route-control'
ANALYSIS=ROOT/'analyses/round20-route-control'
original_prepare=previous.prepare


def prepare(tr,ts,raw,a,b,y,variant):
    prepared=original_prepare(tr,ts,raw,a,b,y,'aux_route_embedding')
    if variant=='matched_control':
        prepared=tuple(f.drop(columns=['category_Flight Distance','category_fd10']) for f in prepared)
    elif variant!='route_onehot':raise ValueError('Unknown matched arm')
    return prepared


def freeze():
    if (OUT/'protocol.json').exists():raise ValueError('Already frozen')
    p=load(ROOT/'experiments/round19-route-tabm/protocol.json')
    p.update(round=20,created_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__),variants=['matched_control','route_onehot'],
        category_encoding='Unmodified PyTabKit OneHotEncoding0d, original TabM Model',
        comparison='Matched control drops the two distance categorical columns. '
                   'Both arms have batch_size=512, model_seed=43 and identical params.',
        motivation='Round19 changed both category architecture and batch size; '
                   'its negative result does not isolate distance categories.')
    p.pop('category_embedding')
    p['source_sha256'].update({s:sha(ROOT/s) for s in
         ['round19_route_tabm.py','experiments/round19-route-tabm/protocol.json']})
    save(OUT/'protocol.json',p)
    previous.msg(stage='frozen',round=20,protocol_sha256=sha(OUT/'protocol.json'))


def check():
    p=load(OUT/'protocol.json')
    if p['script_sha256']!=sha(__file__):raise ValueError('Frozen matched source changed')
    for s,h in {**p['protected_sha256'],**p['source_sha256']}.items():
        if sha(ROOT/s)!=h:raise ValueError('Changed source or protected artifact: '+s)
    for s,h in p['input_sha256'].items():
        if sha(ROOT/'data'/s)!=h:raise ValueError('Changed input')
    if sha(ROOT/'.venv/Lib/site-packages/pytabkit/models/nn_models/tabm.py')!=p['installed_model_sha256']:
        raise ValueError('Installed architecture changed')
    return p


if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('action',choices=['freeze','pilot','full'])
    arg.add_argument('--variant',choices=['matched_control','route_onehot'],default='matched_control')
    a=arg.parse_args()
    if a.action=='freeze':freeze()
    else:
        previous.OUT=OUT;previous.ANALYSIS=ANALYSIS;previous.check=check
        previous.prepare=prepare
        previous.RouteEmbeddingTabM=previous.OriginalModel
        previous.RouteCategories=OneHotEncoding0d
        previous.run(a.variant,a.action=='pilot')
