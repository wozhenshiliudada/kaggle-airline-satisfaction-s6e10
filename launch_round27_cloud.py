"""One private GPU launch, with a structured official receipt and no retries."""
from datetime import datetime,timezone
import json
import re
from pathlib import Path
import kaggle_tls13_transport
from kaggle.api.kaggle_api_extended import KaggleApi
from round18_joint import ROOT,sha,save


def clean(value):return re.sub(r'KG(?:AT|RT)_[A-Za-z0-9._-]+','[REDACTED]',str(value))


def main():
    folder=ROOT/'cloud-round27-access';meta=json.loads((folder/'kernel-metadata.json').read_text('utf-8'))
    assert meta['is_private'] is True and meta['enable_gpu'] is True
    assert meta['machine_shape']=='NvidiaL4' and meta['id'].startswith('woshiliudada/')
    compile((folder/meta['code_file']).read_text('utf-8'),meta['code_file'],'exec')
    path=ROOT/'evidence/round27-cloud-push.json'
    if path.exists():raise ValueError('Launch has already been attempted; inspect receipt/status instead of retrying')
    receipt=dict(attempted_at_utc=datetime.now(timezone.utc).isoformat(),kernel=meta['id'],
        requested_gpu=meta['machine_shape'],is_private=True,timeout_seconds=600,
        code_sha256=sha(folder/meta['code_file']),metadata_sha256=sha(folder/'kernel-metadata.json'),
        protected_sha256={n:sha(ROOT/n) for n in ['best-submission.json','last-submission.json',
            'outputs-round13/submission.csv','team-request.json']})
    api=KaggleApi();api.authenticate()
    quota=api.quota_view()
    receipt['quota']=dict(refresh_at=str(quota.quota_refresh_time),gpu_used_seconds=quota.gpu_quota.time_used.total_seconds(),
        gpu_total_seconds=quota.gpu_quota.total_time_allowed.total_seconds())
    save(path,dict(**receipt,status='dispatching'))
    try:
        response=api.kernels_push(str(folder),timeout='600',acc='NvidiaL4')
        result=dict(ref=response.ref,url=response.url,version=response.version_number,error=response.error,
            kernel_id=response.kernel_id,invalid_competition_sources=response.invalid_competition_sources,
            invalid_dataset_sources=response.invalid_dataset_sources)
        receipt.update(status='rejected' if response.error else 'accepted',response=result)
    except Exception as e:
        response=getattr(e,'response',None)
        detail=None
        if response is not None:
            try: detail=response.json().get('error',{})
            except ValueError: detail=clean(response.text)[:1200]
        receipt.update(status='exception_check_server_before_retry',error=clean(type(e).__name__+': '+str(e))[:1800],server_error=detail)
    save(path,receipt);print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
