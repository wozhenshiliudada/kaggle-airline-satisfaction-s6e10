"""Official CLI with Windows TLS transport for Kaggle API calls.

Certificate verification stays enabled. Authorization headers travel only over
stdin to curl, never command arguments, console logs, or persistent files.
No account config or installed library is changed. Requests to other hosts use
the original SDK transport. Mutating requests are never automatically retried.
"""
import io
import re
import sys
import subprocess
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlsplit
import requests

original_send = requests.Session.send


def quote(value):
    value = str(value)
    if '\n' in value or '\r' in value: raise ValueError('Newline in curl config value')
    return '"'+value.replace('\\','\\\\').replace('"','\\"')+'"'


def send(self, prepared, **kwargs):
    if urlsplit(prepared.url).hostname != 'api.kaggle.com':
        return original_send(self, prepared, **kwargs)
    with tempfile.TemporaryDirectory(prefix='kaggle-transport-') as temp:
        folder=Path(temp); body=folder/'body'; headers=folder/'headers'; content=folder/'content'
        config=['url = '+quote(prepared.url), 'request = '+quote(prepared.method),
                'output = '+quote(content), 'dump-header = '+quote(headers)]
        for key,value in prepared.headers.items():config.append('header = '+quote(key+': '+value))
        if prepared.body is not None:
            payload=prepared.body.encode('utf-8') if isinstance(prepared.body,str) else prepared.body
            if hasattr(payload,'read'):payload=payload.read()
            if not isinstance(payload,bytes):raise ValueError('Unsupported SDK body')
            if len(payload)<100000 and b'\n' not in payload and b'\r' not in payload:
                config.append('data-binary = '+quote(payload.decode('utf-8')))
            else:
                body.write_bytes(payload);config.append('data-binary = '+quote('@'+str(body)))
        result=subprocess.run(['curl.exe','--silent','--show-error','--compressed','--connect-timeout','10',
            '--max-time','45','--tlsv1.2','--tls-max','1.2','--noproxy','*',
            '--write-out','%{http_code}','--config','-'],
            input=('\n'.join(config)+'\n').encode('utf-8'),capture_output=True,timeout=50)
        if result.returncode:
            # Error content is not echoed, because URLs may contain upload keys.
            raise requests.ConnectionError('Windows TLS request failed (curl code '+str(result.returncode)+')')
        response=requests.Response();response.status_code=int(result.stdout[-3:])
        response._content=content.read_bytes();response._content_consumed=True
        response.url=prepared.url;response.request=prepared;response.raw=io.BytesIO(response._content)
        sections=re.split(r'\r?\n\r?\n',headers.read_text('latin-1'))
        last=next(x for x in reversed(sections) if x.startswith('HTTP/'))
        for line in last.splitlines()[1:]:
            if ':' in line:
                key,value=line.split(':',1);response.headers[key]=value.strip()
        response.encoding=requests.utils.get_encoding_from_headers(response.headers)
        return response


requests.Session.send=send

# The installed SDK applies its 30-minute refresh tolerance after expiry,
# retaining expired tokens. Refresh shortly before expiry within this process.
from kagglesdk.kaggle_creds import KaggleCredentials
def expired(self):
    return (not self._access_token_expiration or
            self._access_token_expiration <= datetime.now(timezone.utc)+timedelta(minutes=2))
KaggleCredentials.access_token_has_expired=expired

if __name__=='__main__':
    from kaggle.cli import main
    main()
