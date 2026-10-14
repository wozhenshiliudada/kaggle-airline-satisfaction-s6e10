"""Credential-safe, verified-TLS GET adapter for official output downloads."""
import io
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlsplit
import requests
import kaggle_tls13_transport
from kaggle_windows_transport import quote

_original_send = requests.Session.send


def send(self, prepared, **kwargs):
    if prepared.method != 'GET' or urlsplit(prepared.url).hostname != 'www.kaggleusercontent.com':
        return _original_send(self, prepared, **kwargs)
    if urlsplit(prepared.url).scheme != 'https':
        raise requests.ConnectionError('Output download requires HTTPS')
    with tempfile.TemporaryDirectory(prefix='kaggle-read-output-') as temp:
        path = Path(temp) / 'content'
        config = ['url = ' + quote(prepared.url), 'output = ' + quote(path)]
        # The SDK output URL is signed; no account authorization header is sent.
        r = subprocess.run(['curl.exe', '--silent', '--show-error', '--location',
            '--connect-timeout', '10', '--max-time', '45', '--tlsv1.3', '--tls-max', '1.3',
            '--noproxy', '*', '--write-out', '%{http_code}', '--config', '-'],
            input=('\n'.join(config) + '\n').encode(), capture_output=True, timeout=50)
        if r.returncode or r.stdout[-3:] != b'200':
            raise requests.ConnectionError('Output GET failed without exposing its signed URL')
        response = requests.Response()
        response.status_code = 200
        response._content = path.read_bytes()
        response._content_consumed = True
        response.raw = io.BytesIO(response._content)
        response.request = prepared
        response.url = prepared.url
        return response


requests.Session.send = send
