"""TLS-verified public artifacts, checksummed before installing/loading.
Only this project's virtual environment is modified; existing ML versions stay.
"""
from pathlib import Path
import subprocess
import sys
import hashlib
import json
import importlib.metadata as metadata
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'references/round18-foundation'


def fetch(url, target, seconds=25):
    target = Path(target)
    for route in (['--noproxy', '*'], ['--proxy', 'http://127.0.0.1:7897']):
        resume = ['--continue-at', '-'] if target.suffix == '.ckpt' else []
        command = ['curl.exe', '-sS', '--fail', '--location', '--connect-timeout', '8', '--max-time', str(seconds),
                   *resume,
                   '--tlsv1.2', '--tls-max', '1.2', *route, url, '-o', str(target)]
        result = subprocess.run(command, capture_output=True, timeout=seconds+5)
        if result.returncode == 0: return
    raise RuntimeError('TLS-verified public download failed: '+url.split('?')[0])


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608), b''):h.update(b)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    specs = {'tabicl':'2.2.0', 'einops':'0.8.1', 'huggingface-hub':'0.34.4', 'hf-xet':'1.1.9'}
    wheels = []; audit = {}; supported = set(sys_tags())
    for package, version in specs.items():
        try:
            if metadata.version(package) == version:
                print(json.dumps(dict(stage='already_installed', package=package, version=version)), flush=True)
                continue
            raise RuntimeError('Unexpected preexisting dependency version: '+package)
        except metadata.PackageNotFoundError: pass
        f = DEST / f'{package}-{version}-pypi.json'
        if not f.exists(): fetch(f'https://pypi.org/pypi/{package}/{version}/json', f)
        j = json.loads(f.read_text('utf-8'))
        items = [x for x in j['urls'] if x['filename'].endswith('.whl') and
                 parse_wheel_filename(x['filename'])[3] & supported]
        if not items: raise RuntimeError('No compatible wheel '+package)
        item = items[0]; wheel = DEST / item['filename']
        if not wheel.exists() or sha(wheel) != item['digests']['sha256']: fetch(item['url'], wheel)
        if sha(wheel) != item['digests']['sha256']: raise RuntimeError('Wheel checksum mismatch')
        wheels.append(wheel); audit[package] = dict(version=version, url=item['url'], sha256=sha(wheel),
                                                   requires_dist=j['info']['requires_dist'])
        print(json.dumps(dict(stage='wheel_verified', package=package, version=version)), flush=True)
    if wheels:subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', *map(str, wheels)], check=True)
    subprocess.run([sys.executable, '-c', 'from tabicl import TabICLClassifier; import inspect; print(inspect.signature(TabICLClassifier))'], check=True)
    metadata_file = DEST / 'TabICL-hf-metadata.json'
    if not metadata_file.exists(): fetch('https://huggingface.co/api/models/jingang/TabICL?blobs=true', metadata_file)
    model = json.loads(metadata_file.read_text('utf-8'))
    filename = 'tabicl-classifier-v2-20260212.ckpt'
    info = next(x for x in model['siblings'] if x['rfilename'] == filename)
    expected = info['lfs']['sha256']
    weights = DEST / filename
    url = f"https://huggingface.co/jingang/TabICL/resolve/{model['sha']}/{filename}"
    if not weights.exists() or sha(weights) != expected:fetch(url, weights, seconds=900)
    if sha(weights) != expected:raise RuntimeError('Weight checksum mismatch')
    audit['checkpoint'] = dict(repo='jingang/TabICL', revision=model['sha'], filename=filename,
                               sha256=expected, bytes=weights.stat().st_size, source_url=url)
    (DEST / 'download-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(dict(stage='foundation_ready', checkpoint_sha256=expected, bytes=weights.stat().st_size)), flush=True)


if __name__ == '__main__':main()
