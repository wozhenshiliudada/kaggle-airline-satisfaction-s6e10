"""Add pinned fine-tuning dependencies without upgrading the ML environment."""
from pathlib import Path
import json
import subprocess
import sys
import importlib.metadata as md
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.requirements import Requirement
from round18_joint import ROOT,sha,save

DEST=ROOT/'references/round23-finetune'
SPECS={'transformers':'4.57.1','tokenizers':'0.22.1','safetensors':'0.6.2','regex':'2025.9.18'}


def fetch(url,file):
    urls=[url]
    if url.startswith('https://files.pythonhosted.org/packages/'):
        urls.insert(0,url.replace('https://files.pythonhosted.org/','https://mirrors.aliyun.com/pypi/'))
    for address in urls:
        for version in ['1.3','1.2']:
            r=subprocess.run(['curl.exe','-sS','--fail','--location','--connect-timeout','8','--max-time','180',
                '--noproxy','*','--tlsv'+version,'--tls-max',version,address,'-o',str(file)],capture_output=True,timeout=185)
            if r.returncode==0:return
    raise ValueError('Verified public download failed: '+url.split('?')[0])


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    before={p:md.version(p) for p in ['torch','pytabkit','numpy','scikit-learn','lightgbm','tabicl','huggingface-hub']}
    wheels=[];audit={};supported=set(sys_tags())
    for package,version in SPECS.items():
        file=DEST/(package+'-'+version+'-pypi.json')
        if not file.exists():fetch(f'https://pypi.org/pypi/{package}/{version}/json',file)
        j=json.loads(file.read_text('utf-8'))
        available=[x for x in j['urls'] if x['filename'].endswith('.whl') and parse_wheel_filename(x['filename'])[3]&supported]
        if not available:raise ValueError('No compatible wheel: '+package)
        item=available[0];wheel=DEST/item['filename']
        if not wheel.exists() or sha(wheel)!=item['digests']['sha256']:fetch(item['url'],wheel)
        if sha(wheel)!=item['digests']['sha256']:raise ValueError('Wheel digest mismatch')
        for text in j['info']['requires_dist'] or []:
            req=Requirement(text)
            if req.marker and not req.marker.evaluate({'extra':''}):continue
            actual=SPECS.get(req.name)
            if actual is None:actual=md.version(req.name)
            if actual not in req.specifier:raise ValueError('Dependency conflict: '+str(req)+' installed '+actual)
        wheels.append(str(wheel));audit[package]=dict(version=version,url=item['url'],sha256=sha(wheel))
        print(json.dumps(dict(stage='wheel_verified',package=package,version=version)),flush=True)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps',*wheels],check=True)
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    after={p:md.version(p) for p in before}
    if before!=after:raise ValueError('Existing ML dependency changed')
    save(DEST/'download-audit.json',dict(packages=audit,existing_ml_versions=before,existing_ml_versions_unchanged=True))
    subprocess.run([sys.executable,'-c','from tabicl import FinetunedTabICLClassifier; print("Fine-tuning import PASS")'],check=True)


if __name__=='__main__':main()
