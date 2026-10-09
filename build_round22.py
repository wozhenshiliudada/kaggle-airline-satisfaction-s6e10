"""Create one prospectively fixed lower-regularization follow-up; no source edits."""
from round18_joint import ROOT,sha,load

previous=load(ROOT/'analyses/round21-preserved/crossfit.json')
if previous['eligible']:raise ValueError('Complete eligible round21 before a follow-up')
source=ROOT/'round21_preserved.py';target=ROOT/'round22_preserved.py'
if target.exists():raise ValueError('Follow-up source already exists')
text=source.read_text('utf-8')
replacements={
    "round21-preserved":"round22-preserved",
    "dict(round=21,":"dict(round=22,",
    "l2_to_best=1e-3":"l2_to_best=1e-4",
    "round=21,protocol_sha256":"round=22,protocol_sha256",
    "outputs-round21":"outputs-round22",
}
for old,new in replacements.items():
    if old not in text:raise ValueError('Expected source marker absent: '+old)
    text=text.replace(old,new)
marker="    if (OUT/'protocol.json').exists():raise ValueError('Already frozen')"
text=text.replace(marker, marker+"\n    predecessor=load(ROOT/'analyses/round21-preserved/crossfit.json')"
    "\n    if predecessor['eligible']:raise ValueError('Complete round21 selection first')")
marker="        previous_outcomes_sha256=sha(ROOT/'evidence/round19-20-outcomes.json'))"
text=text.replace(marker,"        previous_outcomes_sha256=sha(ROOT/'evidence/round19-20-outcomes.json'),"
    "\n        predecessor_result_sha256=sha(ROOT/'analyses/round21-preserved/crossfit.json'),"
    "\n        predecessor_source_sha256=sha(ROOT/'round21_preserved.py'),"
    "\n        followup_rule='Only L2 shrinks from .001 to .0001; same members, folds, bounds, pairs and gates.')")
target.write_text(text,encoding='utf-8')
print('Follow-up source created:',target.name,'SHA256',sha(target))
