"""Archive four completed, unused V11 weights before retiring their server26 copies."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command

PF=PROJECT/'results/preflight'
ROOT26='/data/gaob/Re-ID/DeMo-DualAxis'
ROOT27='/data/gb/Re-ID/DeMo-DualAxis'
PY26='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
PY27='/data/gb/Re-ID/conda-envs/tri_reid/bin/python'
PREFIX='runs/shared_identity_v11_trial_20261003/development'
NAMES=[PREFIX+'/MSVR310_'+v+'_s42/best.pth' for v in ('twins_shared','axis_shared','demo_shared','frequency_shared')]
DEST=ROOT27+'/archives/closed_V11_weights_20261007'


def remote(host,python,code):
    return json.loads(command(['ssh',*OPTIONS,host,shlex.quote(python)+' -'],input=code))


def main():
    proof_path=PF/'closed_V11_weights_archive27_actual_20261007.json'
    assert not proof_path.exists()
    review=json.loads((PF/'closed_V11_weights_archive27_source_review_20261007.json').read_text(encoding='utf-8'))
    assert review['status']=='PASS' and not review['blocking_findings']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==review['source_sha256']
    protected=json.loads((PF/'r201j_nonselected_N3_weights_retired_actual_20261007.json').read_text(encoding='utf-8'))['protected_anchors_and_I42_best']
    assert len(protected)==9 and set(NAMES).isdisjoint(protected)
    before=remote('2026',PY26,f'''import hashlib,json,shutil
from pathlib import Path
root=Path({ROOT26!r});names={NAMES!r};prefix=root/{PREFIX!r};protected={protected!r}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in protected.items())
for p in Path('/proc').iterdir():
 if p.name.isdigit() and (p/'cmdline').exists():
  args=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))
  assert not any(str(prefix) in a for a in args)
records=[]
for name in names:
 p=(root/name).resolve();assert p.is_relative_to(prefix.resolve()) and p.name=='best.pth'
 result=p.parent/'result.json';d=json.loads(result.read_text());assert d['status']=='COMPLETE' and d['epochs']==50
 records.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),result_sha256=hashlib.sha256(result.read_bytes()).hexdigest()))
print(json.dumps(dict(records=records,free=shutil.disk_usage(root).free)))''')
    records=before['records'];assert len(records)==4
    total=sum(r['bytes'] for r in records)
    remote('2027',PY27,f'''import json,shutil
from pathlib import Path
root=Path({DEST!r});assert not root.exists() and root.resolve().is_relative_to(Path({ROOT27!r})/'archives')
assert shutil.disk_usage(root.parent).free>{total}+1073741824
for r in {records!r}:(root/r['path']).parent.mkdir(parents=True,exist_ok=True)
print(json.dumps(dict(status='NEW_V11_ARCHIVE_READY')))''')
    for row in records:
        command(['scp','-3',*OPTIONS,'2026:'+ROOT26+'/'+row['path'],'2027:'+DEST+'/'+row['path']])
        print('V11_WEIGHT_COPIED',row['path'],flush=True)
    ack=remote('2027',PY27,f'''import hashlib,json
from pathlib import Path
root=Path({DEST!r});records={records!r}
assert all((root/r['path']).stat().st_size==r['bytes'] and hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in records)
manifest=root/'archive_manifest.json';assert not manifest.exists();manifest.write_text(json.dumps(records,indent=2)+'\\n')
print(json.dumps(dict(status='ALL_FOUR_V11_WEIGHTS_SIZE_SHA_ACK',manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))''')
    assert ack['status']=='ALL_FOUR_V11_WEIGHTS_SIZE_SHA_ACK'
    after=remote('2026',PY26,f'''import hashlib,json,shutil
from pathlib import Path
root=Path({ROOT26!r});prefix=root/{PREFIX!r};records={records!r};protected={protected!r}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in protected.items())
for r in records:
 p=(root/r['path']).resolve();assert p.is_relative_to(prefix.resolve()) and p.name=='best.pth'
 assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 assert hashlib.sha256((p.parent/'result.json').read_bytes()).hexdigest()==r['result_sha256']
for r in records:(root/r['path']).unlink()
assert all(not (root/r['path']).exists() for r in records)
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in protected.items())
print(json.dumps(dict(status='EXACT_FOUR_V11_SOURCE_COPIES_RETIRED',free=shutil.disk_usage(root).free)))''')
    proof=dict(status='ACTUAL_FOUR_UNUSED_CLOSED_V11_WEIGHTS_ARCHIVED27_AND_SERVER26_COPIES_RETIRED',verified_at=datetime.now().isoformat(timespec='seconds'),records=records,bytes=total,archive_root=DEST,archive_ack=ack,before_free=before['free'],after_free=after['free'],protected_anchors_and_I42_best=protected,new_neural_calls=0,new_optimizer_updates=0,unique_weights_lost=0,limits='Four completed historical fit/dev V11 weights only; old source paths require explicit archive27 restore for neural replay. No current I/J/original/G weights, metrics, raw or other projects modified.')
    proof_path.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('records','protected_anchors_and_I42_best')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
