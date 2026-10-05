"""One source-reviewed native launch; copy evidence before bounded raw clear."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified


REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/rgbnt201_identity_outlet_native_20261005'
LOCAL=PROJECT/'results/rgbnt201_identity_outlet_native_20261005'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/rgbnt201_identity_outlet_native_20261005')
SOURCES=('rgbnt201_identity_outlet.py','run_rgbnt201_identity_outlet.py',
    'evaluate_rgbnt201_identity_outlet49.py','verify_rgbnt201_identity_outlet.py',
    'launch_rgbnt201_identity_outlet_native.py','results/preflight/rgbnt201_identity_outlet_native_deploy_20261005.py')


def remote(code):
    return json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))


def main():
    pf=PROJECT/'results/preflight'
    proof=pf/'rgbnt201_identity_outlet_actual_native_session_20261005.json'
    assert not proof.exists() and not LOCAL.exists() and not ARCHIVE.exists()
    review=json.loads((pf/'rgbnt201_identity_outlet_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status']=='PASS' and not review['blocking_findings']
    sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in SOURCES}
    assert review['sources_sha256']==sources
    parents=review['directly_reused_sources_sha256']
    remote(f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r})
assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {parents!r}.items())
print(json.dumps({{'parents_exact':True}}))''')
    for name in SOURCES:
        if not name.startswith('results/'):
            command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    deployed={name:digest for name,digest in sources.items() if not name.startswith('results/')}
    remote(f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {deployed!r}.items())
print(json.dumps({{'new_sources_exact':True}}))''')
    argv=[PYTHON,'-u','launch_rgbnt201_identity_outlet_native.py',
        '--data-root','/data/gaob/Re-ID/dataset',
        '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-run-dir',REMOTE+'/runs/full_official_baselines_20261004/training/RGBNT201_demo_s42',
        '--output',ROOT]
    print('RGBNT201_IDENTITY_OUTLET_NATIVE_START',flush=True)
    process=subprocess.Popen(['ssh',*OPTIONS,'2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)])
    exit_code=process.wait()
    inventory=remote(f'''import hashlib,json
from pathlib import Path
root=Path({ROOT!r})
files={{p.relative_to(root).as_posix():{{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}} for p in root.rglob('*') if p.is_file()}}
assert not any(name.endswith('.pth') for name in files)
print(json.dumps(files))''')
    LOCAL.mkdir(parents=True,exist_ok=False)
    copied={}
    for name,info in inventory.items():
        if name.endswith('.npz'):
            copied[name]=dict(file=info,local=copy_verified(ROOT+'/'+name,info,Path(ROOT),ARCHIVE))
        else:
            assert Path(name).suffix in ('.json','.csv','.log')
            destination=LOCAL/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            command(['scp',*OPTIONS,'2026:'+ROOT+'/'+name,str(destination)])
            assert destination.stat().st_size==info['bytes'] and hashlib.sha256(destination.read_bytes()).hexdigest()==info['sha256']
    receipt=dict(status='RECEIVED_IDENTITY_OUTLET_NATIVE_EVIDENCE',exit_code=exit_code,
        remote_root=ROOT,local_text_root=str(LOCAL),local_binary_root=str(ARCHIVE),sources_sha256=sources,
        text_and_raw_inventory=inventory,archives=copied,formal50_started=0,
        received_at=datetime.now().isoformat(timespec='seconds'))
    proof.write_bytes((json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
    assert exit_code==0,'Read actual received native logs before another attempt'
    result=json.loads((LOCAL/'controller_result.json').read_text())
    assert result['status']=='COMPLETE_IDENTITY_OUTLET_NATIVE_ONLY' and result['actual_updates']==12
    assert len(copied)==4 and result['formal50_started']==result['new_weight_files']==0
    own_raw={name:info for name,info in inventory.items() if name.endswith('.npz')}
    cleared=remote(f'''import hashlib,json
from pathlib import Path
root=Path({ROOT!r})
files={own_raw!r}
for name,info in files.items():
    path=root/name
    assert path.is_file() and path.stat().st_size==info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
for name in files:
    (root/name).unlink()
print(json.dumps({{'raw_cleared_after_local_verify':list(files),'bytes':sum(info['bytes'] for info in files.values())}}))''')
    receipt.update(status='COMPLETE_IDENTITY_OUTLET_NATIVE_LOCAL_RAW_VERIFIED_REMOTE_CLEARED',remote_cleanup=cleared,
        actual_updates=12,new_weight_files=0)
    proof.write_bytes((json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
    print('RGBNT201_IDENTITY_OUTLET_NATIVE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
