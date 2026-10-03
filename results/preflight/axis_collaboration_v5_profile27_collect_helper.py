import base64
from datetime import datetime
import hashlib
import json
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
launch=json.loads((PROJECT/'results/preflight/axis_collaboration_v5_profile_launch.json').read_text(encoding='utf-8'))
code=f'''import base64,hashlib,json,subprocess
from pathlib import Path
root=Path({HOSTS['2027'][0]!r});output=Path({launch['output']!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {launch['source_sha256']!r}.items())
exit_file=Path(str(output)+'_exit.json');log=Path(str(output)+'.log')
record=dict(controller_live=bool(subprocess.run(['ps','-p',{str(launch['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()),log_tail=log.read_text()[-3500:] if log.exists() else '',case_files=len(list(output.glob('*_operators.json'))))
if exit_file.exists():
 record['exit']=json.loads(exit_file.read_text())
 paths=[*sorted(output.glob('*.json')),exit_file,log]
 record['files']={{}}
 for path in paths:
  data=path.read_bytes();record['files'][path.name]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
print(json.dumps(record))
'''
record=json.loads(remote_python('2027',code));record['observed_at']=datetime.now().isoformat(timespec='seconds')
files=record.pop('files',{})
if files:
    target=PROJECT/'profile_output/axis_v4_v5_inference_cost_20261003';assert not target.exists()
    target.mkdir(parents=True)
    manifest={}
    for name,proof in files.items():
        data=base64.b64decode(proof['data'])
        assert len(data)==proof['bytes'] and hashlib.sha256(data).hexdigest()==proof['sha256']
        (target/name).write_bytes(data);manifest[name]={key:proof[key] for key in ('sha256','bytes')}
    (target/'intake.json').write_bytes(json.dumps(dict(observed_at=record['observed_at'],host='2027',files=manifest,exit=record['exit']),indent=2).encode('utf-8'))
    if record['exit']['exit_code']!=0:
        print('V5_PROFILE_PRIMARY_FAILURE',record['log_tail'],flush=True)
    assert record['exit']['exit_code']==0
    summary=json.loads((target/'summary.json').read_text(encoding='utf-8'))
    assert summary['status']=='COMPLETE' and summary['conditions']==15 and summary['optimizer_updates']==0 and summary['official_test_uses']==0
    assert len(summary['rows'])==15 and len(manifest)==33 and not summary['complete_model_FLOPs']
    print('V5_PROFILE_ALL15_COLLECTED',json.dumps(dict(observed_at=record['observed_at'],conditions=15,files=len(manifest),exit_code=0)),flush=True)
else:
    assert record['controller_live'],record['log_tail']
    print('V5_PROFILE_RUNNING',json.dumps(dict(observed_at=record['observed_at'],case_files=record['case_files'])),flush=True)
