from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, copy_file, remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
launches=[json.loads((PROJECT/('results/preflight/anytoany49_'+host+'_launch.json')).read_text()) for host in ('2027',)]
done=set()
while len(done)<len(launches):
    for launch in launches:
        host=launch['host']
        if host in done:continue
        code=f'''import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path({launch['output']!r});project=root.parent.parent
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {launch['source_sha256']!r}.items())
record={{'controller_live':bool(subprocess.run(['ps','-p',{str(launch['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()),'rows':[]}}
for dataset in ('MSVR310','RGBNT201','RGBNT100'):
 name=dataset+'_'+{launch['variant']!r}+'_s42';folder=root/name
 for stage in ('smoke','full'):
  row={{'name':name,'stage':stage}}
  for key,path in [('launch',folder/(stage+'_launch.json')),('exit',folder/(stage+'_exit.json')),('result',folder/stage/('smoke.json' if stage=='smoke' else 'result.json'))]:
   if path.exists():
    value=json.loads(path.read_text())
    row[key]={{key:value[key] for key in ('status','optimizer_updates','normal_feature_max_error','state_tensor_versions_unchanged') if key in value}} if key=='result' else value
  if row.get('exit',{{}}).get('exit_code',0)!=0:row['failure_log']=(folder/(stage+'.log')).read_text()[-5000:]
  record['rows'].append(row)
terminal=root/'controller_result.json'
if terminal.exists():
 record['controller_result']=json.loads(terminal.read_text())
 archive=Path(str(root)+'.tar.gz');assert not archive.exists()
 with tarfile.open(archive,'x:gz') as bundle:
  for path in sorted(root.rglob('*')):
   if path.is_file():bundle.add(path,arcname=path.relative_to(root).as_posix())
 record['archive']={{'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}}
record['controller_log_tail']=Path({launch['log']!r}).read_text()[-3000:]
print(json.dumps(record))
'''
        record=json.loads(remote_python(host,code));record['observed_at']=datetime.now().isoformat(timespec='seconds')
        (PROJECT/('results/preflight/anytoany49_'+host+'_latest_snapshot.json')).write_bytes(json.dumps(record,indent=2).encode('utf-8'))
        print('ANYTOANY49_OBSERVATION',json.dumps(dict(host=host,time=record['observed_at'],live=record['controller_live'],stages=[dict(name=row['name'],stage=row['stage'],exit=row.get('exit',{}).get('exit_code')) for row in record['rows']])),flush=True)
        failures=[row for row in record['rows'] if row.get('exit',{}).get('exit_code',0)!=0]
        assert not failures,failures
        terminal=record.get('controller_result',{}).get('status')=='COMPLETE'
        assert record['controller_live'] or terminal,record['controller_log_tail']
        if terminal:
            campaign='anytoany49_'+launch['variant']+'_20261003'
            local=Path('C:/Users/gb/.codex_tmp')/(campaign+'.tar.gz')
            assert not local.exists()
            copy_file(host,record['archive']['path'],local)
            assert local.stat().st_size==record['archive']['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==record['archive']['sha256']
            target=PROJECT/'results'/campaign;assert not target.exists();target.mkdir()
            with tarfile.open(local,'r:gz') as bundle:
                for member in bundle.getmembers():
                    destination=(target/member.name).resolve()
                    assert destination.is_relative_to(target.resolve()) and member.isfile()
                bundle.extractall(target,filter='data')
            (target/'intake.json').write_bytes(json.dumps(dict(host=host,observed_at=record['observed_at'],archive=record['archive'],
                extracted_files={path.relative_to(target).as_posix():dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in target.rglob('*') if path.is_file()}),indent=2).encode('utf-8'))
            done.add(host)
            print('ANYTOANY49_RAW_TERMINAL_COLLECTED',host,campaign,flush=True)
    if len(done)<len(launches):time.sleep(240)
