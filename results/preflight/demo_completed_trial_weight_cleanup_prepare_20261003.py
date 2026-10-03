from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python

campaigns={
 'axis_collaboration_v7_metric_interface_trial':'MSVR310_axis_metric_fullref_s42',
 'axis_collaboration_v8_routed_metric_trial':'MSVR310_axis_metric_routed_fullref_s42',
 'axis_collaboration_v9_relation_frequency_trial':'MSVR310_axis_relation_frequency_fullref_s42',
}
root,_=HOSTS['2026']
code=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r}).resolve();rows=[];keepers=[]
for campaign,name in {campaigns!r}.items():
 folder=root/'runs'/campaign
 assert json.loads((folder/'controller_result.json').read_text())['status']=='COMPLETE'
 run=folder/'development'/name
 result=json.loads((run/'result.json').read_text())
 assert result['status']=='COMPLETE' and result['epochs']==50
 assert json.loads((run.parent/(name+'_exit.json')).read_text())['exit_code']==0
 best=run/'best.pth';assert best.is_file()
 keepers.append(dict(path=str(best),bytes=best.stat().st_size,sha256=hashlib.sha256(best.read_bytes()).hexdigest(),best=result['best']))
 for path in (folder/'preflight/smoke/smoke.pth',run/'last.pth'):
  assert path.resolve().is_relative_to(root/'runs'/campaign) and path.is_file()
  rows.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),campaign=campaign))
 process_list=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
 assert not any(str(folder) in line for line in process_list.splitlines() if 'python' in line and ' -c ' not in line)
print(json.dumps(dict(host='2026',root=str(root),candidates=rows,retained=keepers,total_bytes=sum(row['bytes'] for row in rows),free_before_bytes=shutil.disk_usage(root).free)))
'''
record=json.loads(remote_python('2026',code))
record.update(observed_at=datetime.now().isoformat(timespec='seconds'),authorization='User explicitly requested cleaning our useless weights and keeping best checkpoints; completed-trial non-best smoke/last only.',status='EXACT_NONBEST_CLEANUP_PREPARED')
target=PROJECT/'results/preflight/completed_trial_weight_cleanup_plan.json'
assert not target.exists()
target.write_bytes(json.dumps(record,indent=2).encode('utf-8'))
print('NONBEST_WEIGHT_CLEANUP_PREPARED',json.dumps(dict(files=len(record['candidates']),bytes=record['total_bytes'],free_before=record['free_before_bytes'])),flush=True)
