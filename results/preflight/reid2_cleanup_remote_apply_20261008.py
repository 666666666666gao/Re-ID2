"""Apply the explicit user-authorized closed-experiment cleanup; no neural work."""
from datetime import datetime
from pathlib import Path
import json,shlex,sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command

def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    plan=load(pf/'reid2_cleanup_plan_20261008.json')
    local=load(pf/'reid2_cleanup_local_actual_20261008.json')
    assert plan['status']=='VERIFIED_EXPLICIT_CLEANUP_READY_NOT_APPLIED'
    assert local['status']=='ACTUAL_PROJECT_LOCAL_TEMPORARIES_CLEANED_RAW_PRESERVED'
    proof=pf/'reid2_project_cleanup_actual_20261008.json';assert not proof.exists()
    outcomes={}
    for row in plan['remote']:
        host,root=row['host'],row['root']
        python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python' if host=='2026' else '/data/gb/Re-ID/conda-envs/tri_reid/bin/python'
        protected=plan['protected_2026_weights'] if host=='2026' else {}
        targets=[dict(item,category='verified_redundant_transfer_package') for item in row['packages']]
        targets.extend(dict(item,category='completed_smoke_weight' if '/checks/' in item['path'] else 'nonselected_Best_of5_weight') for item in row['weights'])
        name='reid2_cleanup_remote_'+host+'_actual_20261008.json'
        assert not (pf/name).exists()
        code=f'''from datetime import datetime
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path({root!r});targets={targets!r};protected={protected!r};proof=root/'results/preflight'/{name!r}
assert not proof.exists()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
assert all(sha(root/n)==s for n,s in protected.items())
assert not any('run_r201' in line or 'launch_r201' in line or 'evaluate_r201' in line for line in subprocess.check_output(['ps','-eo','args'],text=True).splitlines()[1:])
raw={{str(p):p.stat().st_size for p in root.rglob('*.npz') if p.is_file()}}
before=shutil.disk_usage(root).free
for row in targets:
 p=Path(row['path']);assert p.resolve().is_relative_to(root.resolve()) and p.is_file()
 assert p.name.endswith(('.tar.gz','.pth')) and str(p.relative_to(root)) not in protected
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
removed=[]
for row in targets:
 p=Path(row['path']);p.unlink();assert not p.exists()
 removed.append(dict(path=row['path'],bytes=row['bytes'],sha256=row['sha256'],category=row['category']))
assert all(Path(n).is_file() and Path(n).stat().st_size==size for n,size in raw.items())
assert all(sha(root/n)==s for n,s in protected.items())
result=dict(status='ACTUAL_USER_AUTHORIZED_REDUNDANT_PACKAGES_AND_NONSELECTED_WEIGHTS_REMOVED',host={host!r},finished=datetime.now().astimezone().isoformat(timespec='seconds'),removed=removed,removed_bytes=sum(r['bytes'] for r in removed),protected_current_weights=protected,protected_raw_files=len(raw),protected_raw_bytes=sum(raw.values()),before_free=before,after_free=shutil.disk_usage(root).free,new_neural_calls=0,new_optimizer_updates=0,goal='PAUSED_USER_REQUEST_OBJECTIVE_UNMET')
proof.write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))'''
        result=json.loads(command(['ssh',*OPTIONS,host,shlex.quote(python)+' -'],input=code))
        assert result['status']=='ACTUAL_USER_AUTHORIZED_REDUNDANT_PACKAGES_AND_NONSELECTED_WEIGHTS_REMOVED'
        (pf/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        outcomes[host]=result
        print('ACTUAL_REMOTE_CLEANUP_CLOSED',host,len(result['removed']),result['removed_bytes'],flush=True)
    result=dict(status='ACTUAL_PROJECT_CLEANUP_COMPLETE_RESULTS_AND_SELECTED_WEIGHTS_PRESERVED',finished=datetime.now().astimezone().isoformat(timespec='seconds'),local=local,remote=outcomes,removed_bytes=local['removed_bytes']+sum(r['removed_bytes'] for r in outcomes.values()),retired_nonselected_seed_weights=7,retired_smoke_weights=2,selected_seed_weights_preserved=plan['N5_winners'],unique_diagnostic_arrays_preserved=plan['unique_binary_preserved'],retired_checkpoint_replay_requires_retraining=True,new_neural_calls=0,new_optimizer_updates=0,goal='PAUSED_USER_REQUEST_OBJECTIVE_UNMET')
    proof.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','finished','removed_bytes','retired_nonselected_seed_weights','retired_smoke_weights','goal')}),flush=True)

if __name__=='__main__':main()
