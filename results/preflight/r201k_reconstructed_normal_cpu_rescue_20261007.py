"""Receive the two already exact MSVR normal arrays; no repeated neural work."""
from datetime import datetime
import hashlib,json,shlex,shutil,sys
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201k_retired_normal_reconstruction_20261007'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201k_relation_local_pi_20261007')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    proof=pf/'r201k_reconstructed_normal_cpu_rescue_actual_20261007.json';assert not proof.exists()
    review=load(pf/'r201k_reconstructed_normal_cpu_rescue_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert all(sha(PROJECT/n)==digest for n,digest in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    failure=load(pf/'r201k_retired_normal_reconstruction_failure_actual_20261007.json')
    assert failure['original_driver_exit_code']==1 and failure['failure_source_line']==95 and failure['neural_jobs_executed']==4
    normal=load(pf/'r201k_relation_local_pi_actual_session_20261007.json')
    assert normal['status']=='ACTUAL_K_THREE_NORMAL_DATASETS_SIX_FULL50_GT_RAW_COMPLETE' and normal['exit_code']==0
    assert (normal['additional_epochs'],normal['successful_updates'],normal['native_updates'])==(300,19418,18)
    audit=load(pf/'r201k_raw_retirement_dependency_actual_20261007.json')
    selected={j['name']:j for j in audit['jobs']};weights={j['name']:j['sha256'] for j in audit['server26_selected_weights']}
    original=load(pf/'r201k_retired_normal_reconstruction_actual_session_20261007.json')
    assert set(original['completed'])=={'RGBNT201_r201k_frequency_shared_s42','RGBNT201_r201k_axis_shared_s42'}
    inputs={};jobs=[]
    for variant in ('frequency_shared','axis_shared'):
        name='MSVR310_r201k_'+variant+'_s42';job=selected[name]
        actual=next(row for row in failure['rows'] if row['name']==name)
        assert actual['raw_file']==job['normal_file'] and actual['checkpoint_sha256']==weights[name] and actual['result_sha256']==job['result_sha256']
        assert not Path(job['normal_local']).exists()
        reference=PROJECT/'results/r201i_missing49_20261007/full'/('MSVR310_r201i_'+variant+'_s42')/'q_RNT_g_RNT'
        checks={}
        for suffix in ('.json','.csv'):
            prior=reference/('state_00'+suffix)
            current=PROJECT/'results/r201k_retired_normal_reconstruction_failure_exact_bytes_20261007'/name/('state_00_per_query'+suffix)
            assert sha(prior)==sha(current)==actual['text']['state_00_per_query'+suffix]['sha256']
            inputs[str(prior.relative_to(PROJECT)).replace('\\','/')]=dict(bytes=prior.stat().st_size,sha256=sha(prior))
            checks[suffix]=sha(prior)
        jobs.append(dict(job,checkpoint_sha256=weights[name],state00_same_path_reference_sha256=checks))
    for value in original['completed'].values():
        path=Path(value['local']);assert path.stat().st_size==value['local_file']['bytes'] and sha(path)==value['local_file']['sha256']
    assert shutil.disk_usage(ARCHIVE).free>sum(j['normal_file']['bytes'] for j in jobs)+1024**3
    code=f'''import csv,hashlib,json
from pathlib import Path
import numpy as np
from audit_full_official49 import METRICS,recount,verify
from audit_full_official_condition import installed_context
rows=[]
for job in {jobs!r}:
 run=Path(job['run']);out=Path({ROOT!r})/job['dataset']/'training'/job['name'];raw=out/'best_official_arrays.npz'
 assert hashlib.sha256((run/'best.pth').read_bytes()).hexdigest()==job['checkpoint_sha256']
 assert hashlib.sha256((run/'result.json').read_bytes()).hexdigest()==job['result_sha256']
 assert raw.stat().st_size==job['normal_file']['bytes'] and hashlib.sha256(raw.read_bytes()).hexdigest()==job['normal_file']['sha256']
 trained,dataset,installed=installed_context(run);assert dataset=='MSVR310'
 for suffix,digest in job['state00_same_path_reference_sha256'].items():
  assert hashlib.sha256((out/('state_00_per_query'+suffix)).read_bytes()).hexdigest()==digest
 for suffix in ('.json','.csv'):
  assert hashlib.sha256((out/('best_per_query'+suffix)).read_bytes()).hexdigest()==hashlib.sha256((run/('best_per_query'+suffix)).read_bytes()).hexdigest()
 states={{}}
 for state in ('00','10','01','11'):
  stem='best_per_query' if state=='11' else 'state_'+state+'_per_query'
  report=json.loads((out/(stem+'.json')).read_text());text=list(csv.DictReader((out/(stem+'.csv')).open(newline='')))
  exported=[]
  for row,query in zip(text,installed['query']):
   parsed={{k:(float(v) if k in ('AP','INP') else v=='True' if k=='valid' else v if k=='name' else int(v)) for k,v in row.items()}}
   assert all(parsed[k]==query[k] for k in ('name','identity','camera','scene'))
   exported.append(parsed)
  values,error=verify(exported,report,out/(stem+'.csv'),len(installed['gallery']));states[state]=values
 with np.load(raw) as saved:
  for role in ('query','gallery'):
   for field,key in (('ids','identity'),('cameras','camera'),('scenes','scene'),('names','name')):
    assert np.array_equal(saved[role+'_'+field],np.asarray([r[key] for r in installed[role]]))
  checked,error=verify(recount(saved['distances'],installed['query'],installed['gallery'],dataset),json.loads((out/'best_per_query.json').read_text()),out/'best_per_query.csv',len(installed['gallery']))
 assert all(checked[k]==trained['full_metrics'][k] for k in METRICS)
 rows.append(dict(name=job['name'],raw_path=str(raw),normal_file=job['normal_file'],checkpoint_sha256=job['checkpoint_sha256'],result_sha256=job['result_sha256'],four_state_metrics=states,state00_I_same_path_text_exact=True,state00_minus_original_anchor={{k:states['00'][k]-trained['anchor']['selected_anchor_full_metrics'][k] for k in METRICS}},normal11_installed_GT_exact=True,normal11_original_text_exact=True,new_neural_calls=0,new_optimizer_updates=0))
print(json.dumps(rows))'''
    checked=json.loads(command(['ssh',*OPTIONS,'2026','cd '+shlex.quote(REMOTE)+' && CUDA_VISIBLE_DEVICES= '+shlex.quote(PYTHON)+' -'],input=code))
    assert len(checked)==2
    received={}
    for row in checked:
        job=selected[row['name']];local=copy_verified(row['raw_path'],row['normal_file'],Path(ROOT),ARCHIVE)
        assert Path(local)==Path(job['normal_local'])
        code=f'''import hashlib
from pathlib import Path
p=Path({row['raw_path']!r});root=Path({ROOT!r})
assert p.resolve().is_relative_to(root.resolve()) and p.name=='best_official_arrays.npz'
assert p.stat().st_size=={row['normal_file']['bytes']!r} and hashlib.sha256(p.read_bytes()).hexdigest()=={row['normal_file']['sha256']!r};p.unlink()
print('EXACT_RECONSTRUCTED_MSVR_STAGING_RAW_CLEARED_AFTER_CURRENT_LOCAL_ACK')'''
        print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
        received[row['name']]=dict(local=local,file=row['normal_file'],cpu_checks=row)
    current={}
    for dataset,data in normal['datasets'].items():
        for name,record in data['archives'].items():
            path=Path(record['local']);assert path.resolve().is_relative_to(ARCHIVE.resolve())
            assert path.stat().st_size==record['file']['bytes'] and sha(path)==record['file']['sha256']
            current[name]=record
    assert len(current)==6
    result=dict(status='ACTUAL_FOUR_RETIRED_K_NORMAL_RAW_ORIGINAL_SHA_RESTORED_CURRENT_SIX_INPUTS_READY',completed_at=datetime.now().isoformat(timespec='seconds'),original_reconstruction_driver_exit=1,accepted_201_controls=2,accepted_MSVR_CPU_receipts=received,all_six_current_normal_inputs=current,state00_reference_inputs=inputs,new_neural_calls=0,new_optimizer_updates=0,restored_four_bytes=sum(j['normal_file']['bytes'] for j in audit['jobs']),MSVR_geometry_completed=False,limits='No retry of the four inference jobs. Original four-state recount passed before the obsolete anchor-metric equality assertion; actual MSVR00 JSON/CSV exactly equal previously audited I same-path00, with disclosed tiny deltas versus original anchor. All original11 NPZ and best perquery bytes exact. Two MSVR11 installed-GT CPU recounts repeated from already produced arrays; 00/10/01 CSV/CMC/groups consistency checked, no new feature/distance inference. MSVR selected geometry was not reached and is not claimed complete; old exit1 evidence preserved.')
    proof.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('accepted_MSVR_CPU_receipts','all_six_current_normal_inputs','state00_reference_inputs')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
