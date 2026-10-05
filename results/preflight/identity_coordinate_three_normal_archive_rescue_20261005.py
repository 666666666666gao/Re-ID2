"""Close actual completed normal archives after the original controller disappeared."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
ROOT = REMOTE + '/runs/identity_coordinate_full_normal_three_20261005'
OLD = REMOTE + '/runs/rgbnt201_identity_outlet_r201c_20261005'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
LOCAL = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/identity_coordinate_full_normal_three_20261005')


def main():
    pf = PROJECT / 'results/preflight'
    target = pf / 'identity_coordinate_three_normal_actual_session_20261005.json'
    assert not target.exists()
    review = json.loads((pf / 'identity_coordinate_three_normal_archive_rescue_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == value for name, value in review['sources_sha256'].items())
    failure = json.loads((pf / 'identity_coordinate_three_normal_original_session_failure_20261005.json').read_text(encoding='utf-8'))
    assert failure['session'] == 71594 and failure['exit_code'] == 1
    accepted = json.loads((pf / 'identity_coordinate_three_normal_source_review_20261005.json').read_text(encoding='utf-8'))
    runtime = accepted['sources_sha256'] | accepted['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == value for name, value in runtime.items())
    read = f'''import hashlib,json,subprocess
from pathlib import Path
p=Path({REMOTE!r}); root=Path({ROOT!r}); old=Path({OLD!r})
assert not (root/'controller_result.json').exists()
assert not subprocess.run(['ps','-p','1997315','-o','pid='],capture_output=True,text=True).stdout.strip()
assert not subprocess.run(['pgrep','-f','^.*python.*launch_identity_coordinate_three_normal.py'],capture_output=True,text=True).stdout.strip()
assert all(hashlib.sha256((p/name).read_bytes()).hexdigest()==value for name,value in {runtime!r}.items() if not name.startswith('results/'))
jobs=[]; archives={{}}
for dataset,count,steps in [('MSVR310',1032,705),('RGBNT100',8675,6357)]:
    orders=[]
    for variant,gpu in [('frequency_shared',2),('axis_shared',3)]:
        name=dataset+'_identity_'+variant+'_narrow_s42'; run=root/'training'/name
        d=json.loads((run/'result.json').read_text()); audit=json.loads((run/'normal_cpu_audit.json').read_text())
        train_exit=json.loads((root/'training'/(name+'_exit.json')).read_text()); audit_exit=json.loads((root/'audit'/(name+'_exit.json')).read_text())
        assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps']==steps
        assert d['amp_skipped_steps']==d['training_heldout_identities']==0 and d['descriptor_dim']==5120
        assert d['training_coverage']==dict(eligible=count,visited=count,unvisited=[])
        assert audit['status']=='PASS' and audit['max_metric_error']==0 and audit['full_training_coverage']
        assert train_exit['exit_code']==audit_exit['exit_code']==0
        assert [x.name for x in run.glob('*.pth')]==['best.pth']
        orders.append([(x['epoch'],x['step'],x['names'],x['partial_set']) for x in map(json.loads,(run/'batch_orders.jsonl').read_text().splitlines())])
        archived=run/'normal_local_archive.json'; raw=run/'best_official_arrays.npz'
        if archived.exists():
            a=json.loads(archived.read_text()); assert a['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED' and not raw.exists()
            file=a['file']; state='ALREADY_CLEARED'
        else:
            assert dataset=='RGBNT100' and raw.exists()
            file=dict(bytes=raw.stat().st_size,sha256=hashlib.sha256(raw.read_bytes()).hexdigest()); state='PENDING_RAW'
        archives[name]=dict(file=file,remote=str(raw),state=state)
        jobs.append(dict(name=name,dataset=dataset,variant=variant,gpu=gpu,train=train_exit,audit=audit_exit))
    assert orders[0]==orders[1]
print(json.dumps(dict(jobs=jobs,archives=archives,updates=sum(json.loads((root/'training'/x['name']/'result.json').read_text())['optimizer_steps'] for x in jobs))))'''
    inventory = json.loads(command(['ssh', *OPTIONS, '2026', PYTHON + ' -'], input=read))
    assert len(inventory['jobs']) == len(inventory['archives']) == 4 and inventory['updates'] == 14124
    archives = {}
    for name, info in inventory['archives'].items():
        local = LOCAL / 'training' / name / 'best_official_arrays.npz'
        if info['state'] == 'ALREADY_CLEARED':
            assert local.stat().st_size == info['file']['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest() == info['file']['sha256']
        else:
            local = Path(copy_verified(info['remote'], info['file'], Path(ROOT), LOCAL))
        archives[name] = dict(file=info['file'], local=str(local))
        print('NORMAL_RECOVERY_LOCAL_SELECTED_ARRAY_VERIFIED', name, flush=True)
    close = f'''import hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path({ROOT!r}); old=Path({OLD!r})
assert not (root/'controller_result.json').exists()
assert not subprocess.run(['ps','-p','1997315','-o','pid='],capture_output=True,text=True).stdout.strip()
for name,info in {inventory['archives']!r}.items():
    run=root/'training'/name; raw=Path(info['remote']); file=info['file']
    assert raw.resolve().is_relative_to(root.resolve()) and raw.name=='best_official_arrays.npz'
    if info['state']=='PENDING_RAW':
        assert raw.stat().st_size==file['bytes'] and hashlib.sha256(raw.read_bytes()).hexdigest()==file['sha256']
        raw.unlink()
        (run/'normal_local_archive.json').write_text(json.dumps(dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED',file=file,recovered_closeout=True),indent=2)+'\\n')
    else:
        assert not raw.exists() and json.loads((run/'normal_local_archive.json').read_text())['file']==file
pid=1808059
assert subprocess.check_output(['ps','-p',str(pid),'-o','stat='],text=True).strip().startswith('T')
args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
assert 'launch_rgbnt201_identity_outlet50.py' in args and {OLD!r} in args
os.kill(pid,signal.SIGCONT)
(old/'user_priority_missing_resumed.json').write_text(json.dumps(dict(status='ORIGINAL_MISSING_QUEUE_RESUMED_AFTER_THREE_NORMAL_DATASETS',resumed_at=time.time(),controller_pid=pid,continuing_same_receiver=17662,neural_sources_changed=0,recovered_closeout=True),indent=2)+'\\n')
record=dict(status='COMPLETE',runs={inventory['jobs']!r},new_models=4,additional_epochs=200,successful_updates=14124,native_updates=12,existing_RGBNT201_normal_models=2,normal_datasets_completed=3,normal_archives_local_verified=4,paired_sampling_exact=True,original_missing_controller_resumed=True,completion_actor='ARCHIVE_ONLY_RESCUE_AFTER_ORIGINAL_CONTROLLER_DISAPPEARED',original_session=71594,original_local_session_exit_code=1,new_neural_calls=0,new_optimizer_updates=0,original_controller_exit_cause='UNKNOWN; kernel log access denied',limits='Actual training/audit children completed exit0. Original controller did not close normally; archive-only rescue completed remaining2 copies and resumed old parent after all4 raw verified. Benchmark-selected/one seed/original50+extra50; missing and multiseed success unproven.')
(root/'controller_result.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))'''
    closure = json.loads(command(['ssh', *OPTIONS, '2026', PYTHON + ' -'], input=close))
    record = dict(status='COMPLETE_THREE_NORMAL_DATASETS_AND_FOUR_NEW50_GT_AND_LOCAL_RAW',
        finished=datetime.now().isoformat(timespec='seconds'), exit_code=0, remote_root=ROOT, archives=archives,
        native_updates=12, new_successful_updates=14124, additional_epochs=200,
        old_missing_controller_resumed=1808059, old_receiver=17662, sources_sha256=runtime,
        recovered_closeout=True, original_session=71594, original_session_exit_code=1,
        completion_actor=closure['completion_actor'], new_neural_calls=0, new_optimizer_updates=0,
        original_failure='results/preflight/identity_coordinate_three_normal_original_session_failure_20261005.json')
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (pf / 'identity_coordinate_three_normal_archive_rescue_actual_20261005.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ACTUAL_NORMAL_ARCHIVE_ONLY_RECOVERY_COMPLETE_NO_NEURAL_RESTART', flush=True)


if __name__ == '__main__':
    main()
