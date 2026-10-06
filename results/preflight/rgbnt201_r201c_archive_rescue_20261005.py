"""Archive existing784 audited states after the original SSH reset; no NN."""
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
ROOT = REMOTE + '/runs/rgbnt201_identity_outlet_r201c_20261005'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
LOCAL = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/rgbnt201_identity_outlet_r201c_20261005')


def main():
    pf = PROJECT / 'results/preflight'
    target = pf / 'rgbnt201_identity_outlet50_actual_session_20261005.json'
    assert not target.exists()
    review = json.loads((pf / 'rgbnt201_r201c_archive_rescue_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name, digest in review['sources_sha256'].items())
    failure = json.loads((pf / 'rgbnt201_r201c_original_missing_session_failure_20261005.json').read_text(encoding='utf-8'))
    assert failure['session'] == 17662 and failure['exit_code'] == 1
    approved = json.loads((pf / 'rgbnt201_identity_outlet50_source_review_20261005.json').read_text(encoding='utf-8'))
    runtime = approved['sources_sha256'] | approved['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name, digest in runtime.items())
    inspect = f'''import hashlib,json,subprocess
from pathlib import Path
project=Path({REMOTE!r}); root=Path({ROOT!r})
assert not (root/'controller_result.json').exists()
assert not subprocess.run(['ps','-p','1808059','-o','pid='],capture_output=True,text=True).stdout.strip()
assert not subprocess.run(['pgrep','-f','^.*python.*launch_rgbnt201_identity_outlet50.py'],capture_output=True,text=True).stdout.strip()
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==digest for name,digest in {runtime!r}.items() if not name.startswith('results/'))
jobs=[]; bundles={{}}; normals={{}}; orders=[]
for variant,gpu in [('frequency_shared',2),('axis_shared',3)]:
    for bypass in (0,1):
        name='RGBNT201_identity_'+variant+('_bypass' if bypass else '_narrow')+'_s42'; run=root/'training'/name; folder=root/'frozen49'/name
        trained=json.loads((run/'result.json').read_text()); result=json.loads((folder/'result.json').read_text()); audit=json.loads((folder/'independent_fourstate_cpu_audit.json').read_text())
        assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['steps']==trained['optimizer_steps']==2647
        assert trained['amp_skipped_steps']==trained['training_heldout_identities']==0 and trained['descriptor_dim']==5120
        assert trained['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
        assert result['status']=='COMPLETE' and result['state_cases']==audit['state_cases']==196 and audit['status']=='PASS'
        assert result['selected_epoch']==trained['best']['epoch'] and [p.name for p in run.glob('*.pth')]==['best.pth']
        phases={{phase:json.loads((root/phase/(name+'_exit.json')).read_text()) for phase in ('training','frozen49','audit')}}
        assert all(data['exit_code']==0 for data in phases.values())
        orders.append([(x['epoch'],x['step'],x['names'],x['partial_set']) for x in map(json.loads,(run/'batch_orders.jsonl').read_text().splitlines())])
        archived=folder/'local_archive.json'
        if archived.exists():
            data=json.loads(archived.read_text()); assert data['status']=='ALL49_LOCAL_SIZE_SHA_VERIFIED_SERVER_RAW_CLEARED'
            files=data['files']; assert not list(folder.glob('*.npz')); state='ALREADY_CLEARED'
        else:
            assert bypass==1
            files={{str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(folder.glob('q_*_g_*.npz'))}}; state='PENDING_RAW'
        assert len(files)==49
        bundles[name]=dict(files=files,state=state)
        raw=run/'best_official_arrays.npz'; archived=run/'normal_local_archive.json'
        if archived.exists():
            data=json.loads(archived.read_text()); assert data['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED' and not raw.exists()
            file=data['file']; state='ALREADY_CLEARED'
        else:
            assert bypass==1 and raw.exists()
            file=dict(bytes=raw.stat().st_size,sha256=hashlib.sha256(raw.read_bytes()).hexdigest()); state='PENDING_RAW'
        normals[name]=dict(remote=str(raw),file=file,state=state)
        jobs.append(dict(name=name,variant=variant,bypass=bypass,gpu=gpu,train=phases['training'],evaluation=phases['frozen49'],audit=phases['audit']))
assert len(orders[0])==2647 and all(order==orders[0] for order in orders)
print(json.dumps(dict(jobs=jobs,bundles=bundles,normals=normals)))'''
    inventory = json.loads(command(['ssh', *OPTIONS, '2026', PYTHON + ' -'], input=inspect))
    frozen, normal = {}, {}
    for name, bundle in inventory['bundles'].items():
        frozen[name] = {}
        for remote, info in bundle['files'].items():
            local = LOCAL / Path(remote).relative_to(Path(ROOT))
            if bundle['state'] == 'ALREADY_CLEARED':
                assert local.stat().st_size == info['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest() == info['sha256']
            else:
                local = Path(copy_verified(remote, info, Path(ROOT), LOCAL))
            frozen[name][remote] = dict(file=info, local=str(local))
        print('R201C_RECOVERY_ALL49_LOCAL_RAW_VERIFIED', name, flush=True)
    for name, info in inventory['normals'].items():
        local = LOCAL / Path(info['remote']).relative_to(Path(ROOT))
        if info['state'] == 'ALREADY_CLEARED':
            assert local.stat().st_size == info['file']['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest() == info['file']['sha256']
        else:
            local = Path(copy_verified(info['remote'], info['file'], Path(ROOT), LOCAL))
        normal[name] = dict(file=info['file'], local=str(local))
    close = f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({ROOT!r})
assert not (root/'controller_result.json').exists()
assert not subprocess.run(['ps','-p','1808059','-o','pid='],capture_output=True,text=True).stdout.strip()
for name,bundle in {inventory['bundles']!r}.items():
    folder=root/'frozen49'/name
    for remote,file in bundle['files'].items():
        p=Path(remote); assert p.resolve().is_relative_to(folder.resolve()) and p.suffix=='.npz'
        if bundle['state']=='PENDING_RAW':
            assert p.stat().st_size==file['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==file['sha256']; p.unlink()
        else: assert not p.exists()
    if bundle['state']=='PENDING_RAW': (folder/'local_archive.json').write_text(json.dumps(dict(status='ALL49_LOCAL_SIZE_SHA_VERIFIED_SERVER_RAW_CLEARED',files=bundle['files'],recovered_closeout=True),indent=2)+'\\n')
for name,info in {inventory['normals']!r}.items():
    p=Path(info['remote']); run=root/'training'/name; file=info['file']
    assert p.resolve().is_relative_to(run.resolve()) and p.name=='best_official_arrays.npz'
    if info['state']=='PENDING_RAW':
        assert p.stat().st_size==file['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==file['sha256']; p.unlink()
        (run/'normal_local_archive.json').write_text(json.dumps(dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED',file=file,recovered_closeout=True),indent=2)+'\\n')
    else: assert not p.exists()
record=dict(status='COMPLETE',runs={inventory['jobs']!r},models=4,additional_epochs=200,successful_updates=10588,deployed_frozen_cases=196,closed_state_cases=784,paired_sampling_exact=True,all_raw_local_verified=True,completion_actor='ARCHIVE_ONLY_RESCUE_AFTER_ORIGINAL_SSH_CONNECTION_RESET',original_session=17662,original_session_exit_code=1,new_neural_calls=0,new_optimizer_updates=0,limits='All original training/evaluation/GT childexits0; original collector failed connection reset, no original controller success claim. Archive-only closeout preserves all196 raw and4 normal arrays locally. One seed/benchmark best/original50+extra50;12 disjoint-source conditions unsolved.')
(root/'controller_result.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))'''
    closure = json.loads(command(['ssh', *OPTIONS, '2026', PYTHON + ' -'], input=close))
    record = dict(status='COMPLETE_R201C_FOUR50_ALL784STATE_GT_AND_LOCAL_RAW', exit_code=0,
        remote_root=ROOT, local_root=str(LOCAL), sources_sha256=approved['sources_sha256'],
        frozen=frozen, normal=normal, finished=datetime.now().isoformat(timespec='seconds'),
        recovered_closeout=True, original_session=17662, original_session_exit_code=1,
        completion_actor=closure['completion_actor'], new_neural_calls=0, new_optimizer_updates=0,
        original_failure='results/preflight/rgbnt201_r201c_original_missing_session_failure_20261005.json')
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (pf / 'rgbnt201_r201c_archive_rescue_actual_20261005.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ACTUAL_R201C_ARCHIVE_ONLY_RECOVERY_COMPLETE_NO_NEURAL_RESTART', flush=True)


if __name__ == '__main__':
    main()
