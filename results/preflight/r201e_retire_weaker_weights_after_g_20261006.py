"""Retire the closed AP.01 pair only after G normal/state consumers are closed."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command


def main():
    pf = PROJECT / 'results/preflight'
    proof = pf / 'r201e_weaker_weights_retired_after_g_20261006.json'
    assert not proof.exists()
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    old = load(pf / 'r201e_list_objective_fp32check_actual_session_20261006.json')
    new = load(pf / 'r201g_normal_priority_actual_session_20261006.json')
    states = load(pf / 'r201g_normal_states_actual_20261006.json')
    mature = load(pf / 'r201ef_mature_gradient_actual_20261006.json')
    prior = load(pf / 'r201f_weaker_weights_retired_20261006.json')
    analysis = load(PROJECT / 'results/r201g_normal_priority_20261006/normal_analysis/result.json')
    assert old['exit_code'] == new['exit_code'] == 0 and old['successful_updates'] == new['successful_updates'] == 5294
    assert states['status'] == 'ACTUAL_R201G_TWO_NORMAL_FOURSTATE_GT_RAW_COMPLETE'
    assert load(PROJECT / 'results/r201g_normal_states_20261006/normal_state_analysis/result.json')['status'] == 'ACTUAL_R201G_SELECTED_NORMAL_FOURSTATE_CPU_READOUT_COMPLETE'
    assert mature['status'] == 'ACTUAL_E_F_MATURE_TRAIN_ONLY_DEPLOY_GRADIENT_PAIR_COMPLETE'
    assert mature['optimizer_updates'] == mature['checkpoint_writes'] == 0
    for variant in ('axis_shared','frequency_shared'):
        row = next(row for row in analysis['comparisons'] if row['improved']=='normal_'+variant and row['reference']=='list01_'+variant)
        assert row['mAP'] > 0 and row['Rank-1'] > 0
    for group in (old['archives'],new['archives']):
        for row in group.values():
            path = Path(row['local'])
            assert path.stat().st_size == row['file']['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['file']['sha256']
    remote = '/data/gaob/Re-ID/DeMo-DualAxis'
    protected = ['runs/full_official_baselines_20261004/training/'+name+'/best.pth'
        for name in ('RGBNT201_demo_s42','RGBNT100_demo_s42','MSVR310_demo_s42','MSVR310_demo_shared_s42')]
    protected += ['runs/rgbnt201_identity_outlet_r201c_20261005/training/RGBNT201_identity_'+variant+'_narrow_s42/best.pth'
        for variant in ('frequency_shared','axis_shared')]
    protected += ['runs/r201g_normal_priority_20261006/training/RGBNT201_r201g_'+variant+'_s42/best.pth'
        for variant in ('frequency_shared','axis_shared')]
    target_receipts = {name:row for name,row in prior['protected'].items() if name.startswith('runs/r201e_list_objective_fp32check_20261006/')}
    assert len(target_receipts)==2
    code = f'''from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path({remote!r}).resolve(strict=True)
old=root/'runs/r201e_list_objective_fp32check_20261006';mature=root/'runs/r201ef_mature_gradient_20261006'
for folder in (old,mature,root/'runs/r201g_normal_priority_20261006',root/'runs/r201g_normal_states_20261006'):
 assert json.loads((folder/'controller_result.json').read_text())['status']=='COMPLETE'
 assert subprocess.run(['ps','-p',str(json.loads((folder/'controller_launch.json').read_text())['pid'])],stdout=subprocess.DEVNULL).returncode==1
for variant in ('frequency_shared','axis_shared'):
 launch=json.loads((old/'training'/('RGBNT201_r201e_'+variant+'_s42_launch.json')).read_text())
 assert subprocess.run(['ps','-p',str(launch['pid'])],stdout=subprocess.DEVNULL).returncode==1
for label in ('E','F'):
 assert subprocess.run(['ps','-p',str(json.loads((mature/(label+'_launch.json')).read_text())['pid'])],stdout=subprocess.DEVNULL).returncode==1
def receipt(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
before={{name:receipt(root/name) for name in {protected!r}}};targets={{}}
for name,expected in {target_receipts!r}.items():
 path=(root/name).resolve(strict=True);assert path.is_relative_to(old.resolve(strict=True))
 run=json.loads((path.parent/'result.json').read_text())
 assert run['status']=='COMPLETE' and run['epochs']==50 and run['optimizer_steps']==2647 and run['amp_skipped_steps']==0
 assert json.loads((path.parent/'normal_local_archive.json').read_text())['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
 actual=receipt(path);assert actual==expected;targets[str(path)]=actual
assert targets[str(old/'training/RGBNT201_r201e_axis_shared_s42/best.pth')]=={dict(bytes=mature['checkpoint_receipts']['E']['bytes'],sha256=mature['checkpoint_receipts']['E']['sha256'])!r}
free_before=shutil.disk_usage(root).free
for name in targets:Path(name).unlink()
assert all(not Path(name).exists() for name in targets)
assert before=={{name:receipt(root/name) for name in before}}
print(json.dumps(dict(retired=targets,retired_bytes=sum(r['bytes'] for r in targets.values()),protected=before,protected_unchanged=True,remote_free_before=free_before,remote_free_after=shutil.disk_usage(root).free)))'''
    result = json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code))
    result.update(status='ACTUAL_TWO_CLOSED_WEAKER_AP01_WEIGHTS_RETIRED_AFTER_G',verified_at=datetime.now().isoformat(timespec='seconds'),
        local_D_files_deleted=0,local_primary_raw_archives_preserved=4,new_neural_calls=0,
        reason='Both AP.01 selected models are weaker than the retained matching G models in mAP and Rank-1; original50/GT/raw and mature-gradient consumers closed. No planned AP.01 neural consumer remains.',
        limits='Remote own weight cleanup only. All AP.01 raw/text/curves/source and negative results remain; its future neural replay requires retraining. This does not reclaim D space or establish the research goal.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','verified_at','retired_bytes','protected_unchanged','local_D_files_deleted')}))


if __name__ == '__main__':
    main()
