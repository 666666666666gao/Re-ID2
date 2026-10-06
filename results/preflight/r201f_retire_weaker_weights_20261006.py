"""Retire only the two closed, weaker F weights after all their consumers close."""
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
    proof = pf / 'r201f_weaker_weights_retired_20261006.json'
    assert not proof.exists()
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    closed = load(pf / 'r201f_list_temperature_actual_session_20261006.json')
    states = load(pf / 'r201f_normal_states_actual_20261006.json')
    mature = load(pf / 'r201ef_mature_gradient_actual_20261006.json')
    analysis = load(PROJECT / 'results/r201f_list_temperature_20261006/normal_analysis/result.json')
    assert closed['status'] == 'ACTUAL_R201F_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW'
    assert states['status'] == 'ACTUAL_R201F_TWO_NORMAL_FOURSTATE_GT_RAW_COMPLETE'
    assert mature['status'] == 'ACTUAL_E_F_MATURE_TRAIN_ONLY_DEPLOY_GRADIENT_PAIR_COMPLETE'
    assert mature['optimizer_updates'] == mature['checkpoint_writes'] == 0
    for variant in ('axis_shared', 'frequency_shared'):
        comparison = next(row for row in analysis['comparisons']
                          if row['improved'] == 'list_' + variant and row['reference'] == 'unit_' + variant)
        assert comparison['mAP'] < 0 and comparison['Rank-1'] <= 0
    for group in (closed['archives'], states['archives']):
        for row in group.values():
            path = Path(row['local'])
            assert path.stat().st_size == row['file']['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['file']['sha256']
    remote = '/data/gaob/Re-ID/DeMo-DualAxis'
    protected = [
        'runs/full_official_baselines_20261004/training/' + name + '/best.pth'
        for name in ('RGBNT201_demo_s42', 'RGBNT100_demo_s42', 'MSVR310_demo_s42', 'MSVR310_demo_shared_s42')
    ] + [
        'runs/rgbnt201_identity_outlet_r201c_20261005/training/RGBNT201_identity_' + variant + '_narrow_s42/best.pth'
        for variant in ('frequency_shared', 'axis_shared')
    ] + [
        'runs/r201e_list_objective_fp32check_20261006/training/RGBNT201_r201e_' + variant + '_s42/best.pth'
        for variant in ('frequency_shared', 'axis_shared')
    ]
    code = f'''from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path({remote!r}).resolve(strict=True)
training=root/'runs/r201f_list_temperature_20261006'
state_root=root/'runs/r201f_normal_states_20261006'
mature_root=root/'runs/r201ef_mature_gradient_20261006'
for folder in (training,state_root,mature_root):
 assert json.loads((folder/'controller_result.json').read_text())['status']=='COMPLETE'
for label in ('E','F'):
 launch=json.loads((mature_root/(label+'_launch.json')).read_text())
 assert subprocess.run(['ps','-p',str(launch['pid'])],stdout=subprocess.DEVNULL).returncode==1
assert subprocess.run(['ps','-p',str({mature['controller']['pid']!r})],stdout=subprocess.DEVNULL).returncode==1
def receipt(path):
 return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
before={{name:receipt(root/name) for name in {protected!r}}}
targets={{}}
for variant in ('frequency_shared','axis_shared'):
 path=training/'training'/('RGBNT201_r201f_'+variant+'_s42')/'best.pth'
 assert path.resolve(strict=True).is_relative_to(training.resolve(strict=True))
 result=json.loads((path.parent/'result.json').read_text())
 assert result['status']=='COMPLETE' and result['epochs']==50 and result['amp_skipped_steps']==0
 assert json.loads((path.parent/'normal_local_archive.json').read_text())['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
 targets[str(path)]=receipt(path)
assert targets[str(training/'training/RGBNT201_r201f_axis_shared_s42/best.pth')]=={dict(bytes=mature['checkpoint_receipts']['F']['bytes'],sha256=mature['checkpoint_receipts']['F']['sha256'])!r}
free_before=shutil.disk_usage(root).free
for name in targets:
 Path(name).unlink()
assert all(not Path(name).exists() for name in targets)
assert before=={{name:receipt(root/name) for name in before}}
print(json.dumps(dict(retired=targets,retired_bytes=sum(r['bytes'] for r in targets.values()),
 protected=before,protected_unchanged=True,remote_free_before=free_before,remote_free_after=shutil.disk_usage(root).free)))'''
    result = json.loads(command(['ssh', *OPTIONS, '2026',
        shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python') + ' -'], input=code))
    result.update(status='ACTUAL_TWO_CLOSED_WEAKER_R201F_WEIGHTS_RETIRED',
        verified_at=datetime.now().isoformat(timespec='seconds'), local_D_files_deleted=0,
        local_raw_archives_preserved=4, new_neural_calls=0,
        reason='Both F controls lose to their retained matched unit controls; normal GT, raw archives, four-state and mature-gradient consumers are closed.',
        limits='Remote weight cleanup only; not newly freed D-drive space. Historical F arrays/text remain reproducible for CPU metrics; F neural replay requires retraining because its weights were intentionally retired.')
    proof.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'verified_at', 'retired_bytes', 'local_D_files_deleted', 'protected_unchanged')}))


if __name__ == '__main__':
    main()
