"""Receive six already-completed normal runs after the original session closes."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command


def main():
    pf = PROJECT / 'results/preflight'
    actual = json.loads((pf / 'identity_coordinate_three_normal_actual_session_20261005.json').read_text(encoding='utf-8'))
    assert actual['status'] == 'COMPLETE_THREE_NORMAL_DATASETS_AND_FOUR_NEW50_GT_AND_LOCAL_RAW' and actual['exit_code'] == 0
    assert len(actual['archives']) == 4 and actual['new_successful_updates'] == 14124
    review = json.loads((pf / 'identity_coordinate_three_normal_cpu_closeout_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name, digest in review['sources_sha256'].items())
    remote = '/data/gaob/Re-ID/DeMo-DualAxis'
    root = remote + '/runs/identity_coordinate_full_normal_three_20261005'
    old = remote + '/runs/rgbnt201_identity_outlet_r201c_20261005'
    package_remote = '/tmp/identity_coordinate_three_normal_completed_text_20261005.tar.gz'
    runtime = json.loads((pf / 'identity_coordinate_three_normal_source_review_20261005.json').read_text(encoding='utf-8'))
    hashes = runtime['sources_sha256'] | runtime['directly_reused_sources_sha256']
    code = f'''import hashlib,json,tarfile
from pathlib import Path
project=Path({remote!r}); root=Path({root!r}); old=Path({old!r})
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==digest for name,digest in {hashes!r}.items() if not name.startswith('results/'))
controller=json.loads((root/'controller_result.json').read_text())
assert controller['status']=='COMPLETE' and controller['new_models']==4 and controller['successful_updates']==14124
assert controller['normal_archives_local_verified']==4 and controller['paired_sampling_exact']
assert controller['original_missing_controller_resumed']
paths={{path.relative_to(root).as_posix():path for path in root.rglob('*') if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt')}}
for dataset,count,steps in [('MSVR310',1032,705),('RGBNT100',8675,6357),('RGBNT201',3951,2647)]:
    for variant in ['frequency_shared','axis_shared']:
        name=dataset+'_identity_'+variant+'_narrow_s42'
        origin=old if dataset=='RGBNT201' else root
        run=origin/'training'/name; result=json.loads((run/'result.json').read_text())
        assert result['status']=='COMPLETE' and result['epochs']==50 and result['optimizer_steps']==result['steps']==steps
        assert result['amp_skipped_steps']==result['training_heldout_identities']==0 and result['descriptor_dim']==5120
        assert result['training_coverage']==dict(eligible=count,visited=count,unvisited=[])
        assert json.loads((origin/'training'/(name+'_exit.json')).read_text())['exit_code']==0
        assert [path.name for path in run.glob('*.pth')]==['best.pth']
        assert json.loads((run/'normal_local_archive.json').read_text())['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
        assert not (run/'best_official_arrays.npz').exists()
        if dataset=='RGBNT201':
            frozen=old/'frozen49'/name
            audit=json.loads((frozen/'independent_fourstate_cpu_audit.json').read_text())
            assert audit['status']=='PASS' and audit['state_cases']==196
            for path in run.rglob('*'):
                if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
                    paths['training/'+name+'/'+path.relative_to(run).as_posix()]=path
            for filename in ['independent_fourstate_cpu_audit.json','result.json','local_archive.json']:
                paths['rgbnt201_closed_GT/'+name+'/'+filename]=frozen/filename
        else:
            audit=json.loads((run/'normal_cpu_audit.json').read_text())
            assert audit['status']=='PASS' and audit['max_metric_error']==0
package=Path({package_remote!r}); assert not package.exists()
files={{}}
with tarfile.open(package,'w:gz') as archive:
    for relative,path in sorted(paths.items()):
        files[relative]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        archive.add(path,arcname=relative,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))
'''
    inventory = json.loads(command(['ssh', *OPTIONS, '2026', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -'], input=code))
    package = pf / 'identity_coordinate_three_normal_completed_text_20261005.tar.gz'
    assert not package.exists()
    command(['scp', *OPTIONS, '2026:' + package_remote, str(package)])
    assert package.stat().st_size == inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest() == inventory['package']['sha256']
    local = PROJECT / 'results/identity_coordinate_three_normal_completed_20261005'
    local.mkdir(exist_ok=False)
    with tarfile.open(package, 'r:gz') as archive:
        assert set(archive.getnames()) == set(inventory['files'])
        archive.extractall(local, filter='data')
    assert all((local / name).stat().st_size == row['bytes'] and hashlib.sha256((local / name).read_bytes()).hexdigest() == row['sha256'] for name, row in inventory['files'].items())
    command(['ssh', *OPTIONS, '2026', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -'], input='from pathlib import Path\nPath(' + repr(package_remote) + ').unlink()\n')
    package.unlink()
    receipt = dict(status='ACTUAL_THREE_NORMAL_SIXRUN_PRIMARY_TEXT_LOCAL_VERIFIED', verified_at=datetime.now().isoformat(timespec='seconds'),
        remote_root=root, local_root=str(local), files=len(inventory['files']), manifest=inventory['files'],
        completed_control_runs=6, new_optimizer_updates=0, new_neural_calls=0,
        raw_archive_proof='results/preflight/identity_coordinate_three_normal_actual_session_20261005.json')
    (pf / 'identity_coordinate_three_normal_completed_intake_20261005.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(command([sys.executable, '-X', 'utf8', '-B', '-S', str(PROJECT / 'analyze_identity_coordinate_three_normal.py'),
        '--root', str(local), '--baselines', str(PROJECT / 'results/full_official_baselines_20261004/training')]), end='')


if __name__ == '__main__':
    main()
