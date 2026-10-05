"""Collect terminal text once, then reduce installed-GT results locally; no NN."""
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

PF = PROJECT / 'results/preflight'
REMOTE_PROJECT = '/data/gaob/Re-ID/DeMo-DualAxis'
REMOTE_ROOT = REMOTE_PROJECT + '/runs/rgbnt201_identity_outlet_r201c_20261005'
REMOTE_PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'


def main():
    proof = json.loads((PF / 'rgbnt201_identity_outlet50_actual_session_20261005.json').read_text(encoding='utf-8'))
    assert proof['status'] == 'COMPLETE_R201C_FOUR50_ALL784STATE_GT_AND_LOCAL_RAW'
    assert proof['exit_code'] == 0 and proof['remote_root'] == REMOTE_ROOT
    assert len(proof['frozen']) == len(proof['normal']) == 4
    assert all(len(files) == 49 for files in proof['frozen'].values())
    review = json.loads((PF / 'rgbnt201_identity_outlet50_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources = review['sources_sha256'] | review['directly_reused_sources_sha256']
    remote_package = '/tmp/rgbnt201_r201c_terminal_text_20261005.tar.gz'
    code = f'''import hashlib,json,tarfile
from pathlib import Path
project=Path({REMOTE_PROJECT!r}); root=Path({REMOTE_ROOT!r})
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==digest
    for name,digest in {sources!r}.items() if not name.startswith('results/'))
controller=json.loads((root/'controller_result.json').read_text())
assert controller['status']=='COMPLETE' and controller['models']==4
assert controller['additional_epochs']==200 and controller['successful_updates']==10588
assert controller['closed_state_cases']==784 and controller['paired_sampling_exact']
assert controller['all_raw_local_verified']
for job in controller['runs']:
    name=job['name']; run=root/'training'/name; frozen=root/'frozen49'/name
    data=json.loads((run/'result.json').read_text())
    assert data['status']=='COMPLETE' and data['epochs']==50 and data['optimizer_steps']==2647
    assert data['amp_skipped_steps']==0 and data['descriptor_dim']==5120
    assert data['training_heldout_identities']==0
    assert data['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
    assert [p.name for p in run.glob('*.pth')]==['best.pth']
    assert json.loads((root/'training'/(name+'_exit.json')).read_text())['exit_code']==0
    audit=json.loads((frozen/'independent_fourstate_cpu_audit.json').read_text())
    assert audit['status']=='PASS' and audit['state_cases']==196
    assert json.loads((frozen/'local_archive.json').read_text())['status']=='ALL49_LOCAL_SIZE_SHA_VERIFIED_SERVER_RAW_CLEARED'
    assert json.loads((run/'normal_local_archive.json').read_text())['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
    assert not list(frozen.glob('*.npz')) and not (run/'best_official_arrays.npz').exists()
files={{}}; package=Path({remote_package!r}); assert not package.exists()
with tarfile.open(package,'w:gz') as archive:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
            relative=path.relative_to(root).as_posix(); data=path.read_bytes()
            files[relative]={{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}
            archive.add(path,arcname=relative,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))
'''
    inventory = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(REMOTE_PYTHON) + ' -'], input=code))
    package = PF / 'rgbnt201_r201c_terminal_text_20261005.tar.gz'
    command(['scp', *OPTIONS, '2026:' + remote_package, str(package)])
    assert package.stat().st_size == inventory['package']['bytes']
    assert hashlib.sha256(package.read_bytes()).hexdigest() == inventory['package']['sha256']
    local = PROJECT / 'results/rgbnt201_identity_outlet_r201c_20261005'
    local.mkdir(exist_ok=False)
    with tarfile.open(package, 'r:gz') as archive:
        assert set(archive.getnames()) == set(inventory['files'])
        archive.extractall(local, filter='data')
    assert all((local / name).stat().st_size == info['bytes'] and
        hashlib.sha256((local / name).read_bytes()).hexdigest() == info['sha256']
        for name, info in inventory['files'].items())
    command(['ssh', *OPTIONS, '2026', shlex.quote(REMOTE_PYTHON) + ' -'],
        input='from pathlib import Path\nPath(' + repr(remote_package) + ').unlink()\n')
    package.unlink()
    (PF / 'rgbnt201_r201c_terminal_text_intake_20261005.json').write_bytes((json.dumps(dict(
        status='ACTUAL_R201C_FOUR50_ALL784_GT_TERMINAL_TEXT_LOCAL_VERIFIED',
        verified_at=datetime.now().isoformat(timespec='seconds'), files=len(inventory['files']),
        remote_root=REMOTE_ROOT, local_root=str(local), manifest=inventory['files'],
        raw_npz_archived=200, retained_weights=4, new_neural_calls=0, new_optimizer_updates=0), indent=2) + '\n').encode('utf-8'))
    print(command([sys.executable, '-X', 'utf8', '-B', '-S', str(PROJECT / 'analyze_rgbnt201_identity_outlet50.py'),
        '--root', str(local), '--baseline', str(PROJECT / 'results/full_official_baselines_20261004/frozen49/RGBNT201_demo_s42')]), end='')


if __name__ == '__main__':
    main()
