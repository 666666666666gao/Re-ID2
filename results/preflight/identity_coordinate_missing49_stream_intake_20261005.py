"""Collect already-closed inference text, then reduce full three-dataset results."""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('native', 'full'))
    args = parser.parse_args()
    pf = PROJECT / 'results/preflight'
    actual = json.loads((pf / ('identity_coordinate_missing49_stream_' + args.mode + '_actual_session_20261005.json')).read_text(encoding='utf-8'))
    expected = 4 if args.mode == 'native' else 196
    accepted = 'ACTUAL_FOUR_NATIVE_INFERENCE_CONTROLS_ALL16_NORMAL_GT_AND_LOCAL_RAW' if args.mode == 'native' else 'ACTUAL_FOUR_FULL49_INFERENCE_CONTROLS_ALL784_GT_AND_LOCAL_RAW'
    assert actual['status'] == accepted and actual['exit_code'] == 0 and len(actual['raw']) == expected
    review = json.loads((pf / 'identity_coordinate_missing49_cpu_closeout_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name, digest in review['sources_sha256'].items())
    remote = '/data/gaob/Re-ID/DeMo-DualAxis'
    root = remote + '/runs/identity_coordinate_missing49_stream_20261005'
    python = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
    tag = 'identity_coordinate_missing49_stream_' + args.mode + '_closed_text_20261005'
    remote_package = '/tmp/' + tag + '.tar.gz'
    code = f'''import hashlib,json,tarfile
from pathlib import Path
project=Path({remote!r}); root=Path({root!r}); mode={args.mode!r}
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==digest for name,digest in {actual['sources_sha256']!r}.items() if not name.startswith('results/'))
controller=root/('controller_'+mode+'_result.json'); data=json.loads(controller.read_text())
assert data['status']==('PASS' if mode=='native' else 'COMPLETE') and data['controls']==4
assert data['raw_conditions']=={expected} and data['all_raw_local_verified']
assert data['optimizer_updates']==data['new_weights']==0
paths={{controller.name:controller}}
for path in (root/mode).rglob('*'):
    if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
        paths[path.relative_to(root).as_posix()]=path
for run in data['runs']:
    folder=Path(run['result']).parent; result=json.loads((folder/'result.json').read_text())
    audit=json.loads((folder/'independent_cpu_audit.json').read_text())
    assert result['status']=='COMPLETE' and audit['status']=='PASS'
    assert result['state_cases']==audit['state_cases']==(4 if mode=='native' else 196)
    assert result['optimizer_updates']==result['new_weights']==0
    assert json.loads((root/mode/(run['name']+'_exit.json')).read_text())['exit_code']==0
    assert not list(folder.glob('q_*_g_*/raw.npz'))
package=Path({remote_package!r}); assert not package.exists(); files={{}}
with tarfile.open(package,'w:gz') as archive:
    for relative,path in sorted(paths.items()):
        files[relative]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        archive.add(path,arcname=relative,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(python) + ' -'], input=code))
    package = pf / (tag + '.tar.gz')
    assert not package.exists()
    command(['scp', *OPTIONS, '2026:' + remote_package, str(package)])
    assert package.stat().st_size == inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest() == inventory['package']['sha256']
    local = PROJECT / ('results/identity_coordinate_missing49_stream_' + args.mode + '_completed_20261005')
    local.mkdir(exist_ok=False)
    with tarfile.open(package, 'r:gz') as archive:
        assert set(archive.getnames()) == set(inventory['files'])
        archive.extractall(local, filter='data')
    assert all((local / name).stat().st_size == file['bytes'] and hashlib.sha256((local / name).read_bytes()).hexdigest() == file['sha256'] for name, file in inventory['files'].items())
    command(['ssh', *OPTIONS, '2026', shlex.quote(python) + ' -'], input='from pathlib import Path\nPath(' + repr(remote_package) + ').unlink()\n')
    package.unlink()
    receipt = dict(status='ACTUAL_CLOSED_MISSING_INFERENCE_PRIMARY_TEXT_LOCAL_VERIFIED', mode=args.mode,
        verified_at=datetime.now().isoformat(timespec='seconds'), files=len(inventory['files']), manifest=inventory['files'],
        local_root=str(local), new_neural_calls=0, new_optimizer_updates=0)
    (pf / ('identity_coordinate_missing49_' + args.mode + '_intake_20261005.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.mode == 'full':
        print(command([sys.executable, '-X', 'utf8', '-B', '-S', str(PROJECT / 'analyze_identity_coordinate_missing49.py'),
            '--stream', str(local), '--rgbnt201', str(PROJECT / 'results/rgbnt201_identity_outlet_r201c_20261005'),
            '--baselines', str(PROJECT / 'results/full_official_baselines_20261004'),
            '--normal', str(PROJECT / 'results/identity_coordinate_three_normal_completed_20261005')]), end='')
    else:
        print('ACTUAL_NATIVE_MISSING_INFERENCE_TEXT_INTAKE_COMPLETE', len(inventory['files']), flush=True)


if __name__ == '__main__':
    main()
