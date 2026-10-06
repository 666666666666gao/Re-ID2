"""Collect already-closed four G remaining-dataset inference texts; reduce full49 only after actual completion."""
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
    actual = json.loads((pf / ('r201g_other_two_missing49_stream_' + args.mode + '_actual_session_20261006.json')).read_text(encoding='utf-8'))
    expected = 4 if args.mode == 'native' else 196
    accepted = 'ACTUAL_G_OTHER_TWO_FOUR_NATIVE_INFERENCE_CONTROLS_ALL16_NORMAL_GT_AND_LOCAL_RAW' if args.mode == 'native' else 'ACTUAL_G_OTHER_TWO_FOUR_FULL49_INFERENCE_CONTROLS_ALL784_GT_AND_LOCAL_RAW'
    assert actual['status'] == accepted and actual['exit_code'] == 0 and len(actual['raw']) == expected
    review = json.loads((pf / 'r201g_other_two_missing49_cpu_closeout_source_review_20261006.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name, digest in review['sources_sha256'].items())
    remote = '/data/gaob/Re-ID/DeMo-DualAxis'
    root = remote + '/runs/r201g_other_two_missing49_stream_20261006'
    python = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
    tag = 'r201g_other_two_missing49_stream_' + args.mode + '_closed_text_20261006'
    remote_package = '/tmp/' + tag + '.tar.gz'
    package = pf / (tag + '.tar.gz')
    local = PROJECT / ('results/r201g_other_two_missing49_stream_' + args.mode + '_completed_20261006')
    proof = pf / ('r201g_other_two_missing49_' + args.mode + '_intake_20261006.json')
    assert not package.exists() and not local.exists() and not proof.exists()
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
    command(['scp', *OPTIONS, '2026:' + remote_package, str(package)])
    assert package.stat().st_size == inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest() == inventory['package']['sha256']
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
    proof.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.mode == 'full':
        print(command([sys.executable, '-X', 'utf8', '-B', '-S', str(PROJECT / 'analyze_r201g_other_two_missing49.py'), '--stream', str(local)]), end='')
    else:
        print('ACTUAL_NATIVE_MISSING_INFERENCE_TEXT_INTAKE_COMPLETE', len(inventory['files']), flush=True)


if __name__ == '__main__':
    main()
