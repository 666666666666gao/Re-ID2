"""Execute reviewed normal-only diagnostics after actual F training closure."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TRAINING = REMOTE + '/runs/r201g_normal_priority_20261006'
ROOT = REMOTE + '/runs/r201g_normal_states_20261006'
ARCHIVE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201g_normal_states_20261006')
REFERENCE = ARCHIVE.parent / 'full_official_baselines_20261004/raw_distances/RGBNT201_demo_s42/q_RNT_g_RNT.npz'


def main():
    pf = PROJECT / 'results/preflight'
    proof = pf / 'r201g_normal_states_actual_20261006.json'
    assert not proof.exists() and not ARCHIVE.exists()
    closed = json.loads((pf / 'r201g_normal_priority_actual_session_20261006.json').read_text())
    assert closed['exit_code'] == 0 and closed['successful_updates'] == 5294 and len(closed['archives']) == 2
    analysis = json.loads((PROJECT / 'results/r201g_normal_priority_20261006/normal_analysis/result.json').read_text())
    assert analysis['status'] == 'ACTUAL_R201G_TWO_FULL50_NORMAL_CPU_ANALYSIS_COMPLETE'
    review = json.loads((pf / 'r201g_normal_states_source_review_20261006.json').read_text())
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources = review['sources_sha256']
    reused = review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in (sources | reused).items())
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == review['original_normal_raw_sha256']
    for archived in closed['archives'].values():
        path = Path(archived['local'])
        assert path.stat().st_size == archived['file']['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == archived['file']['sha256']
    assert shutil.disk_usage(ARCHIVE.parent).free > 128 * 1024**2
    for name in sources:
        if not name.startswith('results/'):
            command(['scp', *OPTIONS, str(PROJECT / name), '2026:' + REMOTE + '/' + name])
    deployed = {name: sha for name, sha in (sources | reused).items() if not name.startswith('results/')}
    check = f'''import hashlib,json,shutil
from pathlib import Path
root=Path({REMOTE!r})
assert not Path({ROOT!r}).exists() and shutil.disk_usage(root).free>128*1024**2
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {deployed!r}.items())
assert json.loads((Path({TRAINING!r})/'controller_result.json').read_text())['status']=='COMPLETE'
print('F_NORMAL_STATES_NEW_AND_REUSED_SOURCES_EXACT')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=check), end='')
    argv = [PYTHON, '-u', 'launch_r201g_normal_states.py', '--training-root', TRAINING, '--output', ROOT]
    process = subprocess.Popen(['ssh', *OPTIONS, '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', '2026',
        'cd ' + shlex.quote(REMOTE) + ' && ' + shlex.join(argv)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    archives, cleared = {}, set()
    for line in process.stdout:
        if not line.startswith('R201G_STATES '):
            print(line, end='', flush=True)
            continue
        message = json.loads(line[len('R201G_STATES '):])
        name = message['name']
        if message['event'] == 'RAW_READY':
            assert name not in archives and shutil.disk_usage(ARCHIVE.parent).free > message['file']['bytes']
            local = copy_verified(message['path'], message['file'], Path(ROOT), ARCHIVE)
            normal = Path(closed['archives'][name]['local'])
            with zipfile.ZipFile(local) as diagnostic, zipfile.ZipFile(normal) as selected, zipfile.ZipFile(REFERENCE) as original:
                assert diagnostic.read('distances.npy') == selected.read('distances.npy')
                assert diagnostic.read('distances_00.npy') == original.read('distances.npy')
                for prefix in ('query', 'gallery'):
                    for key in ('names', 'ids', 'cameras', 'scenes'):
                        member = prefix + '_' + key + '.npy'
                        assert diagnostic.read(member) == selected.read(member)
            archives[name] = dict(file=message['file'], local=local,
                full11_distance_equal_selected_normal=True, base00_distance_equal_original_DeMo=True,
                eight_GT_metadata_arrays_equal_selected_normal=True)
            process.stdin.write(json.dumps(dict(event='RAW_ARCHIVED', name=name, file=message['file'])) + '\n')
            process.stdin.flush()
        else:
            assert message['event'] == 'RAW_CLEARED' and name in archives and name not in cleared
            cleared.add(name)
            print('F_NORMAL_STATES_LOCAL_GT_DISTANCE_MATCH_REMOTE_CLEARED', name, flush=True)
    process.stdin.close()
    assert process.wait() == 0 and len(archives) == len(cleared) == 2
    package_name = '/tmp/r201g_normal_states_terminal_text_20261006.tar.gz'
    inventory_code = f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({ROOT!r});package=Path({package_name!r})
result=json.loads((root/'controller_result.json').read_text())
assert result['status']=='COMPLETE' and result['state_cases']==8 and result['original_raw_local_verified']==2
assert not list(root.rglob('*.pth')) and not list(root.rglob('*.npz')) and not package.exists()
files={{}}
with tarfile.open(package,'w:gz') as archive:
 for path in sorted(root.rglob('*')):
  if path.is_file():
   name=path.relative_to(root).as_posix();data=path.read_bytes()
   files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
   archive.add(path,arcname=name,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=inventory_code))
    package = pf / 'r201g_normal_states_terminal_text_20261006.tar.gz'
    local = PROJECT / 'results/r201g_normal_states_20261006'
    assert not package.exists() and not local.exists()
    command(['scp', *OPTIONS, '2026:' + package_name, str(package)])
    assert package.stat().st_size == inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest() == inventory['package']['sha256']
    local.mkdir()
    with tarfile.open(package, 'r:gz') as archive:
        assert set(archive.getnames()) == set(inventory['files'])
        archive.extractall(local, filter='data')
    assert all((local / name).stat().st_size == row['bytes'] and hashlib.sha256((local / name).read_bytes()).hexdigest() == row['sha256'] for name, row in inventory['files'].items())
    command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input='from pathlib import Path\nPath(' + repr(package_name) + ').unlink()\n')
    package.unlink()
    result = dict(status='ACTUAL_R201G_TWO_NORMAL_FOURSTATE_GT_RAW_COMPLETE', verified_at=datetime.now().isoformat(timespec='seconds'),
        state_cases=8, neural_extraction_passes=2, optimizer_updates=0, checkpoint_writes=0, archives=archives,
        text_manifest=inventory['files'], installed_GT_six_CMC50_perquery_groups=True,
        limits='Controlled inference of two selected checkpoints, one seed and one normal condition. No retraining, missing, three-dataset or calibrated-contribution acceptance.')
    proof.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'verified_at', 'state_cases', 'optimizer_updates')}), flush=True)


if __name__ == '__main__':
    main()
