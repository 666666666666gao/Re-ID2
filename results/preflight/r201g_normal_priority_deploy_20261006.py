"""Start one reviewed RGBNT201 normal-priority pair after closed prior consumers."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT = REMOTE + '/runs/r201g_normal_priority_20261006'
ARCHIVE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201g_normal_priority_20261006')


def main():
    pf = PROJECT / 'results/preflight'
    old = json.loads((pf / 'r201f_list_temperature_actual_session_20261006.json').read_text(encoding='utf-8'))
    mature = json.loads((pf / 'r201ef_mature_gradient_actual_20261006.json').read_text(encoding='utf-8'))
    assert old['exit_code'] == 0 and old['successful_updates'] == 5294
    assert mature['status'] == 'ACTUAL_E_F_MATURE_TRAIN_ONLY_DEPLOY_GRADIENT_PAIR_COMPLETE'
    assert json.loads((pf / 'r201f_weaker_weights_retired_20261006.json').read_text())['protected_unchanged']
    state_root = PROJECT / 'results/r201f_normal_states_20261006'
    state_pids = [json.loads((state_root / 'controller_launch.json').read_text())['pid']]
    state_pids += [json.loads((state_root / ('RGBNT201_r201f_' + variant + '_s42_launch.json')).read_text())['pid']
                  for variant in ('frequency_shared', 'axis_shared')]
    proof = pf / 'r201g_normal_priority_actual_session_20261006.json'
    assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free > 128 * 1024**2
    review = json.loads((pf / 'r201g_normal_priority_source_review_20261006.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources, reused = review['sources_sha256'], review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in (sources | reused).items())
    pids = [3810565, 3811689, 3811690, mature['controller']['pid'], *state_pids]
    code = f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({REMOTE!r});output=Path({ROOT!r});previous=root/'runs/r201ef_mature_gradient_20261006'
assert not output.exists() and shutil.disk_usage(root).free>1_100_000_000
assert json.loads((previous/'controller_result.json').read_text())['status']=='COMPLETE'
for label in ('E','F'):
 assert subprocess.run(['ps','-p',str(json.loads((previous/(label+'_launch.json')).read_text())['pid'])],stdout=subprocess.DEVNULL).returncode==1
for pid in {tuple(pids)!r}:
 assert subprocess.run(['ps','-p',str(pid)],stdout=subprocess.DEVNULL).returncode==1
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {reused!r}.items() if not name.startswith('results/'))
print('R201G_PREVIOUS_CONSUMERS_CLOSED_SOURCES_STORAGE_VERIFIED')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code), end='')
    for name in sources:
        if not name.startswith('results/'):
            command(['scp', *OPTIONS, str(PROJECT / name), '2026:' + REMOTE + '/' + name])
    deployed = {name: sha for name, sha in sources.items() if not name.startswith('results/')}
    code = f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {deployed!r}.items())
print('R201G_NEW_SOURCES_EXACT')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code), end='')
    argv = [PYTHON, '-u', 'launch_r201g_normal_priority.py', '--dataset', 'RGBNT201',
        '--data-root', '/data/gaob/Re-ID/dataset',
        '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-root', REMOTE + '/runs/full_official_baselines_20261004/training', '--output', ROOT]
    process = subprocess.Popen(['ssh', *OPTIONS, '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', '2026',
        'cd ' + shlex.quote(REMOTE) + ' && ' + shlex.join(argv)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    archives, cleared, native, complete = {}, set(), False, False
    print('R201G_ACTUAL_CONTROLLER_STARTED', flush=True)
    for line in process.stdout:
        if not line.startswith('R201G_STREAM '):
            print(line, end='', flush=True)
            continue
        message = json.loads(line[len('R201G_STREAM '):])
        event = message['event']
        if event == 'NATIVE_PASS':
            assert not native and message['controls'] == 2 and message['actual_updates'] == 6
            assert message['partial_loss_effective_zero']
            native = True
            (pf / 'r201g_normal_priority_native_actual_20261006.json').write_text(json.dumps(dict(
                status='ACTUAL_TWO_NATIVE3_NORMAL_PRIORITY_PASS_FORMAL_NEXT',
                observed_at=datetime.now().isoformat(timespec='seconds'), remote_root=ROOT,
                actual_native_updates=6, partial_loss_effective_zero=True,
                partial_forward_BN_retained=True, new_formal_results=0), indent=2) + '\n', encoding='utf-8')
            print('R201G_ACTUAL_NATIVE6_PASS', flush=True)
        elif event == 'NORMAL_READY':
            name = message['name']
            assert native and name not in archives and shutil.disk_usage(ARCHIVE.parent).free > message['file']['bytes']
            local = copy_verified(message['path'], message['file'], Path(ROOT), ARCHIVE)
            archives[name] = dict(file=message['file'], local=local)
            process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED', name=name, file=message['file'])) + '\n')
            process.stdin.flush()
        elif event == 'NORMAL_CLEARED':
            name = message['name']
            assert name in archives and name not in cleared
            cleared.add(name)
            print('R201G_NORMAL_LOCAL_VERIFIED_REMOTE_CLEARED', name, flush=True)
        else:
            assert event == 'CONTROLLER_COMPLETE' and not complete and message['controls'] == 2 and message['successful_updates'] == 5294
            complete = True
    process.stdin.close()
    assert process.wait() == 0 and native and complete and len(archives) == len(cleared) == 2
    proof.write_text(json.dumps(dict(status='ACTUAL_R201G_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',
        finished=datetime.now().isoformat(timespec='seconds'), exit_code=0, remote_root=ROOT,
        archives=archives, native_updates=6, successful_updates=5294, additional_epochs=100,
        sources_sha256=sources, limits='No other two dataset, missing, paired multiseed or +2 completion claim.'), indent=2) + '\n', encoding='utf-8')
    print('R201G_TWO50_NORMAL_GT_ARCHIVE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
