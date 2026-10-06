"""After accepted RGBNT201 closure, run the unchanged G pair on both other datasets."""
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
ROOT = REMOTE + '/runs/r201g_other_two_normal_20261006'
ARCHIVE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201g_other_two_normal_20261006')
COUNTS = {'MSVR310': (1032,591,1055,705), 'RGBNT100': (8675,1715,8575,6357)}


def main():
    pf = PROJECT / 'results/preflight'
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    proof = pf / 'r201g_other_two_normal_actual_session_20261006.json'
    assert not proof.exists() and not ARCHIVE.exists()
    closed = load(pf / 'r201g_normal_priority_actual_session_20261006.json')
    assert closed['exit_code'] == 0 and closed['successful_updates'] == 5294 and len(closed['archives']) == 2
    normal = load(PROJECT / 'results/r201g_normal_priority_20261006/normal_analysis/result.json')
    assert normal['status'] == 'ACTUAL_R201G_TWO_FULL50_NORMAL_CPU_ANALYSIS_COMPLETE'
    baseline = next(row for row in normal['comparisons'] if row['improved'] == 'normal_axis_shared' and row['reference'] == 'original_DeMo')
    fair = next(row for row in normal['comparisons'] if row['improved'] == 'normal_axis_shared' and row['reference'] == 'normal_frequency_shared')
    assert baseline['both_plus2'] and fair['mAP'] > 0 and fair['Rank-1'] > 0
    states = load(pf / 'r201g_normal_states_actual_20261006.json')
    assert states['status'] == 'ACTUAL_R201G_TWO_NORMAL_FOURSTATE_GT_RAW_COMPLETE' and states['state_cases'] == 8 and states['optimizer_updates'] == states['checkpoint_writes'] == 0
    state_analysis = load(PROJECT / 'results/r201g_normal_states_20261006/normal_state_analysis/result.json')
    assert state_analysis['status'] == 'ACTUAL_R201G_SELECTED_NORMAL_FOURSTATE_CPU_READOUT_COMPLETE'
    utility = next(row for row in state_analysis['comparisons'] if row['improved_variant'] == row['reference_variant'] == 'axis_shared' and row['improved_state'] == '11' and row['reference_state'] == '00')
    assert utility['mAP'] > 0 and utility['Rank-1'] > 0
    review = load(pf / 'r201g_other_two_normal_source_review_20261006.json')
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources = review['sources_sha256'] | review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in sources.items())
    assert shutil.disk_usage(ARCHIVE.parent).free > 600 * 1024**2
    state_root = PROJECT / 'results/r201g_normal_states_20261006'
    pids = [3986116,3988001,3988002,load(state_root / 'controller_launch.json')['pid']]
    pids.extend(load(state_root / ('RGBNT201_r201g_' + variant + '_s42_launch.json'))['pid'] for variant in ('frequency_shared','axis_shared'))
    check = f'''import hashlib,json,shutil,subprocess
from pathlib import Path
project=Path({REMOTE!r}); output=Path({ROOT!r})
assert not output.exists() and shutil.disk_usage(project).free>2_200_000_000
for pid in {pids!r}:
 assert subprocess.run(['ps','-p',str(pid)],stdout=subprocess.DEVNULL).returncode==1
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items() if not name.startswith('results/'))
for dataset,(train,query,gallery,steps) in {COUNTS!r}.items():
 folder=project/'runs/full_official_baselines_20261004/training'/(dataset+'_demo_s42')
 data=json.loads((folder/'result.json').read_text())
 assert data['status']=='COMPLETE' and data['epochs']==50 and data['optimizer_steps']==steps and data['amp_skipped_steps']==0
 assert (data['train_records'],data['query_records'],data['gallery_records'])==(train,query,gallery)
 assert data['training_coverage']==dict(eligible=train,visited=train,unvisited=[]) and (folder/'best.pth').exists()
print('G201_ACCEPTED_ALL_PRIOR_CONSUMERS_CLOSED_REUSED_SOURCES_EXACT_OTHER_TWO_FULL_BASELINES_READY')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=check), end='', flush=True)
    archives, datasets = {}, {}
    for dataset, counts in COUNTS.items():
        dataset_root = ROOT + '/' + dataset
        argv = [PYTHON,'-u','launch_r201g_normal_priority.py','--dataset',dataset,
            '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',dataset_root]
        process = subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
            'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        (pf / 'r201g_other_two_normal_current_receiver_20261006.json').write_text(json.dumps(dict(
            status='ACTUAL_ORIGINAL_SSH_RECEIVER_STARTED_NOT_YET_NATIVE_ACCEPTED',dataset=dataset,remote_root=dataset_root,
            local_ssh_pid=process.pid,started_at=datetime.now().isoformat(timespec='seconds'),
            completed_datasets=list(datasets),policy='Same original sequential receiver; native pair then50/GT/raw. At most2NN, no restart on observation timeout.'),indent=2)+'\n',encoding='utf-8')
        print('G_OTHER_DATASET_ORIGINAL_RECEIVER_STARTED',dataset,process.pid,flush=True)
        native, complete, cleared, local_names = False, False, set(), set()
        for line in process.stdout:
            if not line.startswith('R201G_STREAM '):
                print(line,end='',flush=True)
                continue
            message = json.loads(line[len('R201G_STREAM '):])
            event = message['event']
            if event == 'NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero']
                native = True
                (pf / ('r201g_'+dataset+'_native_actual_20261006.json')).write_text(json.dumps(dict(
                    status='ACTUAL_UNCHANGED_G_OTHER_DATASET_NATIVE6_PASS_FORMAL_NEXT',dataset=dataset,
                    observed_at=datetime.now().isoformat(timespec='seconds'),actual_native_updates=6,
                    remote_root=dataset_root,partial_loss_effective_zero=True,full_formal_completed=0),indent=2)+'\n',encoding='utf-8')
                print('G_OTHER_NATIVE6_ACTUAL_PASS',dataset,flush=True)
            elif event == 'NORMAL_READY':
                name = message['name']
                assert native and name.startswith(dataset+'_r201g_') and name not in archives
                assert shutil.disk_usage(ARCHIVE.parent).free > message['file']['bytes']
                local = copy_verified(message['path'],message['file'],Path(dataset_root),ARCHIVE/dataset)
                archives[name] = dict(dataset=dataset,file=message['file'],local=local)
                local_names.add(name)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n')
                process.stdin.flush()
            elif event == 'NORMAL_CLEARED':
                name = message['name']
                assert name in local_names and name not in cleared
                cleared.add(name)
                print('G_OTHER_NORMAL_LOCAL_VERIFIED_REMOTE_CLEARED',name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not complete and message['controls']==2 and message['successful_updates']==2*counts[3]
                complete = True
        process.stdin.close()
        assert process.wait()==0 and native and complete and len(local_names)==len(cleared)==2
        datasets[dataset] = dict(status='ACTUAL_TWO_FULL50_NORMAL_GT_RAW_COMPLETE',finished_at=datetime.now().isoformat(timespec='seconds'),
            remote_root=dataset_root,controls=2,additional_epochs=100,native_updates=6,successful_updates=2*counts[3],
            archives={name:archives[name] for name in sorted(local_names)})
        (pf / ('r201g_'+dataset+'_normal_actual_20261006.json')).write_text(json.dumps(datasets[dataset],indent=2)+'\n',encoding='utf-8')
    assert len(archives)==4 and sum(row['successful_updates'] for row in datasets.values())==14124
    proof.write_text(json.dumps(dict(status='ACTUAL_UNCHANGED_G_OTHER_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',
        finished_at=datetime.now().isoformat(timespec='seconds'),exit_code=0,datasets=datasets,archives=archives,
        native_updates=12,successful_updates=14124,additional_epochs=200,sources_sha256=sources,
        limits='One unified G recipe/seed42; original50+additional50. Four normal controls only; no missing49, multiseed, +2 or final goal claim without full CPU readout.'),indent=2)+'\n',encoding='utf-8')
    print('G_OTHER_TWO_NORMAL_FOUR50_GT_RAW_ACTUALLY_CLOSED',flush=True)


if __name__ == '__main__':
    main()
