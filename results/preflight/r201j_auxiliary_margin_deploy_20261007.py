"""Run the reviewed primary-margin factor on three complete normal datasets, 201 first."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201j_auxiliary_margin_20261007'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201j_auxiliary_margin_20261007')
UPDATES={'RGBNT201':5294,'MSVR310':1410,'RGBNT100':12714}


def retire_nonselected_seed_weights(pf):
    receipt = pf / 'r201j_nonselected_N3_weights_retired_actual_20261007.json'
    assert not receipt.exists()
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    import csv
    n3 = PROJECT / 'results/r201i_expert_seeds_20261006/normal_seed_analysis'
    with (n3 / 'best_of_three.csv').open(encoding='utf-8-sig') as h:
        best = list(csv.DictReader(h))
    assert len(best) == 2 and all(int(x['seed']) == 42 for x in best)
    targets = []
    for seed in (43, 44):
        closed = load(pf / f'r201i_expert_s{seed}_actual_20261006.json')
        assert closed['status'] == 'COMPLETE' and closed['additional_epochs'] == 100 and closed['updates'] == 5292
        for name, item in closed['archives'].items():
            raw = Path(item['local'])
            assert raw.stat().st_size == item['file']['bytes'] and hashlib.sha256(raw.read_bytes()).hexdigest() == item['file']['sha256']
            local = PROJECT / f'results/r201i_expert_seeds_20261006/s{seed}/training' / name
            result = load(local / 'result.json')
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50 and result['amp_skipped_steps'] == 0
            assert load(local / 'normal_cpu_audit.json')['status'] == 'PASS'
            targets.append(f'runs/r201i_expert_seeds_20261006/s{seed}/training/{name}/best.pth')
    assert len(targets) == 4
    code = f'''import hashlib,json,shutil
from pathlib import Path
root=Path({REMOTE!r});targets={targets!r}
assert all(not ('r201i_' in (p/'cmdline').read_bytes().decode(errors='replace') and b'python -' not in (p/'cmdline').read_bytes()) for p in Path('/proc').iterdir() if p.name.isdecimal() and (p/'cmdline').is_file())
protected=list(root.glob('runs/full_official_baselines_20261004/training/*_demo_s42/best.pth'))+list(root.glob('runs/r201i_primary_margin_20261006/*/training/*/best.pth'))
assert len(protected)==9
before={{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}}
rows=[]
for name in targets:
 p=root/name;assert p.resolve().is_relative_to((root/'runs/r201i_expert_seeds_20261006').resolve()) and p.name=='best.pth'
 result=json.loads((p.parent/'result.json').read_text());assert result['status']=='COMPLETE' and result['epochs']==50 and result['arguments']['seed'] in (43,44)
 assert json.loads((p.parent/'normal_cpu_audit.json').read_text())['status']=='PASS'
 assert json.loads((p.parent.parent/(p.parent.name+'_exit.json')).read_text())['exit_code']==0
 rows.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
free_before=shutil.disk_usage(root).free
for row in rows:(root/row['path']).unlink()
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in before.items())
print(json.dumps(dict(status='ACTUAL_FOUR_NONSELECTED_N3_NORMAL_WEIGHTS_RETIRED',rows=rows,removed_bytes=sum(r['bytes'] for r in rows),free_before=free_before,free_after=shutil.disk_usage(root).free,protected_anchors_and_I42_best=before,raw_unique_lost=0,limit='NN replay of nonselected43/44 requires retraining; raw/allmetrics/GT and bestof3 seed42 retained. No dataset/code/current anchor removed.')))'''
    value = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code))
    receipt.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print('ACTUAL_NONSELECTED_N3_WEIGHTS_RETIRED_BYTES', value['removed_bytes'], flush=True)


def main():
    pf=PROJECT/'results/preflight'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    old=load(pf/'r201i_missing49_actual_session_20261007.json')
    assert old['status']=='ACTUAL_I_ALL_DECLARED_FIXED_BEST_MISSING49_GT_RAW_CPU_COMPLETE' and old['unique_raw_lost']==0
    n3=load(PROJECT/'results/r201i_expert_seeds_20261006/normal_seed_analysis/result.json')
    assert n3['status']=='ACTUAL_I_RGBNT201_THREE_EXPERT_SEEDS_NORMAL_CPU_COMPLETE'
    review=load(pf/'r201j_auxiliary_margin_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    cpu_review=load(pf/'r201j_normal_cpu_closeout_source_review_20261007.json')
    assert cpu_review['status']=='PASS' and not cpu_review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (cpu_review['sources_sha256']|cpu_review['directly_reused_sources_sha256']).items())
    sources,reused=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'run_r201j_auxiliary_margin.py','launch_r201j_auxiliary_margin.py','results/preflight/r201j_auxiliary_margin_deploy_20261007.py'}
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (sources|reused).items())
    proof=pf/'r201j_auxiliary_margin_actual_session_20261007.json'
    assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>900_000_000
    retire_nonselected_seed_weights(pf)
    code=f'''import hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});output=Path({ROOT!r})
assert not output.exists() and shutil.disk_usage(root).free>4_500_000_000
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {reused!r}.items() if not name.startswith('results/'))
print('ACTUAL_CLOSED_I_REUSED_SOURCES_AND_J_STORAGE_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    for name in sources:
        if not name.startswith('results/'):
            command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    deployed={name:sha for name,sha in sources.items() if not name.startswith('results/')}
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {deployed!r}.items())
print('ACTUAL_REVIEWED_J_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    outcomes={}
    for dataset,updates in UPDATES.items():
        out=ROOT+'/'+dataset
        receipt=pf/('r201j_auxiliary_margin_'+dataset+'_actual_session_20261007.json')
        assert not receipt.exists()
        argv=[PYTHON,'-u','launch_r201j_auxiliary_margin.py','--dataset',dataset,
            '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
            'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        print('J_ORIGINAL_DATASET_CONTROLLER_STARTED',dataset,flush=True)
        for line in process.stdout:
            if not line.startswith('R201J_STREAM '):
                print(line,end='',flush=True)
                continue
            message=json.loads(line[len('R201J_STREAM '):])
            event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero']
                native=True
                (pf/('r201j_auxiliary_margin_'+dataset+'_native_actual_20261007.json')).write_text(json.dumps(dict(
                    status='ACTUAL_TWO_J_NATIVE3_AUXILIARY_MARGIN03_PASS_FORMAL_NEXT',observed_at=datetime.now().isoformat(timespec='seconds'),
                    dataset=dataset,remote_root=out,actual_native_updates=6,primary_margin=.3,auxiliary_M_F_hinge03=True,remaining_base_common_soft_unchanged=True,
                    partial_loss_effective_zero=True,normal_full50_complete=False),indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_J_NATIVE6_PASS',dataset,flush=True)
            elif event=='NORMAL_READY':
                name=message['name']
                assert native and name.startswith(dataset+'_r201j_') and name not in archives
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n')
                process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                name=message['name']
                assert name in archives and name not in cleared
                cleared.add(name)
                print('J_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',dataset,name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==updates
                finished=True
        process.stdin.close()
        assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
        value=dict(status='ACTUAL_J_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().isoformat(timespec='seconds'),
            exit_code=0,dataset=dataset,remote_root=out,archives=archives,native_updates=6,successful_updates=updates,
            additional_epochs=100,sources_sha256=sources,limits='Fixed auxiliary-M/F margin hypothesis; no +2 or missing/multiple-seed success assertion.')
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        outcomes[dataset]=value
        print('ACTUAL_J_TWO_FULL50_CLOSED',dataset,flush=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201j_dataset_normal_complete_intake_20261007.py'),'--dataset',dataset],check=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201j_dataset_normal.py'),'--dataset',dataset],check=True)
    result=dict(status='ACTUAL_J_THREE_NORMAL_DATASETS_SIX_FULL50_GT_RAW_COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),
        exit_code=0,datasets=outcomes,native_updates=18,successful_updates=19418,additional_epochs=300,
        sources_sha256=sources,missing_evaluated=False,limits='Existing hinge auxiliary-M/F factor, three full normal datasets only; full goal requires metrics/fair controls/missing/seeds/costs and is not inferred from completion.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_J_THREE_NORMAL_CONTROLLER_CLOSED',flush=True)


if __name__=='__main__':main()
