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
ROOT=REMOTE+'/runs/r201i_primary_margin_20261006'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201i_primary_margin_20261006')
UPDATES={'RGBNT201':5294,'MSVR310':1410,'RGBNT100':12714}


def main():
    pf=PROJECT/'results/preflight'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    full=load(pf/'r201g_other_two_missing49_stream_full_actual_session_20261006.json')
    queue=load(pf/'r201g_other_two_missing49_closeout_queue_actual_session_20261006.json')
    assert full['exit_code']==queue['exit_code']==0 and len(full['raw'])==196 and full['restored_copies_cleared']
    complete=load(PROJECT/'results/r201g_other_two_missing49_stream_full_completed_20261006/three_dataset_missing_analysis/result.json')
    assert complete['status']=='ACTUAL_G_THREE_FULL_OFFICIAL_FIXED_BEST_ALL49_CPU_READOUT'
    assert complete['dataset_count']==3 and complete['paired_query_rows']==461874
    review=load(pf/'r201i_primary_margin_source_review_20261006.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reused=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'run_r201i_primary_margin.py','launch_r201i_primary_margin.py','results/preflight/r201i_primary_margin_deploy_20261006.py'}
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (sources|reused).items())
    proof=pf/'r201i_primary_margin_actual_session_20261006.json'
    assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>900_000_000
    code=f'''import hashlib,json,shutil
from pathlib import Path
root=Path({REMOTE!r});output=Path({ROOT!r})
assert not output.exists() and shutil.disk_usage(root).free>3_500_000_000
old=root/'runs/r201g_other_two_missing49_stream_20261006'
closed=json.loads((old/'controller_full_result.json').read_text())
assert closed['status']=='COMPLETE' and closed['controls']==4 and closed['missing_GT_state_cases']==784 and closed['all_raw_local_verified']
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {reused!r}.items() if not name.startswith('results/'))
print('ACTUAL_G_FULL_CLOSED_REUSED_SOURCES_AND_I_STORAGE_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    for name in sources:
        if not name.startswith('results/'):
            command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    deployed={name:sha for name,sha in sources.items() if not name.startswith('results/')}
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {deployed!r}.items())
print('ACTUAL_REVIEWED_I_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    outcomes={}
    for dataset,updates in UPDATES.items():
        out=ROOT+'/'+dataset
        receipt=pf/('r201i_primary_margin_'+dataset+'_actual_session_20261006.json')
        assert not receipt.exists()
        argv=[PYTHON,'-u','launch_r201i_primary_margin.py','--dataset',dataset,
            '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
            'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        print('I_ORIGINAL_DATASET_CONTROLLER_STARTED',dataset,flush=True)
        for line in process.stdout:
            if not line.startswith('R201I_STREAM '):
                print(line,end='',flush=True)
                continue
            message=json.loads(line[len('R201I_STREAM '):])
            event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero']
                native=True
                (pf/('r201i_primary_margin_'+dataset+'_native_actual_20261006.json')).write_text(json.dumps(dict(
                    status='ACTUAL_TWO_I_NATIVE3_PRIMARY_MARGIN03_PASS_FORMAL_NEXT',observed_at=datetime.now().isoformat(timespec='seconds'),
                    dataset=dataset,remote_root=out,actual_native_updates=6,primary_margin=.3,auxiliary_soft_unchanged=True,
                    partial_loss_effective_zero=True,normal_full50_complete=False),indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_I_NATIVE6_PASS',dataset,flush=True)
            elif event=='NORMAL_READY':
                name=message['name']
                assert native and name.startswith(dataset+'_r201i_') and name not in archives
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n')
                process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                name=message['name']
                assert name in archives and name not in cleared
                cleared.add(name)
                print('I_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',dataset,name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==updates
                finished=True
        process.stdin.close()
        assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
        value=dict(status='ACTUAL_I_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().isoformat(timespec='seconds'),
            exit_code=0,dataset=dataset,remote_root=out,archives=archives,native_updates=6,successful_updates=updates,
            additional_epochs=100,sources_sha256=sources,limits='Fixed primary margin hypothesis; no +2 or missing/multiple-seed success assertion.')
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        outcomes[dataset]=value
        print('ACTUAL_I_TWO_FULL50_CLOSED',dataset,flush=True)
    result=dict(status='ACTUAL_I_THREE_NORMAL_DATASETS_SIX_FULL50_GT_RAW_COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),
        exit_code=0,datasets=outcomes,native_updates=18,successful_updates=19418,additional_epochs=300,
        sources_sha256=sources,missing_evaluated=False,limits='Existing hinge primary factor, three full normal datasets only; full goal requires metrics/fair controls/missing/seeds/costs and is not inferred from completion.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_I_THREE_NORMAL_CONTROLLER_CLOSED',flush=True)


if __name__=='__main__':main()
