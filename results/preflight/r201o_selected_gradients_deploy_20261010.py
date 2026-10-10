"""Bounded selected-checkpoint task gradients, after original normal and paper-six close."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201o_selected_task_gradients_20261010'


def main():
    pf=PROJECT/'results/preflight'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    proof=pf/'r201o_selected_task_gradients_actual_20261010.json'
    assert not proof.exists()
    normal=load(pf/'r201o_same_state_contribution_actual_session_20261010.json')
    assert normal['status']=='ACTUAL_O_TWO_WEAK_NORMAL_DATASETS_FOUR_FULL50_GT_RAW_COMPLETE' and normal['exit_code']==0
    paper=load(pf/'r201o_paper_missing6_actual_session_20261010.json')
    assert paper['status']=='ACTUAL_O_ALL_DECLARED_FIXED_BEST_PAPER6_GT_RAW_CPU_COMPLETE_K201_RETAINED'
    assert paper['new_full_conditions']==24 and paper['new_full_state_cases']==96 and not paper['new_49_evaluation']
    review=load(pf/'r201o_selected_task_gradients_source_review_20261010.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'probe_r201o_selected_task_gradients.py','results/preflight/r201o_selected_gradients_deploy_20261010.py','results/preflight/r201o_closed_gradient_summary_20261010.py'}
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    local=PROJECT/'results/r201o_selected_task_gradients_20261010'
    assert not local.exists()
    jobs=[]
    for dataset in ('MSVR310','RGBNT100'):
        for variant,gpu in (('frequency_shared',2),('axis_shared',3)):
            name=dataset+'_r201o_'+variant+'_s42'
            trained=load(PROJECT/'results/r201o_same_state_contribution_20261010'/dataset/'training'/name/'result.json')
            assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['amp_skipped_steps']==0
            jobs.append(dict(name=name,dataset=dataset,variant=variant,gpu=gpu,run=trained['arguments']['output'],
                selected_epoch=trained['best']['epoch']))
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});out=Path({ROOT!r});assert not out.exists()
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/') and n not in ('probe_mature_identity_gradient.py','list_retrieval_objective.py'))
out.mkdir();print('ACTUAL_CLOSED_NORMAL_PAPER6_DIAG_SOURCE_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    for name in ('probe_r201o_selected_task_gradients.py','probe_mature_identity_gradient.py','list_retrieval_objective.py'):
        command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert hashlib.sha256((root/'probe_r201o_selected_task_gradients.py').read_bytes()).hexdigest()=={sources['probe_r201o_selected_task_gradients.py']!r}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
print('ACTUAL_REVIEWED_SELECTED_GRADIENT_SOURCE_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    state=dict(status='ACTUAL_SELECTED_O_GRADIENT_DIAGNOSIS_STARTED',started=datetime.now().astimezone().isoformat(timespec='seconds'),
        controls=4,optimizer_updates=0,checkpoint_writes=0,new49=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    local.mkdir()
    # Existing GPU executor waits for our selected physical card before starting;
    # each lane runs its own two datasets serially, at most two diagnostic models.
    def lane(gpu):
        rows=[]
        for job in (j for j in jobs if j['gpu']==gpu):
            output=ROOT+'/'+job['name']+'.json'
            code=f'''from pathlib import Path
from gpu_thermal_execute import execute
root=Path({ROOT!r})
execute([{PYTHON!r},'-u','probe_r201o_selected_task_gradients.py','--run-dir',{job['run']!r},'--output',{output!r}],root,{job['name']!r},{gpu!r})
print('ACTUAL_SELECTED_GRADIENT_WORKER_COMPLETE')'''
            completed=subprocess.run(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
                'cd '+shlex.quote(REMOTE)+' && '+shlex.quote(PYTHON)+' -'],input=code,check=True,
                capture_output=True,text=True,encoding='utf-8')
            print(completed.stdout,end='',flush=True)
            code=f'''import hashlib,json
from pathlib import Path
p=Path({output!r});print(json.dumps(dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))'''
            info=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
            path=local/(job['name']+'.json');command(['scp',*OPTIONS,'2026:'+output,str(path)])
            assert path.stat().st_size==info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
            result=load(path)
            assert result['status']=='ACTUAL_O_SELECTED_TRAIN_ONLY_FULL_TASK_GRADIENT_DIAGNOSIS'
            assert result['dataset']==job['dataset'] and result['variant']==job['variant'] and result['selected_epoch']==job['selected_epoch']
            assert result['batches']==result['neural_forwards']==4 and result['optimizer_updates']==result['checkpoint_writes']==0
            assert result['parameters_unchanged'] and result['frozen_identity_unchanged']
            rows.append(dict(job=job,result=result,file=info,local=str(path)))
        return rows
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,gpu) for gpu in (2,3)]
        rows=[row for future in futures for row in future.result()]
    assert len(rows)==4
    for dataset in ('MSVR310','RGBNT100'):
        pair=[r['result'] for r in rows if r['job']['dataset']==dataset]
        assert len(pair)==2
        assert [(r['names'],r['training_labels']) for r in pair[0]['rows']]==[(r['names'],r['training_labels']) for r in pair[1]['rows']]
    state.update(status='ACTUAL_FOUR_SELECTED_O_TRAIN_ONLY_TASK_GRADIENT_DIAGNOSTICS_COMPLETE',
        finished=datetime.now().astimezone().isoformat(timespec='seconds'),controls=4,batches=16,neural_forwards=16,
        rows=rows,paired_training_names_labels=True,official_query_gallery_neural_uses=0,
        limits='Fixed selected weights and16 train-only local gradient observations; no optimizer/retraining/new retrieval score, not whole-training conflict or cause.')
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:state[k] for k in ('status','controls','batches','neural_forwards','optimizer_updates','checkpoint_writes')}),flush=True)
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201o_closed_gradient_summary_20261010.py')],check=True)


if __name__=='__main__':main()
