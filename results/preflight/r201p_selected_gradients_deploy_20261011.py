"""Selected P task gradients on additionally authorized GPU0/1; original normal stays GPU2/3."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib,json,shlex,subprocess,sys
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('MSVR310','RGBNT100'),required=True)
    dataset=parser.parse_args().dataset;pf=PROJECT/'results/preflight'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    proof=pf/('r201p_'+dataset+'_selected_gradients_actual_20261011.json');assert not proof.exists()
    normal=load(pf/('r201p_auxiliary_balance_'+dataset+'_actual_session_20261010.json'))
    assert normal['status']=='ACTUAL_P_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and normal['exit_code']==0
    intake=load(pf/('r201p_'+dataset+'_normal_terminal_text_intake_20261010.json'))
    assert intake['status']=='ACTUAL_P_DATASET_TWO_FULL50_NORMAL_GT_TEXT_VERIFIED' and intake['dataset']==dataset
    normal_root=PROJECT/'results/r201p_auxiliary_balance_20261010'/dataset
    assert load(normal_root/'normal_analysis/result.json')['status']=='ACTUAL_P_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
    review=load(pf/'r201p_selected_diag_gpu01_source_review_20261011.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'probe_r201p_selected_task_gradients.py','gpu_01_diagnostic_execute.py','results/preflight/r201p_selected_gradients_deploy_20261011.py','results/preflight/r201p_closed_gradient_summary_20261011.py'}
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    local=PROJECT/'results/r201p_selected_task_gradients_20261011'/dataset;assert not local.exists()
    remote_root=REMOTE+'/runs/r201p_selected_task_gradients_20261011/'+dataset
    jobs=[]
    for variant,gpu in (('frequency_shared',0),('axis_shared',1)):
        name=dataset+'_r201p_'+variant+'_s42';run=normal_root/'training'/name;trained=load(run/'result.json')
        assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['amp_skipped_steps']==0
        assert load(run/'normal_cpu_audit.json')['status']=='PASS'
        jobs.append(dict(name=name,dataset=dataset,variant=variant,gpu=gpu,run=trained['arguments']['output'],selected_epoch=trained['best']['epoch']))
    deployed=('probe_r201p_selected_task_gradients.py','gpu_01_diagnostic_execute.py','probe_mature_identity_gradient.py','list_retrieval_objective.py')
    code=f"""import hashlib
from pathlib import Path
root=Path({REMOTE!r});out=Path({remote_root!r});assert not out.exists()
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/') and n not in {deployed!r})
out.mkdir(parents=True);print('ACTUAL_P_CLOSED_DATASET_DIAG_SOURCE_READY')"""
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    for name in deployed:command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    code=f"""import hashlib
from pathlib import Path
root=Path({REMOTE!r});sources={sources!r};reuse={reuse!r}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items() if not n.startswith('results/'))
print('ACTUAL_P_REVIEWED_GPU01_DIAG_SOURCES_EXACT')"""
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    state=dict(status='ACTUAL_SELECTED_P_DATASET_GPU01_GRADIENT_DIAGNOSIS_STARTED',dataset=dataset,started=datetime.now().astimezone().isoformat(timespec='seconds'),receiver_pid=__import__('os').getpid(),controls=2,physical_gpus=[0,1],optimizer_updates=0,checkpoint_writes=0,new49=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');local.mkdir(parents=True)
    def worker(job):
        output=remote_root+'/'+job['name']+'.json'
        code=f"""from pathlib import Path
from gpu_01_diagnostic_execute import execute
root=Path({remote_root!r})
execute([{PYTHON!r},'-u','probe_r201p_selected_task_gradients.py','--run-dir',{job['run']!r},'--output',{output!r}],root,{job['name']!r},{job['gpu']!r})
print('ACTUAL_P_SELECTED_GRADIENT_WORKER_COMPLETE')"""
        finished=subprocess.run(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.quote(PYTHON)+' -'],input=code,check=True,capture_output=True,text=True,encoding='utf-8')
        print(finished.stdout,end='',flush=True)
        code=f"""import hashlib,json
from pathlib import Path
p=Path({output!r});print(json.dumps(dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))"""
        info=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code));path=local/(job['name']+'.json')
        command(['scp',*OPTIONS,'2026:'+output,str(path)])
        assert path.stat().st_size==info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
        result=load(path)
        assert result['status']=='ACTUAL_P_SELECTED_TRAIN_ONLY_FULL_TASK_GRADIENT_DIAGNOSIS'
        assert result['dataset']==dataset and result['variant']==job['variant'] and result['selected_epoch']==job['selected_epoch']
        assert result['batches']==result['neural_forwards']==4 and result['optimizer_updates']==result['checkpoint_writes']==0
        assert result['parameters_unchanged'] and result['frozen_identity_unchanged']
        return dict(job=job,result=result,file=info,local=str(path))
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(worker,jobs))
    assert len(rows)==2
    pair=[r['result'] for r in rows]
    assert [(r['names'],r['training_labels']) for r in pair[0]['rows']]==[(r['names'],r['training_labels']) for r in pair[1]['rows']]
    state.update(status='ACTUAL_TWO_SELECTED_P_TRAIN_ONLY_TASK_GRADIENT_DIAGNOSTICS_COMPLETE',finished=datetime.now().astimezone().isoformat(timespec='seconds'),batches=8,neural_forwards=8,rows=rows,paired_training_names_labels=True,official_query_gallery_neural_uses=0,limits='Two fixed selected P weights, 8 train-only full-primary gradient observations on GPU0/1; no optimizer, retraining, new retrieval score or checkpoint writes. GPU2/3 original training untouched. Four observations per control, not sustained/global conflict or causality; temporary BN statistics discarded.')
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:state[k] for k in ('status','dataset','controls','batches','neural_forwards','optimizer_updates','checkpoint_writes')}),flush=True)
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201p_closed_gradient_summary_20261011.py'),'--dataset',dataset],check=True)

if __name__=='__main__':main()
