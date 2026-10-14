"""Continue the pre-NN AW import failure using its four already restored inputs."""
from pathlib import Path
from datetime import datetime
import hashlib,json,os,shlex,subprocess,sys

P=Path('C:/Users/gb/projects/demo_dual_axis_20261002');PF=P/'results/preflight'
sys.path.insert(0,str(PF))
import r201aw_AV_frequency_outlets_flow_20261014 as original

load=original.load
sha=original.sha
save=original.save
remote=original.remote
REMOTE=original.REMOTE;PYTHON=original.PYTHON;ROOT=original.ROOT;TAG=original.TAG


def main():
    review=load(PF/'r201aw_AV_frequency_outlets_resume_source_review_20261014.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources=review['sources_sha256'];reuse=review['directly_reused_sources_sha256']
    assert all(sha(P/n)==h for n,h in (sources|reuse).items())
    failed=load(PF/'r201aw_original_CPU_import_failure_20261014.json')
    assert failed['original70746_exit_code']==1 and failed['probe_exit_code']==0 and failed['new_NN']==0
    observed=json.loads(failed['stdout'])
    assert observed['cwd']=='/home/gaob' and observed['evaluator_resolution']=='None' and not observed['actual_AW_root_exists']
    assert not original.LOCAL.exists() and not original.ARCHIVE.exists() and not (PF/(TAG+'_actual_session.json')).exists()
    ledger=load(Path('C:/Users/gb/.codex_tmp/r201l_uniform_k8_continuation_20261007.json'))
    assert not ledger['live_native_sessions'] and ledger['current_AW']['status']=='ORIGINAL70746_EXIT1_AT_CPU_WARM_BEFORE_NN'
    refs=original.restored_inputs();assert len(refs)==4
    assert all(Path(v['local']).stat().st_size==v['file']['bytes'] and sha(Path(v['local']))==v['file']['sha256'] for v in refs.values())
    protected=load(PF/'r201av_RGB_frequency_value_decay_20261014_actual_session.json')['protected_weights']
    expected_sources={n:h for n,h in reuse.items() if not n.startswith('results/')}
    # R1's 3.3GB admission included these already present 1.292GB inputs.
    remaining_required=3_300_000_000-sum(v['file']['bytes'] for v in refs.values())
    ready=remote(f'''import hashlib,json,shutil,subprocess
from pathlib import Path
r=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert all(Path(v['remote']).stat().st_size==v['file']['bytes'] and hashlib.sha256(Path(v['remote']).read_bytes()).hexdigest()==v['file']['sha256'] for v in {refs!r}.values())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {expected_sources!r}.items())
assert shutil.disk_usage(r).free>{remaining_required!r}
gpu={{g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in (2,3)}}
assert not any(gpu.values())
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,required_additional_free={remaining_required!r},four_existing_restores_exact=True,source_bytes_exact=True,protected_four_exact=True,compute_processes=gpu,physical_gpus=[2,3],temperature_power_read=False)))''')
    warm=remote(f'''import sys,importlib,json,torch
from pathlib import Path
sys.path.insert(0,{REMOTE!r})
modules=[importlib.import_module(n) for n in ('evaluate_r201aw_AV_frequency_outlets','launch_r201aw_AV_frequency_outlets')]
assert all(Path(m.__file__).parent==Path({REMOTE!r}) for m in modules)
import run_r201av_RGB_frequency_value_decay as recipe
assert recipe.REVISION.startswith('AV RGBNT100') and not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS',project_import_root=sys.path[0],torch_version=torch.__version__,CUDA_initialized=False,models=0,optimizer_updates=0,fix='Explicit project import directory, not working-directory fallback')))''')
    save(PF/'r201aw_actual_CPU_import_R2_20261014.json',warm)
    assert warm['status']=='PASS' and not warm['CUDA_initialized']
    state=dict(status='ACTUAL_AW_R2_PRE_NN_IMPORT_FIXED_STARTING',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        deploy_pid=os.getpid(),capacity=ready,admission=warm,restored_inputs=refs,sources_sha256=sources,
        reused_sources_sha256=reuse,protected_weights=protected,physical_gpus=[2,3],maximum_NN=2,
        optimizer_updates=0,new_weights=0,original_R1_native70746_exit_code=1,original_R1_NN_calls=0,
        assets_retransferred=0,neural_source_changes=0,estimated_wall_seconds=900,goal='ACTIVE_UNMET')
    save(PF/(TAG+'_actual_session.json'),state)
    print(json.dumps(dict(status=state['status'],physical_gpus=[2,3],assets_retransferred=0,optimizer_updates=0,estimate_seconds=900)),flush=True)
    argv=[PYTHON,'-u','launch_r201aw_AV_frequency_outlets.py','--training-root',original.TRAINING,'--output',ROOT+'/RGBNT100']
    code=f'''import subprocess
subprocess.run({argv!r},cwd={REMOTE!r},check=True)
'''
    completed=subprocess.run(['ssh',*original.OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',shlex.quote(PYTHON)+' -'],
        input=code,check=True,capture_output=True,text=True,encoding='utf-8')
    print(completed.stdout,end='',flush=True)
    original.collect()


if __name__=='__main__':main()
