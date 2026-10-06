"""Continue the reviewed seed queue after the actual post-intake capacity failure."""
from datetime import datetime
import ast,hashlib,importlib.util,json,os,subprocess
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')


def main():
    pf=PROJECT/'results/preflight'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    review=load(pf/'r201i_expert_seeds_prelaunch_recovery_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in review['sources_sha256'].items())
    assert not subprocess.run(['powershell','-NoProfile','-Command','Get-Process -Id 13056 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id'],capture_output=True,text=True).stdout.strip()
    previous=load(pf/'r201i_expert_seeds_actual_session_20261006.json')
    assert previous['status']=='WAIT_ORIGINAL_I_THREE_NORMAL_COMPLETE' and previous['receiver_pid']==13056 and previous['new_training_updates']==previous['new_neural_calls']==0
    stderr=Path('C:/Users/gb/.codex_tmp/r201i_expert_seeds_20261006.stderr.log').read_text(encoding='utf-8')
    assert 'line 59, in main' in stderr and 'CalledProcessError' in stderr
    capacity=load(pf/'closed_legacy_M1_weight_capacity_retirement_actual_20261007.json')
    assert capacity['actual']['removed_files']==4 and capacity['actual']['remote_free_after']>3_500_000_000
    original=load(pf/'r201i_primary_margin_actual_session_20261006.json')
    assert original['exit_code']==0 and original['additional_epochs']==300 and original['successful_updates']==19418
    assert load(pf/'r201i_RGBNT100_normal_terminal_text_intake_20261006.json')['status']=='ACTUAL_I_DATASET_TWO_FULL50_NORMAL_GT_TEXT_VERIFIED'
    assert all(load(PROJECT/'results/r201i_primary_margin_20261006'/d/'normal_analysis/result.json')['status']=='ACTUAL_I_DATASET_TWO_FULL50_NORMAL_CPU_READOUT' for d in ('RGBNT201','MSVR310','RGBNT100'))
    failure_path=pf/'r201i_expert_seeds_capacity_failure_actual_20261007.json';assert not failure_path.exists()
    failure_path.write_text(json.dumps(dict(status='ACTUAL_PRELAUNCH_CAPACITY_FAILURE_AFTER_ORIGINAL_RGBNT100_CPU_COMPLETE',recorded_at=datetime.now().isoformat(timespec='seconds'),original_receiver_pid=13056,original_state=previous,stderr=stderr,observed_capacity_before=capacity['actual']['remote_free_before'],required_capacity=3500000000,new_neural_calls=0,new_optimizer_updates=0),indent=2)+'\n',encoding='utf-8')
    old_path=pf/'r201i_expert_seeds_deploy_20261006.py'
    spec=importlib.util.spec_from_file_location('reviewed_original_seed_queue',old_path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    original_review=load(pf/'r201i_expert_seeds_source_review_20261006.json')
    sources,reuse=original_review['sources_sha256'],original_review['directly_reused_sources_sha256']
    assert original_review['status']=='PASS' and not original_review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    assert not old.LOCAL.exists() and not old.ARCHIVE.exists()
    tree=ast.parse(old_path.read_text(encoding='utf-8'));body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main').body
    positions=[i for i,n in enumerate(body) if isinstance(n,ast.Assert) and ast.unparse(n.test)=='shutil.disk_usage(ARCHIVE.parent).free > 500000000']
    assert len(positions)==1
    state=dict(previous,status='ACTUAL_POST_INTAKE_CAPACITY_RECOVERY_REVIEWED_PRELAUNCH',receiver_pid=os.getpid(),resumed_at=datetime.now().isoformat(timespec='seconds'),original_receiver_pid=13056,original_failure=failure_path.relative_to(PROJECT).as_posix())
    proof=pf/'r201i_expert_seeds_actual_session_20261006.json';proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    old.__dict__.update(pf=pf,load=load,proof=proof,review=original_review,sources=sources,reuse=reuse,state=state)
    continuation=ast.Module(body=body[positions[0]:],type_ignores=[])
    print(json.dumps(dict(status=state['status'],receiver_pid=os.getpid(),normal_CPU_replayed=False,original_training_sources_unchanged=True)),flush=True)
    exec(compile(continuation,str(old_path),mode='exec'),old.__dict__)


if __name__=='__main__':main()
