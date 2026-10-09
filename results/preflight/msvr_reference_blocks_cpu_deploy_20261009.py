"""Run one fixed-cache CPU block diagnosis; no current training observation or NN."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/msvr_reference_blocks_p0_20261009'


def main():
    pf=PROJECT/'results/preflight'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    proof=pf/'msvr_reference_blocks_cpu_actual_20261009.json';assert not proof.exists()
    local=PROJECT/'results/msvr_reference_blocks_p0_20261009';assert not local.exists()
    review=load(pf/'msvr_reference_blocks_cpu_source_review_20261009.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'diagnose_msvr_baseline_blocks.py','results/preflight/msvr_reference_blocks_cpu_deploy_20261009.py'}
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    inventory_path=pf/'r201m_MSVR_shared_reference_cpu_input_inventory_20261009.json'
    inventory=load(inventory_path)
    assert inventory['status']=='ACTUAL_CLOSED_TWO_MSVR_BASELINE_CPU_INPUTS_READ_ONLY_INVENTORY'
    assert all(r['counts']==[1032,591,1055] and r['files']['best_official_arrays.npz']['exists'] for r in inventory['records'].values())
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
for entry in {inventory['records']!r}.values():
 for item in entry['files'].values():
  path=Path(item['path']);assert item['exists'] and path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
print('ACTUAL_CLOSED_MSVR_CPU_CACHE_AND_REUSED_SOURCE_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='',flush=True)
    remote_source=REMOTE+'/diagnose_msvr_baseline_blocks.py'
    remote_inputs=REMOTE+'/runs/msvr_reference_blocks_inputs_20261009.json'
    command(['scp',*OPTIONS,str(PROJECT/'diagnose_msvr_baseline_blocks.py'),'2026:'+remote_source])
    command(['scp',*OPTIONS,str(inventory_path),'2026:'+remote_inputs])
    code=f'''import hashlib
from pathlib import Path
assert hashlib.sha256(Path({remote_source!r}).read_bytes()).hexdigest()=={sources['diagnose_msvr_baseline_blocks.py']!r}
assert hashlib.sha256(Path({remote_inputs!r}).read_bytes()).hexdigest()=={hashlib.sha256(inventory_path.read_bytes()).hexdigest()!r}
print('ACTUAL_REVIEWED_MSVR_CPU_SOURCE_AND_INPUT_BINDING_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='',flush=True)
    state=dict(status='ACTUAL_CLOSED_MSVR_CPU_BLOCK_DIAGNOSIS_INVOKED_ONCE',started_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        source_review='PASS_SOURCE_ONLY',cpu_threads=4,new_neural_calls=0,new_optimizer_updates=0,new49=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    argv=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
        PYTHON,'-u','diagnose_msvr_baseline_blocks.py','--baseline-root',REMOTE+'/runs/full_official_baselines_20261004',
        '--output',ROOT,'--input-proof',remote_inputs]
    print(command(['ssh',*OPTIONS,'2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)]),end='',flush=True)
    code=f'''import hashlib,json
from pathlib import Path
root=Path({ROOT!r});files={{str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.iterdir() if p.is_file()}}
assert len(files)==9 and all(Path(n).suffix in ('.json','.csv') for n in files)
print(json.dumps(files))'''
    manifest=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    command(['scp',*OPTIONS,'-r','2026:'+ROOT,str(local.parent)])
    assert set(f.name for f in local.iterdir())==set(manifest)
    assert all((local/n).stat().st_size==r['bytes'] and hashlib.sha256((local/n).read_bytes()).hexdigest()==r['sha256'] for n,r in manifest.items())
    result=load(local/'result.json')
    assert result['status']=='COMPLETE_CPU_MSVR310_SELECTED_BASELINE_BLOCK_DIAGNOSIS'
    assert result['neural_inference_calls']==result['optimizer_updates']==0 and result['full_split_counts']==dict(train=1032,query=591,gallery=1055)
    state.update(status='ACTUAL_CPU_MSVR_SELECTED_BASELINE_FOUR_BLOCKS_EXIT0_TEXT_VERIFIED',finished_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        files=manifest,remote=ROOT,local=str(local),result=result,exit_code=0,
        scope='One old fixed original/shared teacher pair, CPU cached-feature scoring only; no new neural inference, missing49, weights, raw distance files, current M progress read or recipe change.')
    proof.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=state['status'],new_neural_calls=0,new_optimizer_updates=0,
        normal={k:result['measurements']['shared_private_only'][k] for k in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')})),flush=True)


if __name__=='__main__':main()
