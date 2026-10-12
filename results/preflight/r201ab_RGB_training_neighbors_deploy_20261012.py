"""One diagnostic on freed GPU2 after the original X full-normal owner closes."""
from pathlib import Path
from datetime import datetime
import hashlib,json,shlex,shutil,subprocess,sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201ab_RGB_training_neighbors_20261012';ROOT=REMOTE+'/runs/'+TAG
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    closed=load(pf/'r201x_original_owner_normal_closed_actual_20261012.json')
    assert closed['exit_code']==0 and closed['terminal_consumed'] and closed['normal_GT_raw_CPU_complete'] and closed['formal_updates']==13126
    review=load(pf/'r201ab_RGB_training_neighbors_source_review_20261012.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources=review['sources_sha256'];reuse=review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    local=PROJECT/'results'/TAG;proof=pf/'r201ab_RGB_training_neighbors_actual_20261012.json'
    assert not proof.exists() and not local.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>1024**3
    code=f'''from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists() and shutil.disk_usage(root).free>450_000_000
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert not Path('/proc/4176751').exists() and not Path('/proc/4176750').exists()
devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
uuid=next(row.split(',')[1].strip() for row in devices.splitlines() if int(row.split(',')[0])==2)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
assert uuid not in apps.splitlines()
print('ACTUAL_BANK_GPU2_FREE_AND_REUSED_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    core='diagnose_rgbnt100_training_neighbors.py'
    command(['scp',*OPTIONS,str(PROJECT/core),'2026:'+REMOTE+'/'+core])
    argv=[PYTHON,'-u',core,'--anchor-run-dir',REMOTE+'/runs/full_official_baselines_20261004/training/RGBNT100_demo_s42','--output',ROOT]
    ssh=['ssh',*OPTIONS,'2026','cd '+shlex.quote(REMOTE)+' && CUDA_VISIBLE_DEVICES=2 '+shlex.join(argv)]
    process=subprocess.run(ssh,capture_output=True,text=True,encoding='utf-8',timeout=300)
    primary=dict(at=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr)
    (pf/'r201ab_RGB_training_neighbors_primary_actual_20261012.json').write_text(json.dumps(primary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert process.returncode==0
    messages=[json.loads(line[len('TRAIN_NEIGHBOR_RESULT '):]) for line in process.stdout.splitlines() if line.startswith('TRAIN_NEIGHBOR_RESULT ')]
    assert len(messages)==1
    result=messages[0];assert result['forward_batches']==136 and result['record_forwards']==8675 and result['optimizer_updates']==0
    assert set(result['files'])=={'training_features.npz','result.json','official_split_manifest.json'}
    archived={}
    for name,info in result['files'].items():
        archived[name]=str(copy_verified(ROOT+'/'+name,info,Path(ROOT),ARCHIVE))
    local.mkdir(parents=True)
    for name in ('result.json','official_split_manifest.json'):
        (local/name).write_bytes(Path(archived[name]).read_bytes())
    raw=Path(archived['training_features.npz']);info=result['files']['training_features.npz']
    assert raw.stat().st_size==info['bytes'] and hashlib.sha256(raw.read_bytes()).hexdigest()==info['sha256']
    cleanup=f'''from pathlib import Path
import hashlib
root=Path({ROOT!r});raw=root/'training_features.npz';assert raw.resolve().is_relative_to(root.resolve())
assert raw.stat().st_size=={info['bytes']!r} and hashlib.sha256(raw.read_bytes()).hexdigest()=={info['sha256']!r}
raw.unlink();print('ACTUAL_ONE_REMOTE_TRAIN_BANK_DUPLICATE_REMOVED')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=cleanup),flush=True)
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_rgbnt100_training_neighbor_coverage.py')],check=True)
    record=dict(status='ACTUAL_RGBNT100_TRAIN_ONLY_BANK_AND_CLOSED_X_COVERAGE_COMPLETE',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        server='2026',gpu=2,original_X_closed=True,AA_original_65092_unchanged=True,record_forwards=8675,
        forward_batches=136,query_gallery_forwards=0,optimizer_updates=0,new_weights=0,raw_archive=archived,
        raw_file=result['files']['training_features.npz'],remote_unique_raw_lost=0,remote_duplicate_raw_deleted=True,
        sources_sha256=sources,limits='Training-only diagnostic, not a new retrieval result or a trained hard-sampling method; no201 NN, no missing专项, no49.')
    proof.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(record['status'],flush=True)


if __name__=='__main__':main()
