"""One reviewed finite cached MS metric comparison, with actual transport captures."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib.util
import json
import shlex
import shutil
import subprocess
import time

T = Path('C:/Users/gb/.codex_tmp')
spec = importlib.util.spec_from_file_location('transport', T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
P, load, OPTIONS, sha = prior.P, prior.load, prior.OPTIONS, prior.sha
PF = P/'results/preflight'
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT = REMOTE+'/runs/msvr_public_metric_weights_CPU_20261016'
INPUTS = REMOTE+'/runs/msvr_public_weight_inputs_20261016'
ARCHIVE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/msvr_public_metric_weights_CPU_20261016')


def capture(argv, label, text=None):
    out = T/('MSVR_public_weight_'+label+'_private_stdout_20261016.log')
    err = T/('MSVR_public_weight_'+label+'_private_stderr_20261016.log')
    receipt = T/('MSVR_public_weight_'+label+'_transport_actual_20261016.json')
    assert not out.exists() and not err.exists() and not receipt.exists()
    started = time.time()
    with out.open('w', encoding='utf-8') as stdout, err.open('w', encoding='utf-8') as stderr:
        child = subprocess.run(argv, input=text, text=True, stdout=stdout, stderr=stderr, timeout=900)
    result = dict(label=label, started_epoch=started, ended_epoch=time.time(), exit_code=child.returncode,
                  stdout=str(out), stderr=str(err))
    receipt.write_text(json.dumps(result, indent=2)+'\n', 'utf-8')
    child.check_returncode()
    return out.read_text('utf-8')


def remote(code, label):
    return json.loads(capture(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON)+' -'], label, code))


def main():
    proof_path = PF/'msvr_public_metric_weight_CPU_actual_20261016.json'
    assert not proof_path.exists()
    review = load(PF/'msvr_public_metric_weight_CPU_source_review_R2_20261016.json')
    assert review['verdict'] == 'PASS' and not review['blocking_findings']
    request = load(PF/'msvr_public_metric_weight_CPU_source_review_request_R2_20261016.json')
    seals = request['sources_sha256'] | request['directly_reused_sources_sha256']
    assert all(sha(P/name) == digest for name, digest in seals.items())
    plan_path = PF/'msvr_public_metric_weight_candidate_plan_20261016.json'
    plan = load(plan_path)
    assert plan['status'] == 'FINITE_MSVR_EXISTING_PUBLIC_METRIC_WEIGHT_PLAN_NOT_RUN'
    assert plan['public_metric_weights'] == [.125, .25, .5]
    assert sha(P/'diagnose_msvr_public_metric_weights_CPU.py') == plan['source_sha256']
    local = P/'results/msvr_public_metric_weights_CPU_20261016'
    assert not local.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage('D:/').free > 150_000_000
    inputs = {key: value for key, value in plan['records'].items() if key.startswith('AH_')}
    assert len(inputs) == 2
    for row in inputs.values():
        path = Path(row['local'])
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    strong = plan['records']['strong_DeMo']
    code = f'''import hashlib,json,shutil
from pathlib import Path
r=Path({REMOTE!r});inp=Path({INPUTS!r});out=Path({ROOT!r})
assert not inp.exists() and not out.exists() and shutil.disk_usage(r).free>300_000_000
p=Path({strong['remote']!r});assert p.stat().st_size=={strong['bytes']} and hashlib.sha256(p.read_bytes()).hexdigest()=={strong['sha256']!r}
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {request['directly_reused_sources_sha256']!r}.items())
inp.mkdir();print(json.dumps(dict(status='CPU_ONLY_SAVED_CACHE_INPUTS_ADMITTED',project_free=shutil.disk_usage(r).free,strong_original_exact=True,GPU_queries=0)))'''
    capacity = remote(code, '01_admit')
    for index, row in enumerate(inputs.values(), 2):
        capture(['scp', *OPTIONS, row['local'], '2026:'+row['remote']], f'{index:02d}_input')
    source_path = P/'diagnose_msvr_public_metric_weights_CPU.py'
    capture(['scp', *OPTIONS, str(source_path), '2026:'+REMOTE+'/'+source_path.name], '04_source')
    capture(['scp', *OPTIONS, str(plan_path), '2026:'+INPUTS+'/input_proof.json'], '05_proof')
    code = f'''import hashlib,json
from pathlib import Path
for row in {plan['records']!r}.values():
 p=Path(row['remote']);assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
assert hashlib.sha256(Path({REMOTE+'/'+source_path.name!r}).read_bytes()).hexdigest()=={sha(source_path)!r}
assert hashlib.sha256(Path({INPUTS+'/input_proof.json'!r}).read_bytes()).hexdigest()=={sha(plan_path)!r}
print(json.dumps(dict(status='REVIEWED_CPU_SOURCE_AND_THREE_INPUTS_EXACT')))'''
    binding = remote(code, '06_bind')
    started = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    argv = ['env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
        PYTHON, '-u', source_path.name, '--input-proof', INPUTS+'/input_proof.json', '--output', ROOT]
    result_stdout = capture(['ssh', *OPTIONS, '2026', 'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)], '07_compute')
    result_line = json.loads(result_stdout.strip().splitlines()[-1])
    assert result_line['status'] == 'ACTUAL_MSVR_FINITE_CACHED_PUBLIC_WEIGHT_FULL_QG_CPU_COMPLETE'
    code = f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({ROOT!r});result=json.loads((r/'result.json').read_text());assert result['status']=='ACTUAL_MSVR_FINITE_CACHED_PUBLIC_WEIGHT_FULL_QG_CPU_COMPLETE'
assert result['new_neural_calls']==result['optimizer_updates']==result['checkpoint_writes']==0 and result['cached_full_QG_cases']==9 and not result['CUDA_initialized']
text={{str(p.relative_to(r)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in r.iterdir() if p.suffix in ('.json','.csv')}}
assert len(text)==19
pkg=r/'text_intake.tar.gz'
with tarfile.open(pkg,'w:gz') as tar:
 for n in text:tar.add(r/n,arcname=n)
raw=result['raw'];assert len(raw)==3
print(json.dumps(dict(result=result,text=text,raw=raw,package=dict(path=str(pkg),bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()))))'''
    terminal = remote(code, '08_manifest')
    required = sum(row['bytes'] for row in terminal['raw'].values())
    assert shutil.disk_usage('D:/').free > required+100_000_000
    local.mkdir()
    package = local/'text_intake.tar.gz'
    capture(['scp', *OPTIONS, '2026:'+terminal['package']['path'], str(package)], '09_text')
    assert package.stat().st_size == terminal['package']['bytes'] and sha(package) == terminal['package']['sha256']
    import tarfile
    with tarfile.open(package) as tar:
        assert set(tar.getnames()) == set(terminal['text'])
        for name, row in terminal['text'].items():
            path = local/name
            assert path.resolve().is_relative_to(local.resolve())
            path.write_bytes(tar.extractfile(name).read())
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    ARCHIVE.mkdir()
    archived = {}
    for index, (label, row) in enumerate(terminal['raw'].items(), 10):
        path = ARCHIVE/(label+'_cached_weight_distances.npz')
        capture(['scp', *OPTIONS, '2026:'+row['path'], str(path)], f'{index}_raw')
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        archived[label] = dict(local=str(path), remote=row['path'], bytes=row['bytes'], sha256=row['sha256'])
    owned = {row['remote']: dict(bytes=row['bytes'], sha256=row['sha256']) for row in archived.values()}
    owned.update({row['remote']: dict(bytes=row['bytes'], sha256=row['sha256']) for row in inputs.values()})
    owned[terminal['package']['path']] = dict(bytes=terminal['package']['bytes'], sha256=terminal['package']['sha256'])
    code = f'''import hashlib,json
from pathlib import Path
owned={owned!r};roots=[Path({ROOT!r}).resolve(),Path({INPUTS!r}).resolve()]
for n,row in owned.items():
 p=Path(n);assert any(p.resolve().is_relative_to(r) for r in roots) and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
strong=Path({strong['remote']!r});assert hashlib.sha256(strong.read_bytes()).hexdigest()=={strong['sha256']!r}
for n in owned:Path(n).unlink()
assert hashlib.sha256(strong.read_bytes()).hexdigest()=={strong['sha256']!r}
print(json.dumps(dict(duplicate_files_cleared=len(owned),duplicate_bytes_cleared=sum(row['bytes'] for row in owned.values()),strong_original_preserved=True)))'''
    cleared = remote(code, '13_clear_duplicates')
    package.unlink()
    assert all(sha(P/name) == digest for name, digest in seals.items())
    for row in inputs.values():
        assert Path(row['local']).stat().st_size == row['bytes'] and sha(Path(row['local'])) == row['sha256']
    proof = dict(status='ACTUAL_MSVR_FINITE_PUBLIC_WEIGHT_CPU_NINE_FULL_QG_ARCHIVED_COMPLETE', started_at=started,
        finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'), source_review='PASS_SOURCE_ONLY',
        capacity=capacity, binding=binding, local=str(local), remote=ROOT, terminal=terminal, archived=archived,
        cleared=cleared, D_free=shutil.disk_usage('D:/').free, remote_compute_exit_code=0,
        actual_transport_captures=13, source_seals_unchanged=len(seals), new_NN=0, optimizer_updates=0,
        training_progress_queries=0, missing_evaluated=False, new49=False, goal='ACTIVE_UNMET',
        limits='Finite cached-feature weight calibration, no model or new training. Nine cases use same checkpoints/GT and equal weight budget. No deployed result promotion without actual descriptor-path equivalence. Exact copies retired only after all three new raw files and nineteen texts were locally size/SHA verified; original caches and checkpoint files untouched.')
    proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps({key: proof[key] for key in ('status', 'finished_at', 'new_NN', 'D_free')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
