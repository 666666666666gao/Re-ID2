"""Admit only the reviewed finite vehicle baselines after F128 is fully audited."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import tarfile
import time

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF = P/'results/preflight'
T = Path('C:/Users/gb/.codex_tmp')
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT = REMOTE+'/runs/vehicle_atmoe_head_baselines_20261016'
SOURCES = ('vehicle_atmoe_head_configuration.py', 'run_vehicle_atmoe_head_baseline.py',
           'check_vehicle_atmoe_head_default_parity.py', 'launch_vehicle_atmoe_head_baselines.py')
spec = importlib.util.spec_from_file_location('transport', T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
load, sha, OPTIONS = prior.load, prior.sha, prior.OPTIONS
CAPTURES = []


def command(argv, *, input=None):
    number = len(CAPTURES)+1
    out = T/f'vehicle_atmoe_head_deploy_R3_{number:02d}_private_stdout_20261016.log'
    err = T/f'vehicle_atmoe_head_deploy_R3_{number:02d}_private_stderr_20261016.log'
    capture = T/f'vehicle_atmoe_head_deploy_R3_{number:02d}_transport_actual_20261016.json'
    assert not out.exists() and not err.exists() and not capture.exists()
    start = time.time()
    capture.write_text(json.dumps(dict(status='ATTEMPT_STARTED', number=number, started=start,
        stdout=str(out), stderr=str(err)), indent=2)+'\n', 'utf-8')
    with out.open('w', encoding='utf-8') as stdout, err.open('w', encoding='utf-8') as stderr:
        child = subprocess.run(argv, input=input, text=True, encoding='utf-8', stdout=stdout, stderr=stderr, timeout=180)
    record = dict(status='RETURNED', number=number, started=start, finished=time.time(), exit_code=child.returncode, stdout=str(out), stderr=str(err))
    capture.write_text(json.dumps(record, indent=2)+'\n', 'utf-8')
    CAPTURES.append(record)
    child.check_returncode()
    return out.read_text('utf-8')


def main():
    output = PF/'vehicle_atmoe_head_actual_session_20261016.json'
    assert not output.exists()
    gate = load(PF/'vehicle_atmoe_head_deployment_source_review_20261016.json')
    assert gate['status'] == 'FINAL' and gate['verdict'] == 'PASS' and not gate['blocking_findings']
    assert gate['reviewed_revision'] == 'R3'
    audited = load(PF/'rgbnt100_F_rank_result_integrity_audit_20261016.json')
    assert audited['status'] == 'FINAL' and audited['verdict'] in ('PASS', 'WARN') and not audited['blocking_findings']
    actual = load(PF/'rgbnt100_F_rank_complete_intake_20261016.json')
    assert actual['status'] == 'ACTUAL_F_RANK128_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    request = load(PF/'rgbnt100_F_rank_result_integrity_request_20261016.json')
    assert audited['request'] == str(PF/'rgbnt100_F_rank_result_integrity_request_20261016.json')
    for row in request['inputs']:
        file = Path(row['path'])
        assert file.stat().st_size == row['bytes'] and sha(file) == row['sha256']
    source_review = load(PF/'vehicle_atmoe_head_controller_source_review_20261016.json')
    assert source_review['status'] == 'FINAL' and source_review['verdict'] == 'SOURCE_ONLY_PASS'
    assert not source_review['blocking_issues']
    sources = {name: sha(P/name) for name in SOURCES}
    inherited = load(PF/'rgbnt100_F_rank_postprocess_source_review_request_20261016.json')
    seals = inherited['sources_sha256'] | inherited['directly_reused_sources_sha256']
    assert len(seals) == 376 and all(sha(P/name) == value for name, value in seals.items())
    runtime_seals = {name: value for name, value in seals.items() if Path(name).suffix in ('.py', '.yaml', '.yml')}
    runtime_seals['run_full_official_experiment.py'] = sha(P/'run_full_official_experiment.py')
    protected = load(PF/'rgbnt100_F_rank_actual_session_20261016.json')['protected_weights']
    admission_code = f'''import hashlib,json,shutil,subprocess
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});assert not r.exists()
seals={runtime_seals!r};protected={protected!r}
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in seals.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in protected.items())
resources={{}}
for g in (2,3):
 row=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
 values=[int(x.strip()) for x in row.split(',')];assert values[0]==g and values[1]<500
 processes=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip();assert not processes
 resources[str(g)]=dict(memory_used_mib=values[1],memory_total_mib=values[2],utilization=values[3],compute_processes=processes)
free=shutil.disk_usage(p).free;assert free>1800000000
r.mkdir();print(json.dumps(dict(project_free=free,selected_GPU_resources=resources,runtime_sources_exact=len(seals),protected_four_exact=True,temperature_power_read=False)))'''
    capacity = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=admission_code))
    files = list(SOURCES)+['results/preflight/rgbnt100_F_rank_complete_intake_20261016.json',
        'results/preflight/rgbnt100_F_rank_result_integrity_audit_20261016.json']
    package = T/'vehicle_atmoe_head_admission_exact_20261016.tar.gz'
    assert not package.exists()
    manifest = {}
    with tarfile.open(package, 'w:gz') as tar:
        for name in files:
            data = (P/name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            manifest[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    destination = ROOT+'/admission_sources.tar.gz'
    command(['scp', *OPTIONS, str(package), '2026:'+destination])
    launch_code = f'''import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});pkg=r/'admission_sources.tar.gz';manifest={manifest!r}
with tarfile.open(pkg) as tar:
 assert set(tar.getnames())==set(manifest)
 for n in manifest:
  path=p/n;assert path.resolve().is_relative_to(p.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tar.extractfile(n).read())
assert all((p/n).stat().st_size==v['bytes'] and hashlib.sha256((p/n).read_bytes()).hexdigest()==v['sha256'] for n,v in manifest.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {runtime_seals!r}.items())
argv=[{PYTHON!r},'-u','launch_vehicle_atmoe_head_baselines.py','--previous-intake',str(p/'results/preflight/rgbnt100_F_rank_complete_intake_20261016.json'),'--previous-audit',str(p/'results/preflight/rgbnt100_F_rank_result_integrity_audit_20261016.json'),'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--output',str(r/'experiments')]
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
with (r/'controller.log').open('x') as log:child=subprocess.Popen(argv,cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
owner=dict(pid=child.pid,started=time.time(),argv=argv);(r/'remote_owner.json').write_text(json.dumps(owner,indent=2)+'\\n')
pkg.unlink();print(json.dumps(owner))'''
    owner = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=launch_code))
    package.unlink()
    result = dict(status='ACTUAL_VEHICLE_HEAD_CONTROLLER_STARTED_NATIVE_NOT_YET_CONFIRMED',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        remote_root=ROOT, remote_owner=owner, capacity=capacity, sources_sha256=sources,
        protected_weights=protected, actual_transport_captures=CAPTURES,
        F_previous_full_intake_and_fresh_audit_final=True, F_inputs_revalidated=len(request['inputs']),
        local376_sources_exact=True, remote_runtime_source_count=len(runtime_seals),
        first_admission_failed_before_GPU_or_NN=True,
        environment='Warm2026 existing tri_reid conda/current completed F native6/full13126; no installation or environment change',
        physical_gpus=[2, 3], maximum_NN=2, native_updates_confirmed=0, formal_updates_confirmed=0,
        first_native_node_seconds=240,
        historical_full_training_seconds=dict(RGBNT100=9685.604719638824, MSVR310=1579.3985397815704),
        completion_estimate_multiplier=1.2,
        missing_evaluated=False, new49=False, goal='ACTIVE_UNMET')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps(dict(status=result['status'], remote_owner=owner['pid'], new_source_count=len(sources)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
