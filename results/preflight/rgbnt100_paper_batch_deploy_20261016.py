"""Deploy finite reviewed B128/K16 RGBNT100 baselines on physicalGPU2/3."""
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
ROOT = REMOTE+'/runs/rgbnt100_paper_batch_baselines_20261016'
SOURCES = ('run_rgbnt100_paper_batch_baseline.py', 'audit_rgbnt100_paper_batch_normal.py',
           'launch_rgbnt100_paper_batch_baselines.py')
spec = importlib.util.spec_from_file_location('transport', T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
CAPTURES = []


def command(argv, *, input=None):
    prefix = T/f'rgbnt100_paper_batch_deploy_{len(CAPTURES)+1:02d}_20261016'
    out, err, receipt = [prefix.with_suffix(suffix) for suffix in ('.stdout.log', '.stderr.log', '.json')]
    assert not any(file.exists() for file in (out, err, receipt))
    started = time.time()
    receipt.write_text(json.dumps(dict(status='ATTEMPT_STARTED', started=started, stdout=str(out), stderr=str(err)), indent=2)+'\n', 'utf-8')
    with out.open('x', encoding='utf-8') as stdout, err.open('x', encoding='utf-8') as stderr:
        child = subprocess.run(argv, input=input, text=True, encoding='utf-8', stdout=stdout, stderr=stderr, timeout=180)
    capture = dict(status='RETURNED', started=started, finished=time.time(), exit_code=child.returncode, stdout=str(out), stderr=str(err))
    receipt.write_text(json.dumps(capture, indent=2)+'\n', 'utf-8')
    CAPTURES.append(capture)
    child.check_returncode()
    return out.read_text('utf-8')


def main():
    output = PF/'rgbnt100_paper_batch_actual_session_20261016.json'
    assert not output.exists()
    request_path = PF/'rgbnt100_paper_batch_source_request_20261016.json'
    request = prior.load(request_path)
    review = prior.load(PF/'rgbnt100_paper_batch_source_review_20261016.json')
    assert review['status'] == 'FINAL' and review['verdict'] == 'PASS' and not review['blocking_findings']
    assert review['request'] == str(request_path)
    for row in request['sources']:
        file = Path(row['path'])
        assert file.stat().st_size == row['bytes'] and prior.sha(file) == row['sha256']
    audit = prior.load(PF/'vehicle_atmoe_head_RGBNT100_result_integrity_audit_20261016.json')
    assert audit['status'] == 'FINAL' and not audit['blocking_findings']
    inherited = prior.load(PF/'rgbnt100_F_rank_postprocess_source_review_request_20261016.json')
    old = inherited['sources_sha256'] | inherited['directly_reused_sources_sha256']
    assert len(old) == 376 and all(prior.sha(P/name) == value for name, value in old.items())
    runtime = {name: value for name, value in old.items() if Path(name).suffix in ('.py', '.yaml', '.yml')}
    for name in ('run_full_official_experiment.py', 'vehicle_atmoe_head_configuration.py'):
        runtime[name] = prior.sha(P/name)
    original = prior.load(PF/'vehicle_atmoe_head_dynamic_amp_actual_session_20261016.json')
    protected = original['protected_weights']
    admission = f'''import hashlib,json,shutil,subprocess
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});assert not r.exists()
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {runtime!r}.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
rows=[]
for gpu in (2,3):
 values=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
 assert values[0]==gpu and values[1]<500 and (values[2]-values[1])*1048576>2*12174917120
 apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip();assert not apps
 rows.append(dict(gpu=gpu,memory_used_mib=values[1],memory_total_mib=values[2],utilization=values[3],compute_processes=apps))
free=shutil.disk_usage(p).free;assert free>1800000000
r.mkdir();print(json.dumps(dict(project_free=free,selected_GPUs=rows,unchanged_runtime_sources=len({runtime!r}),protected_four_exact=True,previous_candidate_audit_required=True,temperature_power_read=False)))'''
    capacity = json.loads(command(['ssh', *prior.OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=admission))
    package = T/'rgbnt100_paper_batch_sources_20261016.tar.gz'
    assert not package.exists()
    manifest = {}
    stage={name:P/name for name in SOURCES}
    stage['prerequisites/previous_intake.json']=PF/'vehicle_atmoe_head_dynamic_amp_RGBNT100_intake_actual_20261016.json'
    stage['prerequisites/previous_audit.json']=PF/'vehicle_atmoe_head_RGBNT100_result_integrity_audit_20261016.json'
    with tarfile.open(package, 'w:gz') as tar:
        for name,file in stage.items():
            data=file.read_bytes();info=tarfile.TarInfo(name);info.size=len(data)
            tar.addfile(info,io.BytesIO(data));manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    command(['scp', *prior.OPTIONS, str(package), '2026:'+ROOT+'/sources.tar.gz'])
    launch = f'''import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});pkg=r/'sources.tar.gz';manifest={manifest!r}
with tarfile.open(pkg) as tar:
 assert set(tar.getnames())==set(manifest)
 for n in manifest:
  target=(r/n if n.startswith('prerequisites/') else p/n);assert target.resolve().is_relative_to(p.resolve());assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(n).read())
assert all((r/n if n.startswith('prerequisites/') else p/n).stat().st_size==v['bytes'] and hashlib.sha256((r/n if n.startswith('prerequisites/') else p/n).read_bytes()).hexdigest()==v['sha256'] for n,v in manifest.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {runtime!r}.items())
for n in {SOURCES!r}:compile((p/n).read_text(),n,'exec')
argv=[{PYTHON!r},'-u','launch_rgbnt100_paper_batch_baselines.py','--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--previous-intake',str(r/'prerequisites/previous_intake.json'),'--previous-audit',str(r/'prerequisites/previous_audit.json'),'--output',str(r/'experiments')]
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
with (r/'controller.log').open('x') as log:child=subprocess.Popen(argv,cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
owner=dict(pid=child.pid,started=time.time(),argv=argv);(r/'remote_owner.json').write_text(json.dumps(owner,indent=2)+'\\n')
pkg.unlink();print(json.dumps(owner))'''
    owner = json.loads(command(['ssh', *prior.OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=launch))
    package.unlink()
    result = dict(status='ACTUAL_RGBNT100_PAPER_BATCH_CONTROLLER_STARTED_NATIVE_PENDING',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'), remote_root=ROOT,
        remote_owner=owner, capacity=capacity, sources_sha256={name: prior.sha(P/name) for name in SOURCES},
        protected_weights=protected, actual_transport_captures=CAPTURES,
        unrelated_jobs_queried=False, new_native_updates_confirmed=0, formal_updates_confirmed=0,
        physical_gpus=[2,3], maximum_own_NN=2, startup_observation_after_epoch=owner['started']+300,
        warm_environment='Unchanged existing2026 tri_reid conda; no installation or environment mutation',
        missing_evaluated=False, new49=False, goal='ACTIVE_UNMET')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps(dict(status=result['status'], controller=owner['pid'], startup_due_epoch=result['startup_observation_after_epoch'])), flush=True)


if __name__ == '__main__':
    main()
