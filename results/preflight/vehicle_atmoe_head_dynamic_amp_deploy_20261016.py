"""Start the two original head candidates after correcting scheduling/accounting."""
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
ROOT = REMOTE+'/runs/vehicle_atmoe_head_dynamic_amp_20261016'
PREVIOUS = REMOTE+'/runs/vehicle_atmoe_head_baselines_20261016'
SOURCES = ('audit_vehicle_atmoe_head_normal.py','resume_vehicle_atmoe_head_baselines.py')
spec = importlib.util.spec_from_file_location('transport',T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
CAPTURES = []


def command(argv, *, input=None):
    number = len(CAPTURES)+1
    prefix = T/f'vehicle_atmoe_dynamic_amp_{number:02d}_20261016'
    out, err, receipt = [prefix.with_suffix(suffix) for suffix in ('.stdout.log','.stderr.log','.json')]
    assert not any(path.exists() for path in (out,err,receipt))
    started = time.time()
    receipt.write_text(json.dumps(dict(status='ATTEMPT_STARTED',started=started,stdout=str(out),stderr=str(err)),indent=2)+'\n','utf-8')
    with out.open('x',encoding='utf-8') as stdout, err.open('x',encoding='utf-8') as stderr:
        child = subprocess.run(argv,input=input,text=True,encoding='utf-8',stdout=stdout,stderr=stderr,timeout=180)
    capture = dict(status='RETURNED',started=started,finished=time.time(),exit_code=child.returncode,stdout=str(out),stderr=str(err))
    receipt.write_text(json.dumps(capture,indent=2)+'\n','utf-8')
    CAPTURES.append(capture)
    child.check_returncode()
    return out.read_text('utf-8')


def main():
    output = PF/'vehicle_atmoe_head_dynamic_amp_actual_session_20261016.json'
    assert not output.exists()
    review = prior.load(PF/'vehicle_atmoe_head_dynamic_amp_source_review_20261016.json')
    assert review['status']=='FINAL' and review['verdict']=='PASS' and not review['blocking_findings']
    request = prior.load(PF/'vehicle_atmoe_head_dynamic_amp_source_request_20261016.json')
    assert review['request']==str(PF/'vehicle_atmoe_head_dynamic_amp_source_request_20261016.json')
    for row in request['sources']:
        file = Path(row['path'])
        assert file.stat().st_size==row['bytes'] and prior.sha(file)==row['sha256']
    rescue = prior.load(PF/'vehicle_atmoe_head_native_rescue_review_20261016.json')
    assert rescue['status']=='FINAL' and rescue['diagnosis']=='SCHEDULER_CONTRACT_MISMATCH_NO_MODEL_DEFECT_DEMONSTRATED'
    assert rescue['existing_native_smoke_valid_under_parent_contract']
    accounting = prior.load(PF/'vehicle_atmoe_head_dynamic_amp_CPU_accounting_20261016.json')
    assert accounting['status']=='PASS' and accounting['new_neural_calls']==0
    old = prior.load(PF/'vehicle_atmoe_head_actual_session_20261016.json')
    failed = prior.load(PF/'vehicle_atmoe_head_native_gate_actual_20261016.json')
    assert not failed['native_accepted'] and not failed['actual']['controller_live']
    assert not any(row['live'] for row in failed['actual']['children'])
    assert all(prior.sha(P/name)==value for name,value in old['sources_sha256'].items())
    inherited = prior.load(PF/'rgbnt100_F_rank_postprocess_source_review_request_20261016.json')
    seals = inherited['sources_sha256'] | inherited['directly_reused_sources_sha256']
    assert len(seals)==376 and all(prior.sha(P/name)==value for name,value in seals.items())
    runtime = {name:value for name,value in seals.items() if Path(name).suffix in ('.py','.yaml','.yml')}
    runtime['run_full_official_experiment.py'] = prior.sha(P/'run_full_official_experiment.py')
    runtime.update(old['sources_sha256'])
    protected = old['protected_weights']
    admission = f'''import hashlib,json,shutil,subprocess
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});old=Path({PREVIOUS!r});assert not r.exists()
assert not Path('/proc/'+str(json.loads((old/'remote_owner.json').read_text())['pid'])).exists()
assert not list((old/'experiments/training').iterdir())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {runtime!r}.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
resources={{}}
for g in (2,3):
 values=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
 assert values[0]==g and values[1]<500
 apps=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip();assert not apps
 resources[str(g)]=dict(memory_used_mib=values[1],memory_total_mib=values[2],utilization=values[3],compute_processes=apps)
free=shutil.disk_usage(p).free;assert free>1800000000
r.mkdir();print(json.dumps(dict(project_free=free,selected_GPU_resources=resources,unchanged_runtime_sources=len({runtime!r}),protected_four_exact=True,temperature_power_read=False)))'''
    capacity = json.loads(command(['ssh',*prior.OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=admission))
    package = T/'vehicle_atmoe_dynamic_amp_sources_20261016.tar.gz'
    assert not package.exists()
    manifest = {}
    with tarfile.open(package,'w:gz') as tar:
        for name in SOURCES:
            data = (P/name).read_bytes()
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info,io.BytesIO(data))
            manifest[name] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    command(['scp',*prior.OPTIONS,str(package),'2026:'+ROOT+'/sources.tar.gz'])
    launch = f'''import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});pkg=r/'sources.tar.gz';manifest={manifest!r}
with tarfile.open(pkg) as tar:
 assert set(tar.getnames())==set(manifest)
 for n in manifest:
  target=p/n;assert target.resolve().is_relative_to(p.resolve());assert not target.exists();target.write_bytes(tar.extractfile(n).read())
assert all((p/n).stat().st_size==v['bytes'] and hashlib.sha256((p/n).read_bytes()).hexdigest()==v['sha256'] for n,v in manifest.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {runtime!r}.items())
for n in manifest:compile((p/n).read_text(),n,'exec')
argv=[{PYTHON!r},'-u','resume_vehicle_atmoe_head_baselines.py','--previous-root',{PREVIOUS!r},'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--output',str(r/'experiments')]
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
with (r/'controller.log').open('x') as log:child=subprocess.Popen(argv,cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
owner=dict(pid=child.pid,started=time.time(),argv=argv);(r/'remote_owner.json').write_text(json.dumps(owner,indent=2)+'\\n')
pkg.unlink();print(json.dumps(owner))'''
    owner = json.loads(command(['ssh',*prior.OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=launch))
    package.unlink()
    result = dict(status='ACTUAL_DYNAMIC_AMP_ACCOUNTED_VEHICLE_CONTROLLER_STARTED_FORMAL_WITNESS_PENDING',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),remote_root=ROOT,previous_root=PREVIOUS,
        remote_owner=owner,capacity=capacity,sources_sha256={name:prior.sha(P/name) for name in SOURCES},
        protected_weights=protected,actual_transport_captures=CAPTURES,old_scientific_sources_unchanged=True,
        old_native_failure_preserved=True,new_native_updates=0,formal_updates_confirmed=0,
        physical_gpus=[2,3],maximum_NN=2,first_formal_witness_seconds=180,
        historical_full_training_seconds=dict(RGBNT100=9685.604719638824,MSVR310=1579.3985397815704),
        completion_estimate_multiplier=1.2,environment='Warm unchanged2026 tri_reid conda; no install or model/scaler change',
        missing_evaluated=False,new49=False,goal='ACTIVE_UNMET')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(status=result['status'],controller=owner['pid'],new_native_updates=0)),flush=True)


if __name__=='__main__':
    main()
