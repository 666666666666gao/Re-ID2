"""Collect finite speed proof and retire only its exact transfer/dependency duplicate."""
from datetime import datetime
from pathlib import Path
import hashlib,json,shlex,subprocess,sys,tarfile
P=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(P))
from collect_results import OPTIONS
PF=P/'results/preflight';REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201bb_parallel_jobs_20261015';ROOT=REMOTE+'/runs/'+TAG;LOCAL=P/'results'/TAG
load=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def command(argv,**kwargs):
    d=subprocess.run(argv,check=False,capture_output=True,text=True,encoding='utf-8',timeout=300,**kwargs)
    if d.returncode:
        print(d.stdout,flush=True);print(d.stderr,file=sys.stderr,flush=True);d.check_returncode()
    return d.stdout


def main():
    started=load(PF/(TAG+'_actual_session.json'));owner=started['owner']['pid']
    code=f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({ROOT!r});assert not Path('/proc/{owner}').exists()
d=json.loads((r/'profile_output/controller_result.json').read_text());assert d['status']=='COMPLETE_BENCHMARK_NOT_METHOD_RESULT' and d['successful_updates']==228
exits=list((r/'profile_output').rglob('*_exit.json'));assert len(exits)==12 and all(json.loads(f.read_text())['exit_code']==0 for f in exits)
for f in (r/'profile_output').rglob('*_launch.json'):
 p=json.loads(f.read_text())['pid'];assert not Path('/proc/'+str(p)).exists()
assert not list(r.rglob('*.pth')) and not list(r.rglob('*.npz'))
manifest={{str(f.relative_to(r)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.log','.jsonl')}}
pkg=r/'text_intake.tar.gz';assert not pkg.exists()
with tarfile.open(pkg,'w:gz') as tar:
 for n in manifest:tar.add(r/n,arcname=n)
print(json.dumps(dict(controller=d,manifest=manifest,package=dict(bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()),all_consumers_absent=True,controller_OS_exit_code='UNATTESTED')))'''
    terminal=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    assert not LOCAL.exists();LOCAL.mkdir();package=LOCAL/'text_intake.tar.gz'
    command(['scp',*OPTIONS,'2026:'+ROOT+'/text_intake.tar.gz',str(package)])
    assert package.stat().st_size==terminal['package']['bytes'] and sha(package)==terminal['package']['sha256']
    with tarfile.open(package) as tar:tar.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==d['bytes'] and sha(LOCAL/n)==d['sha256'] for n,d in terminal['manifest'].items())
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(P/'analyze_r201bb_parallel_jobs.py'),'--root',str(LOCAL/'profile_output')],check=True)
    bank=started['required_bank'];original=Path(bank['local'])
    assert original.stat().st_size==bank['bytes'] and sha(original)==bank['sha256']
    code=f'''import hashlib,json
from pathlib import Path
r=Path({ROOT!r});assert not Path('/proc/{owner}').exists()
pkg=r/'text_intake.tar.gz';assert pkg.stat().st_size=={terminal['package']['bytes']} and hashlib.sha256(pkg.read_bytes()).hexdigest()=={terminal['package']['sha256']!r}
p=Path({bank['remote']!r});owned=Path({REMOTE!r})/'inputs/r201ba_RGB_training_bank_20261015'
assert p.resolve().parent==owned.resolve() and p.stat().st_size=={bank['bytes']} and hashlib.sha256(p.read_bytes()).hexdigest()=={bank['sha256']!r}
assert all(hashlib.sha256((Path({REMOTE!r})/n).read_bytes()).hexdigest()==h for n,h in {started['protected_weights']!r}.items())
pkg.unlink();p.unlink();owned.rmdir();print(json.dumps(dict(matching_bank_and_package_retired=True,protected_four_exact=True)))'''
    cleaned=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code));package.unlink()
    proof=dict(status='ACTUAL_BB_SERIAL_PARALLEL_THREE_REPEATS_COLLECTED',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        terminal=terminal,cleanup=cleaned,new_weights_written=0,unique_weights_deleted=0,
        actual_benchmark_updates=228,actual_benchmark_QG_forward_records=123480,
        performance_claim='Finite workload/card scaling only; not method performance or full50 proof.',goal='ACTIVE_UNMET')
    (PF/(TAG+'_complete_intake.json')).write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=proof['status'],new_weights_written=0,cleanup=cleaned)),flush=True)


if __name__=='__main__':main()
