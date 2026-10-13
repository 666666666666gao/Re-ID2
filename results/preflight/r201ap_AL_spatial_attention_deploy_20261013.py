"""Restore exact AL axis cache/best, native/full once, then retire restored duplicates."""
from datetime import datetime
from pathlib import Path
import hashlib,json,shlex,subprocess,sys,tarfile

P=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(P))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201ap_AL_spatial_attention_20261013';ROOT=REMOTE+'/runs/'+TAG;PF=P/'results/preflight'
MODEL='RGBNT100_r201al_axis_shared_s42'
load=lambda f:json.loads(f.read_text(encoding='utf-8-sig'))
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
remote=lambda code:json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))


def main():
    review=load(PF/'r201ap_AL_spatial_attention_source_review_20261013.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert all(sha(P/n)==h for n,h in (sources|reuse).items())
    closed=load(PF/'r201ao_RGB_hdm_query_guidance_20261013_complete_intake.json')
    assert closed['status']=='ACTUAL_AO_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE' and closed['terminal']['all_children_and_controller_absent']
    best=load(PF/'r201al_closed_best_weights_D_only_actual_20261013.json')['files'][MODEL]
    raw=load(PF/'r201al_RGB_route_temperature_20261013_complete_intake.json')['archives'][MODEL+'/raw']
    assets={'selected_best':best,'selected_reference_arrays':raw}
    assert all(Path(v['local']).stat().st_size==v['bytes'] and sha(Path(v['local']))==v['sha256'] for v in assets.values())
    protected=load(PF/'r201ao_RGB_hdm_query_guidance_20261013_actual_session.json')['protected_weights']
    admission=remote(f'''import hashlib,json,shutil,subprocess
from pathlib import Path
r=Path({REMOTE!r});assets={assets!r};assert not Path({ROOT!r}).exists()
assert all(not Path(v['remote']).exists() for v in assets.values())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {reuse!r}.items() if not n.startswith('results/'))
assert shutil.disk_usage(r).free>sum(v['bytes'] for v in assets.values())+300_000_000
gpu=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not gpu
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,physical_gpu=3,compute_processes=gpu,protected_four_exact=True)))''')
    for row in assets.values():
        subprocess.run(['scp',*OPTIONS,row['local'],'2026:'+row['remote']],check=True,capture_output=True,text=True,encoding='utf-8',timeout=900)
    for name in sources:command(['scp',*OPTIONS,str(P/name),'2026:'+REMOTE+'/'+name])
    remote(f'''import hashlib,json
from pathlib import Path
r=Path({REMOTE!r});assets={assets!r}
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {sources!r}.items())
assert all(Path(v['remote']).stat().st_size==v['bytes'] and hashlib.sha256(Path(v['remote']).read_bytes()).hexdigest()==v['sha256'] for v in assets.values())
print(json.dumps(dict(restored_assets_exact=True,sources_exact=True)))''')
    receipt=PF/(TAG+'_actual_session.json');assert not receipt.exists()
    state=dict(status='ACTUAL_AP_SPATIAL_ATTENTION_STARTING',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        admission=admission,restored_assets=assets,sources_sha256=sources,reused_sources_sha256=reuse,
        physical_gpus=[3],maximum_NN=1,optimizer_updates=0,checkpoint_writes=0)
    receipt.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    code=f'''import subprocess
subprocess.run([{PYTHON!r},'-u','launch_r201ap_AL_spatial_attention.py','--run-dir',
 {REMOTE!r}+'/runs/r201al_RGB_route_temperature_20261013/RGBNT100/training/'+{MODEL!r},'--output',{ROOT!r}],cwd={REMOTE!r},check=True)'''
    completed=subprocess.run(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',shlex.quote(PYTHON)+' -'],
        input=code,check=True,capture_output=True,text=True,encoding='utf-8',timeout=1200)
    print(completed.stdout,end='',flush=True)
    terminal=remote(f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({ROOT!r});c=json.loads((r/'controller_result.json').read_text())
assert c['status']=='COMPLETE' and c['full_query_records']==1715 and c['full_gallery_records']==8575 and c['output_features_exact']
assert not Path('/proc/'+str(json.loads((r/'controller_launch.json').read_text())['pid'])).exists()
for phase in ('native','full'):
 for f in (r/phase).glob('*_launch.json'):
  assert not Path('/proc/'+str(json.loads(f.read_text())['pid'])).exists()
  assert json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text())['exit_code']==0
manifest={{str(f.relative_to(r)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file()}}
pkg=r/'text_intake.tar.gz';assert not pkg.exists()
with tarfile.open(pkg,'w:gz') as tar:
 for n in manifest:tar.add(r/n,arcname=n)
print(json.dumps(dict(controller=c,manifest=manifest,package=dict(bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()),original_children_controller_absent=True)))''')
    local=P/'results'/TAG;local.mkdir(exist_ok=False);package=local/'text_intake.tar.gz'
    command(['scp',*OPTIONS,'2026:'+ROOT+'/text_intake.tar.gz',str(package)])
    assert package.stat().st_size==terminal['package']['bytes'] and sha(package)==terminal['package']['sha256']
    with tarfile.open(package) as tar:tar.extractall(local,filter='data')
    assert all((local/n).stat().st_size==v['bytes'] and sha(local/n)==v['sha256'] for n,v in terminal['manifest'].items())
    assert all(sha(Path(v['local']))==v['sha256'] for v in assets.values())
    cleared=remote(f'''import hashlib,json,shutil
from pathlib import Path
r=Path({REMOTE!r});assets={assets!r};out=Path({ROOT!r})
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
assert all(Path(v['remote']).stat().st_size==v['bytes'] and hashlib.sha256(Path(v['remote']).read_bytes()).hexdigest()==v['sha256'] for v in assets.values())
for v in assets.values():
 f=Path(v['remote']);assert f.resolve().is_relative_to((r/'runs/r201al_RGB_route_temperature_20261013').resolve()) and f.name in ('best.pth','best_official_arrays.npz')
 f.unlink()
(out/'text_intake.tar.gz').unlink()
print(json.dumps(dict(restored_owned_duplicates_removed=2,bytes=sum(v['bytes'] for v in assets.values()),protected_four_exact=True,project_free=shutil.disk_usage(r).free)))''')
    package.unlink()
    state.update(status='ACTUAL_AP_AL_FULL_QG_SPATIAL_ATTENTION_COMPLETE',completed_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        terminal=terminal,cleared=cleared,native_records=64,full_query_records=1715,full_gallery_records=8575,
        full_output_features_exact=True,unique_weights_lost=0,new_missing_evaluation=False,new49=False,
        limits='Existing selected AL only, full QG unchanged output. No retraining or new method gain, no foreground truth or causal conclusion from attention entropy.')
    receipt.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:state[k] for k in ('status','full_query_records','full_gallery_records','optimizer_updates','cleared')}),flush=True)


if __name__=='__main__':main()
