"""Archive the genuinely completed normal pair; never delete unique/raw weights."""
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tarfile

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF = P / 'results/preflight'
T = Path('C:/Users/gb/.codex_tmp')
TAG = 'rgbnt100_B128_reference_experts_normfix_20261026'
LOCAL = P / 'results' / TAG
D = Path('D:/Program Files/UserCache/gb/ReID2-retained/RGBNT100_B128_reference_experts_20261026')
PY = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
spec = importlib.util.spec_from_file_location('captured', PF / 'intake_vehicle_atmoe_dynamic_amp_component_20261016.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
t.DATASET = 'RGB_B128_reference_normfix_terminal_intake_20261026'


def main():
    output = PF / 'rgbnt100_B128_reference_normfix_intake_actual_20261026.json'
    assert not output.exists() and not LOCAL.exists() and not D.exists()
    review = t.prior.load(PF / 'rgbnt100_B128_reference_postprocess_source_review_20261026.json')
    request = t.prior.load(PF / 'rgbnt100_B128_reference_postprocess_source_request_20261026.json')
    assert review['status'] == 'FINAL' and review['verdict'] == 'PASS' and not review['blocking_findings']
    assert all(Path(v['path']).stat().st_size == v['bytes'] and t.prior.sha(Path(v['path'])) == v['sha256'] for v in request['sources'])
    session = t.prior.load(PF / 'rgbnt100_B128_reference_experts_normfix_actual_session_20261026.json')
    root = session['actual']['remote_root']
    assert root == '/data/gaob/Re-ID/DeMo-DualAxis/runs/' + TAG
    export = f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({root!r});e=r/'experiments';owner=json.loads((r/'remote_owner.json').read_text())
assert owner['pid']=={session['actual']['pid']!r} and not Path('/proc/'+str(owner['pid'])).exists()
c=json.loads((e/'controller_result.json').read_text())
assert c['status']=='COMPLETE' and c['controls']==2 and c['successful_updates']==13126 and c['native_updates']==6
assert c['additional_epochs']==100 and c['paired_sampling_exact'] and c['states_cases']==8 and c['reference00_full_parity']
assert c['teacher_reference_head']==8 and c['teacher_reference_training_batch']==128 and c['physical_gpus']==[2,3] and c['maximum_NN']==2
assert {{v['name'] for v in c['runs']}}=={{'RGBNT100_B128ref_AV_frequency_shared_s42','RGBNT100_B128ref_AV_axis_shared_s42'}}
children=[]
for phase in ('reference','native','training','audit','states'):
 files=list((e/phase).glob('*_launch.json'));assert len(files)==2
 for f in files:
  launch=json.loads(f.read_text());assert not Path('/proc/'+str(launch['pid'])).exists()
  ended=json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text());assert ended['exit_code']==0
  children.append(dict(phase=phase,pid=launch['pid'],exit_code=ended['exit_code']))
assert len(children)==10
raw={{}}
for row in c['runs']:
 trained=json.loads((e/'training'/row['name']/'result.json').read_text())
 assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['steps']==trained['optimizer_steps']==6563
 assert trained['amp_skipped_steps']==trained['training_heldout_identities']==0 and trained['descriptor_dim']==5120
 assert trained['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[])
 assert row['best']['selected_epoch']==trained['best']['epoch']
 for key in ('raw','states_raw','best'):
  item=row[key];f=Path(item['path']);assert f.resolve().is_relative_to(r.resolve())
  value=dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
  assert all(value[k]==item[k] for k in value)
  raw[row['name']+'/'+key]=value
assert len(raw)==6
manifest={{f.relative_to(r).as_posix():dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.jsonl','.log')}}
package=r/'terminal_text_intake.tar.gz';assert not package.exists()
with tarfile.open(package,'w:gz') as tar:
 for n in manifest:tar.add(r/n,arcname=n,recursive=False)
print(json.dumps(dict(controller=c,children=children,manifest=manifest,raw=raw,package=dict(path=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()),new_NN=0)))
'''
    compile(export, 'terminal_pair_export', 'exec')
    actual = json.loads(t.command(['ssh', *t.prior.OPTIONS, '2026', shlex.quote(PY) + ' -'], input=export + chr(10)))
    needed = sum(v['bytes'] for v in actual['raw'].values())
    assert shutil.disk_usage(D.anchor).free > needed + 100000000
    assert shutil.disk_usage(P).free > sum(v['bytes'] for v in actual['manifest'].values()) + actual['package']['bytes'] + 100000000
    LOCAL.mkdir()
    D.mkdir()
    package = T / 'RGBNT100_B128_normfix_terminal_texts_20261026.tar.gz'
    assert not package.exists()
    t.command(['scp', *t.prior.OPTIONS, '2026:' + actual['package']['path'], str(package)])
    assert package.stat().st_size == actual['package']['bytes'] and t.prior.sha(package) == actual['package']['sha256']
    with tarfile.open(package) as tar:
        assert set(tar.getnames()) == set(actual['manifest'])
        for n in actual['manifest']:
            f = LOCAL / n
            assert f.resolve().is_relative_to(LOCAL.resolve())
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(tar.extractfile(n).read())
    assert all((LOCAL / n).stat().st_size == v['bytes'] and t.prior.sha(LOCAL / n) == v['sha256'] for n, v in actual['manifest'].items())
    archives = {}
    for name, v in actual['raw'].items():
        target = D / name / Path(v['path']).name
        target.parent.mkdir(parents=True)
        t.command(['scp', *t.prior.OPTIONS, '2026:' + v['path'], str(target)])
        assert target.stat().st_size == v['bytes'] and t.prior.sha(target) == v['sha256']
        archives[name] = dict(local=str(target), remote=v['path'], bytes=v['bytes'], sha256=v['sha256'])
    subprocess.run([sys.executable, '-X', 'utf8', '-B', '-S', str(P / 'analyze_rgbnt100_B128_reference_experts_normal.py')], cwd=P, check=True)
    clear = f'''import hashlib,json
from pathlib import Path
r=Path({root!r});p=Path({actual['package']['path']!r});assert p.resolve().is_relative_to(r.resolve())
assert p.stat().st_size=={actual['package']['bytes']!r} and hashlib.sha256(p.read_bytes()).hexdigest()=={actual['package']['sha256']!r}
assert all(Path(v['remote']).stat().st_size==v['bytes'] and hashlib.sha256(Path(v['remote']).read_bytes()).hexdigest()==v['sha256'] for v in {archives!r}.values())
p.unlink();print(json.dumps(dict(only_text_package_removed=True,all_six_binaries_preserved=True,new_NN=0)))
'''
    compile(clear, 'remove_verified_text_package_only', 'exec')
    cleared = json.loads(t.command(['ssh', *t.prior.OPTIONS, '2026', shlex.quote(PY) + ' -'], input=clear + chr(10)))
    assert package.resolve().parent == T.resolve()
    package.unlink()
    result = dict(status='ACTUAL_RGBNT100_B128_REFERENCE_TWO_FULL50_NORMAL_EIGHT_STATES_ARCHIVED',
        at_utc=datetime.now(timezone.utc).isoformat(), controller=actual['controller'], children=actual['children'],
        local_text_root=str(LOCAL), text_manifest=actual['manifest'], archives=archives, cleared=cleared,
        actual_transport_captures=t.CAPTURES, D_free_after=shutil.disk_usage(D.anchor).free,
        native_updates=6, formal_updates=13126, state_cases=8, raw_and_selected_weights_preserved=6,
        new_NN=0, optimizer_updates=0, missing_evaluated=False, new49=False,
        result_audit='Fresh independent raw/GT/result integrity audit remains required before adoption or binary cleanup.', goal='ACTIVE_UNMET')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps(dict(status=result['status'], text_files=len(actual['manifest']), D_archived_bytes=needed)), flush=True)


if __name__ == '__main__':
    main()
