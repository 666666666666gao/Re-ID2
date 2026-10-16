"""Terminal-only RGBNT100 B128 pair archive; preserve four binaries for fresh result audit."""
from datetime import datetime, timezone, timedelta
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import tarfile

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF = P/'results/preflight'
T = Path('C:/Users/gb/.codex_tmp')
TAG = 'rgbnt100_paper_batch_baselines_20261016'
LOCAL = P/'results'/TAG
D = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
spec = importlib.util.spec_from_file_location('verified_transfer', PF/'intake_vehicle_atmoe_dynamic_amp_component_20261016.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
# Only the reusable transport receipt label changes; its dataset-specific main is never called.
transfer.DATASET = 'RGBNT100_B128_pair'
command, prior = transfer.command, transfer.prior


def main():
    output = PF/'rgbnt100_paper_batch_intake_actual_20261016.json'
    assert not output.exists()
    review = prior.load(PF/'rgbnt100_paper_batch_postprocess_source_review_20261016.json')
    assert review['status'] == 'FINAL' and review['verdict'] == 'PASS' and not review['blocking_findings']
    request = prior.load(PF/'rgbnt100_paper_batch_postprocess_source_request_20261016.json')
    assert all(prior.sha(Path(row['path'])) == row['sha256'] for row in request['sources'])
    session = prior.load(PF/'rgbnt100_paper_batch_actual_session_20261016.json')
    node = prior.load(PF/'rgbnt100_paper_batch_completion_actual_20261016.json')['actual']
    assert node['controller_result']['status'] == 'COMPLETE' and not node['controller_live']
    assert len(node['children']) == 6 and all(not row['live'] and row['exit']['exit_code'] == 0 for row in node['children'])
    root = session['remote_root']
    export = f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({root!r});e=r/'experiments';owner=json.loads((r/'remote_owner.json').read_text())
assert not Path('/proc/'+str(owner['pid'])).exists()
c=json.loads((e/'controller_result.json').read_text());assert c['status']=='COMPLETE' and c['cases']==2 and c['formal_epochs']==100 and c['formal_attempts']==6270 and c['paired_sampling_exact'] and c['physical_gpus']==[2,3]
assert {{row['heads'] for row in c['runs']}}=={{8,32}} and all(row['attempts']==3135 and 0<row['successful_updates']<=3135 and row['amp_skips']==3135-row['successful_updates'] for row in c['runs'])
for phase in ('native','training','audit'):
 for f in (e/phase).glob('*_launch.json'):
  launch=json.loads(f.read_text());assert not Path('/proc/'+str(launch['pid'])).exists()
  assert json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text())['exit_code']==0
manifest={{f.relative_to(r).as_posix():dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.jsonl','.log')}}
raw={{}}
for row in c['runs']:
 for key in ('raw','best'):
  item=row[key];file=Path(item['path']);assert file.resolve().is_relative_to(r.resolve())
  actual=dict(path=str(file),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest());assert actual==item
  raw['head'+str(row['heads'])+'/'+key]=actual
assert len(raw)==4
package=r/'text_intake.tar.gz';assert not package.exists()
with tarfile.open(package,'w:gz') as tar:
 for name in manifest:tar.add(r/name,arcname=name,recursive=False)
print(json.dumps(dict(controller=c,manifest=manifest,raw=raw,text_archive=dict(path=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()),new_neural_calls=0,unrelated_jobs_queried=False)))'''
    exported = json.loads(command(['ssh', *prior.OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=export))
    needed = sum(row['bytes'] for row in exported['raw'].values())
    assert shutil.disk_usage(D.anchor).free > needed+100000000
    text_bytes = sum(row['bytes'] for row in exported['manifest'].values())
    assert shutil.disk_usage(P).free > text_bytes+exported['text_archive']['bytes']+100000000
    assert not LOCAL.exists() and not D.exists()
    LOCAL.mkdir()
    D.mkdir()
    package = T/'rgbnt100_B128_pair_texts_20261016.tar.gz'
    assert not package.exists()
    command(['scp', *prior.OPTIONS, '2026:'+exported['text_archive']['path'], str(package)])
    assert package.stat().st_size == exported['text_archive']['bytes'] and prior.sha(package) == exported['text_archive']['sha256']
    with tarfile.open(package) as tar:
        assert set(tar.getnames()) == set(exported['manifest'])
        for name in exported['manifest']:
            target = LOCAL/name
            assert target.resolve().is_relative_to(LOCAL.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(name).read())
    assert all((LOCAL/name).stat().st_size == row['bytes'] and prior.sha(LOCAL/name) == row['sha256'] for name, row in exported['manifest'].items())
    archives = {}
    for name, row in exported['raw'].items():
        destination = D/name/Path(row['path']).name
        destination.parent.mkdir(parents=True)
        command(['scp', *prior.OPTIONS, '2026:'+row['path'], str(destination)])
        assert destination.stat().st_size == row['bytes'] and prior.sha(destination) == row['sha256']
        archives[name] = dict(local=str(destination), remote=row['path'], bytes=row['bytes'], sha256=row['sha256'])
    retire = f'''import hashlib,json
from pathlib import Path
r=Path({root!r});p=Path({exported['text_archive']['path']!r});assert p.resolve().is_relative_to(r.resolve())
assert p.stat().st_size=={exported['text_archive']['bytes']!r} and hashlib.sha256(p.read_bytes()).hexdigest()=={exported['text_archive']['sha256']!r}
assert all(Path(row['remote']).stat().st_size==row['bytes'] and hashlib.sha256(Path(row['remote']).read_bytes()).hexdigest()==row['sha256'] for row in {archives!r}.values())
p.unlink();print(json.dumps(dict(only_text_package_removed=True,all_four_binaries_preserved=True,new_neural_calls=0)))'''
    cleared = json.loads(command(['ssh', *prior.OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=retire))
    package.unlink()
    proof = dict(status='ACTUAL_RGBNT100_B128_TWO_FULL50_GT_ARCHIVED',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        controller=exported['controller'], local_text_root=str(LOCAL), text_manifest=exported['manifest'], archives=archives,
        actual_transport_captures=transfer.CAPTURES, cleared=cleared,
        D_free_after=shutil.disk_usage(D.anchor).free, C_free_after=shutil.disk_usage(P).free,
        new_neural_calls=0, unrelated_jobs_queried=False, result_audit='Fresh independent result audit pending', goal='ACTIVE_UNMET')
    output.write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps(dict(status=proof['status'], text_files=len(exported['manifest']), D_archived_bytes=needed)), flush=True)


if __name__ == '__main__':
    main()
