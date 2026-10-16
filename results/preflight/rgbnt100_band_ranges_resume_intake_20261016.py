"""One bounded intake continuation after the recorded first-file transport failure."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
spec = importlib.util.spec_from_file_location('original_intake', P/'results/preflight/rgbnt100_band_ranges_complete_intake_20261016.py')
i = importlib.util.module_from_spec(spec)
spec.loader.exec_module(i)


def main():
    output = i.PF/'rgbnt100_band_ranges_complete_intake_20261016.json'
    assert not output.exists()
    review = i.load(i.PF/'rgbnt100_band_ranges_resume_intake_source_review_20261016.json')
    assert review['verdict'] == 'PASS' and not review['blocking_findings']
    failure = i.load(i.PF/'rgbnt100_band_ranges_intake_failure_actual_20261016.json')
    assert failure['scp_exit'] == 255 and failure['actual_readonly_reconnect_pass'] and failure['local_text_archive_exact']
    assert sum(r['exists'] for r in failure['local_archives'].values()) == 1
    session = i.load(i.PF/'rgbnt100_band_ranges_actual_session_20261016.json')
    owner = session['remote_owner']['pid']
    source = i.load(i.PF/'rgbnt100_band_ranges_source_review_request_20261016.json')
    seals = source['sources_sha256'] | source['directly_reused_sources_sha256']
    extra = i.load(i.PF/'rgbnt100_band_ranges_source_review_20261016.json')['additional_sources_sha256']
    assert len(seals) == 368 and all(i.sha(P/n) == h for n, h in seals.items())
    assert len(extra) == 8 and all(i.sha(Path(n)) == h for n, h in extra.items())
    terminal = i.remote(f'''import hashlib,json
from pathlib import Path
r=Path({i.ROOT!r});p=r/'RGBNT100';assert not Path('/proc/{owner}').exists()
d=json.loads((p/'controller_result.json').read_text());assert d['status']=='COMPLETE' and d['successful_updates']==13126 and d['state_cases']==8
assert (d['band_low_cutoff'],d['band_high_cutoff'],d['frequency_bin_counts'])==(.25,.375,[25,30,73])
children=[]
for phase in ('parity','native','training','audit','states'):
 for f in (p/phase).glob('*_launch.json'):
  launch=json.loads(f.read_text());assert not Path('/proc/'+str(launch['pid'])).exists()
  ended=json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text());assert ended['exit_code']==0
  children.append(dict(phase=phase,name=f.name,pid=launch['pid'],exit=ended))
assert len(children)==10
manifest={{str(f.relative_to(r)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.log','.jsonl')}}
pkg=r/'text_intake.tar.gz'
print(json.dumps(dict(controller=d,children=children,manifest=manifest,package=dict(bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()),all_children_and_controller_absent=True,controller_OS_exit='UNATTESTED')))''')
    assert terminal['controller'] == i.load(i.LOCAL/'RGBNT100/controller_result.json')
    assert terminal['manifest'] == failure['remote']['manifest'] and terminal['package'] == failure['remote']['package']
    package = i.LOCAL/'text_intake.tar.gz'
    assert package.stat().st_size == terminal['package']['bytes'] and i.sha(package) == terminal['package']['sha256']
    assert all((i.LOCAL/n).stat().st_size == r['bytes'] and i.sha(i.LOCAL/n) == r['sha256'] for n, r in terminal['manifest'].items())
    required = sum(run[key]['bytes'] for run in terminal['controller']['runs'] for key in ('raw', 'states_raw', 'best'))
    assert shutil.disk_usage(i.ARCHIVE).free > required+500_000_000
    archives, transfers = {}, []
    for run in terminal['controller']['runs']:
        for key, filename in [('raw', 'best_official_arrays.npz'), ('states_raw', 'normal_states.npz'), ('best', 'best.pth')]:
            name = run['name']+'/'+key
            old = failure['local_archives'][name]
            row = run[key]
            destination = Path(old['local'])
            if old['exists']:
                assert name == 'RGBNT100_ranges_AV_frequency_shared_s42/raw'
                assert destination.stat().st_size == old['bytes'] == 211124224 and i.sha(destination) == old['sha256']
            else:
                assert not destination.exists()
            destination.parent.mkdir(parents=True, exist_ok=True)
            index = len(transfers)+1
            receipt = i.T/f'RGBNT100_band_ranges_resume_transfer_{index}_actual_20261016.json'
            assert not receipt.exists()
            started = time.time()
            with (i.T/f'RGBNT100_band_ranges_resume_transfer_{index}_private_stdout_20261016.log').open('w') as out, (i.T/f'RGBNT100_band_ranges_resume_transfer_{index}_private_stderr_20261016.log').open('w') as err:
                child = subprocess.run(['scp', *i.OPTIONS, '2026:'+row['path'], str(destination)], stdout=out, stderr=err, timeout=900)
            transfer = dict(name=name, exit_code=child.returncode, started_epoch=started, ended_epoch=time.time(), original_remote=row['path'], local=str(destination))
            receipt.write_text(json.dumps(transfer, indent=2)+'\n', 'utf-8')
            child.check_returncode()
            assert destination.stat().st_size == row['bytes'] and i.sha(destination) == row['sha256']
            transfers.append(transfer)
            archives[name] = dict(local=str(destination), remote=row['path'], bytes=row['bytes'], sha256=row['sha256'])
    assert len(archives) == len(transfers) == 6
    subprocess.run([sys.executable, '-X', 'utf8', '-B', '-S', str(P/'analyze_rgbnt100_band_range_normal.py')], cwd=P, check=True)
    assert all(i.sha(P/n) == h for n, h in seals.items()) and all(i.sha(Path(n)) == h for n, h in extra.items())
    protected = session['protected_weights']
    cleared = i.remote(f'''import hashlib,json,shutil
from pathlib import Path
p=Path({i.REMOTE!r});r=Path({i.ROOT!r});owned={{v['remote']:dict(bytes=v['bytes'],sha256=v['sha256']) for v in {archives!r}.values()}}
owned[str(r/'text_intake.tar.gz')]={terminal['package']!r}
assert not Path('/proc/{owner}').exists()
assert all(Path(n).resolve().is_relative_to(r.resolve()) and Path(n).stat().st_size==v['bytes'] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==v['sha256'] for n,v in owned.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
for n in owned:Path(n).unlink()
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
print(json.dumps(dict(matching_owned_duplicates_cleared=len(owned),bytes=sum(v['bytes'] for v in owned.values()),protected_bests_exact=True,remote_project_free=shutil.disk_usage(p).free)))''')
    package.unlink()
    for name, row in archives.items():
        model, key = name.split('/')
        folder = i.LOCAL/'RGBNT100'/('states' if key == 'states_raw' else 'training')/model
        (folder/('local_best_archive.json' if key == 'best' else 'local_archive.json')).write_text(json.dumps(dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED', **row), indent=2)+'\n', 'utf-8')
    proof = dict(status='ACTUAL_BAND_RANGE_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE', at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        collector_pid=os.getpid(), terminal=terminal, archives=archives, cleared=cleared, D_free=shutil.disk_usage(i.ARCHIVE).free,
        formal_updates=13126, native_updates=6, parity_forward_record_executions=64, state_cases=8, selected_bests_D_archived=2,
        unique_best_deleted=0, source_seals_unchanged=len(seals), missing_evaluated=False, new49=False, goal='ACTIVE_UNMET',
        recovery='One explicit reviewed continuation of the same six completed artifacts; original intake shell failed1/firstscp255; no new NN or automatic retry',
        resumed_transfer_receipts=transfers, preserved_failure='results/preflight/rgbnt100_band_ranges_intake_failure_actual_20261016.json')
    output.write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps({k: proof[k] for k in ('status', 'formal_updates', 'state_cases', 'D_free', 'cleared')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
