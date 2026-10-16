"""Explicitly resume the two successful captures after the actual C-full failure."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib.util
import json
import os
import shutil
import sys
import tarfile

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
T = Path('C:/Users/gb/.codex_tmp')
PF = P/'results/preflight'
spec = importlib.util.spec_from_file_location('original_intake', PF/'rgbnt100_F_rank_complete_intake_20261016.py')
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)


def main():
    review = intake.load(PF/'rgbnt100_F_rank_C_full_rescue_source_review_20261016.json')
    assert review['status'] == 'FINAL' and review['verdict'] == 'PASS' and not review['blocking_findings']
    output = PF/'rgbnt100_F_rank_complete_intake_20261016.json'
    assert not output.exists()
    space = json.loads(Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/ReID2_closed_N5_text_C_space_recovery_20261016.json').read_text(encoding='utf-8-sig'))
    assert space['status'] == 'ACTUAL_C_FULL_RECOVERY_CLOSED_N5_TEXTS_MOVED_TO_D_ORIGINAL_PATH_JUNCTION'
    assert space['all_original_paths_size_sha_exact'] and space['unique_records_deleted'] == 0
    error = (T/'RGBNT100_F_rank_complete_intake_private_stderr_20261016.log').read_text('utf-8-sig')
    assert "tar.extractall(LOCAL,filter='data')" in error and 'OSError: [Errno 28] No space left on device' in error
    for n in (1, 2):
        record = intake.load(T/f'RGBNT100_F_rank_intake_{n:02d}_transport_actual_20261016.json')
        assert record['index'] == n and record['exit_code'] == 0
        intake.CAPTURES.append(record)
    terminal = intake.load(Path(intake.CAPTURES[0]['stdout']))
    assert terminal['controller']['status'] == 'COMPLETE' and terminal['all_children_and_controller_absent']
    assert len(terminal['children']) == 10 and all(c['exit']['exit_code'] == 0 for c in terminal['children'])
    required = sum(run[key]['bytes'] for run in terminal['controller']['runs'] for key in ('raw', 'states_raw', 'best'))
    assert shutil.disk_usage(intake.ARCHIVE).free > required+500_000_000
    assert not list(intake.ARCHIVE.iterdir())
    assert shutil.disk_usage(P).free > sum(row['bytes'] for row in terminal['manifest'].values())+200_000_000
    package = intake.LOCAL/'text_intake.tar.gz'
    assert package.stat().st_size == terminal['package']['bytes'] and intake.sha(package) == terminal['package']['sha256']
    with tarfile.open(package) as tar:
        assert set(tar.getnames()) == set(terminal['manifest'])
        tar.extractall(intake.LOCAL, filter='data')
    assert all((intake.LOCAL/n).stat().st_size == row['bytes'] and intake.sha(intake.LOCAL/n) == row['sha256']
               for n, row in terminal['manifest'].items())
    reviewed = intake.load(PF/'rgbnt100_F_rank_postprocess_source_review_request_20261016.json')
    seals = reviewed['sources_sha256'] | reviewed['directly_reused_sources_sha256']
    assert len(seals) == 376 and all(intake.sha(P/n) == h for n, h in seals.items())
    archives = {}
    for run in terminal['controller']['runs']:
        for key, filename in [('raw', 'best_official_arrays.npz'), ('states_raw', 'normal_states.npz'), ('best', 'best.pth')]:
            row = run[key]
            folder = intake.ARCHIVE/run['name']/key
            folder.mkdir(parents=True)
            destination = folder/filename
            intake.command(['scp', *intake.OPTIONS, '2026:'+row['path'], str(destination)])
            assert destination.stat().st_size == row['bytes'] and intake.sha(destination) == row['sha256']
            archives[run['name']+'/'+key] = dict(local=str(destination), remote=row['path'], bytes=row['bytes'], sha256=row['sha256'])
    assert len(archives) == 6
    intake.command([sys.executable, '-X', 'utf8', '-B', '-S', str(P/'analyze_rgbnt100_frequency_rank_normal.py')], cwd=P)
    assert all(intake.sha(P/n) == h for n, h in seals.items())
    session = intake.load(PF/'rgbnt100_F_rank_actual_session_20261016.json')
    protected = session['protected_weights']
    owner = session['remote_owner']['pid']
    cleared = intake.remote(f'''import hashlib,json,shutil
from pathlib import Path
p=Path({intake.REMOTE!r});r=Path({intake.ROOT!r});owned={{v['remote']:dict(bytes=v['bytes'],sha256=v['sha256']) for v in {archives!r}.values()}}
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
        folder = intake.LOCAL/'RGBNT100'/('states' if key == 'states_raw' else 'training')/model
        (folder/('local_best_archive.json' if key == 'best' else 'local_archive.json')).write_text(
            json.dumps(dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED', **row), indent=2)+'\n', 'utf-8')
    proof = dict(status='ACTUAL_F_RANK128_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE',
        actual_transport_captures=intake.CAPTURES, at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        collector_pid=os.getpid(), terminal=terminal, archives=archives, cleared=cleared,
        D_free=shutil.disk_usage(intake.ARCHIVE).free, formal_updates=13126, native_updates=6,
        parity_forward_record_executions=64, state_cases=8, selected_bests_D_archived=2,
        unique_best_deleted=0, source_seals_unchanged=len(seals), missing_evaluated=False, new49=False,
        recovery=dict(original_session=87387, original_actual_exit_code=1, original_consumed_tool_chunk='673c11',
            actual_error='OSError28 C full during text tar extraction; first two captures exit0',
            original_intake_source_unchanged=True, original_error_preserved=True, automatic_retries=0, NN_restarts=0,
            continuation='Explicit reviewed continuation from preserved valid text package and first two actual captures',
            closed_text_space_proof=space['status']), goal='ACTIVE_UNMET')
    output.write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    failed_owner = PF/'rgbnt100_F_rank_intake_owner_20261016.json'
    assert failed_owner.stat().st_size == 0
    failed_owner.rename(PF/'rgbnt100_F_rank_intake_owner_20261016.write_failed_empty')
    print(json.dumps({k: proof[k] for k in ('status', 'formal_updates', 'state_cases', 'D_free', 'cleared')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
