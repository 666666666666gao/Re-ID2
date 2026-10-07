"""Archive inactive, closed G vehicle distances on 27; do not delete local data."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command

HOST = '2027'
PYTHON = '/data/gb/Re-ID/conda-envs/tri_reid/bin/python'
REMOTE = '/data/gb/Re-ID/DeMo-DualAxis/archives/closed_G_vehicle49_20261007'
LOCAL_D = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201g_other_two_missing49_stream_20261006')
LOCAL_E = Path('E:/ReID2-experiment-artifacts/closed_20261007/r201g_other_two_missing49_stream_20261006')


def main():
    pf = PROJECT / 'results/preflight'
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    review = load(pf / 'closed_G_vehicle49_archive27_source_review_20261007.json')
    assert review['status'] == 'PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha
               for name, sha in (review['sources_sha256'] | review['directly_reused_sources_sha256']).items())
    actual = load(pf / 'r201g_other_two_missing49_stream_full_actual_session_20261006.json')
    assert actual['status'] == 'ACTUAL_G_OTHER_TWO_FOUR_FULL49_INFERENCE_CONTROLS_ALL784_GT_AND_LOCAL_RAW'
    assert actual['exit_code'] == actual['new_optimizer_updates'] == 0 and len(actual['raw']) == 196
    closed = load(pf / 'r201g_other_two_missing49_closeout_queue_actual_session_20261006.json')
    assert closed['status'] == 'ACTUAL_G_OTHER_FULL49_TEXT_AND_THREE_DATASET_CPU_CLOSEOUT_QUEUE_COMPLETE'
    assert closed['exit_code'] == 0 and closed['raw_local_verified'] == 196 and closed['GT_state_cases'] == 784
    assert closed['new_neural_calls'] == closed['new_optimizer_updates'] == 0
    intake = load(pf / 'r201g_other_two_missing49_full_intake_20261006.json')
    assert intake['status'] == 'ACTUAL_CLOSED_MISSING_INFERENCE_PRIMARY_TEXT_LOCAL_VERIFIED'
    text_root = Path(intake['local_root'])
    assert text_root.resolve().is_relative_to((PROJECT / 'results').resolve())
    assert all((text_root / name).stat().st_size == row['bytes'] and
               hashlib.sha256((text_root / name).read_bytes()).hexdigest() == row['sha256']
               for name, row in intake['manifest'].items())
    relocation = load(pf / 'own_D_closed_G100_frequency49_to_E_actual_20261007.json')
    assert relocation['status'] == 'ACTUAL_CLOSED_G100_FREQUENCY49_RAW_RELOCATED_D_TO_E_SIZE_SHA_VERIFIED'
    assert relocation['files'] == 49 and relocation['lost_raw_files'] == 0
    moved = {row['key']: row for row in relocation['records']}
    assert len(moved) == 49
    rows = []
    jobs = {dataset + '_r201g_' + variant + '_s42'
            for dataset in ('MSVR310', 'RGBNT100') for variant in ('frequency_shared', 'axis_shared')}
    for key, item in sorted(actual['raw'].items()):
        job, condition = key.split('/')
        assert job in jobs and condition.startswith('q_') and '_g_' in condition
        if job == 'RGBNT100_r201g_frequency_shared_s42':
            record = moved[key]
            assert Path(record['source']) == Path(item['local'])
            assert record['bytes'] == item['file']['bytes'] and record['sha256'] == item['file']['sha256']
            source = Path(record['destination'])
            assert source.resolve().is_relative_to(LOCAL_E.resolve())
        else:
            assert key not in moved
            source = Path(item['local'])
            assert source.resolve().is_relative_to(LOCAL_D.resolve())
        assert source.name == 'raw.npz' and source.stat().st_size == item['file']['bytes']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['file']['sha256']
        rows.append(dict(key=key, source=str(source.resolve()), destination=REMOTE + '/full/' + key + '/raw.npz', **item['file']))
    assert len(rows) == 196 and len({row['source'] for row in rows}) == 196
    assert all(sum(row['key'].split('/')[0] == job for row in rows) == 49 for job in jobs)
    total = sum(row['bytes'] for row in rows)
    assert total == 15_477_041_156
    proof = pf / 'closed_G_vehicle49_archive27_ack_actual_20261007.json'
    assert not proof.exists()
    code = f'''import json,shutil
from pathlib import Path
root=Path({REMOTE!r});rows={rows!r}
assert not root.exists() and shutil.disk_usage(root.parent.parent).free>{total!r}+1_000_000_000
root.mkdir(parents=True)
for row in rows:
 path=Path(row['destination']);assert path.resolve().is_relative_to(root.resolve())
 path.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps(dict(status='CLOSED_G_ARCHIVE27_EMPTY_CAPACITY_READY',files=len(rows),bytes={total!r})))'''
    ready = json.loads(command(['ssh', *OPTIONS, HOST, shlex.quote(PYTHON) + ' -'], input=code))
    assert ready['files'] == 196 and ready['bytes'] == total
    started = datetime.now().isoformat(timespec='seconds')
    print('CLOSED_G_ARCHIVE27_COPY_STARTED', started, total, flush=True)
    for index, row in enumerate(rows, 1):
        command(['scp', *OPTIONS, row['source'], HOST + ':' + row['destination']])
        if index in (1, 49, 98, 147, 196):
            print('CLOSED_G_ARCHIVE27_FILES_COPIED', index, flush=True)
    code = f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r});rows={rows!r}
assert {{str(p) for p in root.rglob('raw.npz')}}=={{row['destination'] for row in rows}}
assert all(Path(row['destination']).stat().st_size==row['bytes'] and hashlib.sha256(Path(row['destination']).read_bytes()).hexdigest()==row['sha256'] for row in rows)
manifest=root/'archive_manifest.json';assert not manifest.exists()
manifest.write_text(json.dumps(dict(status='ALL196_CLOSED_G_VEHICLE49_SIZE_SHA_VERIFIED',records=rows,files=196,bytes={total!r},new_neural_calls=0,new_optimizer_updates=0),indent=2)+'\\n')
print(json.dumps(dict(status='ALL196_CLOSED_G_VEHICLE49_SIZE_SHA_VERIFIED',files=196,bytes={total!r},manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))'''
    ack = json.loads(command(['ssh', *OPTIONS, HOST, shlex.quote(PYTHON) + ' -'], input=code))
    assert ack['status'] == 'ALL196_CLOSED_G_VEHICLE49_SIZE_SHA_VERIFIED' and ack['files'] == 196 and ack['bytes'] == total
    value = dict(status='ACTUAL_ALL196_CLOSED_G_RAW_ARCHIVE27_SIZE_SHA_ACK_LOCAL_NOT_REMOVED',
                 started=started, verified_at=datetime.now().isoformat(timespec='seconds'), records=rows,
                 files=196, bytes=total, remote_host=HOST, remote_root=REMOTE, remote_ack=ack,
                 local_files_removed=0, unique_raw_lost=0, new_neural_calls=0, new_optimizer_updates=0,
                 limits='Only inactive closed G vehicle49 data; text six metrics/CMC50/per-query/groups/GT and all I/J/original/G201/current inputs retained. Old G distances will remain on27 after the separate reviewed literal-path local retirement.')
    proof.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ACTUAL_ALL196_CLOSED_G_ARCHIVE27_SHA_ACK_LOCAL_NOT_REMOVED', total, flush=True)


if __name__ == '__main__':
    main()
