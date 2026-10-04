"""Archive audited three-model M4 raw distances locally, then clear matching server copies."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, remote_python

analysis = json.loads((PROJECT / 'results/full_official_modality_outlet_m4_20261005/analysis.json').read_text(encoding='utf-8'))
assert analysis['status'] == 'THREE_FULL_OFFICIAL_M4_RUNS_AND882_STATE_CASES_ANALYZED'
assert analysis['state_metric_cases'] == 882
source = '/data/gaob/Re-ID/DeMo-DualAxis/runs/full_official_modality_outlet_m4_20261005'
local = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/full_official_modality_outlet_m4_20261005/raw_distances')
proof = PROJECT / 'results/preflight/full_official_modality_outlet_distances_local_archive_20261005.json'
assert not local.exists() and not proof.exists()
manifest = json.loads(remote_python('2026', f'''import hashlib,json
from pathlib import Path
root=Path({source!r}).resolve()
assert root.is_relative_to(Path('/data/gaob/Re-ID/DeMo-DualAxis/runs').resolve())
done=json.loads((root/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and len(done['runs'])==3 and done['paired_identity_and_partial_sampling_exact']
assert done['all_frozen_state_cpu_audits_passed'] and done['frozen_state_metric_cases']==882
result={{}}
for row in done['runs']:
 name='MSVR310_'+row['mode']+'_'+row['variant']+'_s42'
 frozen=root/'frozen49'/name;diagnosis=root/'diagnosis'/name
 audit=json.loads((frozen/'independent_cpu_audit.json').read_text())
 states=json.loads((diagnosis/'independent_cpu_audit.json').read_text())
 assert audit['status']=='PASS' and audit['cases']==49
 assert states['status']=='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and states['cases']==294
 assert set(audit['conditions'])==set(states['conditions'])
 for phase in ('training','frozen49','audit','diagnosis','diagnosis_audit'):
  assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
 files=sorted(frozen.glob('q_*_g_*.npz'))+sorted(diagnosis.glob('q_*_g_*/raw.npz'))
 assert len(files)==98
 assert {{p.stem for p in frozen.glob('q_*_g_*.npz')}}==set(audit['conditions'])
 assert {{p.parent.name for p in diagnosis.glob('q_*_g_*/raw.npz')}}==set(states['conditions'])
 result[name]={{p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}}
print(json.dumps(result))
'''))
assert len(manifest) == 3 and sum(len(files) for files in manifest.values()) == 294
total_bytes = sum(p['bytes'] for files in manifest.values() for p in files.values())
assert shutil.disk_usage('D:/').free > total_bytes
local.mkdir(parents=True)
record = dict(status='LOCAL_M4_DISTANCE_ARCHIVE_IN_PROGRESS', started_at=datetime.now().isoformat(timespec='seconds'),
    source=source, local_root=str(local), files=manifest, distance_files=294, total_bytes=total_bytes,
    completed_runs=[], removed_primary_bytes=0, neural_execution=0,
    preserved='All three best.pth, best_official_arrays.npz, source, text results, head arrays and independent CPU audits remain on2026.',
    note='147 frozen49 files and147 raw files containing six-state distances are copied and SHA256 verified before their server copies are removed. Diagnosis text/head files are also copied by scp-r; their originals are retained.')
for path in (proof, local / 'archive_manifest.json'):
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('LOCAL_M4_DISTANCE_ARCHIVE_STARTED', total_bytes, flush=True)
for name, expected in manifest.items():
    frozen_destination = local / 'frozen49' / name
    frozen_destination.mkdir(parents=True)
    diagnosis_destination = local / 'diagnosis'
    diagnosis_destination.mkdir(exist_ok=True)
    subprocess.run(['scp', *OPTIONS, '2026:' + source + '/frozen49/' + name + '/q_*_g_*.npz', str(frozen_destination)], check=True, timeout=1800)
    subprocess.run(['scp', *OPTIONS, '-r', '2026:' + source + '/diagnosis/' + name, str(diagnosis_destination)], check=True, timeout=1800)
    copied = sorted(frozen_destination.glob('q_*_g_*.npz')) + sorted((diagnosis_destination / name).glob('q_*_g_*/raw.npz'))
    actual = {p.relative_to(local).as_posix(): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in copied}
    assert actual == expected
    deleted = json.loads(remote_python('2026', f'''import hashlib,json,os
from pathlib import Path
root=Path({source!r}).resolve();name={name!r}
expected={expected!r}
files=sorted((root/'frozen49'/name).glob('q_*_g_*.npz'))+sorted((root/'diagnosis'/name).glob('q_*_g_*/raw.npz'))
actual={{p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}}
assert actual==expected and len(files)==98
assert all(p.resolve().is_relative_to(root/'frozen49'/name) or p.resolve().is_relative_to(root/'diagnosis'/name) for p in files)
assert json.loads((root/'diagnosis'/name/'independent_cpu_audit.json').read_text())['status']=='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT'
for p in files:p.unlink()
assert not list((root/'frozen49'/name).glob('q_*_g_*.npz')) and not list((root/'diagnosis'/name).glob('q_*_g_*/raw.npz'))
assert [p.name for p in (root/'training'/name).glob('*.pth')]==['best.pth']
usage=os.statvfs(root)
print(json.dumps(dict(removed_bytes=sum(p['bytes'] for p in expected.values()),free_bytes=usage.f_bavail*usage.f_frsize)))
'''))
    record['completed_runs'].append(name)
    record['removed_primary_bytes'] += deleted['removed_bytes']
    record['primary_free_bytes'] = deleted['free_bytes']
    record['updated_at'] = datetime.now().isoformat(timespec='seconds')
    for path in (proof, local / 'archive_manifest.json'):
        path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('LOCAL_M4_RUN_VERIFIED_AND_PRIMARY_CLEARED', name, deleted['removed_bytes'], flush=True)
assert len(record['completed_runs']) == 3 and record['removed_primary_bytes'] == total_bytes
record['status'] = 'ALL294_M4_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'
record['completed_at'] = datetime.now().isoformat(timespec='seconds')
for path in (proof, local / 'archive_manifest.json'):
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('ALL_M4_DISTANCE_FILES_LOCAL_ARCHIVED', json.dumps(dict(files=294, bytes=total_bytes, server_free_bytes=record['primary_free_bytes'], local_root=str(local))), flush=True)
