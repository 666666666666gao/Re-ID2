"""Move verified frozen distances to local D; keep best weights and text results."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, remote_python

analysis = json.loads((PROJECT / 'results/full_official_baselines_20261004/baseline_pair_analysis.json').read_text(encoding='utf-8'))
assert analysis['status'] == 'THREE_COMPLETE_OFFICIAL_BASELINE_PAIRS_ALL294_CASES_ANALYZED'
source = '/data/gaob/Re-ID/DeMo-DualAxis/runs/full_official_baselines_20261004'
local = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/full_official_baselines_20261004/raw_distances')
proof = PROJECT / 'results/preflight/full_official_baseline_distances_local_archive_20261004.json'
assert not local.exists() and not proof.exists()
manifest = json.loads(remote_python('2026', f'''import hashlib,json,os
from pathlib import Path
root=Path({source!r}).resolve()
assert root.is_relative_to(Path('/data/gaob/Re-ID/DeMo-DualAxis/runs').resolve())
done=json.loads((root/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and len(done['runs'])==6 and done['paired_identity_sampling_exact']
result={{}}
for row in done['runs']:
 name=row['dataset']+'_'+row['variant']+'_s42';folder=root/'frozen49'/name
 audit=json.loads((folder/'independent_cpu_audit.json').read_text())
 assert audit['status']=='PASS' and audit['cases']==49
 for phase in ('training','frozen49','audit'):
  assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
 files=sorted(folder.glob('q_*_g_*.npz'))
 assert len(files)==49 and {{p.stem for p in files}}==set(audit['conditions'])
 result[name]={{p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}}
print(json.dumps(result))
'''))
assert len(manifest) == 6 and sum(len(files) for files in manifest.values()) == 294
total_bytes = sum(p['bytes'] for files in manifest.values() for p in files.values())
assert shutil.disk_usage('D:/').free > total_bytes
local.mkdir(parents=True)
record = dict(status='LOCAL_DISTANCE_ARCHIVE_IN_PROGRESS', started_at=datetime.now().isoformat(timespec='seconds'),
    source=source, local_root=str(local), files=manifest, total_bytes=total_bytes, completed_runs=[],
    removed_primary_bytes=0, neural_execution=0,
    preserved='All six best.pth, best_official_arrays.npz, source, JSON/CSV/logs and independent CPU audits remain on2026.',
    note='Distances are retained byte-for-byte locally; only matching server copies are removed after SHA256 verification.')
for path in (proof, local / 'archive_manifest.json'):
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('LOCAL_DISTANCE_ARCHIVE_STARTED', total_bytes, flush=True)
for name, expected in manifest.items():
    destination = local / name
    destination.mkdir()
    subprocess.run(['scp', *OPTIONS, '2026:' + source + '/frozen49/' + name + '/q_*_g_*.npz', str(destination)], check=True, timeout=1800)
    actual = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in destination.glob('*.npz')}
    assert actual == expected
    deleted = json.loads(remote_python('2026', f'''import hashlib,json,os
from pathlib import Path
root=Path({source!r}).resolve();folder=(root/'frozen49'/{name!r}).resolve()
assert folder.is_relative_to(root/'frozen49') and folder.name=={name!r}
expected={expected!r};files=sorted(folder.glob('q_*_g_*.npz'))
actual={{p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}}
assert actual==expected and all(p.resolve().parent==folder for p in files)
assert json.loads((folder/'independent_cpu_audit.json').read_text())['status']=='PASS'
for p in files:p.unlink()
assert not list(folder.glob('*.npz'))
assert [p.name for p in (root/'training'/{name!r}).glob('*.pth')]==['best.pth']
usage=os.statvfs(root)
print(json.dumps(dict(removed_bytes=sum(p['bytes'] for p in expected.values()),free_bytes=usage.f_bavail*usage.f_frsize)))
'''))
    record['completed_runs'].append(name)
    record['removed_primary_bytes'] += deleted['removed_bytes']
    record['primary_free_bytes'] = deleted['free_bytes']
    record['updated_at'] = datetime.now().isoformat(timespec='seconds')
    for path in (proof, local / 'archive_manifest.json'):
        path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('LOCAL_RUN_VERIFIED_AND_PRIMARY_COPY_CLEARED', name, deleted['removed_bytes'], flush=True)
assert len(record['completed_runs']) == 6 and record['removed_primary_bytes'] == total_bytes
record['status'] = 'ALL294_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'
record['completed_at'] = datetime.now().isoformat(timespec='seconds')
for path in (proof, local / 'archive_manifest.json'):
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('ALL_DISTANCE_FILES_LOCAL_ARCHIVED', json.dumps(dict(files=294, bytes=total_bytes, server_free_bytes=record['primary_free_bytes'], local_root=str(local))), flush=True)
