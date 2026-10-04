"""Archive98 audited raw distance files locally before removing their server copies."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, remote_python

intake = json.loads((PROJECT / 'results/full_official_control_readout_20261004/intake_receipt.json').read_text(encoding='utf-8'))
assert intake['status'] == 'BOTH_FROZEN_READOUTS_AND1176_CPU_CASES_TEXT_COLLECTED' and intake['metric_cases'] == 1176
source = '/data/gaob/Re-ID/DeMo-DualAxis/runs/full_official_control_readout_20261004'
local = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/full_official_control_readout_20261004/raw_distances')
proof = PROJECT / 'results/preflight/full_official_control_readout_distances_local_archive_20261004.json'
assert not local.exists() and not proof.exists()
manifest = json.loads(remote_python('2026', f'''import hashlib,json
from pathlib import Path
root=Path({source!r}).resolve()
assert root.is_relative_to(Path('/data/gaob/Re-ID/DeMo-DualAxis/runs').resolve())
done=json.loads((root/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and done['metric_cases']==1176 and done['optimizer_updates']==done['new_weights']==0
result={{}}
for mode in ('measurement_only','independent_control'):
 folder=root/'frozen'/mode
 audit=json.loads((folder/'independent_cpu_audit.json').read_text())
 assert audit['status']=='PASS_FULL_OFFICIAL_READOUT_INSTALLED_GT' and audit['cases']==588
 for phase in ('preflight','frozen','audit'):
  assert json.loads((root/phase/(mode+'_exit.json')).read_text())['exit_code']==0
 files=sorted(folder.glob('q_*_g_*/raw.npz'))
 assert len(files)==49 and {{p.parent.name for p in files}}==set(audit['conditions'])
 result[mode]={{p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}}
print(json.dumps(result))
'''))
assert len(manifest) == 2 and sum(len(files) for files in manifest.values()) == 98
total_bytes = sum(value['bytes'] for files in manifest.values() for value in files.values())
assert shutil.disk_usage('D:/').free > total_bytes
local.mkdir(parents=True)
record = dict(status='LOCAL_READOUT_DISTANCE_ARCHIVE_IN_PROGRESS', started_at=datetime.now().isoformat(timespec='seconds'),
    source=source, local_root=str(local), files=manifest, distance_files=98, total_bytes=total_bytes,
    completed_modes=[], removed_primary_bytes=0, neural_execution=0,
    preserved='All source best weights, official feature arrays, source code, text and CPU-audit receipts remain on2026.',
    note='Each raw file contains12 directed/same-stage distance matrices. Only the98 matching raw.npz files are removed after local SHA256/size verification; all other files remain.')
for path in (proof, local / 'archive_manifest.json'):
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('LOCAL_READOUT_DISTANCE_ARCHIVE_STARTED', total_bytes, flush=True)
for mode, expected in manifest.items():
    destination = local / 'frozen'
    destination.mkdir(exist_ok=True)
    subprocess.run(['scp', *OPTIONS, '-r', '2026:' + source + '/frozen/' + mode, str(destination)], check=True, timeout=1800)
    copied = sorted((destination / mode).glob('q_*_g_*/raw.npz'))
    actual = {p.relative_to(local).as_posix(): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in copied}
    assert actual == expected
    deleted = json.loads(remote_python('2026', f'''import hashlib,json,os
from pathlib import Path
root=Path({source!r}).resolve();mode={mode!r};expected={expected!r}
folder=root/'frozen'/mode
files=sorted(folder.glob('q_*_g_*/raw.npz'))
actual={{p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}}
assert actual==expected and len(files)==49 and all(p.resolve().is_relative_to(folder) for p in files)
assert json.loads((folder/'independent_cpu_audit.json').read_text())['status']=='PASS_FULL_OFFICIAL_READOUT_INSTALLED_GT'
reported=json.loads((folder/'result.json').read_text())
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in reported['original_inputs'].items())
for p in files:p.unlink()
assert not list(folder.glob('q_*_g_*/raw.npz'))
usage=os.statvfs(root)
print(json.dumps(dict(removed_bytes=sum(p['bytes'] for p in expected.values()),free_bytes=usage.f_bavail*usage.f_frsize)))
'''))
    record['completed_modes'].append(mode)
    record['removed_primary_bytes'] += deleted['removed_bytes']
    record['primary_free_bytes'] = deleted['free_bytes']
    record['updated_at'] = datetime.now().isoformat(timespec='seconds')
    for path in (proof, local / 'archive_manifest.json'):
        path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('LOCAL_READOUT_MODE_VERIFIED_AND_PRIMARY_CLEARED', mode, deleted['removed_bytes'], flush=True)
assert len(record['completed_modes']) == 2 and record['removed_primary_bytes'] == total_bytes
record['status'] = 'ALL98_READOUT_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'
record['completed_at'] = datetime.now().isoformat(timespec='seconds')
for path in (proof, local / 'archive_manifest.json'):
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('ALL_READOUT_DISTANCE_FILES_LOCAL_ARCHIVED', json.dumps(dict(files=98, bytes=total_bytes, server_free_bytes=record['primary_free_bytes'], local_root=str(local))), flush=True)
