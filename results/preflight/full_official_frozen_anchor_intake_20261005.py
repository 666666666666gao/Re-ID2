"""Collect completed M7 text/head evidence after the verified streaming session closes."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, command, remote_python

pf = PROJECT / 'results/preflight'
review = json.loads((pf / 'full_official_frozen_anchor_closeout_review_20261005.json').read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blocking_findings']
for file, digest in review['sources_sha256'].items():
    assert hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest
session = json.loads((pf / 'full_official_frozen_anchor_training_session_20261005.json').read_text(encoding='utf-8'))
assert session['status'] == 'STREAM_SESSION_COMPLETE_ALL_DECLARED_ARCHIVES_VERIFIED' and session['mode'] == 'train'
assert len(session['raw']) == 196 and len(session['frozen']) == 5
root = '/data/gaob/Re-ID/DeMo-DualAxis/runs/full_official_frozen_identity_anchor_m7_20261005'
destination = PROJECT / 'results/full_official_frozen_identity_anchor_m7_20261005'
assert not destination.exists()
archive_path = pf / 'full_official_frozen_anchor_text_intake_20261005.tar.gz'
remote = json.loads(remote_python('2026', f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({root!r});done=json.loads((root/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and len(done['runs'])==5
assert done['frozen_metric_cases']==245 and done['enhanced_state_metric_cases']==1176
assert done['all_frozen_and_state_cpu_audits_passed'] and done['paired_identity_and_partial_sampling_exact']
assert not list((root/'diagnosis').glob('*/q_*_g_*/raw.npz')) and not list((root/'frozen49').glob('*/q_*_g_*.npz'))
selected=[p for p in sorted(root.rglob('*')) if p.is_file() and
 (p.suffix in ('.json','.jsonl','.csv','.log') or p.name.startswith('heads_') and p.suffix=='.npz')]
assert not any(p.name=='best.pth' or p.name=='raw.npz' for p in selected)
files={{str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in selected}}
archive=root/'text_intake.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for p in selected:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
print(json.dumps(dict(files=files,path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
'''))
command(['scp', *OPTIONS, '2026:' + remote['path'], str(archive_path)])
assert archive_path.stat().st_size == remote['bytes']
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == remote['sha256']
destination.mkdir()
with tarfile.open(archive_path) as archive:
    members = archive.getmembers()
    assert {m.name for m in members} == set(remote['files'])
    for member in members:
        assert member.isfile()
        target = (destination / member.name).resolve()
        assert target.is_relative_to(destination.resolve())
    archive.extractall(destination, filter='data')
for relative, expected in remote['files'].items():
    path = destination / relative
    assert path.stat().st_size == expected['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
receipt = dict(status='FIVE_FULL_OFFICIAL_ANCHOR_RUNS245_FROZEN_AND1176_STATE_CASES_TEXT_COLLECTED',
               collected_at=datetime.now().isoformat(timespec='seconds'), frozen_metric_cases=245, enhanced_state_metric_cases=1176,
               files=remote['files'], archive=dict(bytes=remote['bytes'], sha256=remote['sha256']),
               local_raw_stream_session='results/preflight/full_official_frozen_anchor_training_session_20261005.json',
               new_neural_execution=0, new_optimizer_updates=0,
               limits='Full completed text/head receipts only; local stream already verified196 enhanced and245 frozen raw files. No neural replay or metric superiority implied.')
(destination / 'intake_receipt.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode('utf-8'))
remote_python('2026', f'''import hashlib
from pathlib import Path
p=Path({remote['path']!r});root=Path({root!r})
assert p.parent==root and p.name=='text_intake.tar.gz'
assert p.stat().st_size=={remote['bytes']} and hashlib.sha256(p.read_bytes()).hexdigest()=={remote['sha256']!r}
p.unlink();print('VERIFIED_TEXT_TRANSFER_TEMP_ARCHIVE_CLEARED')
''')
print('FIVE_FULL_OFFICIAL_ANCHOR_TEXT_INTAKE_COMPLETE', len(remote['files']), flush=True)
