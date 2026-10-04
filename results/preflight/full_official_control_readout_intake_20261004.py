"""Collect text only after the two frozen readouts and1176 installed-GT cases pass."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

source = HOSTS['2026'][0] + '/runs/full_official_control_readout_20261004'
destination = PROJECT / 'results/full_official_control_readout_20261004'
archive = Path(__file__).with_suffix('.tar.gz')
assert not destination.exists() and not archive.exists()
terminal = json.loads((PROJECT / 'results/preflight/full_official_control_readout_primary_observer_terminal_20261004.json').read_text(encoding='utf-8'))
assert terminal['controller_complete'] and terminal['metric_cases'] == 1176
assert not terminal['active_neural_processes'] and not terminal['failures']
packed = json.loads(remote_python('2026', f'''import json,tarfile
from pathlib import Path
root=Path({source!r});done=json.loads((root/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and len(done['runs'])==2
assert done['metric_cases']==1176 and done['repeated_condition_query_rows']==695016
assert done['training_heldout_identities']==done['optimizer_updates']==done['new_weights']==0
assert json.loads((root/'preflight_result.json').read_text())['status']=='PASS'
for mode in ('measurement_only','independent_control'):
 smoke=json.loads((root/'preflight'/mode/'smoke.json').read_text())
 result=json.loads((root/'frozen'/mode/'result.json').read_text())
 audit=json.loads((root/'frozen'/mode/'independent_cpu_audit.json').read_text())
 assert smoke['status']=='PASS' and smoke['normal_feature_max_error']==0
 assert result['status']=='COMPLETE' and result['metric_cases']==588
 assert result['previous_all49_deployed_metrics_perquery_exact'] and result['state_tensor_versions_unchanged']
 assert result['optimizer_updates']==result['new_weights']==0
 assert audit['status']=='PASS_FULL_OFFICIAL_READOUT_INSTALLED_GT' and audit['cases']==588
 assert audit['repeated_condition_query_rows']==347508 and audit['max_sixmetric_error_pp']<1e-8
 assert len(result['measurements'])==len(audit['conditions'])==49 and len(result['readouts'])==12
 for phase in ('preflight','frozen','audit'):
  assert json.loads((root/phase/(mode+'_exit.json')).read_text())['exit_code']==0
files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.log')]
archive=root.parent/'full_official_control_readout_20261004_text.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
 for path in files:out.add(path,arcname=path.relative_to(root).as_posix())
print(json.dumps(dict(files=len(files),archive=str(archive),bytes=archive.stat().st_size)))
'''))
command(['scp', *OPTIONS, '2026:' + packed['archive'], str(archive)])
assert archive.stat().st_size == packed['bytes']
destination.mkdir()
with tarfile.open(archive) as handle:
    handle.extractall(destination, filter='data')
receipt = dict(status='BOTH_FROZEN_READOUTS_AND1176_CPU_CASES_TEXT_COLLECTED',
    collected_at=datetime.now().isoformat(timespec='seconds'), source=source, models=2,
    metric_cases=1176, repeated_condition_query_rows=695016, training_updates=0, new_weights=0,
    text_files=packed['files'], text_archive_bytes=packed['bytes'],
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), binary_files_transferred=0,
    limits='Frozen single-seed MSVR stage diagnostics only; no new training or independent neural regeneration.')
(destination / 'intake_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
(destination / 'intake_entrypoint.py').write_bytes(Path(__file__).read_bytes())
print('FULL_OFFICIAL_READOUT_TEXT_COLLECTED', json.dumps(receipt), flush=True)
