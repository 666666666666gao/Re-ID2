"""One text-only intake after six fresh50 runs and all1764 installed-GT audits."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

root, _ = HOSTS['2026']
source = root + '/runs/full_official_control_m3b_20261004'
destination = PROJECT / 'results/full_official_control_m3b_20261004'
archive = Path(__file__).with_suffix('.tar.gz')
assert not destination.exists() and not archive.exists()
code = f'''import json,tarfile
from pathlib import Path
root=Path({source!r});done=json.loads((root/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and len(done['runs'])==6 and done['training_heldout_identities']==0
assert done['paired_identity_and_partial_sampling_exact'] and done['all_frozen_state_cpu_audits_passed']
assert done['frozen_state_metric_cases']==1764 and done['frozen_state_repeated_query_rows']==1042524
assert done['contribution_rows']==173754
for row in done['runs']:
 name='MSVR310_'+row['mode']+'_'+row['variant']+'_s42'
 train=json.loads((root/'training'/name/'result.json').read_text())
 audit=json.loads((root/'frozen49'/name/'independent_cpu_audit.json').read_text())
 states=json.loads((root/'diagnosis'/name/'independent_cpu_audit.json').read_text())
 assert train['status']=='COMPLETE' and train['epochs']==50 and train['training_heldout_identities']==0
 assert train['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
 assert audit['status']=='PASS' and audit['cases']==49 and audit['perquery_count']==28959
 assert states['status']=='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and states['cases']==294
 assert states['repeated_condition_query_rows']==173754 and states['calibration_rows']==28959
 for phase in ('training','frozen49','audit','diagnosis','diagnosis_audit'):
  assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.log')]
archive=root.parent/'full_official_control_m3b_20261004_text.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
 for path in files:out.add(path,arcname=str(path.relative_to(root)))
print(json.dumps(dict(files=len(files),archive=str(archive),bytes=archive.stat().st_size)))
'''
packed = json.loads(remote_python('2026', code))
command(['scp', *OPTIONS, '2026:' + packed['archive'], str(archive)])
destination.mkdir()
with tarfile.open(archive) as tar:
    tar.extractall(destination, filter='data')
receipt = dict(status='SIX_FULL_OFFICIAL_CONTROL_RUNS_AND1764_CPU_STATE_AUDITS_TEXT_COLLECTED',
    collected_at=datetime.now().isoformat(timespec='seconds'), source=source,
    runs=6, metric_cases=1764, repeated_condition_query_rows=1042524, contribution_rows=173754,
    text_files=packed['files'], text_archive_bytes=packed['bytes'],
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), binary_files_transferred=0,
    analysis_executed=False, goal_complete=False,
    limits='Completed single-seed full-data MSVR trial intake only. Raw NPZ and best weights remain on2026. '
        'Three-dataset/multiseed/fair-control success still requires the actual result analysis and further experiments.')
(destination / 'intake_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
(destination / 'intake_entrypoint.py').write_bytes(Path(__file__).read_bytes())
print('FULL_OFFICIAL_CONTROL_TRIAL_TEXT_COLLECTED', json.dumps(dict(cases=1764, goal_complete=False)), flush=True)
