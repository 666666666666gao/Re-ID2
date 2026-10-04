"""Text-only intake after all three fresh50 M4 models and882 GT state cases complete."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

source = HOSTS['2026'][0] + '/runs/full_official_modality_outlet_m4_20261005'
destination = PROJECT / 'results/full_official_modality_outlet_m4_20261005'
archive = Path(__file__).with_suffix('.tar.gz')
assert not destination.exists() and not archive.exists()
packed = json.loads(remote_python('2026', f'''import json,tarfile
from pathlib import Path
root=Path({source!r});done=json.loads((root/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and len(done['runs'])==3
assert done['training_heldout_identities']==0 and done['paired_identity_and_partial_sampling_exact']
assert done['all_frozen_state_cpu_audits_passed'] and done['frozen_state_metric_cases']==882
assert done['frozen_state_repeated_query_rows']==521262 and done['added_model_parameters']==0
preflight=json.loads((root/'preflight_result.json').read_text())
assert preflight['status']=='PASS' and preflight['actual_optimizer_updates']==9
assert {{r['variant'] for r in done['runs']}}=={{'axis_shared','frequency_shared','twins_shared'}}
for row in done['runs']:
 assert row['mode']=='measurement_only'
 name='MSVR310_measurement_only_'+row['variant']+'_s42'
 train=json.loads((root/'training'/name/'result.json').read_text())
 audit=json.loads((root/'frozen49'/name/'independent_cpu_audit.json').read_text())
 states=json.loads((root/'diagnosis'/name/'independent_cpu_audit.json').read_text())
 assert train['status']=='COMPLETE' and train['epochs']==50 and train['training_heldout_identities']==0
 assert train['arguments']['modality_alignment_weight']==.1 and train['arguments']['gate_gradient_mode']=='measurement_only'
 assert (train['train_records'],train['query_records'],train['gallery_records'])==(1032,591,1055)
 assert train['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
 assert audit['status']=='PASS' and audit['cases']==49 and audit['perquery_count']==28959
 assert states['status']=='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and states['cases']==294
 assert states['repeated_condition_query_rows']==173754 and states['calibration_rows']==28959
 for phase in ('contract','preflight','training','frozen49','audit','diagnosis','diagnosis_audit'):
  assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.log')]
archive=root.parent/'full_official_modality_outlet_m4_20261005_text.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as handle:
 for path in files:handle.add(path,arcname=path.relative_to(root).as_posix())
print(json.dumps(dict(files=len(files),archive=str(archive),bytes=archive.stat().st_size)))
'''))
command(['scp', *OPTIONS, '2026:' + packed['archive'], str(archive)])
assert archive.stat().st_size == packed['bytes']
destination.mkdir()
with tarfile.open(archive) as handle:
    handle.extractall(destination, filter='data')
receipt = dict(status='THREE_FULL_OFFICIAL_M4_RUNS_AND882_GT_STATE_CASES_TEXT_COLLECTED',
    collected_at=datetime.now().isoformat(timespec='seconds'), source=source, models=3,
    state_metric_cases=882, repeated_condition_query_rows=521262, contribution_rows=86877,
    text_files=packed['files'], text_archive_bytes=packed['bytes'],
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), binary_files_transferred=0,
    actual_analysis_executed=False, goal_complete=False,
    limits='Completed single-seed full-data MSVR M4 trial text intake; no NN re-execution. '
        'Three-dataset/multiseed/fair-control and actual inference utility still require evidence. '
        'Raw distances and source best weights remain on2026 pending verified local raw archival.')
(destination / 'intake_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
(destination / 'intake_entrypoint.py').write_bytes(Path(__file__).read_bytes())
print('FULL_OFFICIAL_M4_THREE_RUNS_AND882_CASES_TEXT_COLLECTED', flush=True)
