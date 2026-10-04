"""One text-only intake after six full-data baselines and all294 GT audits finish."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

root, _ = HOSTS['2026']
source = root + '/runs/full_official_baselines_20261004'
destination = PROJECT / 'results/full_official_baselines_20261004'
assert not destination.exists()
code = f'''import json,tarfile
from pathlib import Path
root=Path({source!r})
terminal=json.loads((root/'controller_result.json').read_text())
assert terminal['status']=='COMPLETE' and len(terminal['runs'])==6 and terminal['paired_identity_sampling_exact']
files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.log')]
archive=root.parent/'full_official_baselines_20261004_text.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
 for path in files:out.add(path,arcname=str(path.relative_to(root)))
print(json.dumps(dict(files=len(files),archive=str(archive),bytes=archive.stat().st_size)))
'''
packed = json.loads(remote_python('2026', code))
archive = Path(__file__).with_suffix('.tar.gz')
assert not archive.exists()
command(['scp', *OPTIONS, '2026:' + packed['archive'], str(archive)])
destination.mkdir()
with tarfile.open(archive) as tar:
    tar.extractall(destination, filter='data')
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
summary = {}
for dataset in ('MSVR310', 'RGBNT201', 'RGBNT100'):
    references = {}
    for variant in ('demo', 'demo_shared'):
        run_name = dataset + '_' + variant + '_s42'
        run = json.loads((destination / 'training' / run_name / 'result.json').read_text())
        audit = json.loads((destination / 'frozen49' / run_name / 'independent_cpu_audit.json').read_text())
        assert run['status'] == 'COMPLETE' and run['epochs'] == 50 and audit['status'] == 'PASS' and audit['cases'] == 49
        assert run['training_heldout_identities'] == audit['training_heldout_identities'] == 0
        assert run['training_coverage']['unvisited'] == []
        assert all(abs(audit['normal'][m] - run['full_metrics'][m]) < 1e-8 for m in metrics)
        references[variant] = dict(selected_epoch=run['best']['epoch'], normal=audit['normal'],
            all49_equal_condition_mean=audit['equal_condition_mean'],
            full_split_counts=dict(train=run['train_records'], query=run['query_records'], gallery=run['gallery_records']),
            training_coverage=run['training_coverage'], steps=run['steps'], optimizer_steps=run['optimizer_steps'],
            amp_skipped_steps=run['amp_skipped_steps'], parameters=run['parameters'], descriptor_dim=run['descriptor_dim'])
    summary[dataset] = dict(references=references,
        new_full_protocol_original_DeMo_plus2_thresholds={m: references['demo']['normal'][m] + 2 for m in ('mAP', 'Rank-1')},
        augmented_minus_original_normal={m: references['demo_shared']['normal'][m] - references['demo']['normal'][m] for m in metrics})
report = dict(status='SIX_FULL_OFFICIAL_BASELINES_COMPLETE_ALL294_CPU_AUDITED', collected_at=datetime.now().isoformat(timespec='seconds'),
    datasets=summary, cases=294, repeated_condition_query_rows=sum(
        json.loads(p.read_text())['perquery_count'] for p in (destination / 'frozen49').glob('*/independent_cpu_audit.json')),
    paired_identity_sampling_exact=True, source=source, text_files=packed['files'], archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    goal_complete=False, limits='Baseline stage only, benchmark-selected checkpoints, one seed. Old fit/dev scores are not compared to these official scores.')
(destination / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(destination / 'intake_entrypoint.py').write_bytes(Path(__file__).read_bytes())
print('FULL_OFFICIAL_BASELINES_COLLECTED', json.dumps(dict(cases=294, rows=report['repeated_condition_query_rows'], goal_complete=False)), flush=True)
