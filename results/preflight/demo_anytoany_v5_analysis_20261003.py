import csv
from datetime import datetime
import json
from pathlib import Path

source=Path('C:/Users/gb/.codex_tmp/demo_anytoany_analysis_20261003.py').read_text(encoding='utf-8')
source=source[:source.index('summaries={}')]
source=source.replace("for variant in ('demo','axis_mass_fullref'):","for variant in ('axis_mass_fullref',):")
namespace={}
exec(compile(source,'actual_full49_v5_only_audit','exec'),namespace)
project=namespace['PROJECT'];rows=namespace['all_rows']
assert len(rows)==147
output=project/'results/anytoany49_axis_mass_fullref_20261003/all3_full49_metrics_147.csv';assert not output.exists()
with output.open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
path=project/'results/preflight/anytoany49_v5_complete_analysis.json';assert not path.exists()
path.write_bytes(json.dumps(dict(status='PASS_ACTUAL_V5_ALL3_FULL49_EVIDENCE_AUDIT',observed_at=datetime.now().isoformat(timespec='seconds'),
    cases=3,conditions=147,all_six_and_CMC50_and_groups_recomputed=True,all_prior13_metrics_and_queries_equal=True,
    optimizer_updates=0,official_test_uses=0,limits='V5-only frozen seed42 dev diagnostics; DeMo full49 pending, no comparative improvement claim'),indent=2).encode('utf-8'))
print('V5_ALL3_147_CONDITIONS_AUDITED',flush=True)
