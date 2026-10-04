import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

root = Path('C:/Users/gb/projects/demo_dual_axis_20261002/results/measurement_gate_m3a_20261004')
audit = json.loads((root / 'independent_cpu_audit.json').read_text(encoding='utf-8'))
assert audit['status'] == 'PASS' and audit['cases'] == 3087
report = {}
for variant in ('axis_shared', 'frequency_shared', 'twins_shared'):
    name = 'MSVR310_' + variant + '_s42'
    utility = root / 'original_mean/utility' / name / 'full'
    states = root / 'original_mean/controlled_states' / name / 'full'
    availability = {}
    for qset in ('R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT'):
        with (utility / ('scale_' + qset + '.csv')).open(encoding='utf-8', newline='') as f:
            scalars = list(csv.DictReader(f))
        assert len(scalars) == 360
        with (states / ('q_' + qset + '_g_RNT') / 'contributions.csv').open(encoding='utf-8', newline='') as f:
            predictions = list(csv.DictReader(f))
        assert len(predictions) == 210
        channels = {}
        for channel in ('M', 'F', 'I'):
            actual = [float(scalars[int(r['query_index'])]['gate_' + channel]) for r in predictions]
            implied = [1 / (1 + math.exp(-20 * float(r['prediction_' + channel]))) for r in predictions]
            max_error = max(abs(a - b) for a, b in zip(actual, implied))
            assert max_error < 1e-7
            channels[channel] = dict(actual_gate_mean=statistics.mean(actual), actual_gate_std=statistics.pstdev(actual),
                actual_gate_min=min(actual), actual_gate_max=max(actual), max_error_vs_sigmoid20_prediction=max_error)
        availability[qset] = channels
    report[variant] = availability
target = root / 'actual_gate_summary.json'
assert not target.exists()
target.write_text(json.dumps(dict(status='ACTUAL_EXPORTED_GATES_MATCH_SIGMOID20_PREDICTIONS',
    at=datetime.now().isoformat(timespec='seconds'), runs=report, reference_audit_sha256=hashlib.sha256((root / 'independent_cpu_audit.json').read_bytes()).hexdigest(),
    comparisons=63, query_rows_per_comparison=210, neural_executions=0, gate_changes=0,
    scope='Actual 360-row per-availability utility scalar CSVs joined by installed-GT query bank indices; comparison against the same query predictor exports. Gallery availability does not change the sample gate.'), indent=2) + '\n', encoding='utf-8')
entry = root / 'actual_gate_audit_entrypoint.py'
assert not entry.exists()
entry.write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(status='ACTUAL_GATES_VERIFIED_63_CHANNEL_AVAILABILITY_COMPARISONS',normal={v:r['RNT'] for v,r in report.items()})), flush=True)
