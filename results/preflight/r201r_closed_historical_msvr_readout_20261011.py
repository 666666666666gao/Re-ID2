"""Compare closed MSVR normal runs; no neural calls or current P reads."""
from datetime import datetime
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    'I': ('r201i_primary_margin_20261006', 'margin.3, broadcast PI, historical sampler'),
    'K': ('r201k_relation_local_pi_20261007', 'margin.3, relation-local PI, historical sampler'),
    'L': ('r201l_uniform_k8_20261007', 'margin.3, relation-local PI, P8K8'),
    'M': ('r201m_primary_margin06_20261008', 'margin.6, relation-local PI, P8K8'),
    'Q': ('r201q_task_gate_20261011', 'margin.6, M/F auxiliary.01, contribution regression0, final CE.25'),
    'R': ('r201r_primary_ce_20261011', 'same Q, final full CE.05 only'),
}
baseline_map = 47.16468844722773
baseline_r1 = 63.95939086294417
rows, inputs = [], {}
for stage, (folder, factors) in SPECS.items():
    for variant in ('frequency_shared', 'axis_shared'):
        name = f'MSVR310_r201{stage.lower()}_{variant}_s42'
        path = ROOT / 'results' / folder / 'MSVR310' / 'training' / name / 'result.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['status'] == 'COMPLETE' and data['epochs'] == 50
        assert data['optimizer_steps'] == data['steps'] and data['amp_skipped_steps'] == 0
        assert data['training_heldout_identities'] == 0 and data['descriptor_dim'] == 5120
        metrics = data['full_metrics']
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(stage=stage, variant=variant, factors=factors, epochs=50,
                         optimizer_updates=data['optimizer_steps'], best_epoch=data['best']['epoch'],
                         mAP=metrics['mAP'], Rank1=metrics['Rank-1'],
                         delta_mAP=metrics['mAP'] - baseline_map,
                         delta_Rank1=metrics['Rank-1'] - baseline_r1,
                         both_plus1=metrics['mAP'] >= baseline_map + 1 and metrics['Rank-1'] >= baseline_r1 + 1))
assert len(rows) == 12
output = ROOT / 'results/r201r_closed_diagnostics_20261011/historical_msvr_normal.csv'
assert not output.exists()
with output.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
proof = dict(status='ACTUAL_CLOSED_HISTORICAL_MSVR_NORMAL_CPU_COMPARISON',
             at=datetime.now().astimezone().isoformat(timespec='seconds'), rows=len(rows),
             inputs_sha256=inputs, csv=output.relative_to(ROOT).as_posix(),
             csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
             neural_calls=0, optimizer_updates=0, current_P_progress_reads=0,
             all_both_plus1=False,
             limits='Historical I/K have705 updates, L/M/Q/R1000; factor bundles differ. '
                    'No cross-stage causal attribution. Repeating old margin.3 alone is not an untried proposal; '
                    'a new interaction with the current R recipe would require its own matched experiment.')
assert not any(row['both_plus1'] for row in rows)
target = ROOT / 'results/preflight/r201r_closed_historical_msvr_readout_actual_20261011.json'
assert not target.exists()
target.write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
print(json.dumps(proof))
