"""Two finite full50 RGBNT100 B128 baselines, one per physical GPU2/3."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute

CASES = (dict(heads=8, gpu=2, parameters=97952256),
         dict(heads=32, gpu=3, parameters=97780224))


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    for key in ('previous-intake', 'previous-audit', 'data-root', 'pretrained', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    prior = json.loads(Path(args.previous_intake).read_text())
    audit = json.loads(Path(args.previous_audit).read_text())
    assert prior['status'] == 'ACTUAL_ONE_VEHICLE_HEAD_FULL50_GT_COMPONENT_ARCHIVED' and prior['dataset'] == 'RGBNT100'
    assert audit['status'] == 'FINAL' and audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
    assert audit['request'].endswith('vehicle_atmoe_head_RGBNT100_result_integrity_request_20261016.json')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('native', 'training', 'audit'):
        (root/phase).mkdir()
    save(root/'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        cases=CASES, physical_gpus=[2, 3], maximum_NN=2, temperature_power_control=False,
        scope='Finite heads8/32, B128/K16/seed42, full50 from public CLIP. Two baselines, not Ours. '
            'No new architecture/loss/seed search/missing/201. Actual AMP counts, no automatic retries.'))

    def run(case, mode):
        name = 'RGBNT100_B128_atmoe'+str(case['heads'])+'_demo_s42'+('_smoke' if mode == 'smoke' else '')
        phase = 'native' if mode == 'smoke' else 'training'
        out = root/phase/name
        ended = execute([sys.executable, '-u', 'run_rgbnt100_paper_batch_baseline.py',
            '--atmoe-heads', str(case['heads']), '--dataset', 'RGBNT100', '--variant', 'demo',
            '--seed', '42', '--mode', mode, '--data-root', args.data_root, '--pretrained', args.pretrained,
            '--output', str(out)], root/phase, name, case['gpu'])
        return name, out, ended

    def verify(case, data):
        assert data['arguments']['atmoe_heads'] == case['heads'] and data['arguments']['training_batch'] == 128
        assert data['arguments']['dataset'] == 'RGBNT100' and data['arguments']['variant'] == 'demo' and data['arguments']['seed'] == 42
        assert data['parameters'] == case['parameters'] and data['descriptor_dim'] == 5120
        assert (data['train_records'], data['query_records'], data['gallery_records'], data['classes']) == (8675, 1715, 8575, 50)
        assert data['training_heldout_identities'] == 0

    def native(case):
        _, out, ended = run(case, 'smoke')
        data = json.loads((out/'smoke.json').read_text())
        verify(case, data)
        assert data['status'] == 'SMOKE_PASS' and data['steps'] == 3
        assert len(data['details']) == data['attempts'] and sum(row['optimizer_updated'] for row in data['details']) == 3
        assert data['amp_skipped_steps'] == data['attempts']-3
        assert all(isinstance(row['optimizer_updated'], bool) and len(row['names']) == 128 for row in data['details'])
        assert all(math.isfinite(row['loss']) and math.isfinite(row['amp_scale']) and row['amp_scale'] > 0 for row in data['details'])
        assert data['strict_reload_equal'] and all(data['gradients'].values())
        assert data['descriptor_shape'] == [8, 5120] and not list(out.glob('*.pth'))
        return dict(heads=case['heads'], result=data, exit=ended)

    with ThreadPoolExecutor(max_workers=2) as pool:
        natives = list(pool.map(native, CASES))
    common = min(len(row['result']['details']) for row in natives)
    assert [row['names'] for row in natives[0]['result']['details'][:common]] == [row['names'] for row in natives[1]['result']['details'][:common]]
    save(root/'native_acceptance.json', dict(status='PASS', natives=natives, actual_updates=6,
        attempts=sum(row['result']['attempts'] for row in natives),
        amp_skips=sum(row['result']['amp_skipped_steps'] for row in natives),
        strict_reload_equal=True, all_active_gradients=True, common_native_sample_prefix_equal=True,
        batch=128, P=8, K=16, physical_gpus=[2, 3], maximum_NN=2))
    print('RGBNT100_PAPER_BATCH_NATIVE_PASS', flush=True)

    def train(case):
        name, out, ended = run(case, 'train')
        data = json.loads((out/'result.json').read_text())
        verify(case, data)
        assert data['status'] == 'COMPLETE' and data['epochs'] == 50
        assert 0 < data['optimizer_steps'] <= data['steps'] and data['amp_skipped_steps'] == data['steps']-data['optimizer_steps']
        assert data['training_coverage'] == dict(eligible=8675, visited=8675, unvisited=[])
        checked = execute([sys.executable, '-u', 'audit_rgbnt100_paper_batch_normal.py', '--run-dir', str(out)], root/'audit', name, case['gpu'])
        report = json.loads((out/'normal_cpu_audit.json').read_text())
        assert report['status'] == 'PASS' and report['batch_size'] == 128
        assert (report['attempts'], report['successful_optimizer_updates'], report['amp_skips']) == (data['steps'], data['optimizer_steps'], data['amp_skipped_steps'])
        files = {}
        for key, filename in (('raw', 'best_official_arrays.npz'), ('best', 'best.pth')):
            file = out/filename
            files[key] = dict(path=str(file), bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        return dict(name=name, heads=case['heads'], gpu=case['gpu'], train=ended, audit=checked,
            selected_epoch=data['best']['epoch'], full_metrics=data['full_metrics'],
            attempts=data['steps'], successful_updates=data['optimizer_steps'], amp_skips=data['amp_skipped_steps'], **files,
            paper_target=dict(mAP=87.2, Rank_1=98.6),
            paper_both_plus1=data['full_metrics']['mAP'] >= 87.2 and data['full_metrics']['Rank-1'] >= 98.6,
            interpretation='Fresh public-CLIP identity baseline, no dual-axis attribution. B128 changes P and update budget; benchmark selected one checkpoint.')

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(train, CASES))
    orders = []
    for row in results:
        batches = [json.loads(line) for line in (root/'training'/row['name']/'batch_orders.jsonl').read_text().splitlines()]
        orders.append([(row['epoch'], row['step'], row['names']) for row in batches])
    assert orders[0] == orders[1]
    save(root/'controller_result.json', dict(status='COMPLETE', dataset='RGBNT100', runs=results, cases=2,
        native_updates=6, formal_epochs=100, formal_attempts=sum(row['attempts'] for row in results),
        formal_optimizer_updates=sum(row['successful_updates'] for row in results),
        formal_amp_skips=sum(row['amp_skips'] for row in results), paired_sampling_exact=True,
        full_training_coverage=True, normal_GT_audited=True, physical_gpus=[2, 3], maximum_NN=2,
        missing_evaluated=False, new49=False, raw_preserved_pending_local_archive=True,
        limits='Two finite existing-DeMo configurations, not expert gains. Same B128/P8K16/50epochs/seed42; '
            'all six metrics from own earliest mAP-best. Historical B64 comparison changes P and actual update count; no equal-update claim.'))
    print('RGBNT100_PAPER_BATCH_BASELINES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
