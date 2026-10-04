"""Six full-data baseline jobs, serialized on each of physical GPU2/3."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from gpu_thermal_execute import execute

DATASETS = ('MSVR310', 'RGBNT201', 'RGBNT100')
SCHEDULE = {2: 'demo', 3: 'demo_shared'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    for name in ('preflight', 'training', 'frozen49', 'audit'):
        (root / name).mkdir()

    def command(dataset, variant):
        return [sys.executable, '-u', 'run_full_official_experiment.py', '--dataset', dataset, '--variant', variant,
            '--seed', '42', '--data-root', args.data_root, '--pretrained', args.pretrained]

    def smokes(gpu):
        variant = SCHEDULE[gpu]
        rows = []
        for dataset in DATASETS:
            name = dataset + '_' + variant + '_s42'
            row = execute(command(dataset, variant) + ['--mode', 'smoke', '--output', str(root / 'preflight' / name)], root / 'preflight', name, gpu)
            smoke = json.loads((root / 'preflight' / name / 'smoke.json').read_text())
            assert smoke['status'] == 'SMOKE_PASS' and smoke['steps'] == 3 and smoke['strict_reload_equal']
            assert smoke['training_heldout_identities'] == 0 and all(smoke['gradients'].values())
            rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(smokes, gpu) for gpu in SCHEDULE]
        checked = [row for future in futures for row in future.result()]
    assert len(checked) == 6
    save(root / 'preflight_result.json', dict(status='PASS', smokes=checked))

    def jobs(gpu):
        variant = SCHEDULE[gpu]
        rows = []
        for dataset in DATASETS:
            name = dataset + '_' + variant + '_s42'
            run, frozen = root / 'training' / name, root / 'frozen49' / name
            trained = execute(command(dataset, variant) + ['--mode', 'train', '--output', str(run)], root / 'training', name, gpu)
            result = json.loads((run / 'result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50 and result['training_heldout_identities'] == 0
            evaluated = execute([sys.executable, '-u', 'evaluate_full_official49.py', '--run-dir', str(run), '--output', str(frozen)], root / 'frozen49', name, gpu)
            audited = execute([sys.executable, '-u', 'audit_full_official49.py', '--run-dir', str(run), '--evaluation', str(frozen)], root / 'audit', name, gpu)
            audit = json.loads((frozen / 'independent_cpu_audit.json').read_text())
            assert audit['status'] == 'PASS' and audit['cases'] == 49
            rows.append(dict(dataset=dataset, variant=variant, gpu=gpu, train=trained, evaluation=evaluated, audit=audited))
            save(root / ('gpu_' + str(gpu) + '_completed.json'), dict(status='IN_PROGRESS', runs=rows))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jobs, gpu) for gpu in SCHEDULE]
        rows = [row for future in futures for row in future.result()]
    assert len(rows) == 6
    for dataset in DATASETS:
        paired = []
        for variant in SCHEDULE.values():
            orders = root / 'training' / (dataset + '_' + variant + '_s42') / 'batch_orders.jsonl'
            with orders.open() as handle:
                paired.append([(r['epoch'], r['step'], r['names']) for r in map(json.loads, handle)])
        assert paired[0] == paired[1], (dataset, 'baseline identity sampling differs')
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=rows, training_heldout_identities=0,
        paired_identity_sampling_exact=True, temperature_power_control=False))


if __name__ == '__main__':
    main()
