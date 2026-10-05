"""R201B single protected control: native gates, full50, GT49, verified archive."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import archive_frozen, emit, save


NAME = 'RGBNT201_original_shared_ranked_fixed_s42'


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output', 'anchor-run-dir'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--mode', choices=('preflight', 'train'), required=True)
    parser.add_argument('--preflight-root')
    parser.add_argument('--control-root', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        gpus=[2], temperature_power_control=False))
    common = ['--data-root', args.data_root, '--pretrained', args.pretrained, '--anchor-run-dir', args.anchor_run_dir]
    train_command = [sys.executable, '-u', 'run_rgbnt201_ranked_anchor.py', '--dataset', 'RGBNT201',
        '--variant', 'demo_shared', '--freeze-identity-encoder', '1', '--seed', '42', *common]
    if args.mode == 'preflight':
        for phase in ('contract', 'smoke'):
            (root / phase).mkdir()
        execute([sys.executable, '-u', 'verify_rgbnt201_ranked_anchor.py', *common,
            '--output', str(root / 'contract/paired')], root / 'contract', 'paired', 2)
        native = json.loads((root / 'contract/paired/result.json').read_text())
        assert native['status'] == 'PASS_RGBNT201_RANKED_ANCHOR_TWO_NATIVE_UPDATES'
        assert native['actual_optimizer_updates'] == 2 and native['amp_skipped_steps'] == 0
        assert native['matched_initial_state_tensors_exact'] and native['new_weight_files'] == 0
        execute(train_command + ['--mode', 'smoke', '--output', str(root / 'smoke' / NAME)],
            root / 'smoke', NAME, 2)
        smoke = json.loads((root / 'smoke' / NAME / 'smoke.json').read_text())
        assert smoke['status'] == 'SMOKE_PASS' and smoke['steps'] == smoke['attempts'] == 3
        assert smoke['amp_skipped_steps'] == 0 and smoke['strict_reload_equal'] and all(smoke['gradients'].values())
        assert (smoke['train_records'], smoke['query_records'], smoke['gallery_records']) == (3951, 836, 836)
        assert smoke['training_heldout_identities'] == 0
        assert all(d['rank_full_valid_queries'] > 0 and d['rank_partial_valid_queries'] > 0 for d in smoke['details'])
        assert (smoke['parameters'], smoke['trainable_parameters']) == (
            native['controls']['1']['parameters'], native['controls']['1']['trainable_parameters'])
        assert not list(root.rglob('*.pth'))
        save(root / 'preflight_result.json', dict(status='PASS_RGBNT201_RANKED_NATIVE2_SMOKE3',
            native=native, smoke=smoke, actual_native_updates=2, actual_smoke_updates=3,
            amp_skipped_steps=0, full50_started=0, new_weight_files=0))
        return
    previous = Path(args.preflight_root)
    preflight = json.loads((previous / 'preflight_result.json').read_text())
    assert preflight['status'] == 'PASS_RGBNT201_RANKED_NATIVE2_SMOKE3'
    assert preflight['actual_native_updates'] == 2 and preflight['actual_smoke_updates'] == 3
    assert preflight['amp_skipped_steps'] == 0 and preflight['full50_started'] == 0
    native_args = json.loads((previous / 'controller_launch.json').read_text())['arguments']
    assert all(native_args[key] == vars(args)[key] for key in ('data_root', 'pretrained', 'anchor_run_dir', 'control_root'))
    for phase in ('training', 'frozen49', 'audit'):
        (root / phase).mkdir()
    run, frozen = root / 'training' / NAME, root / 'frozen49' / NAME
    trained = execute(train_command + ['--mode', 'train', '--output', str(run)], root / 'training', NAME, 2)
    result = json.loads((run / 'result.json').read_text())
    assert result['status'] == 'COMPLETE' and result['epochs'] == 50
    assert result['steps'] == result['optimizer_steps'] == 2647 and result['amp_skipped_steps'] == 0
    assert result['training_coverage'] == dict(eligible=3951, visited=3951, unvisited=[])
    assert result['training_heldout_identities'] == 0 and result['descriptor_dim'] == 5632
    control = Path(args.control_root) / 'training/RGBNT201_original_shared_fixed_s42'
    baseline = json.loads((control / 'result.json').read_text())
    assert (result['parameters'], result['trainable_parameters']) == (baseline['parameters'], baseline['trainable_parameters'])
    def orders(path):
        rows = [json.loads(line) for line in (path / 'batch_orders.jsonl').read_text().splitlines()]
        return [(r['epoch'], r['step'], r['names'], r['partial_set']) for r in rows]
    assert len(orders(run)) == 2647 and orders(run) == orders(control)
    rows = [json.loads(line) for line in (run / 'batch_orders.jsonl').read_text().splitlines()]
    save(run / 'ranking_coverage.json', dict(batches=len(rows),
        full_valid_queries=sum(r['rank_full_valid_queries'] for r in rows),
        partial_valid_queries=sum(r['rank_partial_valid_queries'] for r in rows),
        full_zero_positive_batches=sum(r['rank_full_valid_queries'] == 0 for r in rows),
        partial_zero_positive_batches=sum(r['rank_partial_valid_queries'] == 0 for r in rows)))
    evaluated = execute([sys.executable, '-u', 'evaluate_rgbnt201_original_anchor49.py',
        '--run-dir', str(run), '--output', str(frozen)], root / 'frozen49', NAME, 2)
    audited = execute([sys.executable, '-u', 'audit_full_official49.py', '--run-dir', str(run),
        '--evaluation', str(frozen)], root / 'audit', NAME, 2)
    checked = json.loads((frozen / 'independent_cpu_audit.json').read_text())
    assert checked['status'] == 'PASS' and checked['cases'] == 49
    archive_frozen(root, frozen, NAME)
    emit(dict(event='RUN_COMPLETE', job=NAME, completed_epochs=50, frozen_conditions=49))
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=[dict(name=NAME, gpu=2,
        train=trained, evaluation=evaluated, audit=audited)], frozen_metric_cases=49,
        paired_identity_and_partial_sampling_exact=True, training_heldout_identities=0,
        all49_raw_local_verified=True, limits='One objective factor; staged50+50, seed42, benchmark-selected. No dual-axis or three-dataset success claim.'))
    emit(dict(event='CONTROLLER_COMPLETE', runs=1, frozen_cases=49))


if __name__ == '__main__':
    main()
