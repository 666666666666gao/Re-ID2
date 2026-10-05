"""M8's three native/smoke gates on the released final M7 GPU3 lane."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import check_limits, execute
from launch_full_official_frozen_anchor_trial import SCHEDULE, emit, name, save
from run_full_official_frozen_public_identity_experiment import VARIANTS


def check_public_identity(detail):
    assert detail['public_identity_weight'] == .1
    assert detail['public_full_ce'] > 0 and detail['public_partial_ce'] > 0
    assert detail['public_identity_extra_parameters'] == 0
    assert detail['public_identity_extra_backbone_passes'] == 0
    assert detail['public_identity_extra_shared_neck_calls'] == 2
    assert detail['freeze_identity_encoder'] and detail['added_losses'] == 1
    assert detail['optimizer_updated'] and not detail['reference_requires_grad']


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'anchor-run-dir', 'm7-root', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    m7 = Path(args.m7_root)
    last = name('twins_shared', 1)
    assert SCHEDULE[3][-1] == ('twins_shared', 1)
    launched = json.loads((m7 / 'controller_train_launch.json').read_text())
    assert launched['arguments']['anchor_run_dir'] == args.anchor_run_dir
    previous_launch = json.loads((m7 / 'diagnosis' / (last + '_launch.json')).read_text())
    previous_exit = json.loads((m7 / 'diagnosis' / (last + '_exit.json')).read_text())
    assert previous_exit['exit_code'] == 0 and previous_exit['gpu'] == 3
    assert not Path('/proc/' + str(previous_launch['pid'])).exists()
    sample = check_limits(3)
    assert sample['memory_used_mib'] < 500
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('contract', 'preflight'):
        (root / phase).mkdir()
    save(root / 'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        gpu=3, initial_resources=sample, previous_last_neural_exit=previous_exit,
        original_M7_gpu2_reserved=True, temperature_power_control=False, formal_training=False))
    print('M8_NATIVE_CONTROLLER_STARTED', os.getpid(), 'GPU3', flush=True)
    rows = []
    for variant in VARIANTS:
        run = name(variant, 1)
        common = ['--dataset', 'MSVR310', '--variant', variant, '--freeze-identity-encoder', '1',
                  '--seed', '42', '--data-root', args.data_root, '--pretrained', args.pretrained,
                  '--anchor-run-dir', args.anchor_run_dir]
        execute([sys.executable, '-u', 'verify_full_official_frozen_public_identity.py', *common,
                 '--output', str(root / 'contract' / run)], root / 'contract', run, 3)
        contract = json.loads((root / 'contract' / run / 'result.json').read_text())
        assert contract['status'] == 'PASS_FULL_OFFICIAL_ANCHOR_NATIVE_UPDATE_AND_IDENTITY_REFERENCE'
        assert contract['actual_optimizer_updates'] == 1 and contract['amp_skipped_steps'] == 0
        assert contract['strict_reload_equal'] and contract['fixed_encoder_unchanged_after_update_and_reload']
        assert len(contract['initial_all7_base_descriptors_exact']) == 7 and all(contract['initial_all7_base_descriptors_exact'].values())
        assert contract['full_split_counts'] == dict(train=1032, query=591, gallery=1055)
        assert contract['training_heldout_identities'] == 0 and all(contract['active_gradients'].values())
        original = json.loads((m7 / 'contract' / run / 'result.json').read_text())
        assert (contract['parameters'], contract['trainable_parameters']) == (original['parameters'], original['trainable_parameters'])
        assert set(contract['active_gradients']) == set(original['active_gradients'])
        check_public_identity(contract['detail'])
        execute([sys.executable, '-u', 'run_full_official_frozen_public_identity_experiment.py', *common,
                 '--mode', 'smoke', '--output', str(root / 'preflight' / run)], root / 'preflight', run, 3)
        smoke = json.loads((root / 'preflight' / run / 'smoke.json').read_text())
        assert smoke['status'] == 'SMOKE_PASS' and smoke['steps'] == smoke['attempts'] == 3 and smoke['amp_skipped_steps'] == 0
        assert smoke['strict_reload_equal'] and all(smoke['gradients'].values())
        assert smoke['arguments']['public_identity_weight'] == .1
        for detail in smoke['details']:
            check_public_identity(detail)
        assert set(smoke['gradients']) == set(contract['active_gradients'])
        assert (smoke['parameters'], smoke['trainable_parameters']) == (contract['parameters'], contract['trainable_parameters'])
        rows.append(dict(name=run, variant=variant, parameters=contract['parameters'],
            trainable_parameters=contract['trainable_parameters'], active_names=sorted(contract['active_gradients']),
            contract_updates=1, smoke_updates=3, amp_skipped_steps=0, public_identity_weight=.1,
            own_adam_replay_max_error=contract['own_adam_replay_max_error'], initial_all7_exact=True,
            frozen_encoder_exact=True, strict_reload_equal=True))
    assert len(rows) == 3 and all((r['parameters'], r['trainable_parameters'], r['active_names']) ==
        (rows[0]['parameters'], rows[0]['trainable_parameters'], rows[0]['active_names']) for r in rows)
    assert not list(root.rglob('*.pth'))
    save(root / 'preflight_result.json', dict(status='PASS_THREE_M8_FIXED_NATIVE_AND_PUBLIC_CE_SMOKES',
        runs=rows, actual_native_updates=3, actual_smoke_updates=9, amp_skipped_steps=0,
        frozen_metric_cases=0, new_weights=0, formal50_started=0, goal_complete=False,
        limits='Only real native and three-update gates; no completed50, missing49 evaluation or scientific success. Existing M7 continues independently.'))
    emit(dict(event='PREFLIGHT_COMPLETE', runs=3, native_updates=3, smoke_updates=9, formal50_started=0))


if __name__ == '__main__':
    main()
