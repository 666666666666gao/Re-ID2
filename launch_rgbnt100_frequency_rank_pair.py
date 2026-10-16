"""One global AV42 F-width pair after the cutoff-only experiment closes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import save
from rgbnt100_expert_seed_contracts import verify_detail


def verify_model_record(d, key, seed):
    assert key == 'AV' and seed == 42
    assert (d['parameters'], d['trainable_parameters'], d['descriptor_dim']) == (99947398, 2249606, 5120)
    assert (d['train_records'], d['query_records'], d['gallery_records']) == (8675, 1715, 8575)
    assert d['arguments']['seed'] == seed and d['arguments']['expert_recipe'] == key
    assert d['arguments']['frequency_rank'] == 128
    assert d['anchor']['anchor_seed'] == 42 and d['anchor']['expert_seed'] == seed and d['anchor']['selected_recipe'] == key

CONTROLS = (('frequency_shared', 2), ('axis_shared', 3))


def main():
    parser = argparse.ArgumentParser()
    for key in ('previous-controller', 'data-root', 'pretrained', 'anchor-root', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    previous = json.loads(Path(args.previous_controller).read_text())
    assert previous['status'] == 'COMPLETE' and previous['parent_recipe'] == 'AV'
    assert (previous['band_low_cutoff'], previous['band_high_cutoff'], previous['frequency_bin_counts']) == (.25, .375, [25,30,73]) and previous['state_cases'] == 8
    assert previous['seed'] == previous['teacher_seed'] == 42
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('parity', 'native', 'training', 'audit', 'states'):
        (root/phase).mkdir()
    save(root/'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        parent_recipe='AV', seed=42, teacher_seed=42, frequency_rank=128, parameters=99947398, trainable_parameters=2249606,
        physical_gpus=[2, 3], maximum_NN=2, temperature_power_control=False,
        scope='One paired F rank128 configuration on original global AV42/default FFT. Same-seed64 references; no new module, seed sweep or missing evaluation.'))
    anchor = str(Path(args.anchor_root)/'RGBNT100_demo_s42')

    def parity(pair):
        variant, gpu = pair
        name = 'RGBNT100_frank128_AV_'+variant+'_default_parity'
        out = root/'parity'/name
        ended = execute([sys.executable, '-u', 'check_rgbnt100_frequency_rank_parity.py',
            '--variant', variant, '--data-root', args.data_root,
            '--pretrained', args.pretrained, '--anchor-run-dir', anchor, '--output', str(out)],
            root/'parity', name, gpu)
        data = json.loads((out/'result.json').read_text())
        assert data['status'] == 'ACTUAL_F_RANK_DEFAULT_CONSTRUCTOR_OPTIMIZER_RNG_FP32_AMP_FOUR_STATE_PARITY_PASS'
        assert data['state_tensors_exact'] and data['CPU_CUDA_RNG_exact'] and data['default_optimizer_groups_exact']
        assert data['fp32_four_states_exact'] and data['autocast_four_states_exact']
        assert data['optimizer_updates'] == data['checkpoint_writes'] == 0
        assert data['forward_record_executions'] == 32
        return dict(result=data, exit=ended)

    with ThreadPoolExecutor(max_workers=2) as pool:
        parities = list(pool.map(parity, CONTROLS))

    def run(pair, mode):
        variant, gpu = pair
        name = 'RGBNT100_frank128_AV_'+variant+'_s42'+('_smoke' if mode == 'smoke' else '')
        phase = 'native' if mode == 'smoke' else 'training'
        out = root/phase/name
        ended = execute([sys.executable, '-u', 'run_rgbnt100_frequency_rank.py',
            '--frequency-rank', '128',
            '--dataset', 'RGBNT100', '--variant', variant, '--seed', '42', '--mode', mode,
            '--freeze-identity-encoder', '1', '--data-root', args.data_root,
            '--pretrained', args.pretrained, '--anchor-run-dir', anchor, '--output', str(out)],
            root/phase, name, gpu)
        return name, out, ended

    def native(pair):
        _, out, _ = run(pair, 'smoke')
        data = json.loads((out/'smoke.json').read_text())
        assert data['status'] == 'SMOKE_PASS' and data['steps'] == data['attempts'] == 3
        assert data['amp_skipped_steps'] == 0 and data['strict_reload_equal'] and all(data['gradients'].values())
        assert not list(out.glob('*.pth'))
        verify_model_record(data, 'AV', 42)
        assert data['arguments']['frequency_rank'] == 128
        width = data['anchor']['frequency_hidden_width']
        assert width['rank'] == width['condition_output_rank'] == 128 and width['parameter_delta'] == 362240
        assert width['original_global_readers'] and width['original_cutoffs'] == [.125, .25]
        for detail in data['details']:
            verify_detail(detail, 'AV', True)
            assert detail['frequency_hidden_width'] == width
        return data

    with ThreadPoolExecutor(max_workers=2) as pool:
        natives = list(pool.map(native, CONTROLS))
    assert [(d['names'], d['partial_set']) for d in natives[0]['details']] == [
        (d['names'], d['partial_set']) for d in natives[1]['details']]
    save(root/'native_acceptance.json', dict(status='PASS', parent_recipe='AV', seed=42, frequency_rank=128, parameters=99947398, trainable_parameters=2249606,
        actual_updates=6, parity=parities, parity_forward_record_executions=64,
        strict_reload=True, all_active_gradients=True, paired_sampling=True))
    print('RGBNT100_F_RANK_NATIVE_PASS', flush=True)

    def train(pair):
        name, out, ended = run(pair, 'train')
        data = json.loads((out/'result.json').read_text())
        assert data['status'] == 'COMPLETE' and data['epochs'] == 50
        assert data['steps'] == data['optimizer_steps'] > 0 and data['amp_skipped_steps'] == 0
        assert data['training_heldout_identities'] == 0
        assert data['training_coverage'] == dict(eligible=8675, visited=8675, unvisited=[])
        verify_model_record(data, 'AV', 42)
        width = data['anchor']['frequency_hidden_width']
        assert data['arguments']['frequency_rank'] == 128
        assert width['rank'] == width['condition_output_rank'] == 128 and width['parameter_delta'] == 362240
        rows = [json.loads(line) for line in (out/'batch_orders.jsonl').read_text().splitlines()]
        assert len(rows) == data['steps']
        for detail in rows:
            verify_detail(detail, 'AV', False)
            assert detail['frequency_hidden_width'] == width
        audited = execute([sys.executable, '-u', 'audit_full_official_normal.py', '--run-dir', str(out)],
                          root/'audit', name, pair[1])
        assert json.loads((out/'normal_cpu_audit.json').read_text())['status'] == 'PASS'
        states_out = root/'states'/name
        states = execute([sys.executable, '-u', 'evaluate_rgbnt100_frequency_rank_states.py',
                          '--run-dir', str(out), '--output', str(states_out)], root/'states', name, pair[1])
        frozen = json.loads((states_out/'result.json').read_text())
        assert frozen['status'] == 'COMPLETE' and frozen['state_cases'] == 4
        assert frozen['optimizer_updates'] == frozen['checkpoint_writes'] == 0
        assert (frozen['frequency_rank'], frozen['parameters'], frozen['trainable_parameters']) == (128, 99947398, 2249606)
        files = {}
        for key, file in [('raw', out/'best_official_arrays.npz'), ('best', out/'best.pth'),
                          ('states_raw', states_out/'normal_states.npz')]:
            files[key] = dict(path=str(file), bytes=file.stat().st_size,
                              sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        return dict(name=name, variant=pair[0], gpu=pair[1], train=ended, audit=audited, states=states,
                    successful_updates=data['steps'], selected_epoch=data['best']['epoch'],
                    full_metrics=data['full_metrics'], **files)

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, CONTROLS))
    orders = []
    for row in trained:
        values = [json.loads(line) for line in (root/'training'/row['name']/'batch_orders.jsonl').read_text().splitlines()]
        orders.append([(x['epoch'], x['step'], x['names'], x['partial_set']) for x in values])
    assert orders[0] == orders[1]
    save(root/'controller_result.json', dict(status='COMPLETE', parent_recipe='AV', seed=42,
        teacher_seed=42, frequency_rank=128, parameters=99947398, trainable_parameters=2249606, controls=2, runs=trained, native_updates=6,
        successful_updates=sum(row['successful_updates'] for row in trained), additional_epochs=100,
        parity_forward_record_executions=64, paired_sampling_exact=True, state_cases=8,
        physical_gpus=[2, 3], maximum_NN=2, full_training_coverage=True, normal_GT_audited=True,
        missing_evaluated=False, raw_preserved_pending_local_archive=True,
        limits='F hidden-width128 pair on original global AV42/default FFT; existing AV42 rank64 is the same-seed reference, not a new training. Additional362240 learned parameters apply equally to both controls. Measure actual time/memory; no capacity bottleneck or retrieval gain is assumed. Paper target87.2/98.6. Frozen states do not replace retrained module ablations.'))
    print('RGBNT100_F_RANK_PAIR_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
