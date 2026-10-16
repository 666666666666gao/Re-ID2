"""Full teacher parity, native3 per control, then matched expert50 runs serialized onGPU3."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import save

CONTROLS = ('frequency_shared', 'axis_shared')


def verify_detail(detail, native):
    assert isinstance(detail['optimizer_updated'], bool)
    assert all(math.isfinite(detail[key]) for key in ('loss', 'full_loss', 'amp_scale'))
    assert detail['amp_scale'] > 0 and detail['loss'] == detail['full_loss']
    assert detail['partial_loss_effective'] == detail['partial_CE_weight'] == detail['partial_triplet_weight'] == 0
    assert (detail['sampling_identities'], detail['sampling_instances'], detail['sampling_batch']) == (16, 4, 64)
    assert detail['primary_full_metric'] == 'unit5632_margin06_batch_hard_triplet' and detail['primary_margin'] == .6
    assert detail['M_F_auxiliary_weight'] == .001 and detail['parent_main_CE_weight'] == .05
    assert detail['parent_contribution_regression_weight'] == 0 and detail['auxiliary_original_soft_triplet']
    assert detail['M_F_auxiliary_outlet'] == 'actual_first11_projected_unit_relation_mean512'
    assert detail['descriptor_dim'] == detail['main_loss_deployed_dimension'] == 5632
    assert detail['private_bound_dimension'] == 5120 and detail['retained_public_dimension'] == 512
    assert detail['retained_public_projection_frozen']
    contract = detail['projected_auxiliary_forward_contract']
    assert contract['projection_calls'] == dict(M=3 if native else 2, F=3 if native else 2)
    assert contract['pool_calls'] == dict(M=2, F=2) and contract['neck_calls'] == dict(M=1, F=1)
    assert contract['feature_shape'] == [64, 512]
    assert detail['final_descriptor_bound_full']['bounded_shift_max'] <= .10+1e-6
    if native:
        assert detail['fused_BN_calls'] == 2
        assert all(value['finite'] and value['weighted_auxiliary_l1'] > 0
                   for value in detail['direct_PM_PF_auxiliary_gradient'].values())
        audit = detail['relation_PI_native']
        assert audit['per_relation_outside_slot_errors'] == [0.]*7
        assert audit['state_10_parent_equal'] and audit['state_01_parent_equal']


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'anchor-run-dir', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('reference', 'native', 'training', 'audit', 'states'):
        (root/phase).mkdir()
    save(root/'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        physical_gpus=[3], maximum_own_NN=1, original_RGBNT100_GPU2_untouched=True,
        temperature_power_control=False, experiment='Existing AH controls on new selected head16 full5632 reference'))

    def common(variant, out):
        return ['--variant', variant, '--data-root', args.data_root, '--pretrained', args.pretrained,
                '--anchor-run-dir', args.anchor_run_dir, '--output', str(out)]

    references = []
    for variant in CONTROLS:
        name = 'MSVR310_head16_'+variant+'_reference'
        out = root/'reference'/name
        ended = execute([sys.executable, '-u', 'evaluate_msvr_head16_reference.py', *common(variant, out)], root/'reference', name, 3)
        result = json.loads((out/'result.json').read_text())
        assert result['status'] == 'COMPLETE_HEAD16_FULL5632_FROZEN00_OFFICIAL_TEACHER_PARITY'
        assert result['optimizer_updates'] == result['new_weights'] == 0
        assert result['strict_reload_equal'] and result['per_query_equal']
        assert (result['train_records'], result['query_records'], result['gallery_records'], result['descriptor_dim']) == (1032, 591, 1055, 5632)
        references.append(dict(variant=variant, exit=ended, metrics=result['metrics']))
    assert references[0]['metrics'] == references[1]['metrics']
    save(root/'reference_acceptance.json', dict(status='PASS', controls=2, full00_teacher_parity=True, references=references))

    def run(variant, mode):
        name = 'MSVR310_head16_'+variant+'_s42'+('_smoke' if mode == 'smoke' else '')
        phase = 'native' if mode == 'smoke' else 'training'
        out = root/phase/name
        ended = execute([sys.executable, '-u', 'run_msvr_head16_experts.py', '--dataset', 'MSVR310',
            '--freeze-identity-encoder', '1', '--seed', '42', '--mode', mode, *common(variant, out)], root/phase, name, 3)
        return name, out, ended

    natives = []
    for variant in CONTROLS:
        name, out, ended = run(variant, 'smoke')
        result = json.loads((out/'smoke.json').read_text())
        details = result['details']
        assert result['status'] == 'SMOKE_PASS' and result['steps'] == 3
        assert result['attempts'] == len(details) and sum(row['optimizer_updated'] for row in details) == 3
        assert result['amp_skipped_steps'] == len(details)-3
        assert result['strict_reload_equal'] and result['descriptor_shape'] == [8, 5632] and all(result['gradients'].values())
        assert not list(out.glob('*.pth'))
        assert (result['parameters'], result['trainable_parameters']) == (100787334, 2674822)
        assert result['anchor']['complete_reference']['public_projection_frozen']
        for detail in details:
            verify_detail(detail, True)
        natives.append(result)
    shortest = min(row['attempts'] for row in natives)
    orders = [[(row['names'], row['partial_set']) for row in result['details'][:shortest]] for result in natives]
    assert orders[0] == orders[1]
    save(root/'native_acceptance.json', dict(status='PASS', controls=2, actual_updates=6,
        attempts=sum(row['attempts'] for row in natives), amp_skips=sum(row['amp_skipped_steps'] for row in natives),
        common_attempt_prefix_paired=True, strict_reload_equal=True, all_active_gradients_positive=True,
        parameters=100787334, trainable_parameters=2674822, descriptor_dim=5632, finished=time.time()))
    print('MS_HEAD16_REFERENCE_AND_NATIVE_PASS', flush=True)

    trained = []
    paired_order = None
    for variant in CONTROLS:
        name, out, ended = run(variant, 'train')
        result = json.loads((out/'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['epochs'] == 50 and result['steps'] == 705
        assert 0 < result['optimizer_steps'] <= 705 and result['amp_skipped_steps'] == 705-result['optimizer_steps']
        assert result['training_heldout_identities'] == 0
        assert result['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
        assert (result['parameters'], result['trainable_parameters'], result['descriptor_dim']) == (100787334, 2674822, 5632)
        orders = [json.loads(line) for line in (out/'batch_orders.jsonl').read_text().splitlines()]
        assert len(orders) == 705 and sum(row['optimizer_updated'] for row in orders) == result['optimizer_steps']
        for detail in orders:
            verify_detail(detail, False)
        signature = [(row['epoch'], row['step'], row['names'], row['partial_set']) for row in orders]
        if paired_order is None:
            paired_order = signature
        else:
            assert signature == paired_order
        audited = execute([sys.executable, '-u', 'audit_vehicle_atmoe_head_normal.py', '--run-dir', str(out)], root/'audit', name, 3)
        assert json.loads((out/'normal_cpu_audit.json').read_text())['status'] == 'PASS'
        state_dir = root/'states'/name
        state_exit = execute([sys.executable, '-u', 'evaluate_msvr_head16_states.py', '--run-dir', str(out), '--output', str(state_dir)], root/'states', name, 3)
        states = json.loads((state_dir/'result.json').read_text())
        assert states['status'] == 'COMPLETE' and states['state_cases'] == 4
        assert states['full11_features_distance_exact_selected_normal'] and states['frozen00_full_teacher_six_per_query_equal']
        artifacts = {}
        for key, file in (('raw', out/'best_official_arrays.npz'), ('best', out/'best.pth'), ('states_raw', state_dir/'normal_states.npz')):
            artifacts[key] = dict(path=str(file), bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        row = dict(variant=variant, train=ended, audit=audited, states_exit=state_exit, full_metrics=result['full_metrics'],
            selected_epoch=result['best']['epoch'], attempts=705, successful_updates=result['optimizer_steps'],
            amp_skips=result['amp_skipped_steps'], **artifacts)
        trained.append(row)
        save(root/(variant+'_component_complete.json'), row)
    save(root/'controller_result.json', dict(status='COMPLETE', dataset='MSVR310', runs=trained, references=references,
        controls=2, native_updates=6, formal_attempts=1410, formal_optimizer_updates=sum(row['successful_updates'] for row in trained),
        formal_amp_skips=sum(row['amp_skips'] for row in trained), additional_epochs=100, state_cases=8,
        paired_sampling_exact=True, physical_gpus=[3], maximum_own_NN=1,
        raw_preserved_pending_local_archive=True, missing_evaluated=False, new49=False,
        limits='Baseline gains inherited; same AH experts/objectives on one new frozen head16 reference. Teacher50 plus expert50 and benchmark selection disclosed; actual dynamic AMP updates/skips retained. Normal-only single-seed comparison, not missing robustness or retrained ablations.'))
    print('MS_HEAD16_MATCHED_EXPERTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
