"""Matched normal-priority controls: native3, full50, complete installed GT."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from gpu_01_diagnostic_execute import execute as execute01
from gpu_thermal_execute import execute as execute23
from launch_full_official_frozen_anchor_trial import save

GPU_CONTROLS={'MSVR310':(('frequency_shared',0),('axis_shared',1)),'RGBNT100':(('frequency_shared',2),('axis_shared',3))}
PRINT_LOCK = threading.Lock()
AUXILIARY_WEIGHTS = {'MSVR310': .001, 'RGBNT100': .001}


def emit(value):
    with PRINT_LOCK:
        sys.stdout.write('R201Z_STREAM ' + json.dumps(value) + '\n')
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'anchor-root', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--dataset', choices=('MSVR310',), required=True)
    args = parser.parse_args()
    coefficient = AUXILIARY_WEIGHTS[args.dataset]
    previous_coefficient={'MSVR310':.01,'RGBNT100':.05}[args.dataset]
    outlet={'MSVR310':'actual_first11_projected_unit_relation_mean512','RGBNT100':'actual_first11_projected_unit_relation_score_pool512'}[args.dataset]
    CONTROLS=GPU_CONTROLS[args.dataset]
    execute={'MSVR310':execute01,'RGBNT100':execute23}[args.dataset]
    main_CE={'MSVR310':.05,'RGBNT100':.25}[args.dataset]
    regression={'MSVR310':0.,'RGBNT100':.05}[args.dataset]
    root, anchors = Path(args.output), Path(args.anchor_root)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('native', 'training', 'audit'):
        (root / phase).mkdir()
    save(root / 'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        gpus=[g for _,g in CONTROLS], maximum_NN=2, temperature_power_control=False, contribution_regression_weight=regression, final_full_CE_weight=main_CE, M_F_auxiliary_weight=coefficient, experiment='Z versus completed Y: only existing generalFusion HDM/ATMoE9958912 parameter gradients enabled; all five original modules stay eval, four other identity modules and all generalFusion buffers retained. Same P16K4/705updates per arm/additional50/newAdam/mean512/MFaux.001/mainCE.05/reg0/hinge.6/cap.10/PI/FFT/experts/router/gates/5120/BN2/partial0/two backwards. Total100821638 unchanged; active3122822->13081734, both Z controls matched but not active-budget matched to Y. 00 is adapting base; cap.10 references current base, not the original teacher. OriginalDeMo42 teacher50 plus expert50; normalbenchmarkmAP-best earliesttie/samecheckpoint allsix/fullofficial1032train591Q1055G/noholdout/GT/rawCPU. Native3 each before50; GF two-parameter early gradient diagnostics are observational, no global cause claim. No201NN/new49/missing selection/failedseedsearch; X GPU2/3 unchanged; K201 protected.'))
    def argv(variant, mode, output):
        return [sys.executable, '-u', 'run_r201z_MS_modality_adapt.py', '--dataset', args.dataset, '--variant', variant,
            '--freeze-identity-encoder', '0', '--seed', '42', '--mode', mode, '--data-root', args.data_root,
            '--pretrained', args.pretrained, '--anchor-run-dir', str(anchors / (args.dataset+'_demo_s42')), '--output', str(output)]

    def native(variant, gpu):
        name = args.dataset+'_r201z_' + variant + '_s42_smoke'
        output = root / 'native' / name
        execute(argv(variant, 'smoke', output), root / 'native', name, gpu)
        data = json.loads((output / 'smoke.json').read_text())
        assert (data['parameters'],data['trainable_parameters'])=={'MSVR310':(100821638,13081734),'RGBNT100':(99585158,1887366)}[args.dataset]
        if args.dataset=='MSVR310':
            assert data['anchor']['M_F_auxiliary_outlet']['frozen_parameter_names']==['modality_expert.pool_score.weight','frequency_expert.pool_score.weight'] and data['anchor']['M_F_auxiliary_outlet']['frozen_parameter_count']==1024
        assert data['status'] == 'SMOKE_PASS' and data['steps'] == data['attempts'] == 3
        assert data['amp_skipped_steps'] == 0 and data['strict_reload_equal'] and '  NUM_INSTANCE: 4' in data['config'].splitlines()
        assert data['descriptor_shape'] == [8,5120] and data['descriptor_dim'] == 5120
        assert all(data['gradients'].values()) and not list(output.glob('*.pth'))
        scope=data['anchor']['partial_identity_adaptation']
        assert data['anchor']['freeze_identity_encoder'] is False and scope['adapted_module']=='generalFusion' and scope['adapted_parameters']==9958912 and scope['frozen_modules']==['BACKBONE','rgb_reduce','nir_reduce','tir_reduce']
        assert scope['BN_and_dropout_modes_unchanged'] and scope['original_generalFusion_buffers_unchanged'] and scope['parameters_added']==0
        assert all(d['generalFusion_parameter_adaptation'] and d['generalFusion_trainable_parameters']==9958912 and d['generalFusion_eval_buffers_and_frozen4_retained'] and len(d['generalFusion_main_base_aux_native'])==2 for d in data['details'])
        assert data['details'][-1]['generalFusion_native_gradient_audit']['all_active_nonzero'] and data['details'][-1]['generalFusion_native_gradient_audit']['changed_tensors']
        assert data['anchor']['sampler_control']==dict(parent_K=8,current_K=4,batch_size=64,identities_per_batch=16,old_updates_per_arm=1000,new_updates_per_arm=705,same_epochs_not_equal_updates=True)
        assert data['anchor']['M_F_auxiliary_outlet']['parameters_added']==0 and data['anchor']['M_F_auxiliary_outlet']['extra_neck_projection_calls']==0
        assert all(d['loss']==d['full_loss'] and d['partial_loss_effective']==0 and d['partial_CE_weight']==d['partial_triplet_weight']==0
                   and d['fused_BN_calls']==2 and d['full_unit_max_error']<1e-6 and len(d['names'])==64
                   and d['sampling_identities']==16 and d['sampling_instances']==4 and d['sampling_batch']==64
                   and d['primary_margin']==.6 and d['primary_metric_coefficient']==1.
                   and d['primary_margin_definition_equal'] and d['primary_feature_gradient_finite']
                   and set(d['primary_parameter_gradient_l1'])=={'PM','PF','PI','M','F'}
                   and all(d['primary_vs_aux_'+n]['primary_norm']>=0 and d['primary_vs_aux_'+n]['weighted_auxiliary_norm']>=0 for n in ('M','F'))
                   and d['auxiliary_original_soft_triplet'] and d['PI_outlet']=='shared_relation_local'
                   and d['relation_PI_native']['pi_input_shape']==[64,7,1024]
                   and d['relation_PI_native']['pi_output_shape']==[64,7,512]
                   and d['relation_PI_native']['per_relation_outside_slot_errors']==[0.]*7
                   and d['relation_PI_native']['state_10_parent_equal']
                   and d['relation_PI_native']['state_01_parent_equal'] for d in data['details'])
        assert data['anchor']['contribution_reference']=='Stopped-gradient corresponding gallery state; fixed00 positive/negative indices'
        assert data['anchor']['M_F_auxiliary_weight_previous']==previous_coefficient and data['anchor']['M_F_auxiliary_weight']==coefficient
        assert all(d['M_F_auxiliary_weight_previous']==previous_coefficient and d['M_F_auxiliary_weight']==coefficient
                   and d['primary_vs_aux_M']['auxiliary_weight']==coefficient
                   and d['primary_vs_aux_F']['auxiliary_weight']==coefficient for d in data['details'])
        assert all(d['contribution_reference']=='same_state_gallery_fixed00_indices' and len(d['contribution_target_mean'])==len(d['contribution_prediction_mean'])==3 and d['contribution_loss_raw']>=0 and d['contribution_zero_loss_raw']>=0 for d in data['details'])
        assert data['anchor']['parent_main_CE_weight']==main_CE and data['anchor']['parent_contribution_regression_weight']==regression
        assert all(d['parent_main_CE_weight']==main_CE and d['parent_contribution_regression_weight']==regression and d['xent_counted_calls']==2 and abs(d['main_CE_effective']-main_CE*d['main_CE_unweighted'])<1e-8 for d in data['details'])
        for d in data['details']:
            c=d['projected_auxiliary_forward_contract']
            assert c['main11_first_projection_selected'] and c['controlled_pool_first_unchanged'] and c['extra_neck_projection_calls']==0
            assert c['projection_calls']==dict(M=3,F=3) and c['pool_calls']==dict(M=2,F=2) and c['neck_calls']==dict(M=1,F=1)
            assert c['feature_shape']==[64,512] and c['feature_requires_grad'] and c['feature_dtype']=='torch.float32'
            assert set(d['direct_PM_PF_auxiliary_gradient'])=={'M','F'} and all(g['finite'] and g['weighted_auxiliary_norm']>0 and g['weighted_auxiliary_l1']>0 for g in d['direct_PM_PF_auxiliary_gradient'].values())
            assert d['M_F_auxiliary_outlet']==outlet
        assert data['anchor']['final_unit_descriptor_bound']['epsilon']==.10
        assert data['anchor']['final_unit_descriptor_bound']['parameter_change']==0
        for detail in data['details']:
            fixture=detail['final_descriptor_bound_geometry_fixture']
            assert fixture['status']=='PASS_CPU_GEOMETRY_FIXTURE' and fixture['neural_forwards']==0
            assert fixture['active_and_inactive_cases'] and fixture['finite_derivatives']
            reports=detail['final_descriptor_bound_controlled_reports']
            assert set(reports)=={'10','01','11'}
            assert all(r['epsilon']==.10 and r['bounded_shift_max']<=.10+1e-6 for r in reports.values())
        return data

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(native, variant, gpu) for variant, gpu in CONTROLS]
        native_rows = [future.result() for future in futures]
    assert len({(r['parameters'],r['trainable_parameters']) for r in native_rows}) == 1
    assert [(d['names'],d['partial_set']) for d in native_rows[0]['details']] == [(d['names'],d['partial_set']) for d in native_rows[1]['details']]
    save(root / 'native_acceptance.json', dict(status='PASS', controls=2, actual_updates=6, paired_sampling=True,
        primary_unit_triplet=True, primary_margin=.6, auxiliary_soft_unchanged=True, partial_forward_BN_retained=True, partial_loss_effective_zero=True,
        direct_PM_PF_auxiliary_gradient_positive=True,projected_auxiliary_call_contract=True,third_update_active_gradients=True, extra_weights=0, relation_PI_native_invariance=True, controlled10_01_pre_bound_parent_fuse_equal=True, final_unit_shift_epsilon=.10, controlled_final_shift_bound=True, M_F_auxiliary_weight=coefficient, contribution_regression_weight=regression, final_full_CE_weight=main_CE))
    emit(dict(event='NATIVE_PASS',direct_PM_PF_auxiliary_gradient_positive=True,projected_auxiliary_call_contract=True, controls=2, actual_updates=6, partial_loss_effective_zero=True, PI_outlet='shared_relation_local',final_unit_shift_epsilon=.10,controlled_final_shift_bound=True, M_F_auxiliary_weight=coefficient, contribution_regression_weight=regression, final_full_CE_weight=main_CE))

    def archive(run, name):
        path = run / 'best_official_arrays.npz'
        file = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        emit(dict(event='NORMAL_READY', name=name, path=str(path), file=file))
        assert json.loads(sys.stdin.readline()) == dict(event='NORMAL_ARCHIVED', name=name, file=file)
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size == file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == file['sha256']
        path.unlink()
        save(run / 'normal_local_archive.json', dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED', file=file))
        emit(dict(event='NORMAL_CLEARED', name=name))

    with ThreadPoolExecutor(max_workers=1) as archives:
        def train(variant, gpu):
            name = args.dataset+'_r201z_' + variant + '_s42'
            output = root / 'training' / name
            trained = execute(argv(variant, 'train', output), root / 'training', name, gpu)
            data = json.loads((output / 'result.json').read_text())
            assert data['status'] == 'COMPLETE' and data['epochs'] == 50 and '  NUM_INSTANCE: 4' in data['config'].splitlines()
            assert data['steps'] == data['optimizer_steps'] and data['amp_skipped_steps'] == 0
            assert data['anchor']['M_F_auxiliary_weight_previous']==previous_coefficient and data['anchor']['M_F_auxiliary_weight']==coefficient
            assert all(r['loss']==r['full_loss'] and r['partial_loss_effective']==r['partial_CE_weight']==r['partial_triplet_weight']==0
                       and r['sampling_identities']==16 and r['sampling_instances']==4 and r['sampling_batch']==64
                       and r['primary_margin']==.6 and r['auxiliary_original_soft_triplet'] and r['PI_outlet']=='shared_relation_local'
                       and r['M_F_auxiliary_weight_previous']==previous_coefficient and r['M_F_auxiliary_weight']==coefficient
                       for r in map(json.loads,(output/'batch_orders.jsonl').read_text().splitlines()))
            assert data['anchor']['parent_contribution_regression_weight']==regression
            assert data['anchor']['freeze_identity_encoder'] is False and data['anchor']['partial_identity_adaptation']['adapted_parameters']==9958912
            assert all(r['generalFusion_parameter_adaptation'] and r['generalFusion_trainable_parameters']==9958912 and r['generalFusion_eval_buffers_and_frozen4_retained'] for r in map(json.loads,(output/'batch_orders.jsonl').read_text().splitlines()))
            assert data['anchor']['parent_main_CE_weight']==main_CE
            assert all(r['parent_main_CE_weight']==main_CE and r['xent_counted_calls']==2 and abs(r['main_CE_effective']-main_CE*r['main_CE_unweighted'])<1e-8 for r in map(json.loads,(output/'batch_orders.jsonl').read_text().splitlines()))
            assert all(r['parent_contribution_regression_weight']==regression for r in map(json.loads,(output/'batch_orders.jsonl').read_text().splitlines()))
            assert all(b['M_F_auxiliary_outlet']==outlet and b['parent_main_CE_weight']==main_CE and b['parent_contribution_regression_weight']==regression and b['projected_auxiliary_forward_contract']['projection_calls']==dict(M=2,F=2) and b['projected_auxiliary_forward_contract']['neck_calls']==dict(M=1,F=1) and b['projected_auxiliary_forward_contract']['pool_calls']==dict(M=2,F=2) for b in map(json.loads,(output/'batch_orders.jsonl').read_text().splitlines()))
            assert data['training_heldout_identities'] == 0 and data['training_coverage'] == dict(eligible=data['train_records'],visited=data['train_records'],unvisited=[])
            audited = execute([sys.executable,'-u','audit_full_official_normal.py','--run-dir',str(output)], root / 'audit', name, gpu)
            assert json.loads((output / 'normal_cpu_audit.json').read_text())['status'] == 'PASS'
            return dict(name=name,variant=variant,gpu=gpu,train=trained,audit=audited), archives.submit(archive,output,name)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train,variant,gpu) for variant,gpu in CONTROLS]
            results = [future.result() for future in futures]
        for _, future in results:
            future.result()

    def orders(variant):
        file = root / 'training' / (args.dataset+'_r201z_' + variant + '_s42') / 'batch_orders.jsonl'
        return [(r['epoch'],r['step'],r['names'],r['partial_set']) for r in map(json.loads,file.read_text().splitlines())]

    assert orders('frequency_shared') == orders('axis_shared')
    updates=sum(json.loads((root/'training'/r['name']/'result.json').read_text())['optimizer_steps'] for r,_ in results)
    save(root / 'controller_result.json', dict(status='COMPLETE', dataset=args.dataset, controls=2, runs=[r for r,_ in results],
        additional_epochs=100, successful_updates=updates, native_updates=6, paired_sampling_exact=True,
        normal_archives_local_verified=2, M_F_auxiliary_weight=coefficient, contribution_regression_weight=regression, final_full_CE_weight=main_CE, missing_evaluation='After weakdatasets normal closure; only DeMo paper six symmetric missing masks, no new49',
        limits='Z only existing generalFusion9958912 parameter gradients enabled versus completedY while alloriginalmodules eval, frozen4/buffers exact. Same P16K4/705/newAdam/50epoch/loss/mean512/FFT/router/PI/cap.10/5120. Effective trainables13081734 notY3122822; both Z controls matched. 00 adapts; capreferences currentbase notoriginalteacher. Original50+new50/benchmarkmAP-best earliestties. Native GFtwo-parameter mainCEplusHinge versusbaseaux gradients are early-only, not globalcause. FullnormalGT/raw/CPU first/no201NN/new49/missing reselect/failedseedsearch.'))
    emit(dict(event='CONTROLLER_COMPLETE', controls=2, successful_updates=updates))


if __name__ == '__main__':
    main()
