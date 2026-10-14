"""Matched RGB coordinate-metric controls: new native3, full50, full GT; only GPU2/3."""
import argparse,hashlib,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import save

CONTROLS=(('frequency_shared',2),('axis_shared',3))


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-root','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    for phase in ('native','training','audit','states'):(root/phase).mkdir()
    save(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),
        physical_gpus=[2,3],maximum_NN=2,temperature_power_control=False,
        experiment='BC returns to AV and changes only existing relation-local PI hidden64 via 2sigmoid(Wb unit(stopgrad(original base relation512))). One shared zero-initialized64x512 weight32768 parameters, identical addition in ordinary and dual-axis models. Initial AV values/existing tensors/RNG retained; PM/PF/auxiliary/conditions/routing/5120/cap.10 unchanged. Known conditional modulation, not new information, calibrated utility or guaranteed gain. BA prototype initialization and AZ corrected-state metric not retained. Full official normal only, teacher50 plus fresh expert42/newAdam/extra50, benchmarkmAP earliesttie.'))
    def run(variant,gpu,mode):
        name='RGBNT100_r201bc_'+variant+'_s42'+('_smoke' if mode=='smoke' else '')
        phase='native' if mode=='smoke' else 'training';out=root/phase/name
        row=execute([sys.executable,'-u','run_r201bc_RGB_reference_interaction.py','--dataset','RGBNT100','--variant',variant,
            '--freeze-identity-encoder','1','--seed','42','--mode',mode,'--data-root',args.data_root,
            '--pretrained',args.pretrained,'--anchor-run-dir',str(Path(args.anchor_root)/'RGBNT100_demo_s42'),
            '--output',str(out)],root/phase,name,gpu)
        return name,out,row
    def verify_detail(d,native):
        assert d['loss']==d['full_loss'] and d['optimizer_updated']
        assert d['partial_loss_effective']==d['partial_CE_weight']==d['partial_triplet_weight']==0
        assert d['freeze_identity_encoder'] and d['routing_temperature']==2. and d['routing_parameters_added']==0
        if native:
            probes=d['routing_temperature_native_probes']
            assert probes['temperature1_exact_parent'] and probes['no_psi_product_max_error']<1e-6
            assert probes['temperature2_vs1_max_probability_difference']>0
        if native:assert d['fused_BN_calls']==2
        q=d['modality_projected_query_normalization']
        assert q['policy']=='per-head L2 times sqrt(8)' and q['parameters_added']==0 and q['attention_calls']==84
        assert q['projected_keys_values_unchanged'] and q['frequency_attention_unchanged'] and q['pre_norm_min']>0
        assert abs(q['post_norm_min']-8**.5)<1e-5 and abs(q['post_norm_max']-8**.5)<1e-5
        policy=d['attention_projection_decay']
        assert policy['old']==1e-4 and policy['new']==0. and policy['packed_includes_V']
        assert len(policy['targets'])==5 and all(len(policy[k])==15 for k in ('weight_norm_before','weight_norm_after','unscaled_task_gradient_norm','counterfactual_old_L2_gradient_norm'))
        value=d['frequency_value_input_decay']
        assert value['old']==1e-4 and value['new']==0. and value['parameters_added']==0
        assert value['targets']==['frequency_expert.values.1.1.weight','frequency_expert.values.2.1.weight']
        assert value['low_band_shared_M_bias_LN_policy_unchanged']
        assert all(len(value[k])==2 for k in ('weight_norm_before','weight_norm_after','unscaled_task_gradient_norm','counterfactual_old_L2_gradient_norm'))
        gate=d['identity_reference_interaction'];assert gate['parameters_added']==32768
        assert gate['base_reference_stop_gradient'] and gate['final_descriptor_dim']==5120
        if native:assert gate['unscaled_task_gradient_norm']>0
        assert d['sampling_identities']==d['sampling_instances']==8 and d['sampling_batch']==64
        assert d['sampling_treatment']=='hard' and d['primary_margin']==.6 and d['auxiliary_original_soft_triplet']
        assert d['M_F_auxiliary_weight']==.001 and d['parent_main_CE_weight']==.25 and d['parent_contribution_regression_weight']==.05
        assert d['M_F_auxiliary_Triplet_outlet']=='actual_first11_per512_unit_relation_coordinates3584'
        assert d['M_F_auxiliary_CE_outlet']=='unchanged_existing_score_FP32_pool512'
        assert d['auxiliary_metric_feature_dimension']==3584 and d['auxiliary_CE_feature_dimension']==512
        c=d['projected_auxiliary_forward_contract']
        assert c['main11_first_projection_selected'] and c['extra_neck_projection_calls']==0
        assert c['pool_calls']==dict(M=2,F=2) and c['neck_calls']==dict(M=1,F=1)
        expected_calls=3 if native else 2
        assert c['projection_calls']==dict(M=expected_calls,F=expected_calls)
        assert c['metric_feature_shape']==[64,3584] and c['feature_shape']==[64,512] and c['feature_dtype']=='torch.float32'
        for n in ('M','F'):
            v=d[n+'_auxiliary_scorer_fp32']
            assert v['input_shape']==[64,7,512] and v['input_dtype']==v['output_dtype']=='torch.float32' and v['output_requires_grad']
        h=d['hard_identity_sampling'];assert h['post_sampler_RNG_exact'] and h['original_record_multiset_exact'] and h['new_random_draws']==0
        assert h['P']==h['K']==8
        if native:
            assert all(v['finite'] and v['weighted_auxiliary_l1']>0 for v in d['direct_PM_PF_auxiliary_gradient'].values())
            assert d['relation_PI_native']['per_relation_outside_slot_errors']==[0.]*7
            assert d['relation_PI_native']['state_10_parent_equal'] and d['relation_PI_native']['state_01_parent_equal']
    def native(pair):
        name,out,row=run(*pair,'smoke');d=json.loads((out/'smoke.json').read_text())
        assert d['status']=='SMOKE_PASS' and d['steps']==d['attempts']==3 and d['amp_skipped_steps']==0
        assert d['strict_reload_equal'] and all(d['gradients'].values()) and not list(out.glob('*.pth'))
        assert (d['parameters'],d['trainable_parameters'],d['descriptor_dim'])==(99617926,1920134,5120)
        assert (d['train_records'],d['query_records'],d['gallery_records'])==(8675,1715,8575)
        for detail in d['details']:verify_detail(detail,True)
        # Each of the five selection operators must receive an actual Q/K task gradient in native3.
        assert all(any(detail['attention_projection_decay']['unscaled_task_gradient_norm'][3*i]>0 or detail['attention_projection_decay']['unscaled_task_gradient_norm'][3*i+1]>0 for detail in d['details']) for i in range(5))
        assert all(any(v['frequency_value_input_decay']['unscaled_task_gradient_norm'][i]>0 for v in d['details']) for i in range(2))
        assert d['anchor']['frequency_value_decay_targets']==['frequency_expert.values.1.1.weight','frequency_expert.values.2.1.weight']
        assert d['anchor']['frequency_value_parameters_added']==0
        assert d['anchor']['auxiliary_metric_parameters_added']==0
        assert d['anchor']['routing_temperature']==2. and d['anchor']['routing_parameters_added']==0
        init=d['anchor']['identity_reference_interaction']
        assert init['parameters_added']==32768 and init['reference_weight_shape']==[64,512]
        assert init['initial_gate_weight_zero'] and init['CPU_CUDA_RNG_exact']
        assert init['existing_parameter_objects_versions_exact'] and init['same_for_ordinary_and_axis']
        av=Path('runs/r201av_RGB_frequency_value_decay_20261014/RGBNT100/native')/('RGBNT100_r201av_'+pair[0]+'_s42_smoke')/'smoke.json'
        reference=json.loads(av.read_text())['details'][0];current=d['details'][0]
        assert current['names']==reference['names'] and current['partial_set']==reference['partial_set']
        for key in ('primary_triplet_loss','route_entropy_mean','route_nonindependence_mean','full_loss','main_CE_unweighted'):
            assert abs(current[key]-reference[key])<1e-6
        assert current['primary_margin_violations']==reference['primary_margin_violations']
        return d
    with ThreadPoolExecutor(max_workers=2) as pool:natives=list(pool.map(native,CONTROLS))
    assert [(d['names'],d['partial_set']) for d in natives[0]['details']]==[(d['names'],d['partial_set']) for d in natives[1]['details']]
    assert natives[0]['anchor']['identity_reference_interaction']==natives[1]['anchor']['identity_reference_interaction']
    save(root/'native_acceptance.json',dict(status='PASS',controls=2,actual_updates=6,paired_sampling=True,
        parameters=99617926,trainable_parameters=1920134,descriptor_dim=5120,
        identity_reference_parameters_added=32768,initial_AV_gate_RNG_exact=True,scored512CE_retained=True,actual_first11_unit3584_metric=True,strict_reload_equal=True,all_active_gradients_positive=True))
    print('BC_NATIVE_PASS',flush=True)
    def train(pair):
        variant,gpu=pair;name,out,row=run(variant,gpu,'train');d=json.loads((out/'result.json').read_text())
        assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps']==6563
        assert d['amp_skipped_steps']==d['training_heldout_identities']==0
        assert d['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[])
        assert (d['parameters'],d['trainable_parameters'],d['descriptor_dim'])==(99617926,1920134,5120)
        rows=[json.loads(line) for line in (out/'batch_orders.jsonl').read_text().splitlines()]
        assert len(rows)==6563
        for detail in rows:verify_detail(detail,False)
        original=Path('runs/r201ag_RGB_coordinate_metric_20261012/RGBNT100/training')/('RGBNT100_r201ag_'+variant+'_s42')/'batch_orders.jsonl'
        prior=[json.loads(line) for line in original.read_text().splitlines()]
        order=lambda values:[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in values]
        assert order(rows)==order(prior)
        al=Path('runs/r201al_RGB_route_temperature_20261013/RGBNT100/training')/('RGBNT100_r201al_'+variant+'_s42')/'batch_orders.jsonl'
        assert order(rows)==order([json.loads(line) for line in al.read_text().splitlines()])
        audited=execute([sys.executable,'-u','audit_full_official_normal.py','--run-dir',str(out)],root/'audit',name,gpu)
        assert json.loads((out/'normal_cpu_audit.json').read_text())['status']=='PASS'
        states=execute([sys.executable,'-u','evaluate_r201bc_RGB_normal_states.py','--run-dir',str(out),'--output',str(root/'states'/name)],root/'states',name,gpu)
        sd=json.loads((root/'states'/name/'result.json').read_text())
        assert sd['status']=='COMPLETE' and sd['state_cases']==4
        assert sd['full11_features_distance_equal_selected_normal'] and sd['state_tensor_versions_unchanged']
        sp=root/'states'/name/'normal_states.npz'
        path=out/'best_official_arrays.npz'
        return dict(name=name,gpu=gpu,train=row,audit=audited,states=states,states_raw=dict(path=str(sp),bytes=sp.stat().st_size,sha256=hashlib.sha256(sp.read_bytes()).hexdigest()),raw=dict(path=str(path),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()),best=dict(path=str(out/'best.pth'),bytes=(out/'best.pth').stat().st_size,sha256=hashlib.sha256((out/'best.pth').read_bytes()).hexdigest(),selected_epoch=d['best']['epoch']))
    with ThreadPoolExecutor(max_workers=2) as pool:trained=list(pool.map(train,CONTROLS))
    save(root/'controller_result.json',dict(status='COMPLETE',dataset='RGBNT100',runs=trained,controls=2,
        native_updates=6,successful_updates=13126,additional_epochs=100,paired_sampling_exact=True,
        AG_training_record_order_exact=True,AL_training_record_order_exact=True,attention_packed_QKV_decay=0.,attention_decay_targets=5,modality_projected_query_normalization='per-head L2 times sqrt(8)',modality_query_parameters_added=0,correction_budget=.10,states_cases=8,normal_state_forward_record_executions=20580,routing_temperature=2.,routing_parameters_added=0,physical_gpus=[2,3],maximum_NN=2,
        identity_reference_parameters_added=32768,frequency_value_weight_decay=0.,frequency_value_decay_targets=2,frequency_value_parameters_added=0,raw_preserved_pending_local_archive=True,missing_evaluated=False,new49=False,
        limits='BC returns to AV and changes only existing relation-local PI hidden64 via 2sigmoid(Wb unit(stopgrad(original base relation512))). One shared zero-initialized64x512 weight32768 parameters, identical addition in ordinary and dual-axis models. Initial AV values/existing tensors/RNG retained; PM/PF/auxiliary/conditions/routing/5120/cap.10 unchanged. Known conditional modulation, not new information, calibrated utility or guaranteed gain. BA prototype initialization and AZ corrected-state metric not retained. Full official normal only, teacher50 plus fresh expert42/newAdam/extra50, benchmarkmAP earliesttie. Teacher50 plus extra50; benchmark selection; normal only.'))
    print('BC_CONTROLLER_COMPLETE',flush=True)


if __name__=='__main__':main()
