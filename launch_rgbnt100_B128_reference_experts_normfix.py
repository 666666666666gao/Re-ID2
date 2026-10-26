"""Existing AV pair on one B128/head8 teacher: full00, native3, full50, installed GT and four states."""
import argparse,hashlib,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import save

CONTROLS=(('frequency_shared',2),('axis_shared',3))


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-run-dir','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    for phase in ('reference','native','training','audit','states'):(root/phase).mkdir()
    save(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),
        physical_gpus=[2,3],maximum_NN=2,temperature_power_control=False,
        experiment='Same AV pair on completed B128/head8 reference and its rebuilt training-only bank; no new module or loss'))
    def reference(pair):
        variant,gpu=pair;name='RGBNT100_B128ref_'+variant+'_reference';out=root/'reference'/name
        ended=execute([sys.executable,'-u','evaluate_rgbnt100_B128_reference_normfix.py','--variant',variant,
            '--data-root',args.data_root,'--pretrained',args.pretrained,'--anchor-run-dir',args.anchor_run_dir,
            '--output',str(out)],root/'reference',name,gpu)
        value=json.loads((out/'result.json').read_text())
        assert value['status']=='COMPLETE_B128_HEAD8_FULL5120_FROZEN00_OFFICIAL_TEACHER_PARITY'
        assert value['optimizer_updates']==value['new_weights']==0
        assert value['strict_reload_equal'] and value['per_query_equal']
        assert (value['train_records'],value['query_records'],value['gallery_records'],value['descriptor_dim'])==(8675,1715,8575,5120)
        return dict(variant=variant,exit=ended,metrics=value['metrics'])
    with ThreadPoolExecutor(max_workers=2) as pool:references=list(pool.map(reference,CONTROLS))
    save(root/'reference_acceptance.json',dict(status='PASS',controls=2,full00_teacher_parity=True,references=references))
    def run(variant,gpu,mode):
        name='RGBNT100_B128ref_AV_'+variant+'_s42'+('_smoke' if mode=='smoke' else '')
        phase='native' if mode=='smoke' else 'training';out=root/phase/name
        row=execute([sys.executable,'-u','run_rgbnt100_B128_reference_experts.py','--dataset','RGBNT100','--variant',variant,
            '--freeze-identity-encoder','1','--seed','42','--mode',mode,'--data-root',args.data_root,
            '--pretrained',args.pretrained,'--anchor-run-dir',args.anchor_run_dir,
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
        assert (d['parameters'],d['trainable_parameters'],d['descriptor_dim'])==(99585158,1887366,5120)
        assert (d['train_records'],d['query_records'],d['gallery_records'])==(8675,1715,8575)
        for detail in d['details']:verify_detail(detail,True)
        # Each of the five selection operators must receive an actual Q/K task gradient in native3.
        assert all(any(detail['attention_projection_decay']['unscaled_task_gradient_norm'][3*i]>0 or detail['attention_projection_decay']['unscaled_task_gradient_norm'][3*i+1]>0 for detail in d['details']) for i in range(5))
        assert all(any(v['frequency_value_input_decay']['unscaled_task_gradient_norm'][i]>0 for v in d['details']) for i in range(2))
        assert d['anchor']['frequency_value_decay_targets']==['frequency_expert.values.1.1.weight','frequency_expert.values.2.1.weight']
        assert d['anchor']['frequency_value_parameters_added']==0
        assert d['anchor']['auxiliary_metric_parameters_added']==0
        assert d['anchor']['routing_temperature']==2. and d['anchor']['routing_parameters_added']==0
        return d
    with ThreadPoolExecutor(max_workers=2) as pool:natives=list(pool.map(native,CONTROLS))
    assert [(d['names'],d['partial_set']) for d in natives[0]['details']]==[(d['names'],d['partial_set']) for d in natives[1]['details']]
    save(root/'native_acceptance.json',dict(status='PASS',controls=2,actual_updates=6,paired_sampling=True,
        parameters=99585158,trainable_parameters=1887366,descriptor_dim=5120,
        scored512CE_retained=True,actual_first11_unit3584_metric=True,strict_reload_equal=True,all_active_gradients_positive=True))
    print('B128_AV_NATIVE_PASS',flush=True)
    def train(pair):
        variant,gpu=pair;name,out,row=run(variant,gpu,'train');d=json.loads((out/'result.json').read_text())
        assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps']==6563
        assert d['amp_skipped_steps']==d['training_heldout_identities']==0
        assert d['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[])
        assert (d['parameters'],d['trainable_parameters'],d['descriptor_dim'])==(99585158,1887366,5120)
        rows=[json.loads(line) for line in (out/'batch_orders.jsonl').read_text().splitlines()]
        assert len(rows)==6563
        for detail in rows:verify_detail(detail,False)
        audited=execute([sys.executable,'-u','audit_full_official_normal.py','--run-dir',str(out)],root/'audit',name,gpu)
        assert json.loads((out/'normal_cpu_audit.json').read_text())['status']=='PASS'
        states=execute([sys.executable,'-u','evaluate_rgbnt100_B128_reference_states.py','--run-dir',str(out),'--output',str(root/'states'/name)],root/'states',name,gpu)
        sd=json.loads((root/'states'/name/'result.json').read_text())
        assert sd['status']=='COMPLETE' and sd['state_cases']==4
        assert sd['full11_features_distance_equal_selected_normal'] and sd['state_tensor_versions_unchanged']
        teacher=json.loads((Path(args.anchor_run_dir)/'result.json').read_text())
        metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
        assert all(abs(sd['measurements']['00'][key]-teacher['full_metrics'][key])<1e-8 for key in metrics)
        assert (root/'states'/name/'state_00.csv').read_bytes()==(root/'reference'/('RGBNT100_B128ref_'+variant+'_reference')/'actual00.csv').read_bytes()
        sp=root/'states'/name/'normal_states.npz'
        path=out/'best_official_arrays.npz'
        return dict(name=name,gpu=gpu,train=row,audit=audited,states=states,states_raw=dict(path=str(sp),bytes=sp.stat().st_size,sha256=hashlib.sha256(sp.read_bytes()).hexdigest()),raw=dict(path=str(path),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()),best=dict(path=str(out/'best.pth'),bytes=(out/'best.pth').stat().st_size,sha256=hashlib.sha256((out/'best.pth').read_bytes()).hexdigest(),selected_epoch=d['best']['epoch']))
    with ThreadPoolExecutor(max_workers=2) as pool:trained=list(pool.map(train,CONTROLS))
    orders=[]
    for value in trained:
        rows=[json.loads(line) for line in (root/'training'/value['name']/'batch_orders.jsonl').read_text().splitlines()]
        orders.append([(r['epoch'],r['step'],r['names'],r['partial_set']) for r in rows])
    assert len(orders[0])==len(orders[1])==6563 and orders[0]==orders[1]
    save(root/'controller_result.json',dict(status='COMPLETE',dataset='RGBNT100',runs=trained,controls=2,
        native_updates=6,successful_updates=13126,additional_epochs=100,paired_sampling_exact=True,
        teacher_reference_head=8,teacher_reference_training_batch=128,reference00_full_parity=True,attention_packed_QKV_decay=0.,attention_decay_targets=5,modality_projected_query_normalization='per-head L2 times sqrt(8)',modality_query_parameters_added=0,correction_budget=.10,states_cases=8,normal_state_forward_record_executions=20580,routing_temperature=2.,routing_parameters_added=0,physical_gpus=[2,3],maximum_NN=2,
        frequency_value_weight_decay=0.,frequency_value_decay_targets=2,frequency_value_parameters_added=0,raw_preserved_pending_local_archive=True,missing_evaluated=False,new49=False,
        limits='Two full official normal50 existing AV controls on the same selected B128/head8 teacher and training-only bank. No new architecture/module/parameter/loss/descriptor. Historical AV comparison changes identity weights and their deterministic neighbor arrangement, not one factor. Teacher50 plus extra50, official benchmark used for teacher/config/epoch selection. Paper87.2/98.6 target remains; no missing/49/201NN or seed search.'))
    print('B128_AV_CONTROLLER_COMPLETE',flush=True)


if __name__=='__main__':main()
