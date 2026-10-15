"""Existing AV/BC native and normal contracts for one bounded expert-seed wave."""
from rgbnt100_expert_seed_recipe import PARAMETERS


def verify_detail(d,key,native):
    assert d['loss']==d['full_loss'] and d['optimizer_updated'] and d['freeze_identity_encoder']
    assert d['partial_loss_effective']==d['partial_CE_weight']==d['partial_triplet_weight']==0
    assert d['routing_temperature']==2. and d['routing_parameters_added']==0
    assert d['sampling_identities']==d['sampling_instances']==8 and d['sampling_batch']==64
    assert d['sampling_treatment']=='hard' and d['primary_margin']==.6 and d['auxiliary_original_soft_triplet']
    assert d['M_F_auxiliary_weight']==.001 and d['parent_main_CE_weight']==.25 and d['parent_contribution_regression_weight']==.05
    assert d['M_F_auxiliary_Triplet_outlet']=='actual_first11_per512_unit_relation_coordinates3584'
    assert d['M_F_auxiliary_CE_outlet']=='unchanged_existing_score_FP32_pool512'
    assert d['auxiliary_metric_feature_dimension']==3584 and d['auxiliary_CE_feature_dimension']==512
    assert d['final_descriptor_bound_full']['epsilon']==.10 and d['final_descriptor_bound_full']['bounded_shift_max']<=.10+1e-6
    q=d['modality_projected_query_normalization']
    assert q['parameters_added']==0 and q['attention_calls']==84 and q['pre_norm_min']>0
    assert q['projected_keys_values_unchanged'] and q['frequency_attention_unchanged']
    assert abs(q['post_norm_min']-8**.5)<1e-5 and abs(q['post_norm_max']-8**.5)<1e-5
    decay=d['attention_projection_decay'];assert decay['old']==1e-4 and decay['new']==0. and decay['packed_includes_V']
    assert len(decay['targets'])==5
    value=d['frequency_value_input_decay']
    assert value['old']==1e-4 and value['new']==0. and value['parameters_added']==0
    assert value['targets']==['frequency_expert.values.1.1.weight','frequency_expert.values.2.1.weight']
    assert value['low_band_shared_M_bias_LN_policy_unchanged']
    h=d['hard_identity_sampling']
    assert h['post_sampler_RNG_exact'] and h['original_record_multiset_exact'] and h['new_random_draws']==0 and h['P']==h['K']==8
    c=d['projected_auxiliary_forward_contract']
    assert c['main11_first_projection_selected'] and c['extra_neck_projection_calls']==0
    assert c['pool_calls']==dict(M=2,F=2) and c['neck_calls']==dict(M=1,F=1)
    assert c['projection_calls']==dict(M=3 if native else 2,F=3 if native else 2)
    assert c['metric_feature_shape']==[64,3584] and c['feature_shape']==[64,512] and c['feature_dtype']=='torch.float32'
    if key=='BC':
        gate=d['identity_reference_interaction']
        assert gate['parameters_added']==32768 and gate['base_reference_stop_gradient'] and gate['final_descriptor_dim']==5120
        if native:assert gate['unscaled_task_gradient_norm']>0
    if native:
        assert d['fused_BN_calls']==2
        p=d['routing_temperature_native_probes']
        assert p['temperature1_exact_parent'] and p['no_psi_product_max_error']<1e-6 and p['temperature2_vs1_max_probability_difference']>0
        assert all(v['finite'] and v['weighted_auxiliary_l1']>0 for v in d['direct_PM_PF_auxiliary_gradient'].values())
        pi=d['relation_PI_native']
        assert pi['per_relation_outside_slot_errors']==[0.]*7 and pi['state_10_parent_equal'] and pi['state_01_parent_equal']


def verify_model_record(d,key,seed):
    assert (d['parameters'],d['trainable_parameters'],d['descriptor_dim'])==(*PARAMETERS[key],5120)
    assert (d['train_records'],d['query_records'],d['gallery_records'])==(8675,1715,8575)
    assert d['arguments']['seed']==seed and d['arguments']['expert_recipe']==key
    assert d['anchor']['anchor_seed']==42 and d['anchor']['expert_seed']==seed and d['anchor']['selected_recipe']==key
