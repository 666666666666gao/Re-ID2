"""U: preserve T mean512 CE outlet; supervise metric on unit3584 first11 coords."""
from types import MethodType
import torch
from torch.nn import functional as F


def install_projected_auxiliary(model):
    keys=set(model.state_dict())
    frozen=("modality_expert.pool_score.weight","frequency_expert.pool_score.weight")
    parameters=dict(model.named_parameters())
    assert sum(parameters[n].numel() for n in frozen)==1024
    for n in frozen:parameters[n].requires_grad_(False)
    original_forward=model.forward
    original_mpool=model.modality_expert.pool
    original_fpool=model.frequency_expert.pool
    frame={};active=False

    def projected_hook(name):
        def hook(module,inputs,output):
            if not active:return
            frame[name+'_projection_calls']+=1
            if name not in frame:
                assert output.shape[1:]==(7,512) and output.requires_grad
                # Same FP32 per-relation normalization used immediately in fuse.
                frame[name]=F.normalize(output.float(),dim=-1)
        return hook

    def neck_hook(name):
        def hook(module,inputs):
            if active:
                frame[name+'_neck_calls']+=1
                assert torch.equal(inputs[0],frame[name+'_pooled'])
        return hook

    def mpool(self,evidence,eligible):
        if not active:return original_mpool(evidence,eligible)
        frame['M_pool_calls']+=1
        assert frame['M_pool_calls'] in (1,2)
        if frame['M_pool_calls']==1:return original_mpool(evidence,eligible)
        assert eligible.all()
        result=frame['M']
        frame['M_pooled']=result.mean(1)
        return result

    def fpool(self,evidence,available,structured):
        if not active:return original_fpool(evidence,available,structured)
        frame['F_pool_calls']+=1
        assert frame['F_pool_calls'] in (1,2)
        if frame['F_pool_calls']==1:return original_fpool(evidence,available,structured)
        assert available.all()
        frame['F_pooled']=frame['F'].mean(1)
        return frame['F_pooled']

    def forward(self,x,label=None,cam_label=None,view_label=None,partial=False,return_states=False):
        nonlocal active
        active=bool(self.training and not partial)
        if active:
            frame.clear()
            frame.update({name+'_'+key:0 for name in ('M','F') for key in ('projection_calls','pool_calls','neck_calls')})
        output=original_forward(x,label=label,cam_label=cam_label,view_label=view_label,
                                partial=partial,return_states=return_states)
        if active:
            expected_projection_calls=3 if self.normal_priority_smoke else 2
            assert all(frame[n+'_projection_calls']==expected_projection_calls and frame[n+'_pool_calls']==2 and frame[n+'_neck_calls']==1 for n in ('M','F'))
            assert torch.equal(output[3],frame['M_pooled']) and torch.equal(output[5],frame['F_pooled'])
            assert output[3].shape==output[5].shape==(len(x['RGB']),512)
            assert output[3].requires_grad and output[5].requires_grad
            self.projected_auxiliary_features=(output[3],output[5])
            self.projected_auxiliary_metric_features=tuple(F.normalize(frame[n].flatten(1),dim=1) for n in ('M','F'))
            assert all(v.shape==(len(x['RGB']),3584) and v.requires_grad for v in self.projected_auxiliary_metric_features)
            self.projected_auxiliary_forward_audit=dict(
                source='First main11 actual PM/PF output, per512 unit direction, uniform relation mean512',
                projection_calls={n:frame[n+'_projection_calls'] for n in ('M','F')},
                pool_calls={n:frame[n+'_pool_calls'] for n in ('M','F')},
                neck_calls={n:frame[n+'_neck_calls'] for n in ('M','F')},
                feature_shape=list(output[3].shape),feature_requires_grad=True,
                metric_feature_shape=list(self.projected_auxiliary_metric_features[0].shape),
                metric_unit_norm_error=max(float((v.norm(dim=1)-1).abs().max().detach()) for v in self.projected_auxiliary_metric_features),
                metric_source='actual_first11_per512_unit_directions_flattened_whole_unit3584',
                feature_dtype=str(output[3].dtype),controlled_pool_first_unchanged=True,
                main11_first_projection_selected=True,extra_neck_projection_calls=0,auxiliary_pool_scores_frozen=list(frozen),frozen_parameters=1024)
            frame.clear()
        active=False
        return output

    model.modality_projection.register_forward_hook(projected_hook('M'))
    model.frequency_projection.register_forward_hook(projected_hook('F'))
    model.modality_neck.register_forward_pre_hook(neck_hook('M'))
    model.frequency_neck.register_forward_pre_hook(neck_hook('F'))
    model.modality_expert.pool=MethodType(mpool,model.modality_expert)
    model.frequency_expert.pool=MethodType(fpool,model.frequency_expert)
    model.forward=MethodType(forward,model)
    assert set(model.state_dict())==keys
    model.anchor_record['M_F_auxiliary_outlet']=dict(
        previous='Independent pre-condition raw M/F pools512',
        current='Actual first11 projected per-relation unit directions, uniform relation mean512',
        heads_and_coefficients_unchanged=True,parameters_added=0,extra_neck_projection_calls=0,
        auxiliary_dtype='FP32 deployed directions; previous raw pools under native AMP',
        pooling_domain='Uniform mean over seven actual projected relation directions; old independent auxiliary pool_score weights frozen because only stopped controlled targets still use them',
        frozen_parameter_names=list(frozen),frozen_parameter_count=1024,
        controlled10_01_pool='Unchanged first independent pools',
        inference='Hooks/pool replacement inactive in eval; original5120 graph')
    model.anchor_record['M_F_auxiliary_metric_outlet']=dict(source='First actual11 per512 units flattened to unit3584 for existing softTriplet',
        CE='Existing mean512 scores/head/neck unchanged',parameters_added=0,normal_forward_changed=False)
    return model
