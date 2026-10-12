"""AG: preserve scored512 CE; retain actual first11 unit3584 auxiliary metric coordinates."""
from types import MethodType
import torch
from torch.nn import functional as F


def install_projected_auxiliary(model):
    keys=set(model.state_dict())
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
        weights=self.pool_score(frame['M']).squeeze(-1).softmax(1)
        result=frame['M']*weights[...,None]*7
        frame['M_pooled']=result.mean(1)
        return result

    def fpool(self,evidence,available,structured):
        if not active:return original_fpool(evidence,available,structured)
        frame['F_pool_calls']+=1
        assert frame['F_pool_calls'] in (1,2)
        if frame['F_pool_calls']==1:return original_fpool(evidence,available,structured)
        assert available.all()
        weights=self.pool_score(frame['F']).squeeze(-1).softmax(1)
        frame['F_pooled']=(frame['F']*weights[...,None]).sum(1)
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
                source='First main11 actual PM/PF output, per512 unit direction, existing-score relation pool512',
                projection_calls={n:frame[n+'_projection_calls'] for n in ('M','F')},
                pool_calls={n:frame[n+'_pool_calls'] for n in ('M','F')},
                neck_calls={n:frame[n+'_neck_calls'] for n in ('M','F')},
                feature_shape=list(output[3].shape),feature_requires_grad=True,
                feature_dtype=str(output[3].dtype),controlled_pool_first_unchanged=True,
                main11_first_projection_selected=True,extra_neck_projection_calls=0,existing_pool_scores_reused=True,
                metric_feature_shape=[len(x['RGB']),3584],metric_feature_source='First main11 PM/PF per512 unit directions flattened and whole-normalized')
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
        current='Actual first11 projected per-relation unit directions, existing-score relation pool512',
        heads_and_coefficients_unchanged=True,parameters_added=0,extra_neck_projection_calls=0,
        auxiliary_dtype='FP32 deployed directions; previous raw pools under native AMP',
        metric_outlet='AG softTriplet on actual first11 per512 units flattened+whole-normalized3584; scored512 CE unchanged',
        pooling_domain='Existing M/F pool_score reused over seven projected relation directions, previously raw bands/source-band evidence; no new score parameters/calls',
        controlled10_01_pool='Unchanged first independent pools',
        inference='Hooks/pool replacement inactive in eval; original5120 graph')
    return model
