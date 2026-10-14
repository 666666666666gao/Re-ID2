"""AX: apply the existing projected-query normalization to the three F spatial readers."""
import math
from types import MethodType
import torch
import run_r201av_RGB_frequency_value_decay as parent
from run_r201as_RGB_query_normalization import modality_query_unit_attention

configuration=parent.configuration
VARIANTS=parent.VARIANTS
make_optimizer=parent.make_optimizer
TARGETS=tuple('frequency_expert.attention.'+str(i) for i in range(3))
REVISION=('AX RGBNT100 changes only projected Q of the three existing F spatial attentions: '
    'per-head FP32 L2 times sqrt(8), using the existing AS packed projection/SDPA operation. '
    'M query unit, F K/V/output projections, cross-band attention, AV two F value weight '
    'decay0 groups, all other optimizer/loss/sampling/head/budget policies remain unchanged. '
    'Both ordinary mixed-source and structured controls receive this change. AW actual '
    'selected AV fullQG retained F spatial variation but near-uniform F spatial probabilities '
    'and small projected Q. This motivates a reading-scale test, not causal evidence or '
    'guaranteed identity gain. No added parameters, loss, forward, descriptor dimensions '
    'or normalization of keys/values. Original teacher42 frozen/freshheads42/newAdam/'
    'extra50/full8675train/1715Q8575G/hardP8K8/B64/T2/cap.10/mainCE.25/hinge.6/'
    'reg.05/MFaux.001 actualunit3584+scored512/BN2/partial0/two backwards/5120. '
    'NormalbenchmarkmAP/earliesttie/all six same checkpoint. Only2026physicalGPU2/3/'
    'max2/onepercard; preserveK201/strongMS/Dunique; no201NN/missing49/seedhunt/T-P.')


def build(args,cfg,classes,cameras):
    model=parent.build(args,cfg,classes,cameras)
    for attention in model.frequency_expert.attention:
        attention.projected_query_unit_audit=[]
        attention.forward=MethodType(modality_query_unit_attention,attention)
    model.anchor_record.update(method=REVISION,
        frequency_projected_query_normalization='per-head L2 times sqrt(8)',
        frequency_query_targets=list(TARGETS),frequency_query_parameters_added=0,
        frequency_cross_band_unchanged=True,
        modality_query_limit='AS M projectedQ unit retained; AX also normalizes F spatial Q. K/V/output/crossband unchanged.')
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    for attention in model.frequency_expert.attention:attention.projected_query_unit_audit=[]
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    records=[]
    for attention in model.frequency_expert.attention:
        values=torch.stack(attention.projected_query_unit_audit).detach().float().cpu()
        assert torch.isfinite(values).all() and values[:,0].min()>0
        assert (values[:,2:]-math.sqrt(8)).abs().max()<1e-5
        records.append(dict(attention_calls=len(values),pre_norm_min=float(values[:,0].min()),
            pre_norm_mean=float(values[:,1].mean()),post_norm_min=float(values[:,2].min()),
            post_norm_max=float(values[:,3].max())))
    detail['modality_projected_query_normalization']['frequency_attention_unchanged']=False
    detail['frequency_projected_query_normalization']=dict(policy='per-head L2 times sqrt(8)',
        targets=list(TARGETS),parameters_added=0,bands=records,
        projected_keys_values_unchanged=True,cross_band_unchanged=True,
        normalization_dtype='torch.float32',attention_dtype='existing autocast',
        extra_forwards=0,extra_losses=0,extra_updates=0)
    return detail


if __name__=='__main__':
    import run_full_official_frozen_anchor_experiment as runner
    assert runner.make_optimizer is parent.parent.parent.original_make_optimizer
    runner.make_optimizer=make_optimizer
    runner.configuration=configuration
    runner.assert_anchor_unchanged=parent.parent.parent.parent.parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json=parent.parent.parent.parent.parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader=parent.parent.parent.parent.parent.parent.parent.make_loader
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='AV BN2/partial0/two backwards retained; only F spatial projected Q normalized.')
