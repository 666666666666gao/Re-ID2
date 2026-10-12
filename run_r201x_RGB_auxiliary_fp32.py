"""X: RGB W .001 recipe, with only the actual F auxiliary scorer in FP32."""
import run_r201w_reduced_auxiliary as w
from rgb_auxiliary_frequency_score_fp32 import install

configuration=w.configuration
VARIANTS=w.VARIANTS
REVISION=('RGB-only numerical repair of failed W native: actual projected second F auxiliary scoring in FP32; '
    'all 4D original/controlled01 calls and normal evaluation unchanged. Same W M/Faux.001, mainCE.25/reg.05, '
    'S score pooling512, hinge.6/cap.10/PI/FFT/two experts/router/gates/params, frozen originalDeMo50/newheads42/Adam/additional50, '
    'P8K8/order/BN2/partial0/two backwards. Both ordinary/axis changed identically and require fresh native3 each before formal50. '
    'Compared with completed S, both aux coefficient and scorer precision differ; no single-factor performance causality claim. '
    'W failure and observed3-step precision sensitivity retained; allgradient gate not relaxed, no missing/201/49/seed rescue.')


def build(args,cfg,classes,cameras):
    assert args.dataset=='RGBNT100'
    model=install(w.build(args,cfg,classes,cameras))
    model.anchor_record['method']=REVISION
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    scorer=model.frequency_expert.pool_score;before=scorer.auxiliary_fp32_calls
    detail=w.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    assert scorer.auxiliary_fp32_calls-before==1
    audit=dict(scorer.auxiliary_fp32_audit)
    assert audit['input_shape']==[64,7,512] and audit['input_dtype']==audit['output_dtype']=='torch.float32' and audit['output_requires_grad']
    detail['F_auxiliary_scorer_fp32']=audit
    return detail


if __name__=='__main__':
    runner=w.runner;runner.configuration=configuration;runner.assert_anchor_unchanged=w.assert_anchor_unchanged
    runner.write_json=w.s.p.parent.parent.parent.k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Unchanged W RGB partial0/BN2/two backwards; only actual second F auxiliary scorer FP32.')
