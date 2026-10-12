"""AD random candidate: match M/F auxiliary scorer precision before sampling comparison."""
import run_r201x_RGB_auxiliary_fp32 as parent
from rgb_auxiliary_modality_score_fp32 import install

configuration=parent.configuration
VARIANTS=parent.VARIANTS
REVISION=('AD random sampling candidate; actual second M and F projected auxiliary scorer FP32. '
    'Existing raw4D/main-eval calls unchanged, no parameter/RNG additions. All X model/loss/BN2/partial0/teacher42/heads42/newAdam/additional50/5120/P8K8 held. '
    'Fair sampling claim requires new random and hard ordinary/axis controls with identical arithmetic, native3 each first. '
    'AC failed ordinary Mscore gradient gate retained; FP16 underflow cause not proved. No zero-gradient gate relaxation/artificialgrad/newloss/missing专项/201NN/49/seedhunt.')


def build(args,cfg,classes,cameras):
    model=install(parent.build(args,cfg,classes,cameras))
    model.anchor_record.update(method=REVISION,sampling_treatment='random')
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    scorer=model.modality_expert.pool_score;before=scorer.auxiliary_fp32_calls
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    assert scorer.auxiliary_fp32_calls-before==1
    audit=dict(scorer.auxiliary_fp32_audit)
    assert audit['input_shape']==[64,7,512] and audit['input_dtype']==audit['output_dtype']=='torch.float32' and audit['output_requires_grad']
    detail.update(M_auxiliary_scorer_fp32=audit,sampling_treatment='random')
    return detail


if __name__=='__main__':
    runner=parent.w.runner
    runner.configuration=configuration
    runner.assert_anchor_unchanged=parent.w.assert_anchor_unchanged
    runner.write_json=parent.w.s.p.parent.parent.parent.k.parent.g.record

    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Matched AD M/Fscorer FP32, unchanged BN2/partial0/two backwards; only random versus deterministic train-neighbor group arrangement differs.')
