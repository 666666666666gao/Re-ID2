"""R: only final full-view CE0.25->0.05 relative to the completed Q MSVR recipe."""
import run_r201q_task_gate as parent

FINAL_FULL_CE_WEIGHT=.05
REVISION=(
    'R201R single final-full-CE-weight factor relative to completed Q MSVR310: '
    '0.25->0.05 only for the first/full deployed5120D classifier objective. '
    'The second zero-weight partial CE is not rescaled, auxiliary CE/soft '
    'Triplets use the original loss_fn unchanged. Main unit hinge.6, MFaux.01, '
    'contribution regression0/task-gate head/sigmoid20/same-state raw targets, '
    'cap.10/partial0 forwards/BN2/two backwards, all network/params/FFT/expert/'
    'route/PM-PF-PI/5120, originalDeMo42 teacher50 frozen, fresh heads/experts/'
    'Adam/seed42/P8K8/orders/augmentations/additional50 stay. Native verifies '
    'the observed two-call xent contract and effective coefficient before '
    'formal. Config ID_LOSS_WEIGHT.25 remains for auxiliary loss construction; '
    'effective final-full coefficient.05 is explicitly logged. Completed '
    'Pselected4batch PM/PF CE gradients~3x metric with near-zero positive '
    'cosines motivate a controlled test, not proof of harmful CE or Qgradients. '
    'Qnormal reg0 gave no+1; do not expand failed seeds. Only MSVR matched '
    'ordinary/axis GPU0/1, preserve live P RGB1002/3 and solepaperqueue/K201 '
    'best. Full officialtrain/Q/G, earliest normal benchmark-mAP-best same '
    'all6/CMC50/perquery/groups/best-last; normal first and paper-six later, '
    'no new49, calibrated/causal/global conflict or guaranteedgain claim.')

configuration=parent.configuration
VARIANTS=parent.VARIANTS

def build(args,cfg,classes,cameras):
    assert args.dataset=='MSVR310' and cfg.MODEL.ID_LOSS_WEIGHT==.25
    model=parent.build(args,cfg,classes,cameras)
    assert model.contribution_loss_weight==0. and model.loss_weights[1:3]==[.01,.01]
    model.final_full_CE_weight=FINAL_FULL_CE_WEIGHT
    model.anchor_record.update(method=REVISION,final_full_CE_weight_previous=.25,
        final_full_CE_weight=FINAL_FULL_CE_WEIGHT,auxiliary_config_ID_LOSS_WEIGHT=.25,
        zero_weight_partial_CE_not_rescaled=True)
    return model

def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    calls=0;observed={}
    def full_only_xent(score,target):
        nonlocal calls
        calls+=1
        raw=xent(score,target)
        if calls==1:
            observed['final_full_CE_unweighted']=float(raw.detach())
            return (model.final_full_CE_weight/.25)*raw
        assert calls==2
        return raw
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,full_only_xent,retained,variant)
    assert calls==2 and model.final_full_CE_weight==FINAL_FULL_CE_WEIGHT
    detail.update(observed,final_full_CE_weight_previous=.25,final_full_CE_weight=FINAL_FULL_CE_WEIGHT,
        final_full_CE_effective=FINAL_FULL_CE_WEIGHT*observed['final_full_CE_unweighted'],
        xent_observed_calls=calls,zero_weight_partial_CE_not_rescaled=True,auxiliary_config_ID_LOSS_WEIGHT=.25)
    return detail

if __name__=='__main__':
    native=parent.parent.parent.parent.parent;runner=native.runner
    runner.configuration=configuration;runner.assert_anchor_unchanged=native.assert_anchor_unchanged
    runner.write_json=native.k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Exactly unchanged Q partial0 computation/CE/mining/BN2/two backwards; only final full-view CE.25->.05.')
