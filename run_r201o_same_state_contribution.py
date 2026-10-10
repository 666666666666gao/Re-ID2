"""O: same-state query/gallery contribution scores on the unchanged N recipe."""
import torch
from torch.nn import functional as F
import run_r201n_bounded_identity_shift as parent

REVISION=('R201O single contribution-reference factor relative to completed N: '
    'each detached controlled query is scored against the corresponding detached '
    'controlled gallery state, using identical positive/negative indices mined '
    'once from stopped-gradient00. N final unit5120 shift cap.10, main CE.25+'
    'hinge.6, auxiliarysoft/contribution.05/sigmoid20, partial0 forward/BN2/'
    'two backwards, K relation-local PI, P8K8/fresh heads/Adam/seed42/additional50 '
    'and frozen originalDeMo42 identity remain. No new parameters/forwards or '
    'copied teacher queries. MSVR310/RGBNT100 only; preserve K201 best. Full '
    'official train/query/gallery, benchmark mAP-best earliest ties/same six '
    'metrics, fixed-best six symmetric DeMo missing settings later; no new49. '
    'Same-state empirical margin is not dataset AP, a causal effect or a '
    'guarantee of calibration, distinct experts or a +1 improvement.')

configuration=parent.configuration
VARIANTS=parent.VARIANTS


@torch.no_grad()
def same_state_targets(states,labels):
    vectors={key:F.normalize(value.float(),dim=1) for key,value in states.items()}
    reference=vectors['00']
    with torch.autocast('cuda',enabled=False):
        similarity=reference@reference.T
    same=labels[:,None].eq(labels[None])
    positive=same & ~torch.eye(len(labels),device=labels.device,dtype=torch.bool)
    negative=~same
    assert positive.any(1).all() and negative.any(1).all()
    pos=similarity.masked_fill(~positive,torch.inf).argmin(1)
    neg=similarity.masked_fill(~negative,-torch.inf).argmax(1)
    scores={key:(value*(value[pos]-value[neg])).sum(1) for key,value in vectors.items()}
    target=torch.stack((scores['11']-scores['01'],scores['11']-scores['10'],
                        scores['11']-scores['10']-scores['01']+scores['00']),1)
    return target,pos,neg,scores


def build(args,cfg,classes,cameras):
    model=parent.build(args,cfg,classes,cameras)
    model.calibrator.targets=same_state_targets
    model.anchor_record['method']=REVISION
    model.anchor_record['contribution_reference']='Stopped-gradient corresponding gallery state; fixed00 positive/negative indices'
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    audit=model.contribution_audit
    assert not audit['target_requires_grad']
    detail['contribution_reference']='same_state_gallery_fixed00_indices'
    detail['contribution_target_mean']=[sum(row[i] for row in audit['targets'])/len(audit['targets']) for i in range(3)]
    detail['contribution_prediction_mean']=[sum(row[i] for row in audit['predictions'])/len(audit['predictions']) for i in range(3)]
    detail['contribution_loss_raw']=audit['loss']
    targets=torch.tensor(audit['targets'],dtype=torch.float32,device='cpu')
    detail['contribution_zero_loss_raw']=float(F.smooth_l1_loss(torch.zeros_like(targets),targets))
    return detail


if __name__=='__main__':
    runner=parent.parent.runner
    runner.configuration=configuration
    runner.assert_anchor_unchanged=parent.parent.assert_anchor_unchanged
    runner.write_json=parent.parent.k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Unchanged N partial0 computation; same-state contribution gallery only.')
