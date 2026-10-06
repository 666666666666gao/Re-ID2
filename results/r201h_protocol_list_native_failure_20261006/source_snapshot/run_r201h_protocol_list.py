"""R201H: matched full-view AP objectives on the unchanged G201 zero-partial fixture."""
import argparse
import sys

import torch
import run_full_official_frozen_anchor_experiment as runner
import run_r201g_normal_priority as g
from original_identity_anchor import assert_anchor_unchanged
from protocol_list_objective import smooth_ap
from rgbnt201_projection_probe import VARIANTS


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-mask',required=True,choices=('label','camera'))
    policy,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    reference=policy.reference_mask
    revision=('R201H primary full-view Smooth-AP on the closed G201 fixture. '
        'The label/camera arms differ only in same-ID/same-camera junk eligibility; '
        'temperature.01/coefficient1/fusedCE.25 fixed. Original DeMo50 reference and '
        'identity encoder frozen, fresh experts/heads/newAdam/additional50, seed42/P8K8/B64, '
        '5120D unit descriptor, full classification/auxiliary/contribution unchanged. '
        'Partial CE and cross-triplet weights zero; forwards/BN2/mining/two backwards retained. '
        'All3951 official training records/171 identities remain; no-positive AP query/batch '
        'omits only that ranking contribution, not data or classification. '
        'Camera-aware AP is a candidate training-protocol comparison, not a code fix or '
        'new loss invention. Label-only AP here must be retrained: prior E/F used nonzero '
        'partial objectives. Highest official normal mAP/earliesttie selects one best; '
        'benchmark-selected, not untouched test. Normal and missing +2 never guaranteed. '
        'Reference arm: '+reference)

    def build(args,cfg,classes,cameras):
        assert args.dataset=='RGBNT201'
        args.primary_reference_mask=reference
        model=g.build(args,cfg,classes,cameras)
        model.anchor_record['method']=revision
        return model

    def update(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
        assert len(batch[-1])==len(set(batch[-1]))==64
        calls,metric=0,{}

        def losses(score,feature,target,target_cam):
            nonlocal calls
            calls+=1
            if calls!=1:
                return loss_fn(score,feature,target,target_cam)
            assert feature.shape==(64,5120)
            ap_loss,info=smooth_ap(feature,target,target_cam,reference)
            metric.update(info,full_AP_loss=float(ap_loss.detach()),primary_full_metric='protocol_smooth_AP_unit5120',metric_coefficient=1.)
            if model.normal_priority_smoke:
                gradient=torch.autograd.grad(ap_loss,feature,retain_graph=True)[0]
                assert torch.isfinite(gradient).all()
                if info['valid_AP_queries']:
                    assert gradient.abs().sum()>0
                else:
                    assert gradient.abs().sum()==0 and ap_loss==0
                metric['AP_feature_gradient_finite']=True
                metric['AP_feature_gradient_l1']=float(gradient.abs().sum())
            return .25*xent(score,target)+ap_loss

        detail=g.step(model,batch,optimizer,scaler,losses,xent,retained,variant)
        assert calls==len(model.loss_weights)
        detail.update(metric,reference_mask=reference,all64_observations_in_classification=True,added_losses=0)
        return detail

    runner.assert_anchor_unchanged=assert_anchor_unchanged
    runner.write_json=g.record
    runner.main(builder=build,update=update,variants=VARIANTS,method_revision=revision,
        partial_training='G zero weights unchanged; all partial images/forwards/BN2/mining/backwards still computed. Only primary full-list reference policy changes between arms.')


if __name__=='__main__':
    main()
