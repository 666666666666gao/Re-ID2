"""N: one bounded final-descriptor factor on the paired weak-dataset M recipe."""
import run_r201m_primary_margin06 as parent
from bounded_identity_descriptor import MAX_UNIT_SHIFT,install_bound
from rgbnt201_projection_probe import VARIANTS


REVISION=('R201N one-factor final unit5120 shift bound epsilon0.10 relative to M: '
    'bound complete parent M/F/I candidate against fixed available identity base, '
    'same ordinary/axis variants, inactive exact candidate, no new trainable parameters. '
    'M final CE0.25+hinge0.6, auxiliary soft losses/contribution, partial0 forward/BN2/'
    'two backwards, K relation-local PI, L B64P8K8, respective frozen originalDeMo42 '
    'teacher50, fresh heads/Adam/scheduler/AMP512/seed42/additional50 and5120 stay. '
    'Only MSVR310/RGBNT100; preserve K201 best and its source/weights. Full official '
    'train/query/gallery, normal mAP-best earliest ties, fixed best for six symmetric '
    'DeMo missing settings; no new49 or five-seed search from failure. '
    'Geometric protection is not new identity evidence, calibrated contribution, '
    'causal diagnosis or guaranteed retrieval improvement. No new source/runtime '
    'acceptance is implied by preparation.')

configuration=parent.configuration


def build(args,cfg,classes,cameras):
    model=install_bound(parent.build(args,cfg,classes,cameras))
    model.anchor_record['method']=REVISION
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    report=model.descriptor_bound_reports[('11',True)]
    assert report['bounded_shift_max']<=MAX_UNIT_SHIFT+1e-6
    detail['final_descriptor_bound_full']=dict(report)
    if model.normal_priority_smoke:
        controlled={state:dict(model.descriptor_bound_reports[(state,True)]) for state in ('10','01','11')}
        assert all(value['bounded_shift_max']<=MAX_UNIT_SHIFT+1e-6 for value in controlled.values())
        detail['final_descriptor_bound_controlled_reports']=controlled
        detail['final_descriptor_bound_geometry_fixture']=dict(model.descriptor_bound_fixture)
        detail['K_controlled_PI_equalities_scope']='Parent pre-bound PI evidence; final10/01 receive the same new descriptor bound'
    return detail


if __name__=='__main__':
    parent.runner.configuration=configuration
    parent.runner.assert_anchor_unchanged=parent.assert_anchor_unchanged
    parent.runner.write_json=parent.k.parent.g.record
    parent.runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Unchanged M partial0 computation and auxiliarysoft; only complete final unit5120 descriptor shift bound0.10.')
