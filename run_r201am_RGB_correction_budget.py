"""AM: one final correction-budget factor versus the completed AL pair."""
import run_r201al_RGB_route_temperature as parent
import rgbnt100_correction_budget as budget

configuration=parent.configuration
VARIANTS=parent.VARIANTS
REVISION=('AM RGBNT100 final unit-descriptor correction budget epsilon0.10->0.20 versus completed AL. '
    'Only the existing bound parameter changes for both ordinary/axis controls; joint-logit temperature2, '
    'all existing network/modules/heads/parameters/frozen original teacher42, mainCE.25/hinge.6/reg.05/'
    'MFaux.001/hardP8K8/B64/BN2/partial0/two backwards/5120 and additional50/newAdam stay. '
    'Actual AL selected-epoch training clipping .985568 and active margin fraction .838263, plus '
    'saved full-QG shifts near .10, motivate a controlled budget test; not proof of causation or gain. '
    'No classifier initialization change, additional expert/loss/descriptor dimension, backbone adaptation '
    'or unbounded correction. Real native3 each, full50 and fixed8GTstates; highest normal benchmarkmAP '
    'earliesttie/sameweight all6. Only2026physicalGPU2/3/max2, keepK201/strongMS, no201NN/missing专项/'
    'new49/failedseedhunt or temperature-power read/control.')


def build(args,cfg,classes,cameras):
    assert args.dataset=='RGBNT100'
    model=parent.build(args,cfg,classes,cameras)
    assert model.anchor_record['final_unit_descriptor_bound']['epsilon']==budget.EPSILON
    model.anchor_record.update(method=REVISION,correction_budget_previous=.10,
        correction_budget=budget.EPSILON,correction_budget_parameters_added=0)
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    report=detail['final_descriptor_bound_full']
    assert report['epsilon']==budget.EPSILON and report['bounded_shift_max']<=budget.EPSILON+1e-6
    if model.normal_priority_smoke:
        fixture=detail['final_descriptor_bound_geometry_fixture']
        assert fixture['epsilon']==budget.EPSILON and fixture['finite_derivatives']
        assert all(r['epsilon']==budget.EPSILON and r['bounded_shift_max']<=budget.EPSILON+1e-6
            for r in detail['final_descriptor_bound_controlled_reports'].values())
    detail.update(correction_budget=budget.EPSILON,correction_budget_parameters_added=0)
    return detail


if __name__=='__main__':
    runner=parent.parent.parent.parent.x.w.runner
    runner.configuration=configuration
    runner.assert_anchor_unchanged=parent.parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json=parent.parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader=parent.parent.parent.parent.make_loader
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Exactly AL full/partial computation, BN2/two backwards/routerT2 retained; only existing final correction budget .10->.20.')
