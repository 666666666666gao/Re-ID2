"""AA: scale only existing generalFusion update learning rates by .1."""
import run_r201z_MS_modality_adapt as z

configuration=z.configuration
VARIANTS=z.VARIANTS
REVISION='AA MS-only actual optimizer-step learning-rate factor versus closed Z: existing generalFusion parameter groups use .1 of their current nominal scheduled LR only during the same optimizer update, then restore nominal group LR. This scales warmup/cosine/noise/floor/bias ratios consistently without changing scheduler state, nonGF groups, gradients, loss, forwards, BN, optimizer calls, initialization or sampling. Same originalDeMo42 teacher50/freshheads42/newAdam/additional50/705updates/P16K4/5120/twoaxes/FFT/PI/route/gates/mean512/MFaux.001/mainCE.05/reg0/hinge.6/cap.10/currentadapting00. GF9958912 remains active/eval, frozen4/GFbuffers retained; total100821638/active13081734 unchanged and two controls matched. Closed ZbestE2 axis47.122164892/63.620981387, lastE50 41.471663388/56.683587140, loss1.708/hinge.110 vs early; closednormal4 expertincrement-.017077649/noR1changes. This motivates smaller pretrained-module updates, not proves excessiveLR causal or guaranteesgain; knownhyperparametercontrol notpaperinnovation. Freshnative3each/allactive/strictreload/directPM PF/actualLRgroupcontract before50; allofficial1032train591Q1055G/noholdout/normalbenchmarkmAP-best earliesttie/all6/CMC/perquery/groups/curves/pairedZorders. No201NN/new49/missingselection/failedseedhunt; X55293 GPU2/3 unchanged. Original50+extra50 disclosed.'
FACTOR=.1

def build(args,cfg,classes,cameras):
    model=z.build(args,cfg,classes,cameras)
    model.anchor_record['method']=REVISION
    model.anchor_record['generalFusion_learning_rate_control']=dict(factor=FACTOR,scope='Existing GF groups only during original optimizer step',
        nominal_scheduler_unchanged=True,nonGF_LR_unchanged=True,gradients_losses_forward_BN_unchanged=True,
        warmup_cosine_noise_floor_and_bias_ratios_all_scaled=True,nominal_group_LR_restored_after_update=True,parameters_added=0)
    return model

def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    gf_ids={id(p) for p in model.generalFusion.parameters()}
    assert all(len(g['params'])==1 for g in optimizer.param_groups)
    groups=[g for g in optimizer.param_groups if id(g['params'][0]) in gf_ids]
    assert {id(g['params'][0]) for g in groups}==gf_ids
    nominal=[g['lr'] for g in groups]
    nonGF=[g['lr'] for g in optimizer.param_groups if id(g['params'][0]) not in gf_ids]
    for group,rate in zip(groups,nominal):group['lr']=FACTOR*rate
    effective=[g['lr'] for g in groups]
    detail=z.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    for group,rate in zip(groups,nominal):group['lr']=rate
    assert [g['lr'] for g in groups]==nominal
    assert [g['lr'] for g in optimizer.param_groups if id(g['params'][0]) not in gf_ids]==nonGF
    detail['generalFusion_learning_rate_control']=dict(factor=FACTOR,groups=len(groups),parameter_elements=sum(g['params'][0].numel() for g in groups),
        nominal_min=min(nominal),nominal_max=max(nominal),effective_min=min(effective),effective_max=max(effective),
        nominal_group_LR_restored_after_update=True,nonGF_LR_unchanged=True)
    return detail

if __name__=='__main__':
    runner=z.y.w.runner;runner.configuration=configuration;runner.assert_anchor_unchanged=z.assert_retained;runner.write_json=z.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Exact Z partial0/BN2/two backwards; only GF update learning rate .1, every nominal scheduled group restored afterward.')
