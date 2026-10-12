"""AE: matched strong-reference controls with a fixed outlet/relationship schedule."""
import json
from run_experiment import write_json
import run_full_official_frozen_anchor_experiment as runner
import run_r201y_MS_K4 as y
import msvr_strong_staged_identity as stage

REVISION=('AE normal MSVR310 only. Strong shared-DeMo50 teacher private5120 initialization, '
    'then additional50/newAdam and fixed five outlet epochs plus45 limited generalFusion epochs at .1LR. '
    'Axis frozen50 separates stage adaptation from stronger initialization; staged ordinary/axis '
    'have identical data/order/augmentation/budget/heads/loss/parameters. Staged noexpert DeMo '
    'has the same private5120 interface, teacher, full/main/base objectives, stages and705 updates '
    'but no M/F/route/projection/auxiliary objectives; parameter/compute differences disclosed. '
    'Original BACKBONE/three reducers and all core eval modes/GF buffers fixed. '
    'One optimizer/scheduler with GF groups registered initially, no GF gradient/state/weight-decay '
    'during outlet phase; no optimizer reset/addgroups at epoch6. Native staged smoke explicitly '
    'one successful outlet step then two relation steps, not five trained epochs. '
    'Full1032train591Q1055G/noholdout/benchmarkmAP-best earliestties/all6 sameweight/CMC/perquery/groups/'
    'curves/best-last/fourstate net gain. Strong private47.763/65.482 differs from full563249.426/69.036 '
    'and both are inherited baseline scores, not added-expert gain. Currentbase00 changes after stage2; '
    'cap.10 is not a teacher-drift bound. KeepY mean512/MFaux.001/mainCE.05/reg0/hinge.6/P16K4/'
    'BN2/partial0/two backwards. No new public block/local regions/FFT/loss/201NN/49/missingdesign/seedhunt.')

MODEL=None
ORIGINAL_LOADER=runner.make_loader
ORIGINAL_OPTIMIZER=runner.make_optimizer


def build(args,cfg,classes,cameras):
    global MODEL
    MODEL=stage.build(args,cfg,classes,cameras)
    MODEL.anchor_record['method']=REVISION
    return MODEL


def optimizer(cfg,model,center):
    if model.staged_adaptation:model.generalFusion.requires_grad_(True)
    result=ORIGINAL_OPTIMIZER(cfg,model,center)
    model.generalFusion.requires_grad_(False)
    return result


def loader(records,cfg,training,seed):
    if training:
        MODEL.stage_epoch=0 if MODEL.normal_priority_smoke else seed-42
    return ORIGINAL_LOADER(records,cfg,training,seed)


def record(path,value):
    if 'descriptor_dim' in value:value['descriptor_dim']=5120
    if 'pretraining' in value:
        value['pretraining']='Existing strong shared-DeMo50 E37 private5120 plus extra50; inherited baseline gain separated from adaptation and experts'
        value['registered_trainable_parameters']=value['trainable_parameters']+(9958912 if MODEL.staged_adaptation else 0)
        value['trainable_parameters_by_phase']=dict(outlets=value['trainable_parameters'],relations=value['trainable_parameters']+(9958912 if MODEL.staged_adaptation else 0))
    if value.get('status') in ('RUNNING','COMPLETE','SMOKE_PASS'):stage.assert_retained(MODEL)
    write_json(path,value)


if __name__=='__main__':
    runner.configuration=y.configuration
    runner.make_optimizer=optimizer
    runner.make_loader=loader
    runner.assert_anchor_unchanged=stage.assert_retained
    runner.write_json=record
    runner.main(builder=build,update=stage.step,variants=stage.CONTROLS,method_revision=REVISION,
        partial_training='Existing zero partial loss/BN2/two backwards. No missing training objective or new missing evaluation.')
