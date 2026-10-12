"""AK: warm the existing RGB AG outlets, then adapt the last CLIP visual block."""
import run_full_official_frozen_anchor_experiment as runner
import run_r201ag_RGB_coordinate_metric as parent
from run_r201ac_RGB_hard_identity import make_loader as hard_loader
from run_experiment import write_json
import rgbnt100_lastblock_staged as stage

configuration = parent.configuration
VARIANTS = stage.CONTROLS
REVISION = ('AK RGBNT100 last-existing-visual-block staged adaptation relative to AG. '
    'Same original DeMo50 teacher42, fresh expert/heads42, whole5120 identity-coordinate outlet, '
    'FFT/twoaxis/conditional-update/joint-route/relation-localPI/cap.10, M/F scored512 CE and unit3584 metric, '
    'mainCE.25/hinge.6/reg.05/MFaux.001, ADhard P8K8/B64 exact record/augmentation/partial sequence, '
    'BN2/partial0/two backwards/newAdam/additional50. '
    'Only existing BACKBONE.base.transformer.resblocks.11 becomes trainable after five outlet epochs, '
    'at .1 of the current scheduled visual group LR (registered nominal5e-6; native warmup3.5e-6); '
    'existing warmup/cosine noise retained, actual LR logged; other11 visual blocks, '
    'three reducers/generalFusion/allcoreevalmodes and buffers retained. No new parameters/head/publicblock/loss. '
    'Both ordinary/axis staged controls; native one outlet+two visual updates before each formal50. '
    'Currentbase00 can move afterstage2; not original frozen teacher, cap is not teacher-drift protection. '
    'Full8675train1715Q8575G/noholdout/50benchmarkmAP-best earliesttie/all6 andcurves/GT/fourstates. '
    'Source/profile/native are not a promisedgain or proof that frozen features caused failure. '
    'Only2026GPU2/3/no201NN/missing专项/49/seedhunt/T-P; keepK201 and strongMS results.')
MODEL = None
ORIGINAL_OPTIMIZER = runner.make_optimizer


def build(args, cfg, classes, cameras):
    global MODEL
    MODEL = stage.build(args, cfg, classes, cameras)
    MODEL.anchor_record['method'] = REVISION
    return MODEL


def optimizer(cfg, model, center):
    stage.tail(model).requires_grad_(True)
    result = ORIGINAL_OPTIMIZER(cfg, model, center)
    ids = {id(p) for p in stage.tail(model).parameters()}
    groups = [g for g in result[0].param_groups if id(g['params'][0]) in ids]
    assert {id(g['params'][0]) for g in groups} == ids
    assert all(g['lr'] == 5e-6 for g in groups)
    stage.tail(model).requires_grad_(False)
    return result


def loader(records, cfg, training, seed):
    if training:
        MODEL.stage_epoch = 0 if MODEL.normal_priority_smoke else seed - 42
    return hard_loader(records, cfg, training, seed)


def record(path, value):
    if 'descriptor_dim' in value:
        value['descriptor_dim'] = 5120
    if 'pretraining' in value:
        value['pretraining'] = 'Original fullofficial RGBNT100 DeMo50 teacher42, then additional50/newAdam; five outlet epochs plus45 last-visual-block adaptation'
        value['registered_trainable_parameters'] = value['trainable_parameters'] + MODEL.tail_parameter_count
        value['trainable_parameters_by_phase'] = dict(outlets=value['trainable_parameters'],
            visual_tail=value['trainable_parameters'] + MODEL.tail_parameter_count)
        value['freeze_identity_scope'] = MODEL.anchor_record['retained_identity_scope']
    if value.get('status') in ('RUNNING', 'COMPLETE', 'SMOKE_PASS'):
        stage.assert_retained(MODEL)
        value['effective_trainable_parameters_at_record'] = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    write_json(path, value)


if __name__ == '__main__':
    runner.configuration = configuration
    runner.make_optimizer = optimizer
    runner.make_loader = loader
    runner.assert_anchor_unchanged = stage.assert_retained
    runner.write_json = record
    runner.main(builder=build, update=stage.step, variants=VARIANTS, method_revision=REVISION,
        partial_training='AG same hard-schedule images/partial0/BN2/two backwards. Last visual block: no gradient or Adam state in first5 epochs, .1 visualLR afterward; no missing objective.')
