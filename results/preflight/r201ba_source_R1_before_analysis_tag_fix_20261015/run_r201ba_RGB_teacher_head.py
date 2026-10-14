"""BA: train-only identity initialization of the existing final 5120 classifier."""
import run_r201av_RGB_frequency_value_decay as parent
import run_r201ac_RGB_hard_identity as sampling
from rgbnt100_training_identity_head import initialize

configuration = parent.configuration
VARIANTS = parent.VARIANTS
make_optimizer = parent.make_optimizer
step = parent.step
REVISION = ('BA RGBNT100 changes only the initialization directions of existing final fused_classifier.weight '
    'versus AV. Whiten original frozen DeMo teacher42 full-official TRAIN class means by TRAIN population '
    'variance plus existing fused BN epsilon and gamma; normalize and retain each fresh random row norm. '
    'Other heads, final BN parameters/stats, all experts, conditions, route, FFT, optimizer, losses, sampling, '
    'params/RNG and final5120 inference graph unchanged. Uses existing immutable AB8675x5120 TRAIN bank '
    'from the same originalteacher; no query/gallery features or labels or new encoder passes for initialization. '
    'CPU preprocessing/init cost changes, no extra module forward/loss or optimizer updates. Final classifier '
    'remains trainable. This is known prototype initialization, not a new frequency mechanism, proof that '
    'random heads caused negative gains, or promised +1. AZ corrected10/01 metric failed and is not retained; '
    'AV Mqueryunit/FrawQ/fivepackedQKVdecay0/twoFvaluedecay0/T2/hardP8K8/B64/cap.10/CE.25/hinge.6/'
    'reg.05/MFaux.001 first11pureunit3584+scored512/BN2/partial0/two backwards retained. '
    'Teacher50+fresh expert-head42/newAdam/extra50, full8675train1715Q8575G/normalbenchmark mAP '
    'earliesttie/samebest6/CMC50/perquery/groups/bestlast/eightstates. Matched ordinary and structured '
    'controls use same bank, initialization and budget. Only26physicalGPU2/3/max2onepercard/noT-P/'
    'no201NN/missing49/failedseedhunt; preserve201K/AV/MSstrong/uniqueD.')


def build(args, cfg, classes, cameras):
    model = parent.build(args, cfg, classes, cameras)
    keys = set(model.state_dict())
    model = initialize(model, sampling.BANK)
    assert set(model.state_dict()) == keys
    model.anchor_record['method'] = REVISION
    return model


if __name__ == '__main__':
    import run_full_official_frozen_anchor_experiment as runner
    assert runner.make_optimizer is parent.parent.parent.original_make_optimizer
    runner.make_optimizer = make_optimizer
    runner.configuration = configuration
    runner.assert_anchor_unchanged = parent.parent.parent.parent.parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json = parent.parent.parent.parent.parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader = parent.parent.parent.parent.parent.parent.parent.make_loader
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
        partial_training='Same AV BN2/partial0/two backwards; only final classifier initial directions use existing TRAIN bank, not AZ corrected-state metric.')
