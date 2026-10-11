"""Q: isolate contribution regression on completed P MSVR310, retaining task-trained gates."""
import run_r201p_auxiliary_balance as parent

CONTRIBUTION_WEIGHTS = {'MSVR310': 0., 'RGBNT100': .05}
REVISION = (
    'R201Q single contribution-regression-weight factor relative to P: MSVR310 '
    '.05->0, RGBNT100 retains.05. The existing prediction/gate head and sigmoid20 '
    'are unchanged and receive the original final retrieval-task gradient. '
    'Same-state stopped-gallery targets, fixed00 indices and raw regression '
    'diagnostics are still computed; no calibrated-contribution claim. P M/F '
    'auxiliary weights MSVR.01/RGB100.05, all other full losses, CE.25+hinge.6, '
    'partial0 computation/BN2/two backwards, relation-local PI, cap.10, network, '
    'parameters/5120D, frozen originalDeMo42, fresh experts/heads/Adam/seed42, '
    'P8K8/orders/augmentations/additional50 remain. First test only MSVR matched '
    'ordinary/axis on additionally authorized GPU0/1, while original P RGB100 '
    'GPU2/3 proceeds unchanged. Preserve K201 best. Four selected P batches '
    'motivate this control, not sustained/global conflict, causal proof or a '
    'guaranteed normal/missing +1. Full official train/query/gallery, fixed '
    'normal benchmark-mAP-best earliest ties/six metrics; paper-six only after '
    'normal stage closure, no new49 or failing-direction seed sweep.'
)

configuration = parent.configuration
VARIANTS = parent.VARIANTS


def build(args, cfg, classes, cameras):
    model = parent.build(args, cfg, classes, cameras)
    assert model.contribution_loss_weight == .05
    coefficient = CONTRIBUTION_WEIGHTS[args.dataset]
    model.contribution_loss_weight = coefficient
    model.anchor_record['method'] = REVISION
    model.anchor_record['contribution_regression_weight_previous'] = .05
    model.anchor_record['contribution_regression_weight'] = coefficient
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    coefficient = model.anchor_record['contribution_regression_weight']
    assert model.contribution_loss_weight == coefficient
    detail = parent.step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    detail['contribution_regression_weight_previous'] = .05
    detail['contribution_regression_weight'] = coefficient
    detail['contribution_regression_effective'] = coefficient * detail['contribution_loss_raw']
    return detail


if __name__ == '__main__':
    native = parent.parent.parent.parent
    runner = native.runner
    runner.configuration = configuration
    runner.assert_anchor_unchanged = native.assert_anchor_unchanged
    runner.write_json = native.k.parent.g.record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
                partial_training='Unchanged P computation; only MSVR contribution regression weight zero, existing task gates retained.')
