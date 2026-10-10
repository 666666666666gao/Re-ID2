"""P: dataset-specific M/F auxiliary weights on the unchanged, completed O recipe."""
import run_r201o_same_state_contribution as parent

AUXILIARY_IDENTITY_WEIGHTS = {'MSVR310': .01, 'RGBNT100': .05}
REVISION = (
    'R201P single auxiliary-identity-weight factor relative to completed O: '
    'M/F weights .1->.01 for MSVR310 and .1->.05 for RGBNT100, identically in '
    'ordinary and axis controls. Network, initial teacher42/fresh experts42, '
    'P8K8/order/augmentations/Adam/additional50, full CE.25+hinge.6, auxiliary '
    'soft Triplet, other full losses, contribution.05/sigmoid20/same-state '
    'gallery targets, partial0 forwards/BN2/two backwards, relation-local PI '
    'and final unit5120 cap.10 stay unchanged. No new parameters or forwards. '
    'The selected O four-batch gradient ratios motivate this controlled test; '
    'they do not prove global conflict, a causal failure or improved retrieval. '
    'Preserve K201 best and completed evidence. Full official train/query/gallery, '
    'normal benchmark-mAP-best earliest ties/same six metrics, fixed-best six '
    'symmetric DeMo missing settings later, no new49 or failed seed search.'
)

configuration = parent.configuration
VARIANTS = parent.VARIANTS


def build(args, cfg, classes, cameras):
    model = parent.build(args, cfg, classes, cameras)
    original = list(model.loss_weights)
    assert original[1:3] == [.1, .1]
    coefficient = AUXILIARY_IDENTITY_WEIGHTS[args.dataset]
    model.loss_weights = original.copy()
    model.loss_weights[1:3] = [coefficient, coefficient]
    model.anchor_record['method'] = REVISION
    model.anchor_record['M_F_auxiliary_weight_previous'] = .1
    model.anchor_record['M_F_auxiliary_weight'] = coefficient
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    coefficient = model.anchor_record['M_F_auxiliary_weight']
    assert model.loss_weights[1:3] == [coefficient, coefficient]
    detail = parent.step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    detail['M_F_auxiliary_weight_previous'] = .1
    detail['M_F_auxiliary_weight'] = coefficient
    return detail


if __name__ == '__main__':
    native = parent.parent.parent
    runner = native.runner
    runner.configuration = configuration
    runner.assert_anchor_unchanged = native.assert_anchor_unchanged
    runner.write_json = native.k.parent.g.record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
                partial_training='Unchanged O partial0 computation; only M/F auxiliary weights change.')
