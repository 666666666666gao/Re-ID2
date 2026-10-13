"""AL: soften the existing joint router on the exact frozen AG RGBNT100 recipe."""
import run_r201ag_RGB_coordinate_metric as parent
import rgbnt100_tempered_router as tempered

configuration = parent.configuration
VARIANTS = parent.VARIANTS
REVISION = ('AL RGBNT100 joint-logit Softmax temperature2 versus frozen AG temperature1. '
    'All joint a_S+c_b+psi logits divided by2; no uniform-mixture floor, additional expert, '
    'head, loss, parameter, descriptor dimension or backbone adaptation. '
    'Same original DeMo50 teacher42/freshheads42/newAdam/additional50, hardP8K8/B64, '
    'mainCE.25/hinge.6/reg.05/MFaux.001/cap.10/BN2/partial0/two backwards/5120. '
    'Both ordinary mixed-source and structured dual-axis controls use temperature2. '
    'Current AG/AK selected routes concentrate on low bands; this is a testable routing '
    'hypothesis, not proof that high-band information is absent or temperature improves retrieval. '
    'Fresh native3 percontrol checks gradients/reload and routing-only FP32 T1 parity/no-psi '
    'factorization before formal50; highest normal benchmarkmAP/earliesttie/all6sameweight. '
    'Only2026physicalGPU2/3/max2/onepercard; preserveK201/strongMS; no201NN/missing专项/49/seedhunt.')


def build(args, cfg, classes, cameras):
    model = tempered.build(args, cfg, classes, cameras)
    model.anchor_record['method'] = REVISION
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    model.router.capture_temperature_native = model.normal_priority_smoke
    model.router.temperature_native_inputs = None
    detail = parent.step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    detail['routing_temperature'] = tempered.TEMPERATURE
    detail['routing_parameters_added'] = 0
    if model.normal_priority_smoke:
        detail['routing_temperature_native_probes'] = tempered.native_probes(model.router)
    model.router.temperature_native_inputs = None
    return detail


if __name__ == '__main__':
    runner = parent.parent.parent.x.w.runner
    runner.configuration = configuration
    runner.assert_anchor_unchanged = parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json = parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader = parent.parent.parent.make_loader
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
        partial_training='AG hard record/augmentation/partial sequence, BN2/partial0/two backwards retained; only joint-logit temperature2. Existing independent10/01 paths remain unchanged.')
