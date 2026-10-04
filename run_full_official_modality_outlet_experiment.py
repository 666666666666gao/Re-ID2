"""Single added PM-outlet supervision on full official fresh50, matched across experts."""
from modality_outlet_alignment_axis import build
from run_full_official_control_experiment import step as control_step
from run_full_official_experiment import main


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    detail = control_step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    detail.update(model.modality_alignment_audit)
    return detail


def builder(args, cfg, classes, cameras):
    args.gate_gradient_mode, args.pooling = 'measurement_only', 'original_mean'
    args.relation_weight, args.alignment_weight, args.alignment_temperature = .1, .1, .07
    args.modality_alignment_weight = .1
    return build(args, cfg, classes, cameras)


if __name__ == '__main__':
    main(builder=builder, update=step, variants=('axis_shared', 'frequency_shared', 'twins_shared'),
        method_revision='Full-official M4 versus existing matched M3a: add only .1 positive-identity public alignment on actual routed PM weighted normalized anchor/relationship mean7. Stopped same-full-view public gallery, different observations; existing F relation/alignment, FFT, experts, joint routing, contribution measurement-only gates and descriptor unchanged. No new parameters or historical weights.',
        partial_training='Unchanged one uniform proper subset per batch versus stopped full gallery, .25 CE/.5 triplet; PM alignment is applied only on the full training view, identically for all three expert variants.')
