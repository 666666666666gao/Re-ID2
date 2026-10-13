"""AO: a single teacher-semantic query condition on the complete AL RGB recipe."""
import run_r201al_RGB_route_temperature as parent
from rgbnt100_hdm_query_guidance import install

configuration = parent.configuration
VARIANTS = parent.VARIANTS
REVISION = ('AO RGBNT100 adds only pre-ATMoE frozen HDM relation evidence to M attention queries. '
    'Reuse existing router.to_modality_query at .1, shared across bands, for initial and final M reads; '
    'frequency-specific/source-specific values stay filtered. No new parameter, backbone/HDM forward, '
    'head, unfiltered value path, descriptor dimension or loss. Original teacher42 frozen, '
    'AL T2/cap.10/mainCE.25/hinge.6/reg.05/MFaux.001/unit3584/hardP8K8/BN2/partial0/two backwards unchanged. '
    'Matched ordinary and structured controls both receive the same guidance. Extra50/full8675train/1715Q/8575G; '
    'highest normal benchmarkmAP/earliesttie/allsix same selected weight and fixed fourstates. '
    'Hypothesis from previously absent HDM query condition and limited AL task-gradient diagnostics, not a proven cause. '
    'Only2026 GPU2/3, preserveK201/strongMS, no missing专项/49/seedsearch/T-P.')


def build(args, cfg, classes, cameras):
    model = install(parent.build(args, cfg, classes, cameras))
    model.anchor_record['method'] = REVISION
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    model.hdm_query_native_reads = []
    detail = parent.step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    detail.update(HDM_query_guidance=True, HDM_query_guidance_parameters_added=0,
        HDM_query_guidance_coefficient=.1, HDM_query_guidance_values='unchanged_filtered_band_source_tokens')
    if model.normal_priority_smoke:
        reads = model.hdm_query_native_reads
        assert len(reads) == 4
        assert [r['previous_cross_condition_present'] for r in reads] == [False, True, False, True]
        assert all(r['guide_l2_mean'] > 0 for r in reads)
        detail['HDM_query_guidance_native'] = list(reads)
    model.hdm_query_native_reads = []
    return detail


if __name__ == '__main__':
    runner = parent.parent.parent.parent.x.w.runner
    runner.configuration = configuration
    runner.assert_anchor_unchanged = parent.parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json = parent.parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader = parent.parent.parent.parent.make_loader
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
        partial_training='Same AL two forwards/BN2/partial0; only M queries get the source-specific frozen HDM guide. No missing training design.')
