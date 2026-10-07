"""Only replace I's broadcast PI with a shared per-relation PI projection."""
import identity_coordinate_three_dataset as coordinate
import run_full_official_frozen_anchor_experiment as runner
import run_r201i_primary_margin as parent
from original_identity_anchor import assert_anchor_unchanged
from relation_local_identity_outlet import RelationLocalPIAxis
from rgbnt201_projection_probe import VARIANTS

REVISION = ('R201K single PI-outlet factor relative to completed I. Same existing '
    '1024->64->512 projection tensors now process concat(M_S,F_S) at each of seven '
    'relations, normalized per512; replace mean(M_S)+globalF and one broadcast vector. '
    'No new parameters/experts or output coordinates. Existing conditions, joint route, '
    'PM/PF, gates, anchors, final5120 norm and controlled00/10/01 unchanged as code. '
    'I primary unit5120 hinge.3, M/F/base/common auxiliary soft and CE/weights, partial0 '
    'forward/BN2/mining/two backwards, frozen original DeMo42, fresh heads/Adam, '
    'B64/K8-4-16/additional50 and data sampling unchanged. Full official train/query/gallery, '
    'normal mAP-best earliest tie, benchmark selected; three normal datasets before missing49. '
    'Direct PI input isolation does not imply strict end-to-end source independence: '
    'conditions/gates/global normalization still couple sources. Benefit/cause/novelty '
    'and all3+2 remain unproven until actual fair training/GT/missing/seeds.')


def build(args, cfg, classes, cameras):
    # This assignment affects only this new process; original source files and
    # the running J process remain unchanged.
    coordinate.IdentityCoordinateAxis = RelationLocalPIAxis
    model = parent.build(args, cfg, classes, cameras)
    model.relation_pi_native_audit = {}
    model.anchor_record['method'] = REVISION
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    if model.normal_priority_smoke:
        model.relation_pi_native_audit = {}
    detail = parent.step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    detail['PI_outlet'] = 'shared_relation_local'
    if model.normal_priority_smoke:
        audit = model.relation_pi_native_audit
        assert audit['pi_input_shape'] == [64, 7, 1024]
        assert audit['pi_output_shape'] == [64, 7, 512]
        assert audit['per_relation_outside_slot_errors'] == [0.] * 7
        assert audit['state_10_parent_equal'] and audit['state_01_parent_equal']
        detail['relation_PI_native'] = dict(audit)
    return detail


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = parent.g.record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
        partial_training='Exactly I partial0 with the same forwards/BN2/mining/two backwards; only PI aggregation changes.')
