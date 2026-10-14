"""AZ: supervise existing corrected identity states with the existing auxiliary metric loss."""
from types import MethodType
import torch
from torch.nn import functional as F
import run_r201av_RGB_frequency_value_decay as parent

configuration = parent.configuration
VARIANTS = parent.VARIANTS
make_optimizer = parent.make_optimizer
REVISION = ('AZ RGBNT100 changes only M/F auxiliary softTriplet feature versus AV: '
    'pure first11 PM/PF per512 unit correction directions -> existing controlled10/01 '
    'corrected identity relation coordinates1536:5120, whole-normalized3584. '
    '10/01 are independent expert states with the other expert conditions, psi and I off; '
    'they are not claimed to be main11 conditional PF features. Scored512 CE remains '
    'the original first11 pure projected directions. Actual corrected-state amplitude, '
    'relation weighting, gates and cap.10 already exist and are reused without extra '
    'attention/projection/pool/neck/backbone forward or loss/optimizer updates. Two '
    'additional array normalizations replace the metric inputs, not a new neural branch. '
    'No parameter, RNG, final5120 eval graph, teacher freeze, BN or missing change. '
    'Same AV M-query-unit/F-raw-query/fivepackedQKVdecay0/twoFvaluedecay0/'
    'T2/hardP8K8/B64/mainCE.25/hinge.6/reg.05/MFaux.001/BN2/partial0/two backwards. '
    'Fresh original teacher42/fresh expert-head42/newAdam/extra50; full8675train/'
    '1715Q8575G, normalbenchmarkmAP earliesttie/all six same selected checkpoint, '
    'paired ordinary/axis and frozen normal states. AY fixed AX reads stronger F '
    'spatial selection and some independent correction rescues without net deployment '
    'gain; this motivates a metric-target test, not proof of a sole cause or guaranteed '
    'gain. AV is the stronger parent, failed AX F-query factor is not retained. '
    'Only26physicalGPU2/3/max2/onepercard/noT-P/no201NN/missing49/failedseedhunt.')


def build(args, cfg, classes, cameras):
    model = parent.build(args, cfg, classes, cameras)
    keys = set(model.state_dict())
    previous_states = model.controlled_states
    previous_forward = model.forward
    captured = {}

    def controlled_states(self, base, m0, f0, eligible, available, full):
        states = previous_states(base, m0, f0, eligible, available, full)
        if self.training:
            assert not captured
            features = tuple(F.normalize(states[key][:, 1536:5120].float(), dim=1) for key in ('10', '01'))
            assert all(v.shape == (len(base), 3584) and v.requires_grad and torch.isfinite(v).all() for v in features)
            captured['features'] = features
        return states

    def forward(self, x, label=None, cam_label=None, view_label=None, partial=False, return_states=False):
        active = bool(self.training and not partial)
        if active:
            assert not captured
        output = previous_forward(x, label=label, cam_label=cam_label, view_label=view_label,
                                  partial=partial, return_states=return_states)
        if active:
            features = captured.pop('features')
            self.projected_auxiliary_metric_features = features
            self.corrected_state_metric_gradients = {}
            self.corrected_state_metric_visits = {}
            if self.normal_priority_smoke:
                def metric_gradient(name):
                    def record(gradient):
                        value = gradient.detach().float()
                        visits = self.corrected_state_metric_visits.setdefault(name, [])
                        assert len(visits) < 3
                        assert torch.isfinite(value).all() and value.abs().sum() > 0
                        phase = ('projection_auxiliary_probe', 'expert_auxiliary_probe', 'scaled_full_loss_backward')[len(visits)]
                        visits.append(dict(phase=phase, finite=True, norm=float(value.norm()), l1=float(value.abs().sum())))
                        if len(visits) == 3:
                            self.corrected_state_metric_gradients[name] = dict(finite=True, native_hook_visits=3,
                                scaled_backward_norm=visits[-1]['norm'], scaled_backward_l1=visits[-1]['l1'],
                                existing_diagnostic_visits=visits[:2], actual_backward_visit=visits[-1]['phase'])
                    return record
                for name, value in zip(('M', 'F'), features):
                    value.register_hook(metric_gradient(name))
            error = max(float((v.norm(dim=1)-1).abs().max().detach()) for v in features)
            assert error < 1e-6
            self.corrected_state_metric_audit = dict(states=['10', '01'], relationship_slice=[1536, 5120],
                feature_dimension=3584, feature_requires_grad=True, metric_unit_norm_error=error,
                controlled_state_calls=1, extra_module_forwards=0, existing_CE512_unchanged=True,
                source='Existing independent corrected10/01 final descriptors; relation3584 subvector whole-normalized. Not pure delta or main11 conditioned branches.')
            self.projected_auxiliary_forward_audit.update(metric_feature_source=self.corrected_state_metric_audit['source'],
                                                         metric_feature_shape=list(features[0].shape))
        return output

    model.controlled_states = MethodType(controlled_states, model)
    model.forward = MethodType(forward, model)
    assert set(model.state_dict()) == keys
    model.anchor_record.update(method=REVISION, auxiliary_metric_feature='existing_corrected10_01_relation_coordinates_unit3584',
        auxiliary_CE_feature='unchanged_first11_pure_projected_existing_score_FP32_pool512', auxiliary_metric_parameters_added=0)
    model.anchor_record['M_F_auxiliary_outlet']['metric_outlet'] = 'AZ existing corrected10/01 relationship3584, whole-normalized, existing auxiliary softTriplet.'
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    detail = parent.step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    assert model.projected_auxiliary_metric_features is None
    detail.update(M_F_auxiliary_Triplet_outlet='existing_corrected10_01_relation_coordinates_unit3584',
                  corrected_state_metric_audit=dict(model.corrected_state_metric_audit))
    if model.normal_priority_smoke:
        assert set(model.corrected_state_metric_gradients) == {'M', 'F'}
        assert all(len(v) == 3 for v in model.corrected_state_metric_visits.values())
        detail['corrected_state_metric_native_gradients'] = dict(model.corrected_state_metric_gradients)
    return detail


if __name__ == '__main__':
    import run_full_official_frozen_anchor_experiment as runner
    assert runner.make_optimizer is parent.parent.parent.original_make_optimizer
    runner.make_optimizer = make_optimizer
    runner.configuration = configuration
    runner.assert_anchor_unchanged = parent.parent.parent.parent.parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json = parent.parent.parent.parent.parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader = parent.parent.parent.parent.parent.parent.parent.make_loader
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
        partial_training='AV BN2/partial0/two backwards retained; only full M/F auxiliary metric uses existing independent corrected10/01 relation coordinates instead of pure projected corrections.')
