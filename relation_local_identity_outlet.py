"""R201K: share the existing PI projection across seven relation-specific inputs."""
import torch
from torch.nn import functional as F

from rgbnt201_identity_outlet import IdentityCoordinateAxis


class RelationLocalPIAxis(IdentityCoordinateAxis):
    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible,
             relation_mass=None, relation_frequency=None):
        original_inputs = (base, modality, frequency, gates, use_m, use_f,
                           eligible, relation_mass, relation_frequency)
        with torch.autocast('cuda', enabled=False):
            base, modality, frequency, gates = (value.float() for value in (base, modality, frequency, gates))
            batch, relations, dim = modality.shape
            anchor = base[:, 1536:5120].reshape(batch, relations, dim).norm(dim=-1, keepdim=True).detach()
            if relation_mass is None:
                assert use_m and not use_f
                logits = self.router.relation_score(modality).squeeze(-1)
                relation_mass = logits.masked_fill(~eligible, -torch.inf).softmax(1)
            weights = relation_mass.float() * eligible.sum(1, keepdim=True)
            delta = torch.zeros_like(base[:, 1536:5120]).reshape(batch, relations, dim)
            if use_m:
                direction = F.normalize(self.modality_projection(modality), dim=-1)
                delta = delta + self.residual_scale[0] * gates[:, :1, None] * direction * anchor
            if use_f:
                assert relation_frequency is not None
                direction = F.normalize(self.frequency_projection(relation_frequency.float()), dim=-1)
                delta = delta + self.residual_scale[1] * gates[:, 1:2, None] * direction * anchor
            if use_m and use_f:
                pi_input = torch.cat((modality, relation_frequency.float()), -1)
                interaction = F.normalize(self.interaction_projection(pi_input), dim=-1)
                delta = delta + self.residual_scale[2] * gates[:, 2:, None] * interaction * anchor
                if self.normal_priority_smoke:
                    assert pi_input.shape == (batch, 7, 1024) and interaction.shape == (batch, 7, 512)
                    errors = []
                    # The normalized PI vector of S must not read other slots.
                    # This does not assert invariance of the final globally normalized
                    # descriptor, conditions or gates, which still couple relations.
                    with torch.no_grad():
                        pattern = torch.linspace(.01, .17, 1024, device=pi_input.device)
                        indices = torch.arange(relations, device=pi_input.device)
                        for relation in range(relations):
                            changed = pi_input.detach().clone()
                            changed[:, indices != relation] += pattern
                            changed_pi = F.normalize(self.interaction_projection(changed), dim=-1)
                            error = float((changed_pi[:, relation] - interaction.detach()[:, relation]).abs().max())
                            assert error == 0.
                            errors.append(error)
                    self.relation_pi_native_audit.update(pi_input_shape=list(pi_input.shape),
                        pi_output_shape=list(interaction.shape), per_relation_outside_slot_errors=errors)
            delta = delta * weights[..., None] * eligible[..., None]
            private = torch.cat((base[:, :1536], base[:, 1536:5120] + delta.flatten(1)), 1)
            output = F.normalize(private, dim=1)
            if self.normal_priority_smoke and not (use_m and use_f):
                with torch.no_grad():
                    original = IdentityCoordinateAxis.fuse(self, *original_inputs)
                assert torch.equal(output.detach(), original)
                state = '10' if use_m else '01'
                self.relation_pi_native_audit['state_' + state + '_parent_equal'] = True
            return output
