"""Train interface B's residuals to improve a fixed-reference retrieval margin."""
import torch
from torch.nn import functional as F

from relation_frequency_interface import RelationFrequencyInterfaceDeMo


def joint_retrieval_gain(states, labels, margin=.02, temperature=.1):
    with torch.autocast('cuda', enabled=False):
        base = states['00'].float()
        reference = F.normalize(base.detach(), dim=1)
        similarity = reference @ reference.T
        same = labels[:, None].eq(labels[None])
        positive = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        negative = ~same
        assert positive.any(1).all() and negative.any(1).all()
        pos = similarity.masked_fill(~positive, torch.inf).argmin(1)
        neg = similarity.masked_fill(~negative, -torch.inf).argmax(1)
        direction = reference[pos] - reference[neg]
        scores = {key: (F.normalize(states[key].detach().float(), dim=1) * direction).sum(1)
                  for key in ('00', '10', '01')}
        target = torch.stack(tuple(scores.values()), 1).max(1).values + margin
        weights = (-scores['00'] / temperature).sigmoid()
        # Identical forward value; remove the direct base-coordinate gradient.
        # Expert paths may still update the shared backbone through their inputs.
        query = states['11'].float() - (base - base.detach())
        current = (F.normalize(query, dim=1) * direction).sum(1)
        loss = (weights * (target - current).relu()).sum() / weights.sum()
    audit = dict(loss=float(loss.detach()), margin=margin, temperature=temperature,
                 weight_mean=float(weights.mean()), active_fraction=float((target > current).float().mean()),
                 base_margin_mean=float(scores['00'].mean()), full_margin_mean=float(current.detach().mean()),
                 stopped_target_mean=float(target.mean()), positive_indices=pos.tolist(), negative_indices=neg.tolist(),
                 stopped_reference=True, stopped_targets=True, direct_base_gradient_removed=True,
                 reference='One stopped current-training-batch base00 gallery and the same hardest legal positive/negative for all states')
    return loss, audit


class RetrievalUtilityDeMo(RelationFrequencyInterfaceDeMo):
    gain_weight = .25
    gain_margin = .02
    gain_temperature = .1

    def controlled_states(self, *args, **kwargs):
        states = super().controlled_states(*args, **kwargs)
        if self.training:
            self._gain_states = states
        return states

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3):
        output = super().forward(x, label=label, cam_label=cam_label, view_label=view_label,
                                 return_pattern=return_pattern)
        if not self.training:
            return output
        assert label is not None
        states = self._gain_states
        del self._gain_states
        gain, self.gain_audit = joint_retrieval_gain(states, label, self.gain_margin, self.gain_temperature)
        return (*output[:-1], output[-1] + self.gain_weight * gain)
