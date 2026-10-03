"""P1-A: give the existing frequency block an explicit retrieval metric weight."""
import torch
from torch.nn import functional as F

from mass_axis_collaboration import MassAxisCollaborationDeMo


class MetricMassAxisCollaborationDeMo(MassAxisCollaborationDeMo):
    def __init__(self, classes, cfg, cameras):
        super().__init__(classes,cfg,cameras)
        # Reuse the existing F scalar as a metric-budget logit. No parameters,
        # modules or descriptor coordinates are added. Initial budget=0.05;
        # the original noncompetitive F gate still controls its actual use.
        with torch.no_grad():
            self.residual_scale[1].copy_(torch.logit(self.residual_scale.new_tensor(.05)))

    def fuse(self,base,modality,frequency,gates,use_m,use_f,eligible,relation_mass=None):
        original=super().fuse(base,modality,frequency,gates,use_m,use_f,eligible,relation_mass)
        if not use_f:
            return original
        with torch.autocast('cuda',enabled=False):
            identity=original[:,:-512].float()
            direction=F.normalize(self.frequency_projection(frequency.float()),dim=1)
            weight=self.residual_scale[1].sigmoid()*gates[:,1:2].float()
            tail=direction*identity.norm(dim=1,keepdim=True).detach()*(weight/(1-weight)).sqrt()
            # Final global normalization gives [sqrt(1-w)*identity_hat,
            # sqrt(w)*frequency_hat]. The original identity coordinates and
            # M/I amplitudes before final normalization remain unchanged.
            return torch.cat((identity,tail),1)
