"""Condition the existing relation-local interaction bottleneck on its identity base."""
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F


def install_identity_reference_interaction(model):
    """Add one 512->64 gate inside existing PI; zero init preserves AV at admission."""
    projection=model.interaction_projection
    assert len(projection)==4 and isinstance(projection[2],nn.GELU)
    assert projection[1].weight.shape==(64,1024) and projection[3].weight.shape==(512,64)
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    projection.register_parameter('identity_reference_weight',nn.Parameter(
        torch.zeros(64,512,device=projection[1].weight.device,dtype=projection[1].weight.dtype)))
    assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),cuda_rng)
    parent_fuse=model.fuse
    reference={}

    def condition_hidden(module,inputs,output):
        assert output.shape[1:]==(7,64)
        gate=2*torch.sigmoid(F.linear(reference['unit'],projection.identity_reference_weight.float()))
        return output*gate

    def fuse(self,base,modality,frequency,gates,use_m,use_f,eligible,
             relation_mass=None,relation_frequency=None):
        # The original frozen relation coordinates supply semantics, not just norm.
        reference['unit']=F.normalize(base[:,1536:5120].detach().float().reshape(-1,7,512),dim=-1)
        return parent_fuse(base,modality,frequency,gates,use_m,use_f,eligible,
                           relation_mass,relation_frequency)

    projection[2].register_forward_hook(condition_hidden)
    model.fuse=MethodType(fuse,model)
    model.anchor_record['identity_reference_interaction']=dict(
        placement='Existing relation-local PI GELU hidden64, before existing output Linear64->512',
        formula='Wo[GELU(We LN(concat(M_S,F_S))) * 2 sigmoid(Wb normalize(stopgrad(b_S)))]',
        reference='Original frozen seven identity relation coordinates, matching S; no query/gallery pair or label input',
        gate_shape=[7,64],reference_weight_shape=[64,512],parameters_added=32768,
        initialization='Exact zero Wb => multiplicative1; all existing tensors and CPU/CUDA RNG retained',
        CPU_CUDA_RNG_exact=True,initial_gate_weight_zero=bool(torch.count_nonzero(projection.identity_reference_weight)==0),
        same_for_ordinary_and_axis=True,standalone_PM_PF_unchanged=True,
        inactive_states='00/10/01 never execute PI; existing identity and single-expert paths retained',
        cap='Existing final unit5120 .10 bound remains inside original fuse after full PI aggregation',
        limit='Known conditional modulation. Not calibrated utility, causal rescue proof, new information, or guaranteed gain. Ordinary control receives the same extra capacity.')
    return model
