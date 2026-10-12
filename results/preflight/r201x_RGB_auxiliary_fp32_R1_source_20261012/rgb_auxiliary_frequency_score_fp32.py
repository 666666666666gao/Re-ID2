"""FP32 scoring only for the actual projected F auxiliary relation pool."""
from types import MethodType
import torch


def install(model):
    scorer=model.frequency_expert.pool_score
    original=scorer.forward
    assert scorer.bias is None
    scorer.auxiliary_fp32_calls=0

    def forward(self,value):
        # Existing raw/controlled evidence is 4D; only actual auxiliary relations are 3D.
        if value.ndim==3:
            assert self.training and value.shape[1:]==(7,512)
            with torch.autocast('cuda',enabled=False):
                result=original(value.float())
            self.auxiliary_fp32_calls+=1
            self.auxiliary_fp32_audit=dict(input_shape=list(value.shape),input_dtype=str(value.dtype),output_dtype=str(result.dtype),output_requires_grad=result.requires_grad)
            return result
        return original(value)

    scorer.forward=MethodType(forward,scorer)
    model.anchor_record['F_auxiliary_scorer_precision']=dict(scope='Actual second projected F relation scoring only',precision='FP32',parameters_added=0,
        unchanged_calls='All 4D independent/raw/controlled01 scoring uses the original method',normal_inference_graph_unchanged=True,
        evidence='W ordinary native3 nnz0/3/0 versus sameZ/weight/scaledfull upstream FP32 localVJP512/512/512; original native failure retained, new pair native required')
    return model
