"""AS: normalize projected queries of only the existing modality spatial reader."""
import math
from types import MethodType
import torch
from torch.nn import functional as F
import run_r201aq_RGB_attention_decay as parent

configuration=parent.configuration
VARIANTS=parent.VARIANTS
REVISION=('AS RGBNT100 changes only the new modality-axis spatial attention projected Q: '
    'per-head L2 normalization times sqrt(head_dim), then the existing scaled dot-product '
    'attention and unchanged K/V/out projections. No normalization of keys/values or frequency '
    'spatial/crossband operators. This removes query amplitude from M spatial selection; '
    'query directions and conditional message directions remain trainable. No new parameter. '
    'Same fresh AQ teacher42/newAdam/expert50/packedQKVdecay0/T2/cap.10/hardP8K8/B64/'
    'mainCE.25/hinge.6/reg.05/aux.001/unit3584/scored512/BN2/partial0/two backwards/5120. '
    'AR actual full10290 AQ reads measured nonzero processed M key spatial variance but '
    'small projected queries/logit variation and near-uniform spatial probabilities; QK '
    'weight preservation alone did not yield +1. No foreground/identity/gain claim. '
    'Both source-structured and ordinary mixed-source controls receive the same Q policy. '
    'Complete8675train/1715Q8575G,50 epochs, same benchmarkmAP/earliesttie/allsix selectedweight; '
    'only2026GPU2/3/max2/onepercard, no201NN/missing专项/49/seedhunt/T-P.')


def modality_query_unit_attention(self,query,key,value,need_weights=False):
    assert self.batch_first and self.embed_dim==64 and self.num_heads==8
    assert self.dropout==0 and not need_weights and self.bias_k is None and self.bias_v is None and not self.add_zero_attn
    assert key is value
    batch,length,_=query.shape;head=self.embed_dim//self.num_heads
    query_sequence=query.transpose(0,1);key_sequence=key.transpose(0,1)
    q,k,v=F._in_projection_packed(query_sequence,key_sequence,key_sequence,self.in_proj_weight,self.in_proj_bias)
    q=q.view(length,batch*self.num_heads,head).transpose(0,1).view(batch,self.num_heads,length,head)
    k=k.view(key.shape[1],batch*self.num_heads,head).transpose(0,1).view(batch,self.num_heads,key.shape[1],head)
    v=v.view(value.shape[1],batch*self.num_heads,head).transpose(0,1).view(batch,self.num_heads,value.shape[1],head)
    q32=F.normalize(q.float(),dim=-1)*math.sqrt(head)
    if self.training:
        before=q.detach().float().norm(dim=-1);after=q32.detach().norm(dim=-1)
        self.projected_query_unit_audit.append(torch.stack((before.min(),before.mean(),after.min(),after.max())))
    q=q32.to(q.dtype)
    attended=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
    attended=attended.permute(2,0,1,3).contiguous().view(batch*length,self.embed_dim)
    attended=F.linear(attended,self.out_proj.weight,self.out_proj.bias).view(length,batch,self.embed_dim)
    return attended.transpose(0,1),None


def build(args,cfg,classes,cameras):
    model=parent.build(args,cfg,classes,cameras)
    attention=model.modality_expert.attention
    attention.projected_query_unit_audit=[]
    attention.forward=MethodType(modality_query_unit_attention,attention)
    model.anchor_record.update(method=REVISION,modality_projected_query_normalization='per-head L2 times sqrt(8)',
        modality_query_parameters_added=0,modality_query_existing_parameter_names_unchanged=True,
        modality_query_limit='Only M spatial projected Q amplitude removed. K/V/F/crossband unchanged. Not the full QKNorm method or an established identity-relevance mechanism.')
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    attention=model.modality_expert.attention;attention.projected_query_unit_audit=[]
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    values=torch.stack(attention.projected_query_unit_audit).detach().float().cpu()
    assert torch.isfinite(values).all() and values[:,0].min()>0
    assert (values[:,2:]-math.sqrt(8)).abs().max()<1e-5
    detail['modality_projected_query_normalization']=dict(policy='per-head L2 times sqrt(8)',parameters_added=0,
        attention_calls=len(values),pre_norm_min=float(values[:,0].min()),pre_norm_mean=float(values[:,1].mean()),
        post_norm_min=float(values[:,2].min()),post_norm_max=float(values[:,3].max()),
        projected_keys_values_unchanged=True,frequency_attention_unchanged=True,
        normalization_dtype='torch.float32',attention_dtype='existing autocast',
        limit='Norms measured in FP32 before casting to existing AMP dtype. No added forward/loss/update; no entropy or foreground target.')
    return detail


if __name__=='__main__':
    runner=parent.parent.parent.parent.parent.x.w.runner
    assert runner.make_optimizer is parent.original_make_optimizer
    runner.make_optimizer=parent.make_optimizer
    runner.configuration=configuration
    runner.assert_anchor_unchanged=parent.parent.parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json=parent.parent.parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader=parent.parent.parent.parent.parent.make_loader
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Same AQ BN2/partial0/two backwards. Only M projected query amplitude normalized; no missing training design.')
