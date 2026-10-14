"""Read actual fixed AX attention: both M and three F spatial Q are normalized."""
import math
import torch
from torch.nn import functional as F
from dual_axis import RELATIONS
from probe_r201at_AS_spatial_attention import PreviousAudit, spatial_statistics, STATISTICS, READING


class AttentionAudit(PreviousAudit):
    def __init__(self,model,n):
        super().__init__(model,n,())
        self.structured=model.structured
        self.handles.append(model.modality_expert.value.register_forward_hook(self.modality_values))
        for band,network in enumerate(model.frequency_expert.values):
            for stage,module in enumerate(network):
                self.handles.append(module.register_forward_hook(self.frequency_values(band,stage)))

    def modality_values(self,module,args,out):
        assert out.shape==(self.batch,3,3,self.n,64)
        for source in range(3):
            for band in range(3):
                self.fields[f'M_source{source}_band{band}_after_GELU_spatial_variance']=out[:,source,band].float().var(-2,unbiased=False).mean(-1).detach()

    def frequency_values(self,band,stage):
        def hook(module,args,out):
            assert out.shape[:3]==(self.batch,3,self.n)
            labels=('after_LN','after_linear','after_GELU')
            for source in range(3):
                prefix=f'F_source{source}_band{band}_'
                value=out[:,source].float()
                self.fields[prefix+labels[stage]+'_spatial_variance']=value.var(-2,unbiased=False).mean(-1).detach()
                if stage==0:
                    raw=args[0][:,source].float();assert raw.shape[-1]==512
                    self.fields[prefix+'input_spatial_variance']=raw.var(-2,unbiased=False).mean(-1).detach()
                    self.fields[prefix+'input_mean_square']=raw.square().mean((-2,-1)).detach()
                if stage==2:self.fields[prefix+'after_GELU_mean_square']=value.square().mean((-2,-1)).detach()
        return hook

    def expert(self,kind,band=None):
        def hook(module,args,kwargs):
            query,key,value=args[:3]
            assert key is value and module.batch_first and module.dropout==0
            assert module.bias_k is module.bias_v is None and not module.add_zero_attn
            assert kwargs.get('attn_mask') is kwargs.get('key_padding_mask') is None and not kwargs.get('is_causal',False)
            assert query.dtype==key.dtype==module.in_proj_weight.dtype==torch.float32
            dim,heads=module.embed_dim,module.num_heads;depth=dim//heads
            b,qlen,_=query.shape;klen=key.shape[1]
            query_sequence=query.transpose(0,1)
            # Initial cross-band self attention aliases Q=K=V; preserve that packed path.
            key_sequence=query_sequence if query is key else key.transpose(0,1)
            projected=F._in_projection_packed(query_sequence,key_sequence,key_sequence,module.in_proj_weight,module.in_proj_bias)
            q,k,v=[x.view(length,b*heads,depth).transpose(0,1).view(b,heads,length,depth)
                for x,length in zip(projected,(qlen,klen,klen))]
            effective_q=F.normalize(q.float(),dim=-1)*math.sqrt(depth) if kind in ('M','F') else q
            logits=effective_q@k.transpose(-1,-2)/math.sqrt(depth)
            probability=logits.softmax(-1)
            if kind in ('M','F'):
                sources=klen//self.n;assert klen==sources*self.n and qlen==1
                reading=(query[:,0].norm(dim=-1),q[:, :,0].flatten(1).norm(dim=-1),
                    effective_q[:,:,0].flatten(1).norm(dim=-1),
                    key.reshape(b,sources,self.n,dim).var(-2,unbiased=False).mean((1,2)),
                    k.reshape(b,heads,sources,self.n,depth).var(-2,unbiased=False).mean((1,2,3)),
                    v.reshape(b,heads,sources,self.n,depth).var(-2,unbiased=False).mean((1,2,3)),
                    logits.reshape(b,heads,qlen,sources,self.n).std(-1,unbiased=False).mean((1,2,3)))
            if kind=='M':
                call=self.m_calls;stage=call//21;relation=(call%21)//3;current_band=call%3
                assert stage in (0,1) and b==self.batch
                assert sources==(len(RELATIONS[relation]) if self.structured else 3)
                p=probability[:,:,0].reshape(b,heads,sources,self.n)
                average=probability[:,:,0].mean(1)
                mean_map=average.reshape(b,sources,self.n)
                head_mean_stats=tuple(x.mean(1).detach() for x in spatial_statistics(mean_map))
                per_head_stats=tuple(x.mean((1,2)).detach() for x in spatial_statistics(p))
                self.m_stats.setdefault((stage,current_band),[]).append(head_mean_stats+per_head_stats+reading)
                if stage==0:self.initial_probabilities.append(average.detach())
                else:
                    first=self.initial_probabilities[call-21];middle=(first+average)/2
                    js=.5*(first*(first.clamp_min(1e-20).log()-middle.clamp_min(1e-20).log())+
                        average*(average.clamp_min(1e-20).log()-middle.clamp_min(1e-20).log())).sum(-1)
                    self.js.append(js.detach())
                self.m_calls+=1
            elif kind=='F':
                assert b==self.batch*3 and sources==(1 if self.structured else 3)
                p=probability[:,:,0].reshape(self.batch,3,heads,sources,self.n)
                mean_stats=tuple(x.mean(2) for x in spatial_statistics(p.mean(2)))
                head_stats=tuple(x.mean((2,3)) for x in spatial_statistics(p))
                for slot in range(3):
                    for convention,stats in [('head_mean_probability',mean_stats),('per_head',head_stats)]:
                        for label,x in zip(STATISTICS,stats):self.fields[f'F_{slot}_band{band}_'+convention+'_'+label]=x[:,slot].detach()
                    for label,x in zip(READING,reading):self.fields[f'F_{slot}_band{band}_'+label]=x.reshape(self.batch,3)[:,slot].detach()
                self.f_calls+=1
            else:
                assert qlen==klen==3 and b==self.batch*3 and self.cross_calls in (0,1)
                h,_,_=spatial_statistics(probability)
                self.fields[f'cross_{self.cross_calls}_per_head_normalized_entropy']=h.reshape(self.batch,3,heads,3).mean((1,2,3)).detach()
                self.fields[f'cross_{self.cross_calls}_diagonal_mass']=probability.diagonal(dim1=-2,dim2=-1).reshape(self.batch,3,heads,3).mean((1,2,3)).detach()
                self.cross_calls+=1
            # None leaves all original attention arguments and outputs unchanged.
        return hook

    def finish(self):
        assert (self.m_calls,self.f_calls,self.cross_calls)==(42,3,2) and len(self.initial_probabilities)==len(self.js)==21
        labels=[convention+'_'+label for convention in ('head_mean_probability','per_head') for label in STATISTICS]+list(READING)
        for (stage,band),values in self.m_stats.items():
            assert len(values)==7
            for index,label in enumerate(labels):self.fields[f'M_{stage}_band{band}_'+label]=torch.stack([x[index] for x in values]).mean(0)
        self.fields['M_initial_conditioned_JS_mean']=torch.stack(self.js).mean(0)
        assert len(self.fields)==275
        assert all(x.shape==(self.batch,) and torch.isfinite(x).all() for x in self.fields.values())
        return {k:x.cpu().tolist() for k,x in self.fields.items()}

