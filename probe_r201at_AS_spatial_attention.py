"""Non-mutating full normal reads of the two fixed selected AS controls."""
import argparse,csv,json,math,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from dual_axis import RELATIONS
from experiment_data import make_loader
from official_training_data import full_records
from original_identity_anchor import assert_anchor_unchanged
from probe_r201ar_AQ_spatial_attention import AttentionAudit as PreviousAudit,spatial_statistics
from run_r201as_RGB_query_normalization import build,configuration
from run_experiment import write_json

STATISTICS=('normalized_entropy','effective_support_fraction','maximum_lift')
READING=('input_query_norm','projected_query_norm','effective_query_norm',
    'input_key_spatial_variance','projected_key_spatial_variance',
    'projected_value_spatial_variance','logit_centered_spatial_std')


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
            effective_q=F.normalize(q,dim=-1)*math.sqrt(depth) if kind=='M' else q
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


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--phase',choices=('native','full'),required=True)
    args=parser.parse_args();run,out=Path(args.run_dir),Path(args.output);assert not out.exists()
    trained=json.loads((run/'result.json').read_text());arguments=argparse.Namespace(**trained['arguments'])
    assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['optimizer_steps']==6563
    assert arguments.dataset=='RGBNT100' and arguments.variant in ('axis_shared','frequency_shared') and arguments.seed==42
    epoch=23 if arguments.variant=='axis_shared' else 22
    assert trained['best']['epoch']==epoch and trained['descriptor_dim']==5120
    assert trained['amp_skipped_steps']==trained['training_heldout_identities']==0
    cfg=configuration(arguments);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    train,query,gallery,classes,cameras,manifest=full_records(arguments.data_root,'RGBNT100')
    assert (len(train),len(query),len(gallery),classes)==(8675,1715,8575,50)
    assert manifest==json.loads((run/'official_split_manifest.json').read_text())
    model=build(arguments,cfg,classes,cameras)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    model.eval();assert_anchor_unchanged(model)
    versions={n:x._version for n,x in model.state_dict().items()}
    root=run.parent.parent
    def state_rows(state):
        with (root/'states'/run.name/('state_'+state+'.csv')).open(newline='') as f:return list(csv.DictReader(f))
    before,after=state_rows('00'),state_rows('11');assert len(before)==len(after)==1715
    audit=AttentionAudit(model,model.bands.height*model.bands.width)
    records=query+gallery;metadata_records=manifest['query']+manifest['gallery']
    limit=64 if args.phase=='native' else len(records)
    with np.load(run/'best_official_arrays.npz') as saved:
        assert saved['query_names'].tolist()==[x['name'] for x in manifest['query']]
        assert saved['gallery_names'].tolist()==[x['name'] for x in manifest['gallery']]
        expected=np.concatenate((saved['query_features'],saved['gallery_features']))
    rows=[];offset=0
    with torch.no_grad():
        for images,_,cam,scene,names in make_loader(records[:limit],cfg,False,42):
            audit.begin(offset,len(names));images={k:x.cuda(non_blocking=True) for k,x in images.items()}
            features=F.normalize(model(images,cam_label=cam.cuda(),view_label=scene.cuda()).float(),dim=1).cpu().numpy()
            assert np.array_equal(features,expected[offset:offset+len(names)])
            measures=audit.finish()
            for i,name in enumerate(names):
                index=offset+i;metadata=metadata_records[index];assert name==metadata['name']
                if index<len(query):
                    assert before[index]['name']==after[index]['name']==name
                    original,full=int(before[index]['Rank-1']),int(after[index]['Rank-1'])
                    role=('correct_preserved' if full else 'newly_harmed') if original else ('rescued' if full else 'remaining_original_error')
                else:role='gallery'
                rows.append(dict(index=index,side='query' if index<len(query) else 'gallery',**metadata,role=role,**{k:x[i] for k,x in measures.items()}))
            offset+=len(names)
    audit.close();assert offset==limit and versions=={n:x._version for n,x in model.state_dict().items()}
    assert_anchor_unchanged(model)
    keys=list(measures);groups={role:[x for x in rows if x['role']==role] for role in sorted({x['role'] for x in rows})}
    summaries={role:dict(records=len(members),statistics={k:dict(mean=statistics.mean(x[k] for x in members),median=statistics.median(x[k] for x in members),minimum=min(x[k] for x in members),maximum=max(x[k] for x in members)) for k in keys}) for role,members in groups.items()}
    out.mkdir()
    with (out/'spatial_attention_records.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:format(x,'.8g') if isinstance(x,float) else x for k,x in row.items()} for row in rows)
    write_json(out/'result.json',dict(status='SMOKE_PASS' if args.phase=='native' else 'COMPLETE',phase=args.phase,model='AS_'+arguments.variant,
        records=limit,query_records=min(limit,1715),gallery_records=max(limit-1715,0),forward_batches=math.ceil(limit/64),selected_epoch=epoch,
        parameters_unchanged=True,frozen_identity_unchanged=True,output_features_exact_saved_selected_cache=True,optimizer_updates=0,checkpoint_writes=0,
        full_six_metrics_equal_selected=trained['full_metrics'] if args.phase=='full' else None,attention_fields=len(keys),summaries=summaries,
        statistic_convention='AS actual sequence-first packed QKV, including aliased initial crossband self-attention. Only M projected Q receives per-head L2 times sqrt(8); raw projected_query_norm and effective_query_norm are separately labelled. Compare teacher/M/F using head_mean_probability, then condition over each input-source spatial block. Separately labelled per_head statistics are computed before averaging heads. M equal relation/source averages; M JS on joint source-position distribution. F slots are physical sources ONLY for axis; ordinary has three query slots, each reading all three sources. Within-source statistics average those physical input blocks; ordinary slots are not source-specific experts.',
        limits='Non-mutating prehooks/forward hooks, original output equals the selected cache exactly. F_source processing statistics are measured BEFORE source embedding and ordinary mixing, on real source/band raw input, LayerNorm, Linear and GELU. M_source is after its value GELU. Variance means centered spatial variance averaged over channels; mean_square includes spatially constant components. These are learned patch-feature bands, not sensor spectral bands, foreground truth, causal identity importance or raw image frequency. Error roles use official labelled state00/11 only for diagnostics, never deployment routing. This is fixed selected AS eval, no new accuracy, training, optimizer, checkpoint, missing49 or seeds.'))
    print(json.dumps(dict(status='SMOKE_PASS' if args.phase=='native' else 'COMPLETE',records=limit,output_features_exact=True,optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
