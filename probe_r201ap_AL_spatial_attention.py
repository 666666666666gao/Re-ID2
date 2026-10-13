"""Measure frozen selected AL spatial reads; hooks never change model outputs."""
import argparse,csv,json,math,statistics
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dual_axis import RELATIONS
from experiment_data import make_loader
from official_training_data import full_records
from original_identity_anchor import assert_anchor_unchanged
from run_r201al_RGB_route_temperature import build,configuration
from run_experiment import write_json


def spatial_statistics(p):
    assert p.shape[-1]>1 and torch.isfinite(p).all() and (p>=0).all()
    mass=p.sum(-1,keepdim=True);assert (mass>0).all()
    p=p/mass
    entropy=-(p*p.clamp_min(1e-20).log()).sum(-1)
    return entropy/math.log(p.shape[-1]),entropy.exp()/p.shape[-1],p.max(-1).values*p.shape[-1]


class AttentionAudit:
    def __init__(self,model,n,selected):
        self.model,self.n,self.selected=model,n,set(selected)
        self.handles=[];self.maps={};self.offset=0
        for index,name in enumerate(('r','n','t')):
            self.handles.append(getattr(model.generalFusion,name).register_forward_hook(self.teacher(index)))
        self.handles.append(model.modality_expert.attention.register_forward_pre_hook(self.expert('M'),with_kwargs=True))
        for band,module in enumerate(model.frequency_expert.attention):
            self.handles.append(module.register_forward_pre_hook(self.expert('F',band),with_kwargs=True))
        self.handles.append(model.frequency_expert.cross_band.register_forward_pre_hook(self.expert('cross'),with_kwargs=True))

    def begin(self,offset,batch):
        self.offset,self.batch=offset,batch;self.fields={};self.m_calls=0;self.f_calls=0;self.cross_calls=0
        self.m_stats={};self.initial_probabilities=[];self.js=[]

    def teacher(self,index):
        def hook(module,args,output):
            weight=output[1]
            assert weight.shape==(self.batch,1,self.n+1)
            patch=weight[:,0,1:].float();mass=patch.sum(-1)
            h,s,lift=spatial_statistics(patch)
            for key,value in [('patch_mass',mass),('head_mean_probability_normalized_entropy',h),('head_mean_probability_effective_support_fraction',s),('head_mean_probability_maximum_lift',lift)]:
                self.fields['teacher_'+str(index)+'_'+key]=value.detach()
            for i in range(self.batch):
                position=self.offset+i
                if position in self.selected:
                    self.maps.setdefault(str(position),{})[str(index)]=(patch[i]/mass[i]).detach().cpu().tolist()
        return hook

    def expert(self,kind,band=None):
        def hook(module,args,kwargs):
            query,key,_=args[:3]
            assert module.batch_first and module.bias_k is module.bias_v is None and not module.add_zero_attn
            assert kwargs.get('attn_mask') is kwargs.get('key_padding_mask') is None and not kwargs.get('is_causal',False)
            assert query.dtype==key.dtype==module.in_proj_weight.dtype==torch.float32
            dim=module.embed_dim;heads=module.num_heads;depth=dim//heads;b,qlen,_=query.shape;klen=key.shape[1]
            weight,bias=module.in_proj_weight,module.in_proj_bias
            q=F.linear(query,weight[:dim],bias[:dim]).reshape(b,qlen,heads,depth).transpose(1,2)
            k=F.linear(key,weight[dim:2*dim],bias[dim:2*dim]).reshape(b,klen,heads,depth).transpose(1,2)
            probability=(q@k.transpose(-1,-2)/math.sqrt(depth)).softmax(-1)
            if kind=='M':
                call=self.m_calls;stage=call//21;relation=(call%21)//3;current_band=call%3
                assert stage in (0,1) and qlen==1 and klen==len(RELATIONS[relation])*self.n
                p=probability[:,:,0].reshape(self.batch,heads,len(RELATIONS[relation]),self.n)
                average=probability[:,:,0].mean(1)
                mean_map=average.reshape(self.batch,len(RELATIONS[relation]),self.n)
                head_mean_stats=tuple(v.mean(1).detach() for v in spatial_statistics(mean_map))
                per_head_stats=tuple(v.mean((1,2)).detach() for v in spatial_statistics(p))
                self.m_stats.setdefault((stage,current_band),[]).append(head_mean_stats+per_head_stats)
                if stage==0:self.initial_probabilities.append(average.detach())
                else:
                    first=self.initial_probabilities[call-21];middle=(first+average)/2
                    js=.5*(first*(first.clamp_min(1e-20).log()-middle.clamp_min(1e-20).log())+
                        average*(average.clamp_min(1e-20).log()-middle.clamp_min(1e-20).log())).sum(-1)
                    self.js.append(js.detach())
                self.m_calls+=1
            elif kind=='F':
                assert qlen==1 and klen==self.n and b==self.batch*3
                p=probability[:,:,0].reshape(self.batch,3,heads,self.n)
                head_mean_stats=spatial_statistics(p.mean(2))
                per_head_stats=spatial_statistics(p)
                for index in range(3):
                    for convention,stats in [('head_mean_probability',head_mean_stats),('per_head',per_head_stats)]:
                        for label,value in zip(('normalized_entropy','effective_support_fraction','maximum_lift'),stats):
                            self.fields[f'F_{index}_band{band}_'+convention+'_'+label]=(value[:,index] if convention=='head_mean_probability' else value[:,index].mean(1)).detach()
                self.f_calls+=1
            else:
                assert qlen==klen==3 and b==self.batch*3 and self.cross_calls in (0,1)
                h,_,_=spatial_statistics(probability)
                self.fields['cross_'+str(self.cross_calls)+'_per_head_normalized_entropy']=h.reshape(self.batch,3,heads,3).mean((1,2,3)).detach()
                self.fields['cross_'+str(self.cross_calls)+'_diagonal_mass']=probability.diagonal(dim1=-2,dim2=-1).reshape(self.batch,3,heads,3).mean((1,2,3)).detach()
                self.cross_calls+=1
            # Returning None leaves original query/key/value, flags and forward untouched.
        return hook

    def finish(self):
        assert (self.m_calls,self.f_calls,self.cross_calls)==(42,3,2) and len(self.initial_probabilities)==len(self.js)==21
        for (stage,band),values in self.m_stats.items():
            assert len(values)==7
            labels=[convention+'_'+label for convention in ('head_mean_probability','per_head')
                for label in ('normalized_entropy','effective_support_fraction','maximum_lift')]
            for index,label in enumerate(labels):
                self.fields[f'M_{stage}_band{band}_'+label]=torch.stack([v[index] for v in values]).mean(0)
        self.fields['M_initial_conditioned_JS_mean']=torch.stack(self.js).mean(0)
        assert all(v.shape==(self.batch,) and torch.isfinite(v).all() for v in self.fields.values())
        return {k:v.cpu().tolist() for k,v in self.fields.items()}

    def close(self):
        for handle in self.handles:handle.remove()


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--phase',choices=('native','full'),required=True)
    args=parser.parse_args();run,out=Path(args.run_dir),Path(args.output);assert not out.exists()
    trained=json.loads((run/'result.json').read_text());assert trained['status']=='COMPLETE' and trained['epochs']==50
    arguments=argparse.Namespace(**trained['arguments'])
    assert arguments.dataset=='RGBNT100' and arguments.variant=='axis_shared' and arguments.seed==42
    assert trained['best']['epoch']==20 and trained['descriptor_dim']==5120 and trained['optimizer_steps']==6563
    assert trained['amp_skipped_steps']==trained['training_heldout_identities']==0
    cfg=configuration(arguments);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    train,query,gallery,classes,cameras,manifest=full_records(arguments.data_root,'RGBNT100')
    assert (len(train),len(query),len(gallery),classes)==(8675,1715,8575,50)
    assert manifest==json.loads((run/'official_split_manifest.json').read_text())
    model=build(arguments,cfg,classes,cameras)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    model.eval();assert_anchor_unchanged(model)
    versions={n:v._version for n,v in model.state_dict().items()}
    casepath=Path('results/rgbnt100_AG_visual_cases_CPU_20261013/result.json')
    cases=json.loads(casepath.read_text());assert cases['status']=='ACTUAL_CACHED_AG_FOUR_ERROR_CASES_SELECTED'
    selected=[c['query_index'] for c in cases['cases']]
    for case in cases['cases']:assert case['query']==manifest['query'][case['query_index']]
    root=run.parent.parent
    def csvrows(state):
        with (root/'states'/run.name/('state_'+state+'.csv')).open(newline='') as f:return list(csv.DictReader(f))
    before,after=csvrows('00'),csvrows('11');assert len(before)==len(after)==1715
    n=model.bands.height*model.bands.width;audit=AttentionAudit(model,n,selected)
    records=query+gallery;metadata_records=manifest['query']+manifest['gallery']
    limit=64 if args.phase=='native' else len(records)
    with np.load(run/'best_official_arrays.npz') as saved:
        assert saved['query_names'].tolist()==[r['name'] for r in manifest['query']]
        assert saved['gallery_names'].tolist()==[r['name'] for r in manifest['gallery']]
        expected=np.concatenate((saved['query_features'],saved['gallery_features']))
    rows=[];offset=0
    with torch.no_grad():
        for images,_,cam,scene,names in make_loader(records[:limit],cfg,False,42):
            audit.begin(offset,len(names));images={k:v.cuda(non_blocking=True) for k,v in images.items()}
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
                rows.append(dict(index=index,side='query' if index<len(query) else 'gallery',**metadata,role=role,
                    **{k:v[i] for k,v in measures.items()}))
            offset+=len(names)
    audit.close();assert offset==limit and versions=={n:v._version for n,v in model.state_dict().items()}
    assert_anchor_unchanged(model)
    keys=list(measures)
    groups={role:[r for r in rows if r['role']==role] for role in sorted({r['role'] for r in rows})}
    summaries={role:dict(records=len(members),statistics={k:dict(mean=statistics.mean(r[k] for r in members),median=statistics.median(r[k] for r in members),minimum=min(r[k] for r in members),maximum=max(r[k] for r in members)) for k in keys}) for role,members in groups.items()}
    out.mkdir()
    with (out/'spatial_attention_records.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:format(v,'.8g') if isinstance(v,float) else v for k,v in row.items()} for row in rows)
    write_json(out/'teacher_case_maps.json',dict(height=model.bands.height,width=model.bands.width,maps=audit.maps,
        selected_cases=cases['cases'],limits='Already preselected four AG diagnostic queries; maps from frozen AL teacher, not segmentation/foreground truth or a representative error estimate. Images stay private.'))
    write_json(out/'result.json',dict(status='SMOKE_PASS' if args.phase=='native' else 'COMPLETE',phase=args.phase,
        records=limit,query_records=min(limit,1715),gallery_records=max(limit-1715,0),forward_batches=math.ceil(limit/64),
        model='AL_axis_shared',selected_epoch=20,parameters_unchanged=True,frozen_identity_unchanged=True,
        output_features_exact_saved_selected_cache=True,optimizer_updates=0,checkpoint_writes=0,
        full_six_metrics_equal_selected=trained['full_metrics'] if args.phase=='full' else None,
        attention_fields=len(keys),summaries=summaries,
        statistic_convention='Direct teacher/M/F comparison uses head_mean_probability: average raw attention probabilities across heads FIRST, then condition separately over patches within each source. M averages equal source/relation summaries. Separately labelled per_head fields compute spatial statistics per head before averaging and are not directly compared with teacher. Crossband entropy is per_head; M JS uses head-mean joint source-position probabilities.',
        limits='Teacher averaged-head weights are already returned by existing HDM. New expert weights are independently calculated QK-softmax in non-mutating prehooks with actual FP32 projection parameters, no mask/dropout/bias-kv/causal modification. Original attention forwards are unchanged; all measured output features match saved cache exactly. Conditional spatial entropy separates within-source spatial selection from source selection. Entropy/JS are diagnostic, not foreground proof, causal conflict or improved retrieval. Only one fixed mature AL model, no retraining, missing/49 or multiseed claims.'))
    print(json.dumps(dict(status='SMOKE_PASS' if args.phase=='native' else 'COMPLETE',records=limit,output_features_exact=True,optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
