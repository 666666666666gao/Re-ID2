"""Read closed O MSVR normal archives locally; no NumPy, models or SSH."""
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import zipfile

from analyze_identity_coordinate_three_normal import load,query_rows,table

PROJECT=Path(__file__).resolve().parent


def values(archive,key,shape,dtype,cast):
    data=archive.read(key+'.npy')
    assert sys.byteorder=='little' and data[:8]==b'\x93NUMPY\x01\x00'
    length=struct.unpack('<H',data[8:10])[0]
    header=ast.literal_eval(data[10:10+length].decode('latin1'))
    assert header==dict(descr=dtype,fortran_order=False,shape=shape)
    result=memoryview(data)[10+length:].cast(cast)
    assert len(result)==math.prod(shape)
    return result


def dot(a,b):
    return math.fsum(x*y for x,y in zip(a,b))


def describe(numbers):
    return dict(mean=statistics.mean(numbers),median=statistics.median(numbers),maximum=max(numbers))


def main():
    root=PROJECT/'results/r201o_same_state_contribution_20261010/MSVR310'
    assert load(root/'normal_analysis/result.json')['status']=='ACTUAL_O_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
    closed=load(PROJECT/'results/preflight/r201o_same_state_contribution_MSVR310_actual_session_20261010.json')
    assert closed['status']=='ACTUAL_O_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and closed['exit_code']==0
    reference=load(PROJECT/'results/preflight/msvr_two_closed_normal_caches_local_preserved_20261009.json')['files']['demo']
    inputs={'original_DeMo':dict(path=reference['local'],bytes=reference['bytes'],sha256=reference['sha256'])}
    for variant in ('frequency_shared','axis_shared'):
        saved=closed['archives']['MSVR310_r201o_'+variant+'_s42']
        inputs['O_'+variant]=dict(path=saved['local'],**saved['file'])
    for entry in inputs.values():
        file=Path(entry['path'])
        assert file.stat().st_size==entry['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==entry['sha256']
    out=root/'normal_geometry'
    assert not out.exists()
    baseline_rows,_=query_rows(PROJECT/'results/full_official_baselines_20261004/training/MSVR310_demo_s42','MSVR310')
    metadata=[side+'_'+field for side in ('query','gallery') for field in ('ids','cameras','scenes','names')]
    samples,pairs,summary=[],[],{}
    with zipfile.ZipFile(inputs['original_DeMo']['path']) as original:
        base={side:values(original,side+'_features',(count,5120),'<f4','f') for side,count in (('query',591),('gallery',1055))}
        old_dist=values(original,'distances',(591,1055),'<f4','f')
        qids=values(original,'query_ids',(591,),'<i8','q');gids=values(original,'gallery_ids',(1055,),'<i8','q')
        qscene=values(original,'query_scenes',(591,),'<i8','q');gscene=values(original,'gallery_scenes',(1055,),'<i8','q')
        references=[]
        for i in range(591):
            positives=[j for j in range(1055) if gids[j]==qids[i] and gscene[j]!=qscene[i]]
            negatives=[j for j in range(1055) if gids[j]!=qids[i]]
            assert positives and negatives
            references.append((min(positives,key=lambda j:old_dist[i*1055+j]),min(negatives,key=lambda j:old_dist[i*1055+j])))
        for variant in ('frequency_shared','axis_shared'):
            name='O_'+variant
            trained=load(root/'training'/('MSVR310_r201o_'+variant+'_s42')/'result.json')
            current_rows,_=query_rows(root/'training'/('MSVR310_r201o_'+variant+'_s42'),'MSVR310')
            assert trained['best']['epoch']==3
            model={}
            with zipfile.ZipFile(inputs[name]['path']) as saved:
                assert all(original.read(key+'.npy')==saved.read(key+'.npy') for key in metadata)
                fused={side:values(saved,side+'_features',(count,5120),'<f4','f') for side,count in (('query',591),('gallery',1055))}
                new_dist=values(saved,'distances',(591,1055),'<f4','f')
                for side,count in (('query',591),('gallery',1055)):
                    for i in range(count):
                        b=base[side][i*5120:(i+1)*5120];z=fused[side][i*5120:(i+1)*5120]
                        bn,zn=dot(b,b),dot(z,z)
                        assert abs(bn-1)<1e-5 and abs(zn-1)<1e-5
                        shift=math.sqrt(math.fsum((y-x)**2 for x,y in zip(b,z)))
                        assert shift<=.10+1e-6
                        scale=math.sqrt(dot(b[:1536],b[:1536])/dot(z[:1536],z[:1536]))
                        residual=[scale*y-x for x,y in zip(b,z)]
                        global_error=max(abs(x) for x in residual[:1536]);assert global_error<2e-6
                        energy=dot(residual,residual);assert energy>0
                        radial=dot(b,residual)
                        samples.append(dict(model=name,side=side,index=i,unit_shift=shift,
                            angle_degrees=math.degrees(math.acos(min(1,max(-1,dot(b,z)/math.sqrt(bn*zn))))),
                            relative_raw_correction_norm=math.sqrt(energy),radial_energy_fraction=radial*radial/(bn*energy),
                            unchanged_global_max_error=global_error))
                    selected=[r for r in samples if r['model']==name and r['side']==side]
                    model[side]=dict(count=count,**{key:describe([r[key] for r in selected]) for key in
                        ('unit_shift','angle_degrees','relative_raw_correction_norm','radial_energy_fraction','unchanged_global_max_error')})
                for i,(p,n) in enumerate(references):
                    bq=base['query'][i*5120:(i+1)*5120];zq=fused['query'][i*5120:(i+1)*5120]
                    bp=base['gallery'][p*5120:(p+1)*5120];zp=fused['gallery'][p*5120:(p+1)*5120]
                    bn=base['gallery'][n*5120:(n+1)*5120];zn=fused['gallery'][n*5120:(n+1)*5120]
                    dq=[y-x for x,y in zip(bq,zq)]
                    old_difference=[x-y for x,y in zip(bp,bn)]
                    gallery_delta=[(yp-xp)-(yn-xn) for xp,yp,xn,yn in zip(bp,zp,bn,zn)]
                    qterm=dot(dq,old_difference);gterm=dot(bq,gallery_delta);cross=dot(dq,gallery_delta)
                    before=(old_dist[i*1055+n]-old_dist[i*1055+p])/2
                    after=(new_dist[i*1055+n]-new_dist[i*1055+p])/2
                    assert abs(dot(bq,old_difference)-before)<3e-6
                    assert abs(qterm+gterm+cross-(after-before))<3e-6
                    original_row,current_row=baseline_rows[i],current_rows[i]
                    assert all(original_row[key]==current_row[key] for key in ('query_index','name','identity','camera','scene'))
                    sample=next(r for r in samples if r['model']==name and r['side']=='query' and r['index']==i)
                    pairs.append(dict(model=name,query_index=i,name=original_row['name'],identity=int(original_row['identity']),
                        camera=int(original_row['camera']),scene=int(original_row['scene']),unit_shift=sample['unit_shift'],
                        original_nearest_positive_index=p,original_nearest_negative_index=n,reference_cosine_margin=before,
                        O_same_indices_cosine_margin=after,delta_same_indices_cosine_margin=after-before,
                        query_only_cosine_term=qterm,gallery_only_cosine_term=gterm,joint_cosine_term=cross,
                        delta_AP_pp=100*(float(current_row['AP'])-float(original_row['AP'])),
                        delta_rank1=int(current_row['Rank-1'])-int(original_row['Rank-1'])))
            rows=[r for r in pairs if r['model']==name]
            denominator=math.fsum(abs(r[key]) for r in rows for key in ('query_only_cosine_term','gallery_only_cosine_term','joint_cosine_term'))
            assert denominator>0
            model['fixed_reference_pairs']=dict(queries=len(rows),margin_improved=sum(r['delta_same_indices_cosine_margin']>0 for r in rows),
                margin_worsened=sum(r['delta_same_indices_cosine_margin']<0 for r in rows),AP_improved=sum(r['delta_AP_pp']>0 for r in rows),
                AP_worsened=sum(r['delta_AP_pp']<0 for r in rows),rank1_rescued=sum(r['delta_rank1']==1 for r in rows),
                rank1_harmed=sum(r['delta_rank1']==-1 for r in rows),
                net_over_components_absolute_sum=math.fsum(abs(r['delta_same_indices_cosine_margin']) for r in rows)/denominator,
                **{key:describe([r[key] for r in rows]) for key in ('delta_same_indices_cosine_margin','query_only_cosine_term','gallery_only_cosine_term','joint_cosine_term')})
            summary[name]=model
    assert len(samples)==3292 and len(pairs)==1182
    for entry in inputs.values():assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    source_names=('analyze_r201o_saved_normal_geometry.py','analyze_identity_coordinate_three_normal.py')
    result=dict(status='ACTUAL_CACHED_O_MSVR_SELECTED_NORMAL_GEOMETRY_AND_PAIRED_MARGIN_CPU_COMPLETE',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        inputs=inputs,input_arrays_preserved=True,samples=3292,paired_query_rows=1182,summary=summary,
        sources_sha256={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in source_names},
        new_neural_calls=0,new_optimizer_updates=0,new_weights=0,remote_queries=0,
        limits='Already selected E3 O normal and E32 original DeMo only; full591Q/1055G and original same-ID/same-scene exclusion. Fixed positive/negative indices from the original reference are diagnostic, not list AP or training targets. Query/gallery/cross decomposition is an algebraic readout, not intervention or causal attribution. Final geometry is the entire5120D descriptor, not isolated M/F/I evidence; radial reconstruction uses verified unchanged global direction. No lateE50 features, missing conditions, new selection or independently repeated seeds.')
    out.mkdir()
    table(out/'per_sample_geometry.csv',samples);table(out/'paired_query_geometry.csv',pairs)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],samples=3292,paired_query_rows=1182,summary=summary),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
