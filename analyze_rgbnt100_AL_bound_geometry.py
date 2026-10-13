"""Measure every saved AL query/gallery displacement; no new model inference."""
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile

from analyze_r201o_saved_normal_geometry import values
from analyze_rgbnt100_common_distance_probe import names

P=Path(__file__).resolve().parent
TAG='r201al_RGB_route_temperature_20261013'


def summarize(rows):
    shifts=sorted(r['unit_shift'] for r in rows)
    return dict(records=len(rows),mean=math.fsum(shifts)/len(shifts),minimum=shifts[0],maximum=shifts[-1],
        median=shifts[(len(shifts)-1)//2],p90=shifts[int(.9*(len(shifts)-1))],
        near_point1_fraction=sum(x>=.1-1e-5 for x in shifts)/len(shifts))


def main():
    intake=json.loads((P/'results/preflight'/ (TAG+'_complete_intake.json')).read_text(encoding='utf-8'))
    assert intake['status']=='ACTUAL_AL_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    all_rows=[];summaries=[]
    for variant in ('frequency_shared','axis_shared'):
        model='RGBNT100_r201al_'+variant+'_s42'
        a,b=(intake['archives'][model+'/'+key] for key in ('raw','states_raw'))
        for row in (a,b):
            f=Path(row['local']);assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
        folder=P/'results'/TAG/'RGBNT100/states'/model
        with (folder/'state_00.csv').open(encoding='utf-8',newline='') as f:before=list(csv.DictReader(f))
        with (folder/'state_11.csv').open(encoding='utf-8',newline='') as f:after=list(csv.DictReader(f))
        assert len(before)==len(after)==1715
        with zipfile.ZipFile(a['local']) as full,zipfile.ZipFile(b['local']) as base:
            for side,n in (('query',1715),('gallery',8575)):
                labels=names(full,side+'_names',n)
                assert labels==names(base,side+'_names',n)
                x=values(full,side+'_features',(n,5120),'<f4','f')
                y=values(base,'base_'+side+'_features',(n,5120),'<f4','f')
                part=[]
                for i in range(n):
                    lo=i*5120;hi=lo+5120;u=x[lo:hi];v=y[lo:hi]
                    shift=math.sqrt(math.fsum((z-t)**2 for z,t in zip(u,v)))
                    assert shift<=.1+1e-5
                    role='gallery'
                    if side=='query':
                        assert labels[i]==before[i]['name']==after[i]['name']
                        first,last=int(before[i]['Rank-1']),int(after[i]['Rank-1'])
                        role=('correct_preserved' if last else 'newly_harmed') if first else ('rescued' if last else 'remaining_original_error')
                    part.append(dict(model=model,side=side,index=i,name=labels[i],role=role,unit_shift=shift))
                all_rows.extend(part)
                groups={role:summarize([r for r in part if r['role']==role]) for role in sorted({r['role'] for r in part})}
                summaries.append(dict(model=model,side=side,**summarize(part),groups=groups))
                del x,y
    out=P/'results/rgbnt100_AL_bound_geometry_CPU_20261013';out.mkdir(exist_ok=False)
    with (out/'all_query_gallery_shifts.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    result=dict(status='ACTUAL_AL_ALL_QUERY_GALLERY_BOUND_GEOMETRY_CPU_COMPLETE',records=len(all_rows),summaries=summaries,
        new_neural_calls=0,new_optimizer_updates=0,epsilon=.1,near_boundary_tolerance=1e-5,
        limits='Complete saved1715Q8575G percontrol. Geometric proximity to the output bound does not itself identify a clipping event or prove a larger budget improves retrieval. Actual full-training clipping fractions are separately logged. Query labels only stratify saved diagnostic roles; no deployable oracle routing or test-label training.')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':
    main()
