"""Selected-checkpoint route readout after AL completion; no per-epoch routing claim."""
import csv,json,math
from pathlib import Path

P=Path(__file__).resolve().parent
TAG='r201al_RGB_route_temperature_20261013'
METRICS=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')

def main():
    completed=json.loads((P/'results/preflight'/ (TAG+'_complete_intake.json')).read_text(encoding='utf-8'))
    assert completed['status']=='ACTUAL_AL_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    assert completed['terminal']['all_children_and_controller_absent'] and completed['formal_updates']==13126 and completed['state_cases']==8
    rows=[];details=[]
    for label,tag in [('AG','r201ag_RGB_coordinate_metric_20261012'),('AK','r201ak_RGB_lastblock_staged_20261013'),('AL',TAG)]:
        for variant in ('frequency_shared','axis_shared'):
            f=P/'results'/tag/'RGBNT100/training'/('RGBNT100_r201'+label.lower()+'_'+variant+'_s42')/'result.json'
            d=json.loads(f.read_text(encoding='utf-8'));r=d['runtime']
            assert d['status']=='COMPLETE' and d['epochs']==50 and d['optimizer_steps']==6563 and d['amp_skipped_steps']==0
            assert (d['train_records'],d['query_records'],d['gallery_records'],d['descriptor_dim'])==(8675,1715,8575,5120)
            assert r['triplets']==10290 and len(r['mean_route'])==7 and all(len(v)==3 for v in r['mean_route'])
            route=r['mean_route'];assert abs(math.fsum(v for row in route for v in row)-1)<1e-5
            rows.append(dict(model=label+'_'+variant,selected_epoch=d['best']['epoch'],**{k:d['full_metrics'][k] for k in METRICS},
                low_mass=math.fsum(v[0] for v in route),mid_mass=math.fsum(v[1] for v in route),high_mass=math.fsum(v[2] for v in route),
                largest_mean_probability=max(v for row in route for v in row),mean_per_sample_entropy=r['mean_route_entropy'],
                mean_per_sample_joint_product_L1=r['mean_joint_minus_marginal_product_L1']))
            details.append(dict(model=label+'_'+variant,source=str(f.relative_to(P)),sample_count=r['triplets'],mean_route=route,mean_outer_gates=r['mean_outer_gates']))
    out=P/'results/rgbnt100_AL_selected_routes_CPU_20261013';out.mkdir(exist_ok=False)
    with (out/'selected_routes.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(status='ACTUAL_AL_COMPLETE_SELECTED_ROUTE_CONTRAST_CPU',records=rows,details=details,new_neural_calls=0,new_optimizer_updates=0,
        limits='Each method uses its own normal mAP-best earliest-tie checkpoint and identical full10290QG scope. '
        'AG/AL matched frozen recipe; AK additionally permits visual-tail adaptation and is a historical reference. '
        'Entropy is mean per-sample entropy, not entropy of the mean table; supplied L1 is mean per-sample joint-minus-product. '
        'No stored per-epoch route curve, same-parameter T1/T2 intervention, per-query route correlation or causal explanation. '
        'More diffuse routing does not prove retrieval utility; high-band evidence may flow through cross-band conditions.')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':main()
