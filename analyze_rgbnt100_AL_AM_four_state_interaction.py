"""Account for fixed-reference four-state retrieval changes from saved full distances."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

from analyze_r201o_saved_normal_geometry import values
from analyze_rgbnt100_common_distance_probe import names

P=Path(__file__).resolve().parent
OUT=P/'results/rgbnt100_AL_AM_four_state_interaction_CPU_20261013'
STATES=('00','10','01','11')
TRIALS=(('AL','r201al_RGB_route_temperature_20261013','r201al',.10),
        ('AM','r201am_RGB_correction_budget_20261013','r201am',.20))
load=lambda f:json.loads(f.read_text(encoding='utf-8-sig'))


def rows(f):
    with f.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def summarize(records):
    keys=('M_standalone_margin_gain','F_standalone_margin_gain','M_given_F_margin_gain',
          'F_given_M_margin_gain','empirical_interaction_margin','joint_margin_gain',
          'M_given_F_AP_gain_pp','F_given_M_AP_gain_pp','joint_AP_gain_pp')
    result=dict(queries=len(records),identities=len({r['identity'] for r in records}),
        baseline_correct=sum(r['R1_00'] for r in records),
        M_only_rescued=sum(not r['R1_00'] and r['R1_10'] for r in records),
        M_only_harmed=sum(r['R1_00'] and not r['R1_10'] for r in records),
        F_only_rescued=sum(not r['R1_00'] and r['R1_01'] for r in records),
        F_only_harmed=sum(r['R1_00'] and not r['R1_01'] for r in records),
        joint_rescued=sum(not r['R1_00'] and r['R1_11'] for r in records),
        joint_harmed=sum(r['R1_00'] and not r['R1_11'] for r in records),
        joint_alone_correct=sum(r['R1_11'] and not r['R1_10'] and not r['R1_01'] for r in records),
        joint_wrong_single_correct=sum(not r['R1_11'] and (r['R1_10'] or r['R1_01']) for r in records),
        joint_AP_above_both_single=sum(r['AP_11']>max(r['AP_10'],r['AP_01'])+1e-12 for r in records),
        joint_AP_below_both_single=sum(r['AP_11']<min(r['AP_10'],r['AP_01'])-1e-12 for r in records),
        fixed_baseline_pair_crosses_positive_but_joint_wrong=sum(r['fixed_margin_11']>0 and not r['R1_11'] for r in records))
    result['effects']={k:dict(mean=statistics.mean(r[k] for r in records),
        median=statistics.median(r[k] for r in records),
        positive=sum(r[k]>1e-12 for r in records),negative=sum(r[k]<-1e-12 for r in records)) for k in keys}
    return result


def main():
    assert not OUT.exists()
    manifest=load(P/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42/official_split_manifest.json')
    query,gallery=manifest['query'],manifest['gallery']
    assert (len(query),len(gallery))==(1715,8575)
    positives={};negatives={}
    for item in query:
        key=(item['identity'],item['camera'])
        if key not in positives:
            positives[key]=[j for j,g in enumerate(gallery) if g['identity']==key[0] and g['camera']!=key[1]]
            negatives[key]=[j for j,g in enumerate(gallery) if g['identity']!=key[0]]
        assert positives[key] and negatives[key]
    all_records=[];summaries=[];inputs={};base_distances=None
    for trial,tag,prefix,budget in TRIALS:
        intake=load(P/'results/preflight'/(tag+'_complete_intake.json'))
        assert intake['status']=='ACTUAL_'+trial+'_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
        for variant in ('frequency_shared','axis_shared'):
            model='RGBNT100_'+prefix+'_'+variant+'_s42'
            root=P/'results'/tag/'RGBNT100'
            trained=load(root/'training'/model/'result.json')
            assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['optimizer_steps']==6563
            raw=intake['archives'][model+'/states_raw'];file=Path(raw['local'])
            assert file.stat().st_size==raw['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==raw['sha256']
            inputs[model]=raw
            per_state={s:rows(root/'states'/model/('state_'+s+'.csv')) for s in STATES}
            assert all(len(r)==1715 for r in per_state.values())
            assert sum(r['Rank-1']=='0' for r in per_state['00'])==58
            with zipfile.ZipFile(file) as archive:
                for side,metadata in (('query',query),('gallery',gallery)):
                    assert names(archive,side+'_names',len(metadata))==[r['name'] for r in metadata]
                    for field,key in (('ids','identity'),('cameras','camera'),('scenes','scene')):
                        assert list(values(archive,side+'_'+field,(len(metadata),),'<i8','q'))==[r[key] for r in metadata]
                distances={s:values(archive,'distances' if s=='11' else 'distances_'+s,(1715,8575),'<f4','f') for s in STATES}
            if base_distances is None:
                base_distances=distances['00']
            else:
                assert base_distances==distances['00']
            records=[]
            for i,item in enumerate(query):
                offset=i*8575;key=(item['identity'],item['camera'])
                p=min(positives[key],key=lambda j:distances['00'][offset+j])
                n=min(negatives[key],key=lambda j:distances['00'][offset+j])
                fixed={s:.5*(distances[s][offset+n]-distances[s][offset+p]) for s in STATES}
                dynamic={}
                for s in STATES:
                    assert all(str(per_state[s][i][k])==str(item[k]) for k in ('name','identity','camera','scene'))
                    pd=min(distances[s][offset+j] for j in positives[key])
                    nd=min(distances[s][offset+j] for j in negatives[key])
                    dynamic[s]=.5*(nd-pd)
                    assert int(dynamic[s]>0)==int(per_state[s][i]['Rank-1'])
                ap={s:float(per_state[s][i]['AP']) for s in STATES}
                rank={s:int(per_state[s][i]['Rank-1']) for s in STATES}
                role=('correct_preserved' if rank['11'] else 'newly_harmed') if rank['00'] else ('rescued' if rank['11'] else 'remaining_original_error')
                records.append(dict(trial=trial,variant=variant,model=model,selected_epoch=trained['best']['epoch'],budget=budget,
                    query_index=i,**item,role=role,baseline_positive_name=gallery[p]['name'],baseline_negative_name=gallery[n]['name'],
                    **{'AP_'+s:ap[s] for s in STATES},**{'R1_'+s:rank[s] for s in STATES},
                    **{'fixed_margin_'+s:fixed[s] for s in STATES},**{'dynamic_margin_'+s:dynamic[s] for s in STATES},
                    M_standalone_margin_gain=fixed['10']-fixed['00'],F_standalone_margin_gain=fixed['01']-fixed['00'],
                    M_given_F_margin_gain=fixed['11']-fixed['01'],F_given_M_margin_gain=fixed['11']-fixed['10'],
                    empirical_interaction_margin=fixed['11']-fixed['10']-fixed['01']+fixed['00'],
                    joint_margin_gain=fixed['11']-fixed['00'],M_given_F_AP_gain_pp=100*(ap['11']-ap['01']),
                    F_given_M_AP_gain_pp=100*(ap['11']-ap['10']),joint_AP_gain_pp=100*(ap['11']-ap['00'])))
            assert abs(statistics.mean(r['AP_11'] for r in records)*100-trained['full_metrics']['mAP'])<1e-8
            assert abs(statistics.mean(r['R1_11'] for r in records)*100-trained['full_metrics']['Rank-1'])<1e-8
            groups={'all':records,'original_errors':[r for r in records if not r['R1_00']],
                    'original_correct':[r for r in records if r['R1_00']]}
            summaries.append(dict(trial=trial,variant=variant,model=model,selected_epoch=trained['best']['epoch'],budget=budget,
                groups={name:summarize(members) for name,members in groups.items()}))
            all_records.extend(records)
            del distances
    OUT.mkdir()
    with (OUT/'all_query_fixed_reference_effects.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_records[0]));writer.writeheader();writer.writerows(all_records)
    result=dict(status='ACTUAL_AL_AM_FULL_FOUR_STATE_FIXED_REFERENCE_CPU_COMPLETE',
        at=datetime.now().astimezone().isoformat(timespec='seconds'),models=4,queries_per_model=1715,gallery_per_model=8575,
        diagnostic_query_rows=len(all_records),baseline_errors_per_model=58,inputs=inputs,summaries=summaries,
        original_00_distances_all_models_exact=True,dynamic_margin_sign_all_states_matches_saved_Rank1=True,
        full11_saved_mAP_Rank1_recount_exact=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed_reference='Closest camera-valid positive and closest different-identity negative under frozen00; the same pair is scored in allfour states of that selected checkpoint.',
        new_neural_calls=0,new_optimizer_updates=0,new_remote_queries=0,
        limitations='Selected checkpoints differ across models/AL/AM. Fixed-pair margins need not determine full Rank1 after competing gallery items move. Empirical interaction also contains existing conditions/routing/gates/PI/bound/normalization effects; it is not an isolated PI ablation, gradient conflict, causality or information-theoretic synergy. Repeated query rows are not independent observations. Test labels only describe saved diagnostics, not training or oracle deployment. AO is still running and is not included.')
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],models=4,diagnostic_query_rows=len(all_records),summaries=summaries)),flush=True)


if __name__=='__main__':main()
