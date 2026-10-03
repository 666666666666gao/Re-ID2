import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT
keys=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
old_conditions={'q_RNT_g_RNT':'clean'}
for missing,retained in {'r':'NT','n':'RT','t':'RN','rn':'T','rt':'N','nt':'R'}.items():
    old_conditions['q_'+retained+'_g_'+retained]='both_missing_'+missing
    old_conditions['q_'+retained+'_g_RNT']='query_missing_'+missing
def read(path):return json.loads((PROJECT/path).read_text(encoding='utf-8'))
def rows(path):
    with (PROJECT/path).open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))
differences=[]
for dataset in ('MSVR310','RGBNT201','RGBNT100'):
    newroot='results/anytoany49_demo_20261003/'+dataset+'_demo_s42/full'
    oldroot=('results/axis_collaboration_v4_missing_development26_repaired/' if dataset=='RGBNT100' else 'results/axis_collaboration_v4_missing_development27/')+dataset+'_demo_s42/full'
    new,old=read(newroot+'/result.json'),read(oldroot+'/result.json')
    assert new['selected_epoch']==old['selected_epoch'] and new['normal_feature_max_error']==old['normal_feature_max_error']==0
    for name,proof in new['original_input_files'].items():assert proof==old['original_input_files'][name]
    for condition,previous in old_conditions.items():
        delta={key:new['measurements'][condition]['metrics'][key]-old['measurements'][previous]['metrics'][key] for key in keys}
        if max(abs(value) for value in delta.values())<=1e-8:continue
        before,after=rows(oldroot+'/'+previous+'.csv'),rows(newroot+'/'+condition+'.csv')
        assert len(before)==len(after) and all(all(a[key]==b[key] for key in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for a,b in zip(before,after))
        valid=[(a,b) for a,b in zip(before,after) if a['valid']=='True']
        differences.append(dict(dataset=dataset,condition=condition,previous=previous,all_six_delta_pp=delta,
            AP_changed=sum(float(a['AP'])!=float(b['AP']) for a,b in valid),Rank1_changed=sum(a['Rank-1']!=b['Rank-1'] for a,b in valid),
            first_match_changed=sum(a['first_match']!=b['first_match'] for a,b in valid),
            AP_max_abs_difference=max(abs(float(a['AP'])-float(b['AP'])) for a,b in valid)))
path=PROJECT/'results/preflight/anytoany49_previous13_difference_diagnostic.json';assert not path.exists()
path.write_bytes(json.dumps(dict(checkpoints_same_four_input_hashes=True,query_GT_order_same=True,differences=differences,
    conclusion='Difference magnitude only; source not yet proven, no widened acceptance tolerance'),indent=2).encode('utf-8'))
print('ACTUAL_PREVIOUS13_DIFFERENCES',json.dumps(differences),flush=True)
