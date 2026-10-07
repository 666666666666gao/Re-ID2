from collections import Counter, defaultdict
from datetime import datetime
import hashlib, json
from pathlib import Path

project=Path(__file__).resolve().parents[2]
output=project/'results/preflight/r201l_closed_training_view_coverage_actual_20261008.json'
assert not output.exists()
settings=[('K','RGBNT201',3951,2647),('K','RGBNT100',8675,6357),('K','MSVR310',1032,705),('L','MSVR310',1032,1000)]
readouts=[];sources={}
for family,dataset,records,steps in settings:
 stage='r201k_relation_local_pi_20261007' if family=='K' else 'r201l_uniform_k8_20261007'
 folder=project/'results'/stage/dataset/'training'/(dataset+'_r201'+family.lower()+'_axis_shared_s42')
 paths=[folder/n for n in ('official_split_manifest.json','result.json','batch_orders.jsonl')]
 contents={p:p.read_bytes() for p in paths}
 manifest=json.loads(contents[paths[0]]);result=json.loads(contents[paths[1]])
 batches=[json.loads(line) for line in contents[paths[2]].decode('utf-8-sig').splitlines()]
 assert manifest['dataset']==dataset and manifest['training_heldout_identities']==0 and len(manifest['train'])==records
 assert result['status']=='COMPLETE' and result['epochs']==50 and result['steps']==result['optimizer_steps']==steps and result['amp_skipped_steps']==0
 assert len(batches)==steps and all(r['optimizer_updated'] for r in batches)
 view='scene' if dataset=='MSVR310' else 'camera'
 byname={r['name']:r for r in manifest['train']};assert len(byname)==records and all(r[view]>=0 for r in byname.values())
 identities=defaultdict(list)
 for record in byname.values():identities[record['identity']].append(record)
 histogram=Counter(len({r[view] for r in group}) for group in identities.values())
 eligible_records=sum(len(group) for group in identities.values() if len({r[view] for r in group})>1)
 observations=[]
 for batch in batches:
  assert len(batch['names'])==64 and all(name in byname for name in batch['names'])
  groups=defaultdict(list)
  for name in batch['names']:groups[byname[name]['identity']].append(byname[name])
  positive_pairs=same_record_pairs=cross_view_pairs=distinct_record_anchors=cross_view_anchors=0
  for group in groups.values():
   size=len(group);names=Counter(r['name'] for r in group);views=Counter(r[view] for r in group)
   positive_pairs+=size*(size-1)
   same_record_pairs+=sum(count*(count-1) for count in names.values())
   cross_view_pairs+=size*size-sum(count*count for count in views.values())
   distinct_record_anchors+=size if len(names)>1 else 0
   cross_view_anchors+=size if len(views)>1 else 0
  assert positive_pairs>0
  observations.append(dict(epoch=batch['epoch'],anchors=64,identity_positive_pairs=positive_pairs,same_record_pairs=same_record_pairs,cross_view_pairs=cross_view_pairs,distinct_record_anchors=distinct_record_anchors,cross_view_anchors=cross_view_anchors,duplicate_record_positions=64-len(set(batch['names']))))
 windows=[]
 for label,first,last in [('all',1,50),('last10',41,50)]:
  selected=[r for r in observations if first<=r['epoch']<=last];assert selected
  totals={key:sum(r[key] for r in selected) for key in ('anchors','identity_positive_pairs','same_record_pairs','cross_view_pairs','distinct_record_anchors','cross_view_anchors','duplicate_record_positions')}
  windows.append(dict(window=label,first_epoch=first,last_epoch=last,batches=len(selected),**totals,cross_view_anchor_fraction=totals['cross_view_anchors']/totals['anchors'],distinct_record_anchor_fraction=totals['distinct_record_anchors']/totals['anchors'],cross_view_positive_pair_fraction=totals['cross_view_pairs']/totals['identity_positive_pairs'],same_record_positive_pair_fraction=totals['same_record_pairs']/totals['identity_positive_pairs']))
 readouts.append(dict(family=family,dataset=dataset,model='axis_shared_s42',GT_view_field=view,train_records=records,train_identities=len(identities),views_per_identity_histogram=dict(sorted(histogram.items())),global_cross_view_eligible_records=eligible_records,global_cross_view_eligible_record_fraction=eligible_records/records,windows=windows))
 for path,body in contents.items():sources[path.relative_to(project).as_posix()]=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
value=dict(status='ACTUAL_CLOSED_TRAINING_GT_VIEW_AND_SAMPLING_COVERAGE_CPU_READOUT',recorded_at=datetime.now().isoformat(timespec='seconds'),formal_closed_models=4,total_actual_batch_rows=sum(steps for _,_,_,steps in settings),readouts=readouts,sources=sources,new_neural_calls=0,new_optimizer_updates=0,remote_queries=0,goal='ACTIVE_UNMET',limits='Closed local official TRAIN manifests and actual50epoch PK batch names only. Cross-view uses original RGBNT camera/MSVR scene metadata; measures potential distinct-record/cross-view positive availability, not which hard positive the current identity-only loss selects. No embeddings, gradients, causal effect or future L100 result inferred. K/L sampler changes positive multiplicity, negative identity pool and update budget; no query/gallery identities used for training, no heldout, NN, parameter or benchmark selection changes. Ordinary/axis paired orders were audited elsewhere; this readout uses their closed axis lane only.')
output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=value['status'],new_neural_calls=0,readouts=readouts),ensure_ascii=False))
