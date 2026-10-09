"""Read-only CPU geometry of already selected N MSVR normal arrays; no NN."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command

pf = PROJECT / 'results/preflight'
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
recovered = load(pf / 'r201n_MSVR_post_transport_collection_actual_20261009.json')
assert recovered['status'] == 'ACTUAL_N_ORIGINAL_MSVR_POST_TRANSPORT_GT_RAW_TEXT_CPU_COMPLETE'
closed = PROJECT / 'results/r201n_bounded_identity_shift_20261009/MSVR310'
assert load(closed / 'normal_analysis/result.json')['status'] == 'ACTUAL_N_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
reference = load(pf / 'msvr_two_closed_normal_caches_local_preserved_20261009.json')['files']['demo']
inputs = {'original_DeMo': dict(path=reference['source'], sha256=reference['sha256'])}
for variant in ('frequency_shared', 'axis_shared'):
    archive = recovered['collection']['archives']['MSVR310_r201n_' + variant + '_s42']
    inputs['N_' + variant] = dict(path=archive['remote'], sha256=archive['file']['sha256'])
output = closed / 'normal_geometry'
assert not output.exists()
code = '''import hashlib,json
from pathlib import Path
import numpy as np
inputs = INPUTS
assert all(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in inputs.values())
metadata = [p+'_'+s for p in ('query','gallery') for s in ('ids','cameras','scenes','names')]
with np.load(inputs['original_DeMo']['path'],allow_pickle=False) as data:
 base = {k:data[k].copy() for k in metadata+['query_features','gallery_features','distances']}
assert base['query_features'].shape==(591,5120) and base['gallery_features'].shape==(1055,5120)
assert base['distances'].shape==(591,1055)
rows,summary=[],{}
for name in ('N_frequency_shared','N_axis_shared'):
 with np.load(inputs[name]['path'],allow_pickle=False) as data:
  assert all(np.array_equal(data[k],base[k]) for k in metadata)
  distances=data['distances'].copy(); model={}
  assert distances.shape==(591,1055) and np.isfinite(distances).all()
  for prefix,count in (('query',591),('gallery',1055)):
   a=base[prefix+'_features'].astype(np.float64);b=data[prefix+'_features'].astype(np.float64)
   assert a.shape==b.shape==(count,5120) and np.isfinite(a).all() and np.isfinite(b).all()
   assert np.max(np.abs(np.linalg.norm(a,axis=1)-1))<1e-5 and np.max(np.abs(np.linalg.norm(b,axis=1)-1))<1e-5
   a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
   shift=np.linalg.norm(b-a,axis=1);angle=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',a,b),-1,1)))
   assert np.max(shift)<=.10+1e-6
   ag=a[:,:1536];bg=b[:,:1536]
   ag/=np.linalg.norm(ag,axis=1,keepdims=True);bg/=np.linalg.norm(bg,axis=1,keepdims=True)
   global_direction_shift=np.linalg.norm(bg-ag,axis=1)
   model[prefix]=dict(count=count,unit_shift_mean=float(shift.mean()),unit_shift_max=float(shift.max()),
    unit_shift_p90=float(np.quantile(shift,.9)),angle_degrees_mean=float(angle.mean()),
    global_direction_shift_max=float(global_direction_shift.max()))
   if prefix=='query':
    for i in range(count):
     positives=np.flatnonzero((base['gallery_ids']==base['query_ids'][i])&(base['gallery_scenes']!=base['query_scenes'][i]))
     negatives=np.flatnonzero(base['gallery_ids']!=base['query_ids'][i])
     assert len(positives) and len(negatives)
     p=int(positives[np.argmin(base['distances'][i,positives])]);n=int(negatives[np.argmin(base['distances'][i,negatives])])
     old_margin=float((base['distances'][i,n]-base['distances'][i,p])/2)
     new_margin=float((distances[i,n]-distances[i,p])/2)
     rows.append(dict(model=name,query_index=i,name=str(base['query_names'][i]),identity=int(base['query_ids'][i]),
      camera=int(base['query_cameras'][i]),scene=int(base['query_scenes'][i]),unit_shift=float(shift[i]),angle_degrees=float(angle[i]),
      original_nearest_positive_index=p,original_nearest_negative_index=n,reference_cosine_margin=old_margin,
      N_same_indices_cosine_margin=new_margin,delta_same_indices_cosine_margin=new_margin-old_margin))
 summary[name]=model
assert len(rows)==1182
assert all(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in inputs.values())
print(json.dumps(dict(summary=summary,rows=rows,input_arrays_preserved=True,new_neural_calls=0,new_optimizer_updates=0)))
'''.replace('INPUTS', repr(inputs))
payload = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python') + ' -'], input=code))
paired = list(csv.DictReader((closed / 'normal_analysis/paired_query_changes.csv').open(encoding='utf-8', newline='')))
changes = {(r['improved'], int(r['query_index'])): r for r in paired if r['reference'] == 'original_DeMo'}
for row in payload['rows']:
    pair = changes[(row['model'], row['query_index'])]
    assert all(str(row[k]) == pair[k] for k in ('name', 'identity', 'camera', 'scene'))
    row.update(delta_AP_pp=float(pair['delta_AP_pp']), delta_rank1=int(pair['delta_rank1']))
for name in payload['summary']:
    rows = [r for r in payload['rows'] if r['model'] == name]
    payload['summary'][name]['fixed_query_reference'] = dict(
        margin_improved=sum(r['delta_same_indices_cosine_margin'] > 0 for r in rows),
        margin_worsened=sum(r['delta_same_indices_cosine_margin'] < 0 for r in rows),
        AP_improved=sum(r['delta_AP_pp'] > 0 for r in rows), AP_worsened=sum(r['delta_AP_pp'] < 0 for r in rows),
        rank1_rescued=sum(r['delta_rank1'] == 1 for r in rows), rank1_harmed=sum(r['delta_rank1'] == -1 for r in rows))
output.mkdir()
with (output / 'paired_query_geometry.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(payload['rows'][0]))
    writer.writeheader(); writer.writerows(payload['rows'])
payload.pop('rows')
payload.update(status='ACTUAL_CACHED_N_MSVR_NORMAL_GEOMETRY_CPU_COMPLETE', collected_at=datetime.now().astimezone().isoformat(timespec='seconds'),
    inputs=inputs, receiver_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(), paired_query_rows=1182,
    limits='Only already selected normal E3 N MSVR and original E32 DeMo arrays. Full official591Q/1055G, metadata exact; original same-ID/same-scene exclusion for positive reference. Each query uses identical original-nearest positive/negative indices in both descriptors. Final normalized shift is not raw expert residual or isolated M/F contribution; fixed-margin change is not AP, causal proof or a new selection criterion. No NN, training, new missing combinations, result replacement or benchmark reselection.')
(output / 'result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(payload['summary'], ensure_ascii=False), flush=True)
