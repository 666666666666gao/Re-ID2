"""CPU output-subspace diagnostic of fixed N MSVR best weights; no forward."""
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
root = PROJECT / 'results/r201n_bounded_identity_shift_20261009/MSVR310'
assert load(root / 'normal_analysis/result.json')['status'] == 'ACTUAL_N_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
reference = load(pf / 'msvr_two_closed_normal_caches_local_preserved_20261009.json')['files']['demo']
weights = {'N_' + j['variant']: dict(path=j['run'] + '/best.pth', sha256=j['best_sha256'], selected_epoch=j['selected_epoch'])
           for j in recovered['collection']['jobs']}
assert set(weights) == {'N_frequency_shared', 'N_axis_shared'} and all(v['selected_epoch'] == 3 for v in weights.values())
output = root / 'projection_subspace'
assert not output.exists()
code = '''import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
weights=WEIGHTS;reference=REFERENCE
protected={reference['source']:reference['sha256']}|{v['path']:v['sha256'] for v in weights.values()}
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in protected.items())
with np.load(reference['source'],allow_pickle=False) as data:
 q=data['query_features'].astype(np.float64);g=data['gallery_features'].astype(np.float64)
 qids=data['query_ids'].copy();gids=data['gallery_ids'].copy();qs=data['query_scenes'].copy();gs=data['gallery_scenes'].copy()
 dist=data['distances'].copy()
assert q.shape==(591,5120) and g.shape==(1055,5120) and dist.shape==(591,1055)
assert np.isfinite(q).all() and np.isfinite(g).all() and np.isfinite(dist).all()
assert np.max(np.abs(np.linalg.norm(q,axis=1)-1))<1e-5 and np.max(np.abs(np.linalg.norm(g,axis=1)-1))<1e-5
q/=np.linalg.norm(q,axis=1,keepdims=True);g/=np.linalg.norm(g,axis=1,keepdims=True)
pos,neg=[],[]
for i in range(591):
 p=np.flatnonzero((gids==qids[i])&(gs!=qs[i]));n=np.flatnonzero(gids!=qids[i]);assert len(p) and len(n)
 pos.append(int(p[np.argmin(dist[i,p])]))
 neg.append(int(n[np.argmin(dist[i,n])]))
directions=(g[pos]-g[neg])[:,1536:].reshape(591,7,512)
query=q[:,1536:].reshape(591,7,512)
relations=('R','N','T','RN','RT','NT','RNT')
rows,summary=[],{}
for name,saved in weights.items():
 state=torch.load(saved['path'],map_location='cpu',weights_only=True)
 spaces={};weight_ranks={}
 for role,prefix in (('M','modality_projection'),('F','frequency_projection'),('I','interaction_projection')):
  w=state[prefix+'.3.weight'].numpy().astype(np.float64);b=state[prefix+'.3.bias'].numpy().astype(np.float64)
  assert w.shape==(512,64) and b.shape==(512,) and np.isfinite(w).all() and np.isfinite(b).all()
  singular=np.linalg.svd(w,compute_uv=False);assert singular[0]>0
  weight_ranks[role]=int((singular>singular[0]*1e-6).sum())
  spaces[role]=np.column_stack((w,b))
 del state
 spaces['M_F_I_union']=np.column_stack([spaces[role] for role in ('M','F','I')])
 summary[name]={'weight_matrix_numerical_ranks':weight_ranks,'spaces':{}}
 for role,columns in spaces.items():
  u,s,_=np.linalg.svd(columns,full_matrices=False);assert s[0]>0
  rank=int((s>s[0]*1e-6).sum());basis=u[:,:rank]
  model_rows=[]
  for r,relation in enumerate(relations):
   c=query[:,r];centered=c-c.mean(0,keepdims=True);d=directions[:,r]
   energies={}
   for label,value in (('reference_query',c),('centered_query',centered),('fixed_p_minus_n',d)):
    total=float(np.square(value).sum());assert total>0
    captured=float(np.square(value@basis).sum())/total
    assert 0<=captured<=1+1e-8
    energies[label+'_energy_captured']=captured
   row=dict(model=name,space=role,relation=relation,output_span_rank=rank,**energies)
   rows.append(row);model_rows.append(row)
  summary[name]['spaces'][role]=dict(output_span_rank=rank,
   reference_query_energy_captured_mean=float(np.mean([r['reference_query_energy_captured'] for r in model_rows])),
   centered_query_energy_captured_mean=float(np.mean([r['centered_query_energy_captured'] for r in model_rows])),
   fixed_p_minus_n_energy_captured_mean=float(np.mean([r['fixed_p_minus_n_energy_captured'] for r in model_rows])))
assert len(rows)==56
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in protected.items())
print(json.dumps(dict(summary=summary,rows=rows,protected_inputs_unchanged=True,
 model_constructions=0,neural_forwards=0,new_optimizer_updates=0)))
'''.replace('WEIGHTS', repr(weights)).replace('REFERENCE', repr(reference))
payload = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python') + ' -'], input=code))
output.mkdir()
with (output / 'relation_subspace_coverage.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(payload['rows'][0])); writer.writeheader(); writer.writerows(payload['rows'])
payload.pop('rows')
payload.update(status='ACTUAL_FIXED_N_MSVR_PROJECTION_SUBSPACE_CPU_COMPLETE', collected_at=datetime.now().astimezone().isoformat(timespec='seconds'),
    reference=reference, weights=weights, rows=56, singular_value_relative_rank_threshold=1e-6,
    receiver_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
    limits='Only fixed normal E3 N weights and original E32 MSVR normal feature cache. CPU torch.load(weights_only,map_location=cpu) and NumPy SVD, no model construction or forward/CUDA. Each affine projection output is contained in span(W2,b); M/F/I union is a separate upper space, not the full identity descriptor. Reference query, centered query and fixed legal-nearest positive-minus-negative directions use all591 queries/1055gallery, same-ID/same-scene positive exclusion. Energy coverage is geometric compatibility, not actual projected expert features, identity accuracy, an instruction to copy teacher coordinates, or causal proof of failure/capacity insufficiency. Final normalization and frozen5120 base remain outside this individual-output-space claim. No retraining, source/checkpoint/selection changes or new missing conditions.')
(output / 'result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(payload['summary'], ensure_ascii=False), flush=True)
