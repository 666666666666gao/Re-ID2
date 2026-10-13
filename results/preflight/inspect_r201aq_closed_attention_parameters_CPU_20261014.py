"""Measure only two already-terminal AQ selected checkpoints on remote CPU during intake."""
from pathlib import Path
import json,shlex,sys

P=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(P))
from collect_results import OPTIONS,command
PRIVATE=Path('C:/Users/gb/.codex_tmp');TAG='r201aq_RGB_attention_decay_20261013'
observed=json.loads((PRIVATE/'AQ_original_controller_near_end_actual_20261013.json').read_text('utf-8-sig'))
assert not observed['actual']['controller_live'] and observed['actual']['result']['status']=='COMPLETE'
assert all(not c['live'] and c['exit']['exit_code']==0 for c in observed['actual']['children'])
code='''import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,torch
from datetime import datetime
from pathlib import Path
torch.set_num_threads(2)
r=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/r201aq_RGB_attention_decay_20261013/RGBNT100')
d=json.loads((r/'controller_result.json').read_text());assert d['status']=='COMPLETE' and d['successful_updates']==13126 and d['states_cases']==8
targets=('modality_expert.attention','frequency_expert.attention.0','frequency_expert.attention.1','frequency_expert.attention.2','frequency_expert.cross_band')
names=['modality_expert.'+k for k in ('source','query','band','value.0.weight','value.0.bias','value.1.weight','value.1.bias')]
names+=['frequency_expert.'+k for k in ('source','query')]
names+=['frequency_expert.values.'+str(i)+'.'+k for i in range(3) for k in ('0.weight','0.bias','1.weight','1.bias')]
def stats(v):
 v=v.double();assert v.isfinite().all()
 return dict(norm=v.norm().item(),absolute_max=v.abs().max().item(),standard_deviation=v.std(unbiased=True).item(),elements=v.numel(),nonzero=torch.count_nonzero(v).item())
rows={}
for run in d['runs']:
 f=r/'training'/run['name']/'best.pth';result=json.loads((f.parent/'result.json').read_text())
 state=torch.load(f,map_location='cpu',weights_only=True)
 inspected={n:dict(shape=list(state[n].shape),**stats(state[n])) for n in names}
 for t in targets:
  for ending in ('in_proj_weight','in_proj_bias'):
   n=t+'.'+ending;v=state[n];assert v.shape[0]==192
   for i,label in enumerate(('Q','K','V')):inspected[n+'_'+label]=stats(v.chunk(3,dim=0)[i])
  n=t+'.out_proj.weight';inspected[n]=dict(shape=list(state[n].shape),**stats(state[n]))
 assert len(inspected)==56 and not torch.cuda.is_initialized()
 rows[run['name']]=dict(checkpoint=dict(remote=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()),selected_epoch=result['best']['epoch'],full_metrics=result['full_metrics'],parameters=inspected)
 del state
print(json.dumps(dict(status='ACTUAL_AQ_CLOSED_SELECTED_ATTENTION_PARAMETERS_CPU_INSPECTION',at=datetime.now().astimezone().isoformat(timespec='seconds'),runs=rows,
 model_forwards=0,optimizer_updates=0,GPU_calls=0,new_checkpoint_writes=0,CUDA_initialized=False,
 estimator='Float32 checkpoint tensors converted to double for CPU norm/std; sample standard deviation.',
 limits='Two actually closed normal-mAP-selected AQ checkpoints only, not activation/attention probabilities, matrix rank, gradient direction, old optimizer replay or proof of causal improvement. Includes QKV biases/query/input/outprojection alongside the five changed packed QKV weights. No live NN progress query or source/policy change.')))'''
data=command(['ssh',*OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code)
result=json.loads(data);assert result['status']=='ACTUAL_AQ_CLOSED_SELECTED_ATTENTION_PARAMETERS_CPU_INSPECTION'
out=P/'results/preflight/r201aq_closed_selected_attention_parameters_CPU_20261014.json';assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],models={n:dict(selected_epoch=r['selected_epoch'],full_metrics=r['full_metrics'],QK_norms={k:v['norm'] for k,v in r['parameters'].items() if 'in_proj_weight' in k and k.endswith(('_Q','_K'))},query_norms={k:r['parameters'][k]['norm'] for k in ('modality_expert.query','modality_expert.band','frequency_expert.query')}) for n,r in result['runs'].items()})),flush=True)
