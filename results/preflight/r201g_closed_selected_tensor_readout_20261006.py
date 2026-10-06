"""CPU-only readout of four closed selected G weights; no neural forward or updates."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command

pf=PROJECT/'results/preflight'
proof=pf/'r201g_closed_selected_tensor_actual_20261006.json'
assert not proof.exists()
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
review=load(pf/'r201g_closed_selected_tensor_source_review_20261006.json')
assert review['status']=='PASS' and not review['blocking_findings']
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
normal=load(pf/'r201g_normal_priority_actual_session_20261006.json')
msvr=load(pf/'r201g_MSVR310_normal_actual_20261006.json')
assert normal['status']=='ACTUAL_R201G_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and normal['exit_code']==0
assert normal['successful_updates']==5294 and len(normal['archives'])==2
assert msvr['status']=='ACTUAL_TWO_FULL50_NORMAL_GT_RAW_COMPLETE' and msvr['successful_updates']==1410 and len(msvr['archives'])==2
code='''import hashlib,io,json,os
from pathlib import Path
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']==''
torch.set_num_threads(4)
root=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs')
rows=[]
for dataset,folder,steps in (('RGBNT201','r201g_normal_priority_20261006',2647),('MSVR310','r201g_other_two_normal_20261006/MSVR310',705)):
 for variant in ('frequency_shared','axis_shared'):
  run=root/folder/'training'/(dataset+'_r201g_'+variant+'_s42')
  result=json.loads((run/'result.json').read_text())
  assert result['status']=='COMPLETE' and result['epochs']==50 and result['optimizer_steps']==steps and result['amp_skipped_steps']==0
  assert json.loads((run/'normal_cpu_audit.json').read_text())['status']=='PASS'
  assert json.loads((run/'normal_local_archive.json').read_text())['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
  assert [p.name for p in run.glob('*.pth')]==['best.pth']
  path=run/'best.pth';content=path.read_bytes();digest=hashlib.sha256(content).hexdigest()
  state=torch.load(io.BytesIO(content),map_location='cpu',weights_only=True)
  scales=state['residual_scale'].double()
  assert scales.shape==(3,) and torch.isfinite(scales).all()
  projections={}
  for module in ('modality_projection','frequency_projection'):
   weight=state[module+'.3.weight'].double();bias=state[module+'.3.bias'].double()
   assert weight.shape==(512,64) and bias.shape==(512,) and torch.isfinite(weight).all() and torch.isfinite(bias).all()
   singular=torch.linalg.svdvals(weight)
   numerical_rank=int(torch.linalg.matrix_rank(weight))
   probability=singular/singular.sum()
   nonzero=probability>0
   entropy_rank=float(torch.exp(-(probability[nonzero]*probability[nonzero].log()).sum()))
   span_rank=int(torch.linalg.matrix_rank(torch.cat((weight,bias[:,None]),dim=1)))
   projections[module]=dict(output_weight_shape=list(weight.shape),output_weight_numerical_rank=numerical_rank,
    output_weight_entropy_effective_rank=entropy_rank,affine_output_subspace_dimension=numerical_rank,
    linear_span_with_bias_rank=span_rank,singular_values=singular.tolist(),bias_norm=float(bias.norm()),
    rank_precision='float64; rank properties of selected output weights, not observed feature ranks')
  assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
  rows.append(dict(dataset=dataset,variant=variant,selected_epoch=result['best']['epoch'],selected_metrics=result['full_metrics'],
   checkpoint=dict(path=str(path),bytes=len(content),sha256=digest,unchanged=True),
   residual_scale=scales.tolist(),initial_residual_scale=[.1,.05,.05],projections=projections))
  del state,content
print(json.dumps(dict(status='ACTUAL_CLOSED_G_FOUR_SELECTED_CHECKPOINT_CPU_TENSOR_READOUT',rows=rows,checkpoints=4,
 CPU_only=True,new_neural_models=0,new_neural_calls=0,new_optimizer_updates=0,weights_modified=0,
 limits='Selected normal-mAP best checkpoints only, not lateE50 weights. Learned scales/output matrix spectra are not effective sample residual magnitude, radial/tangent decomposition, contribution calibration, cause of late collapse or evidence that a wider/bypass projection helps. Internal bypass was already tested in R201C. No RGBNT100 live checkpoint read, no GPU/thermal/power query.')))
'''
result=json.loads(command(['ssh',*OPTIONS,'2026',"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "+shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code))
assert result['status']=='ACTUAL_CLOSED_G_FOUR_SELECTED_CHECKPOINT_CPU_TENSOR_READOUT' and result['checkpoints']==4
assert result['new_neural_calls']==result['new_optimizer_updates']==result['weights_modified']==0
result['verified_at']=datetime.now().isoformat(timespec='seconds')
proof.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:result[key] for key in ('status','verified_at','checkpoints','new_neural_calls','weights_modified')}))
