"""Review-gated CPU loss algebra/gradient checks, without neural model or CUDA."""
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
proof=pf/'r201h_protocol_list_actual_math_20261006.json'
assert not proof.exists()
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
review=load(pf/'r201h_protocol_list_source_review_20261006.json')
assert review['status']=='PASS' and not review['blocking_findings']
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==digest for name,digest in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
helper=(PROJECT/'protocol_list_objective.py').read_text(encoding='utf-8')
old=(PROJECT/'list_retrieval_objective.py').read_text(encoding='utf-8')
code=f'''import json,math,os,types
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']==''
torch.set_num_threads(4)
new=types.ModuleType('new_loss');old=types.ModuleType('old_loss')
exec(compile({helper!r},'protocol_list_objective.py','exec'),new.__dict__)
exec(compile({old!r},'list_retrieval_objective.py','exec'),old.__dict__)
generator=torch.Generator(device='cpu').manual_seed(20261006)
features=torch.randn(6,8,generator=generator,requires_grad=True)
labels=torch.tensor([0,0,0,1,1,1]);cameras=torch.tensor([0,0,1,0,1,1])
left,info=new.smooth_ap(features,labels,cameras,'label')
right=old.smooth_ap(features,labels)
g1=torch.autograd.grad(left,features,retain_graph=True)[0]
g2=torch.autograd.grad(right,features)[0]
assert abs(float(left-right))<1e-7 and float((g1-g2).abs().max())<1e-7
label_check=dict(value_error=abs(float(left-right)),gradient_max_error=float((g1-g2).abs().max()),valid_queries=info['valid_AP_queries'])
scores=torch.linspace(.52,.71,36).reshape(6,6).requires_grad_()
loss,info=new.ap_from_scores(scores,labels,cameras,'camera')
same=labels[:,None].eq(labels[None]);same_camera=cameras[:,None].eq(cameras[None]);junk=same & same_camera
gradient=torch.autograd.grad(loss,scores,retain_graph=True)[0]
assert torch.isfinite(gradient).all() and gradient.abs().sum()>0 and torch.count_nonzero(gradient[junk])==0
def reference(matrix,ids,cams,policy):
 values=[]
 for i in range(len(ids)):
  eligible=[j for j in range(len(ids)) if j!=i and (policy=='label' or not (ids[i]==ids[j] and cams[i]==cams[j]))]
  positive=[j for j in eligible if ids[i]==ids[j]]
  if not positive:continue
  def outranks(j,k):return 1/(1+math.exp(-(matrix[i][j]-matrix[i][k])/.01))
  ratios=[(1+math.fsum(outranks(j,k) for j in positive if j!=k))/(1+math.fsum(outranks(j,k) for j in eligible if j!=k)) for k in positive]
  values.append(math.fsum(ratios)/len(ratios))
 return 1-math.fsum(values)/len(values) if values else 0.
manual=reference(scores.detach().tolist(),labels.tolist(),cameras.tolist(),'camera')
assert abs(float(loss)-manual)<1e-6
camera_check=dict(value_error=abs(float(loss)-manual),valid_queries=info['valid_AP_queries'],used_positive_pairs=info['used_positive_pairs'],junk_score_gradient_nonzero=int(torch.count_nonzero(gradient[junk])))
permutation=torch.tensor([5,0,3,2,4,1])
permuted,_=new.ap_from_scores(scores[permutation][:,permutation],labels[permutation],cameras[permutation],'camera')
assert abs(float(loss-permuted))<1e-7
mixed_labels=torch.tensor([0,0,1,1,2,2]);mixed_cameras=torch.tensor([0,0,0,1,0,0])
mixed_scores=scores.detach().clone().requires_grad_()
mixed,mixed_info=new.ap_from_scores(mixed_scores,mixed_labels,mixed_cameras,'camera')
assert mixed_info['valid_AP_queries']==2 and torch.isfinite(mixed)
mixed_gradient=torch.autograd.grad(mixed,mixed_scores)[0]
assert torch.count_nonzero(mixed_gradient[torch.tensor([0,1,4,5])])==0
empty_features=torch.randn(4,8,generator=generator,requires_grad=True)
empty,empty_info=new.smooth_ap(empty_features,torch.tensor([0,0,1,1]),torch.tensor([0,0,1,1]),'camera')
assert empty==0 and empty_info['zero_valid_batch'] and empty_info['valid_AP_queries']==0
assert torch.count_nonzero(torch.autograd.grad(empty,empty_features)[0])==0
print(json.dumps(dict(status='ACTUAL_CPU_PROTOCOL_AP_ALGEBRA_AND_GRADIENT_PASS',label_matches_old=label_check,camera_matches_scalar_reference=camera_check,permutation_error=abs(float(loss-permuted)),mixed_valid_queries=2,empty_batch_graph_connected_zero=True,torch_version=torch.__version__,CPU_only=True,new_neural_models=0,new_neural_calls=0,new_optimizer_updates=0,synthetic_case_groups=5,limits='Synthetic CPU loss algebra/gradient check only; not native Adam/model/AMP/GT-performance acceptance or a claim that camera-aware AP improves retrieval. All real training records still retained by unchanged G step classification/auxiliary objectives.')))
'''
python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
result=json.loads(command(['ssh',*OPTIONS,'2026','CUDA_VISIBLE_DEVICES=\'\' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 '+shlex.quote(python)+' -'],input=code))
assert result['status']=='ACTUAL_CPU_PROTOCOL_AP_ALGEBRA_AND_GRADIENT_PASS' and result['new_neural_calls']==0
result['verified_at']=datetime.now().isoformat(timespec='seconds')
proof.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
