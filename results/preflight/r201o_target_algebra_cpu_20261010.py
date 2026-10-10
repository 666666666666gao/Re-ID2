"""Actual CPU algebra: deployed same-state scores versus backward-compatible scores."""
import ast,hashlib,json,shlex,sys
from datetime import datetime
from pathlib import Path
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command


def main():
    new_source=(PROJECT/'run_r201o_same_state_contribution.py').read_text(encoding='utf-8')
    old_source=(PROJECT/'axis_collaboration.py').read_text(encoding='utf-8')
    new=next(n for n in ast.parse(new_source).body if isinstance(n,ast.FunctionDef) and n.name=='same_state_targets')
    old_class=next(n for n in ast.parse(old_source).body if isinstance(n,ast.ClassDef) and n.name=='ContributionCalibrator')
    old=next(n for n in old_class.body if isinstance(n,ast.FunctionDef) and n.name=='targets')
    old.decorator_list=[d for d in old.decorator_list if not isinstance(d,ast.Name) or d.id!='staticmethod']
    functions=ast.unparse(new)+'\n'+ast.unparse(old)
    code='''import torch,json,math
from torch.nn import functional as F
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
'''+functions+'''
angles=torch.tensor([0.,.12,1.,1.12]);base=torch.stack((angles.cos(),angles.sin()),1).requires_grad_()
labels=torch.tensor([0,0,1,1])
theta=.05;rotation=torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
states={'00':base,'10':base,'01':base,'11':base@rotation.T}
target,pos,neg,scores=same_state_targets(states,labels)
old,oldpos,oldneg,oldscores=targets(states,labels)
assert pos.tolist()==oldpos.tolist()==[1,0,3,2] and neg.tolist()==oldneg.tolist()==[2,2,1,1]
assert target.abs().max()<1e-6 and old.abs().max()>.01
assert not target.requires_grad and all(not s.requires_grad for s in scores.values())
for key,value in states.items():
 v=F.normalize(value.detach().float(),dim=1);gram=v@v.T
 assert torch.allclose(scores[key],gram[torch.arange(4),pos]-gram[torch.arange(4),neg],atol=1e-6,rtol=0)
changed=base@torch.diag(torch.tensor([2.,1.]))
off={'00':base,'10':changed,'01':base,'11':changed}
offtarget,offpos,offneg,_=same_state_targets(off,labels)
assert offtarget[:,0].abs().max()>.01 and offtarget[:,1:].abs().max()<1e-6
assert torch.equal(pos,offpos) and torch.equal(neg,offneg)
binding=torch.nn.Module();binding.targets=same_state_targets
bound=binding.targets(states,labels)
assert torch.equal(bound[0],target) and len(binding.state_dict())==0
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_CPU_TARGET_ALGEBRA',torch_version=torch.__version__,same_rotation_same_state_target_max=float(target.abs().max()),same_rotation_old_gallery_target_max=float(old.abs().max()),F_off_F_and_interaction_target_max=float(offtarget[:,1:].abs().max()),F_off_M_target_max=float(offtarget[:,0].abs().max()),fixed_indices=True,targets_and_scores_stopped_gradient=True,scores_match_pairwise_state_similarity=True,nn_module_instance_callable_binding=True,CUDA_uninitialized=True,neural_forwards=0,optimizer_updates=0)))
'''
    value=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code))
    value.update(executed_at=datetime.now().astimezone().isoformat(timespec='seconds'),new_source_sha256=hashlib.sha256(new_source.encode()).hexdigest(),old_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),receiver_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(),scope='CPU function-only source extraction, no training model construction or empirical performance result. Common rotation counterexample; fixed indices; unavailable-F interaction equality; callable binding.')
    (PROJECT/'results/preflight/r201o_target_algebra_cpu_actual_20261010.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
