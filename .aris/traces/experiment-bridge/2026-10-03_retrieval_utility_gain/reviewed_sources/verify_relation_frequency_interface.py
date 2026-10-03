"""Actual per-source F injection and closed-state contract, without updates."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from experiment_data import make_loader,split_records
from run_mass_experiment import build as build_v5
from run_relation_frequency_experiment import build,configuration,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    args.dataset='MSVR310';args.variant='axis_relation_frequency_fullref';args.seed=42;args.contribution_weight=.05
    output=Path(args.output);output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cfg=configuration(args)
    _,dev,_,classes,cameras=split_records(args.data_root,args.dataset)
    baseline=build_v5(argparse.Namespace(**{**vars(args),'variant':'axis_mass_fullref'}),cfg,classes,cameras)
    model=build(args,cfg,classes,cameras)
    old,new=baseline.state_dict(),model.state_dict()
    assert set(old)==set(new) and all(torch.equal(old[name],new[name]) for name in old)
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters==sum(p.numel() for p in baseline.parameters()) and model.loss_weights==baseline.loss_weights
    images,_,cam,scene,_=next(iter(make_loader(dev[:8],cfg,False,args.seed)))
    images={key:value.cuda() for key,value in images.items()};cam,scene=cam.cuda(),scene.cuda()
    baseline.eval();model.eval()
    versions={name:value._version for name,value in model.state_dict().items()}
    read_m,read_f=model.modality_expert.read,model.frequency_expert.read
    evidence_m,evidence_f=[],[]
    def capture_m(*arguments,**kwargs):
        value=read_m(*arguments,**kwargs);evidence_m.append(value);return value
    def capture_f(*arguments,**kwargs):
        value=read_f(*arguments,**kwargs);evidence_f.append(value);return value
    model.modality_expert.read=capture_m;model.frequency_expert.read=capture_f
    fuse=model.fuse;calls=[]
    def capture_fuse(*arguments,**kwargs):
        calls.append(arguments);return fuse(*arguments,**kwargs)
    model.fuse=capture_fuse
    with torch.no_grad():
        before=baseline(images,cam_label=cam,view_label=scene)
        after=model(images,cam_label=cam,view_label=scene)
        base,modality,frequency,gates,use_m,use_f,eligible,mass,relation_frequency=calls[0]
        assert before.shape==after.shape==(8,5632) and after[:,-512:].eq(0).all()
        assert torch.equal(before[:,:1536],after[:,:1536])
        conditional=model.last_route/model.last_route.sum(2,keepdim=True)
        expected_relation=(model.router.by_relation(evidence_f[1])*conditional[...,None]).sum(2)
        assert torch.equal(expected_relation,relation_frequency)
        anchor=base[:,1536:5120].reshape(8,7,512).norm(dim=-1,keepdim=True)
        direction=F.normalize(model.frequency_projection(relation_frequency.float()),dim=-1)
        delta=model.residual_scale[1]*gates[:,1:2,None]*direction*anchor*(mass*7)[...,None]
        expected=torch.cat((before[:,:1536],before[:,1536:5120]+delta.flatten(1),torch.zeros_like(before[:,-512:])),1)
        assert torch.equal(expected,after) and torch.isfinite(after).all()
        available=torch.ones((8,3),device='cuda',dtype=torch.bool)
        m0,f0=evidence_m[0],evidence_f[0]
        states=model.controlled_states(base,m0,f0,eligible,available,after)
        changed_m=model.controlled_states(base,m0+999,f0,eligible,available,after)
        changed_f=model.controlled_states(base,m0,f0+999,eligible,available,after)
        assert torch.equal(states['01'],changed_m['01']) and torch.equal(states['10'],changed_f['10'])
        assert len(evidence_m)==len(evidence_f)==2
        m_only=baseline.fuse(base,model.modality_expert.pool(m0,eligible),torch.zeros_like(frequency),
            model.calibrator(base,model.modality_expert.pool(m0,eligible),torch.zeros_like(frequency))[1],True,False,eligible)
        assert torch.equal(states['10'],m_only)
    assert versions=={name:value._version for name,value in model.state_dict().items()}
    write_json(output/'result.json',dict(status='PASS_RELATION_FREQUENCY_TENSOR_CONTRACT',parameters=parameters,
        descriptor_dim=5632,initial_all_state_and_loss_weights_equal_V5=True,source_specific_band_pool_exact=True,
        original_global_coordinates_exact=True,frequency_identity_injection_exact=True,frequency_tail_exact_zero=True,
        state01_invariant_to_independent_M_evidence=True,state10_invariant_to_independent_F_evidence=True,
        controlled_states_no_additional_conditioned_expert_reads=True,state10_exact_original_V5=True,
        state_tensor_versions_unchanged=True,optimizer_updates=0,official_test_uses=0))
    print('RELATION_FREQUENCY_TENSOR_PASS',flush=True)


if __name__=='__main__':main()
