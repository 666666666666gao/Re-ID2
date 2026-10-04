"""Real CUDA initialization/output parity and actual PF relation-loss gradient."""
import argparse
import gc
from pathlib import Path

import torch

from common_outlet_axis import build as original_build
from experiment_data import make_loader, seed_all, split_records
from frequency_relation_axis import build, relation_loss
from run_mass_experiment import configuration, write_json
from run_shared_identity_experiment import PARTIAL_SETS


def main():
    parser = argparse.ArgumentParser()
    for name in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    args.dataset, args.seed, args.pooling = 'MSVR310', 42, 'original_mean'
    torch.set_num_threads(4)
    output = Path(args.output); output.mkdir(exist_ok=False)
    cfg = configuration(args)
    fit, dev, _, classes, cameras = split_records(args.data_root, args.dataset)
    image, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
    image = {k:v.cuda() for k,v in image.items()};cam,scene=cam.cuda(),scene.cuda()
    counts, checks = {}, []
    for variant in ('axis_shared', 'frequency_shared', 'twins_shared'):
        args.variant, args.relation_weight = variant, .1
        baseline = original_build(args,cfg,classes,cameras).eval()
        model = build(args,cfg,classes,cameras).eval()
        assert set(model.state_dict()) == set(baseline.state_dict())
        assert all(torch.equal(v,baseline.state_dict()[n]) for n,v in model.state_dict().items())
        counts[variant]=dict(parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        for retained in (*PARTIAL_SETS, (0,1,2)):
            partial={k:v if i in retained else torch.zeros_like(v) for i,(k,v) in enumerate(image.items())}
            with torch.no_grad():
                expected=baseline(partial,cam_label=cam,view_label=scene,return_states=True)
                actual=model(partial,cam_label=cam,view_label=scene,return_states=True)
            assert all(torch.equal(actual[s],expected[s]) for s in actual)
            checks.append(dict(variant=variant,retained=retained,all_four_states_exact=True))
        images,labels,cam_fit,scene_fit,_=next(iter(make_loader(fit,cfg,True,args.seed)))
        fit_images={k:v.cuda() for k,v in images.items()}
        baseline.train();model.train();model.relation_weight=0
        with torch.no_grad(), torch.autocast('cuda'):
            seed_all(1234)
            expected=baseline(fit_images,label=labels.cuda(),cam_label=cam_fit.cuda(),view_label=scene_fit.cuda())
            seed_all(1234)
            actual=model(fit_images,label=labels.cuda(),cam_label=cam_fit.cuda(),view_label=scene_fit.cuda())
        assert len(expected)==len(actual) and all(torch.equal(a,b) for a,b in zip(actual,expected))
        checks.append(dict(variant=variant,zero_weight_training_outputs_exact=True))
        del baseline,expected,actual
        gc.collect();torch.cuda.empty_cache()
        model.relation_weight=.1
        model.train()
        with torch.autocast('cuda'):
            result=model(fit_images,label=labels.cuda(),cam_label=cam_fit.cuda(),view_label=scene_fit.cuda())
        loss=model.last_frequency_relation_loss
        gradients=torch.autograd.grad(loss,tuple(model.frequency_projection.parameters()))
        assert all(torch.isfinite(g).all() for g in gradients) and sum(float(g.abs().sum()) for g in gradients)>0
        assert model.relation_audit['frequency_relation_target_requires_grad'] is False
        assert model.relation_audit['frequency_relation_pairs']==len(labels)*(len(labels)-1)
        checks.append(dict(variant=variant,actual_PF_relation_gradient_nonzero=True,**model.relation_audit))
        del model,result,loss,gradients
        gc.collect();torch.cuda.empty_cache()
    assert len({tuple(v.values()) for v in counts.values()})==1
    student=torch.randn(8,512,device='cuda',requires_grad=True)
    reference=torch.randn(8,512,device='cuda',requires_grad=True)
    loss,target=relation_loss(student,reference);loss.backward()
    assert reference.grad is None and student.grad is not None and torch.isfinite(student.grad).all() and student.grad.abs().sum()>0
    write_json(output/'result.json',dict(status='PASS_FREQUENCY_RELATION_CONTRACT',checks=checks,
        parameter_counts=counts,initial_states_and_eval_four_states_exact=True,teacher_stopped=True,
        actual_PF_gradient=True,optimizer_updates=0,official_test_uses=0))
    print('FREQUENCY_RELATION_CONTRACT_PASS',flush=True)


if __name__ == '__main__':
    main()
