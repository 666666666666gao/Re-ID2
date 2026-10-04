"""Actual CUDA M2 parity and alignment gradient/duplicate-observation checks."""
import argparse
import gc
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all, split_records
from frequency_relation_axis import build as parent_build
from identity_alignment_axis import build, identity_alignment_loss
from run_mass_experiment import configuration, write_json
from run_shared_identity_experiment import PARTIAL_SETS


def main():
    parser = argparse.ArgumentParser()
    for name in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    args.dataset, args.seed, args.pooling = 'MSVR310', 42, 'original_mean'
    args.relation_weight, args.alignment_temperature = .1, .07
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(exist_ok=False)
    cfg = configuration(args)
    fit, dev, _, classes, cameras = split_records(args.data_root, args.dataset)
    image, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
    image = {k:v.cuda() for k,v in image.items()}; cam, scene = cam.cuda(), scene.cuda()
    counts, checks = {}, []
    for variant in ('axis_shared', 'frequency_shared', 'twins_shared'):
        args.variant, args.alignment_weight = variant, .1
        parent = parent_build(args, cfg, classes, cameras).eval()
        model = build(args, cfg, classes, cameras).eval()
        assert set(model.state_dict()) == set(parent.state_dict())
        assert all(torch.equal(v, parent.state_dict()[n]) for n,v in model.state_dict().items())
        counts[variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        for retained in (*PARTIAL_SETS, (0,1,2)):
            partial = {k:v if i in retained else torch.zeros_like(v) for i,(k,v) in enumerate(image.items())}
            with torch.no_grad():
                expected = parent(partial, cam_label=cam, view_label=scene, return_states=True)
                actual = model(partial, cam_label=cam, view_label=scene, return_states=True)
            assert all(torch.equal(actual[s], expected[s]) for s in actual)
            checks.append(dict(variant=variant, retained=retained, all_four_states_exact=True))
        images, labels, cam_fit, scene_fit, names = next(iter(make_loader(fit, cfg, True, args.seed)))
        images = {k:v.cuda() for k,v in images.items()}; labels = labels.cuda()
        parent.train(); model.train(); model.alignment_weight = 0
        with torch.no_grad(), torch.autocast('cuda'):
            seed_all(1234)
            expected = parent(images, label=labels, cam_label=cam_fit.cuda(), view_label=scene_fit.cuda())
            seed_all(1234)
            actual = model(images, label=labels, cam_label=cam_fit.cuda(), view_label=scene_fit.cuda())
        assert len(expected) == len(actual) and all(torch.equal(a,b) for a,b in zip(actual,expected))
        checks.append(dict(variant=variant, zero_alignment_weight_training_outputs_exact=True))
        del parent, expected, actual
        gc.collect(); torch.cuda.empty_cache()
        model.alignment_weight = .1; model.alignment_names = tuple(names)
        with torch.autocast('cuda'):
            result = model(images, label=labels, cam_label=cam_fit.cuda(), view_label=scene_fit.cuda())
        loss = model.last_identity_alignment_loss
        gradients = torch.autograd.grad(loss, tuple(model.frequency_projection.parameters()))
        assert all(torch.isfinite(g).all() for g in gradients) and sum(float(g.abs().sum()) for g in gradients) > 0
        assert model.alignment_audit['identity_alignment_reference_requires_grad'] is False
        assert model.relation_audit['frequency_relation_weight'] == .1
        checks.append(dict(variant=variant, actual_PF_alignment_gradient=True, **model.alignment_audit))
        del model, result, loss, gradients
        gc.collect(); torch.cuda.empty_cache()
    assert len({tuple(v.values()) for v in counts.values()}) == 1
    student = torch.randn(8,512, device='cuda', requires_grad=True)
    reference = torch.randn(8,512, device='cuda', requires_grad=True)
    labels = torch.tensor([0,0,1,1,2,2,3,3], device='cuda')
    names = ('a','a','b','c','d','e','f','g')
    loss, detail = identity_alignment_loss(student,reference,labels,names,.07)
    loss.backward()
    assert reference.grad is None and student.grad is not None and torch.isfinite(student.grad).all()
    assert student.grad[:2].abs().sum() == 0 and student.grad[2:].abs().sum() > 0
    assert detail['identity_alignment_valid_anchors'] == 6 and detail['identity_alignment_excluded_anchors'] == 2
    q = torch.nn.functional.normalize(student.detach().float(),dim=1)[2:]
    g = torch.nn.functional.normalize(reference.detach().float(),dim=1)
    logits = q @ g.T / .07
    logits[torch.arange(6,device='cuda'),torch.arange(2,8,device='cuda')] = -torch.inf
    target = torch.tensor([3,2,5,4,7,6],device='cuda')
    expected = torch.nn.functional.cross_entropy(logits,target)
    assert torch.allclose(loss.detach(),expected,atol=1e-6,rtol=0)
    checks.append(dict(duplicate_only_identity_anchors_excluded=True,
        stopped_gallery_gradient=True, independently_recomputed_CE_equal=True))
    write_json(out/'result.json',dict(status='PASS_IDENTITY_ALIGNMENT_CONTRACT',checks=checks,
        parameter_counts=counts, state_eval_zero_weight_exact=True, actual_PF_gradient=True,
        optimizer_updates=0, official_test_uses=0))
    print('IDENTITY_ALIGNMENT_CONTRACT_PASS',flush=True)


if __name__ == '__main__':
    main()
