"""Real Adam/reload/all7 gates plus every official RGBNT201 query/gallery."""
import argparse
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_full_official49 import SETS
from experiment_data import make_loader, seed_all
from full_evaluation import full_metrics, distance
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from missing_evaluation import mask_images
from modeling.make_model import make_model
from official_training_data import full_records, metadata
from original_identity_anchor import assert_anchor_unchanged
from rgbnt201_projection_probe import build, VARIANTS
from run_experiment import configuration, write_json
from run_shared_identity_experiment import step, PARTIAL_SETS
from shared_identity_axis import encode_available
from solver.make_optimizer import make_optimizer


@torch.no_grad()
def readout(model, query, gallery, cfg, seed, output):
    """Capture the first PM/PF call, which is conditioned/routed full state11."""
    output.mkdir()
    collected = {name: [] for name in ('00', '10', '01', '11', 'private',
        'M_input', 'M_narrow', 'M_bypass', 'F_input', 'F_narrow', 'F_bypass')}
    geometry = {'M': [], 'F': []}
    model.eval()
    for images, _, cam, scene, _ in make_loader(query + gallery, cfg, False, seed):
        images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
        captured = {}
        def capture(kind):
            def hook(module, inputs, returned):
                if kind in captured:
                    return
                value = inputs[0].float()
                narrow = nn.Sequential.forward(module, value)
                bypass = value + narrow
                expected = bypass if model.bypass else narrow
                assert torch.equal(returned, expected)
                captured[kind] = {name: tensor for name, tensor in
                    (('input', value), ('narrow', narrow), ('bypass', bypass))}
            return hook
        hooks = [model.modality_projection.register_forward_hook(capture('M')),
                 model.frequency_projection.register_forward_hook(capture('F'))]
        states = model(images, cam_label=cam.cuda(), view_label=scene.cuda(), return_states=True)
        for hook in hooks:
            hook.remove()
        assert set(captured) == {'M', 'F'}
        for name, value in states.items():
            collected[name].append(F.normalize(value.float(), dim=1).cpu())
        private = encode_available(model, images, cam.cuda(), scene.cuda())[2][:, :5120]
        collected['private'].append(F.normalize(private.float(), dim=1).cpu())
        for kind, tensors in captured.items():
            for name, value in tensors.items():
                pooled = value.mean(1) if kind == 'M' else value
                collected[kind + '_' + name].append(F.normalize(pooled, dim=1).cpu())
            x, p, xp = (tensors[name] for name in ('input', 'narrow', 'bypass'))
            xhat = F.normalize(x, dim=-1)
            radial = (p * xhat).sum(-1, keepdim=True) * xhat
            geometry[kind].append(dict(samples=len(x),
                narrow_over_input_norm=float((p.norm(dim=-1) / x.norm(dim=-1)).mean()),
                narrow_input_cosine=float(F.cosine_similarity(x, p, dim=-1).mean()),
                bypass_input_cosine=float(F.cosine_similarity(x, xp, dim=-1).mean()),
                narrow_radial_energy_fraction=float(radial.square().sum() / p.square().sum()),
                note='For M, batch statistics include its seven routed relations; not independent queries.'))
    arrays = metadata(query, 'query') | metadata(gallery, 'gallery')
    results, raw = {}, dict(arrays)
    for name, values in collected.items():
        features = torch.cat(values)
        assert features.shape[0] == len(query) + len(gallery)
        assert torch.isfinite(features).all()
        q, g = features[:len(query)], features[len(query):]
        results[name] = full_metrics(distance(q, g), arrays['query_ids'], arrays['gallery_ids'],
            arrays['query_cameras'], arrays['gallery_cameras'], arrays['query_names'],
            arrays['query_cameras'], arrays['query_scenes'], output / name)
        raw[name + '_query_features'], raw[name + '_gallery_features'] = q.numpy(), g.numpy()
        if name.startswith(('M_', 'F_')):
            centered = q.numpy() - q.numpy().mean(0, keepdims=True)
            singular = np.linalg.svd(centered, compute_uv=False)
            results[name]['numerical_rank_rel_1e-5'] = int(np.count_nonzero(singular > singular[0] * 1e-5))
            probability = singular ** 2 / (singular ** 2).sum()
            positive = probability[probability > 0]
            results[name]['effective_rank_entropy_squared_singular'] = float(np.exp(-(positive * np.log(positive)).sum()))
    np.savez_compressed(output / 'readout_arrays.npz', **raw)
    write_json(output / 'result.json', dict(metrics=results, geometry_batches=geometry,
        query_count=len(query), gallery_count=len(gallery), GT='same-ID/same-camera junk exclusion',
        limits='Input and output readouts use the identical routed tensor of this state. They are diagnostics, '
               'not independently trained models. Fresh experts plus only three updates are not a method result.'))
    return results


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'anchor-run-dir', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    args.dataset, args.seed = 'RGBNT201', 42
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args)
    train, query, gallery, classes, cameras, manifest = full_records(args.data_root, args.dataset)
    assert (len(train), len(query), len(gallery)) == (3951, 836, 836)
    write_json(out / 'official_split_manifest.json', manifest)
    models = {}
    for bypass in (0, 1):
        args.bypass = bypass
        models[bypass] = build(args, cfg, classes, cameras)
    fixed, preserved = models[0], models[1]
    assert fixed.state_dict().keys() == preserved.state_dict().keys()
    assert all(torch.equal(value, preserved.state_dict()[name]) for name, value in fixed.state_dict().items())
    reference = make_model(cfg, classes, cameras).float().cuda()
    reference.load_state_dict(torch.load(Path(args.anchor_run_dir) / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    reference.eval()
    images, _, cam, scene, _ = next(iter(make_loader(query[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    cam, scene = cam.cuda(), scene.cuda()
    initial_by_set = {}
    with torch.no_grad():
        expected = reference(images, cam_label=cam, view_label=scene)
        for model in models.values():
            model.eval()
            assert torch.equal(encode_available(model, images, cam, scene)[2][:, :5120], expected)
        for available, missing in SETS.items():
            masked = mask_images(images, missing)
            first = encode_available(fixed, masked, cam, scene)[2][:, :5120]
            second = encode_available(preserved, masked, cam, scene)[2][:, :5120]
            assert torch.equal(first, second)
            initial_by_set[available] = first.clone()
    del reference
    results = {}
    for bypass, model in models.items():
        model.train()
        loss_fn, center = make_loss(cfg, classes)
        optimizer, _ = make_optimizer(cfg, model, center)
        named = {id(parameter): name for name, parameter in model.named_parameters()}
        groups, pairs = [], []
        for group in optimizer.param_groups:
            clones = []
            for parameter in group['params']:
                copy = nn.Parameter(parameter.detach().clone())
                clones.append(copy)
                pairs.append((named[id(parameter)], parameter, copy))
            groups.append({**{key: value for key, value in group.items() if key != 'params'}, 'params': clones})
        assert isinstance(optimizer, torch.optim.Adam)
        replay = torch.optim.Adam(groups, **optimizer.defaults)
        scaler = torch.amp.GradScaler('cuda', init_scale=512)
        seed_all(args.seed)
        details, orders, gradients, max_error = [], [], {}, 0.
        for index, batch in enumerate(make_loader(train, cfg, True, args.seed)):
            assert len(batch[1]) == 64 and torch.unique(batch[1]).numel() == 8
            assert all(int((batch[1] == value).sum()) == 8 for value in torch.unique(batch[1]))
            detail = step(model, batch, optimizer, scaler, loss_fn, CrossEntropyLabelSmooth(classes), PARTIAL_SETS[index])
            details.append(detail)
            orders.append(list(batch[-1]))
            assert detail['optimizer_updated'] and scaler.get_scale() == 512 and not detail['reference_requires_grad']
            state = {}
            for name, parameter, copy in pairs:
                grad = parameter.grad
                state[name] = dict(active=bool(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0),
                    grad_is_none=grad is None, nonzero_count=int(torch.count_nonzero(grad)) if grad is not None else 0)
            gradients[str(index + 1)] = state
            write_json(out / ('gradients_' + str(bypass) + '_' + str(index + 1) + '.json'), state)
            assert all(row['active'] for row in state.values()), [name for name, row in state.items() if not row['active']]
            for name, parameter, copy in pairs:
                copy.grad = parameter.grad.detach().clone()
            replay.step()
            for name, parameter, copy in pairs:
                error = float((parameter.detach() - copy.detach()).abs().max())
                max_error = max(max_error, error)
                assert torch.allclose(parameter, copy, rtol=1e-6, atol=1e-7), name
            assert_anchor_unchanged(model)
            if index == 2:
                break
        model.eval()
        with torch.no_grad():
            for available, missing in SETS.items():
                current = encode_available(model, mask_images(images, missing), cam, scene)[2][:, :5120]
                assert torch.equal(current, initial_by_set[available])
            before = model(images, cam_label=cam, view_label=scene, return_states=True)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, map_location='cuda', weights_only=True), strict=True)
        with torch.no_grad():
            after = model(images, cam_label=cam, view_label=scene, return_states=True)
        assert all(torch.equal(before[key], after[key]) for key in before)
        assert_anchor_unchanged(model)
        del replay, optimizer, pairs, groups, buffer
        metrics = readout(model, query, gallery, cfg, args.seed, out / ('bypass_' + str(bypass)))
        original = json.loads((Path(args.anchor_run_dir) / 'result.json').read_text())['full_metrics']
        assert all(abs(metrics['private'][key] - value) < 1e-8 for key, value in original.items())
        results[str(bypass)] = dict(actual_updates=3, amp_skips=0, native_Adam_replay_max_error=max_error,
            all_active_parameters_nonzero_each_update=True, strict_reload_four_states=True,
            full_private_exact_original_before=True, frozen_available_private_all7_exact_after=True,
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            details=details, batch_names=orders, metrics=metrics, anchor=model.anchor_record)
    assert results['0']['batch_names'] == results['1']['batch_names']
    assert results['0']['parameters'] == results['1']['parameters']
    assert results['0']['trainable_parameters'] == results['1']['trainable_parameters']
    assert not list(out.rglob('*.pth'))
    write_json(out / 'result.json', dict(status='PASS_RGBNT201_PROJECTION_NATIVE3_PER_MODE_FULL_READOUT',
        variant=args.variant, full_split_counts=dict(train=3951, query=836, gallery=836),
        training_heldout_identities=0, matched_initial_state_tensors_exact=True, paired_batches_exact=True,
        actual_updates=6, amp_skips=0, controls=results, new_weight_files=0, formal50_started=0,
        descriptor_dim=5632, finished=time.time(),
        limits='Bounded paired native diagnosis, not full50 or all49. All836 query/gallery records are used. '
               'The fixed public25/fresh public head is common setup, not a claim of protected fused original accuracy.'))
    print('RGBNT201_PROJECTION_NATIVE_READOUT_PASS', args.variant, flush=True)


if __name__ == '__main__':
    main()
