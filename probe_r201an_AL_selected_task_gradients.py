"""AN: actual AL objectives on training observations; no optimizer or new weights."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment_data import seed_all
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from layers.triplet_loss import TripletLoss
from official_training_data import full_records
from original_identity_anchor import assert_anchor_unchanged
from probe_mature_identity_gradient import PREFIXES, SCALE, gradient, norm, compare
from run_r201al_RGB_route_temperature import build, configuration
from run_r201ac_RGB_hard_identity import make_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, choices=(1, 4), required=True)
    args = parser.parse_args()
    run, output = Path(args.run_dir), Path(args.output)
    assert not output.exists()
    trained = json.loads((run / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
    assert trained['arguments']['dataset'] == 'RGBNT100' and trained['arguments']['seed'] == 42
    assert trained['arguments']['variant'] in ('frequency_shared', 'axis_shared')
    assert trained['descriptor_dim'] == 5120 and trained['optimizer_steps'] == 6563
    assert trained['amp_skipped_steps'] == trained['training_heldout_identities'] == 0
    assert trained['method_revision'].startswith('AL RGBNT100 joint-logit Softmax temperature2')
    assert trained['anchor']['final_unit_descriptor_bound']['epsilon'] == .10
    assert trained['anchor']['M_F_auxiliary_weight'] == .001
    assert trained['anchor']['parent_main_CE_weight'] == .25
    assert trained['anchor']['parent_contribution_regression_weight'] == .05
    assert json.loads((run / 'normal_cpu_audit.json').read_text())['status'] == 'PASS'
    assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
    arguments = argparse.Namespace(**trained['arguments'])
    cfg = configuration(arguments)
    assert cfg.MODEL.NO_MARGIN and cfg.MODEL.ID_LOSS_WEIGHT == .25
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    train, _, _, classes, cameras, manifest = full_records(arguments.data_root, 'RGBNT100')
    assert len(train) == 8675 and classes == 50
    assert manifest == json.loads((run / 'official_split_manifest.json').read_text())
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    assert not model.normal_priority_smoke
    loss_fn, _ = make_loss(cfg, classes)
    xent = CrossEntropyLabelSmooth(classes)
    hinge = TripletLoss(margin=.6)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.split('.')[0] in PREFIXES]
    parameters = [p for _, p in named]
    groups = {g: [i for i, (n, _) in enumerate(named) if n.split('.')[0] == g] for g in PREFIXES}
    assert all(groups.values()) and len(groups['residual_scale']) == 1
    versions = {n: p._version for n, p in model.named_parameters()}
    seed_all(43)
    model.train()
    rows = []
    for index, batch in enumerate(make_loader(train, cfg, True, 43), 1):
        images, labels, cam, scene, names = batch
        counts = Counter(labels.tolist())
        assert len(names) == 64 and len(counts) == 8 and set(counts.values()) == {8}
        images = {k: v.cuda(non_blocking=True) for k, v in images.items()}
        labels, cam, scene = labels.cuda(), cam.cuda(), scene.cuda()
        with torch.autocast('cuda'):
            output_values = model(images, label=labels, cam_label=cam, view_label=scene)
            assert len(output_values) == 17 and output_values[1].shape == (64, 5120)
            assert model.loss_weights[:3] == [1., .001, .001]
            m, f = model.projected_auxiliary_metric_features
            assert m.shape == f.shape == (64, 3584)
            components = dict(primary_CE=.25*xent(output_values[0], labels),
                primary_hinge=hinge(output_values[1], labels)[0],
                modality_aux=.001*loss_fn(output_values[2], m, labels, cam),
                frequency_aux=.001*loss_fn(output_values[4], f, labels, cam),
                other_full=sum(model.loss_weights[i//2]*loss_fn(output_values[i], output_values[i+1], labels, cam) for i in range(6, 16, 2)),
                contribution=output_values[-1])
        assert all(torch.isfinite(v) for v in components.values())
        ce_feature, hinge_feature = gradient(components['primary_CE'], [output_values[1]], True), gradient(components['primary_hinge'], [output_values[1]], True)
        grads = {k: gradient(v, parameters, True) for k, v in components.items()}
        grads['fused_primary'] = gradient(components['primary_CE'] + components['primary_hinge'], parameters, True)
        grads['M_F_aux'] = gradient(components['modality_aux'] + components['frequency_aux'], parameters, True)
        grads['full_total'] = gradient(sum(components.values()), parameters, False)
        roles = {}
        for group, indices in groups.items():
            roles[group] = dict(gradient_l2={k: norm(g, indices) for k, g in grads.items()},
                unused_tensors={k: sum(g[i] is None for i in indices) for k, g in grads.items()},
                primary_CE_vs_hinge=compare(grads['primary_CE'], grads['primary_hinge'], indices),
                fused_primary_vs_M_aux=compare(grads['fused_primary'], grads['modality_aux'], indices),
                fused_primary_vs_F_aux=compare(grads['fused_primary'], grads['frequency_aux'], indices),
                M_aux_vs_F_aux=compare(grads['modality_aux'], grads['frequency_aux'], indices),
                fused_primary_vs_M_F_aux=compare(grads['fused_primary'], grads['M_F_aux'], indices),
                fused_primary_vs_contribution=compare(grads['fused_primary'], grads['contribution'], indices),
                fused_primary_vs_full_total=compare(grads['fused_primary'], grads['full_total'], indices))
        base = F.normalize(torch.cat((output_values[11], output_values[13], output_values[15], output_values[7]), 1).detach().float(), dim=1)
        fused = output_values[1].detach().float()
        assert base.shape == fused.shape and float((fused-base).norm(dim=1).max()) <= .1 + 1e-6
        bound = dict(model.descriptor_bound_reports[('11', True)])
        assert bound['epsilon'] == .10 and not model.contribution_audit['target_requires_grad']
        scale_index = groups['residual_scale'][0]
        rows.append(dict(batch=index, names=list(names), training_labels=labels.cpu().tolist(),
            losses={k: float(v.detach()) for k, v in components.items()}, deploy_parameter_roles=roles,
            primary_CE_vs_hinge_descriptor_gradient=compare(ce_feature, hinge_feature, [0]),
            residual_scale_gradients={k: None if g[scale_index] is None else g[scale_index].tolist() for k, g in grads.items()},
            final_descriptor_bound_full=bound,
            gates=model.last_gates.float().mean(0).cpu().tolist(),
            route_band_mass=model.last_route.float().sum(1).mean(0).cpu().tolist()))
        assert_anchor_unchanged(model)
        assert all(p.grad is None for p in model.parameters())
        assert versions == {n: p._version for n, p in model.named_parameters()}
        print('AN_TRAIN_ONLY_GRADIENT_BATCH', arguments.variant, index, flush=True)
        del output_values, components, grads, m, f, base, fused
        if index == args.batches:
            break
    assert len(rows) == args.batches
    result = dict(status='SMOKE_PASS' if args.batches == 1 else 'COMPLETE',
        completed_at=datetime.now().astimezone().isoformat(timespec='seconds'), source_run=str(run),
        variant=arguments.variant, dataset='RGBNT100', selected_epoch=trained['best']['epoch'],
        training_records=8675, observation_seed=43, batches=args.batches, neural_forwards=args.batches,
        optimizer_constructed=False, optimizer_updates=0, checkpoint_writes=0,
        official_query_gallery_neural_uses=0, frozen_identity_unchanged=True, parameters_unchanged=True,
        gradient_scale=SCALE, roles={g: [named[i][0] for i in indices] for g, indices in groups.items()}, rows=rows,
        limits='Fixed AL selected weights, four sampled training batches per control; same labels/order, different selected epochs. Actual mainCE.25/hinge.6 and M/F scored512 CE plus actual unit3584 softTriplet auxiliary .001, contribution .05. No optimizer or new retrieval score; temporary train-mode BN statistics are discarded. Zero-weight partial forward is not replayed. None cosines mean disconnected/zero gradients, not zero-angle agreement. This local observation does not prove sustained conflict or a unique cause; M/F loss conflict differs from M/F data incompatibility.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
