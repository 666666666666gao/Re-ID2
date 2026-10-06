"""Four real training batches, selected weights, actual deploy-parameter gradients."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all
from identity_coordinate_three_dataset import build
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from list_retrieval_objective import smooth_ap
from official_training_data import full_records
from original_identity_anchor import assert_anchor_unchanged
from run_experiment import configuration
from run_shared_identity_experiment import PARTIAL_SETS
from shared_identity_axis import partial_gallery_triplet

BATCHES = 4
SCALE = 512.
PREFIXES = ('residual_scale', 'modality_projection', 'frequency_projection', 'interaction_projection',
            'router', 'calibrator', 'modality_expert', 'frequency_expert',
            'shared_projection', 'fused_neck', 'fused_classifier')


def gradient(loss, parameters, retain_graph):
    with torch.autocast('cuda', enabled=False):
        values = torch.autograd.grad(SCALE * loss, parameters, retain_graph=retain_graph, allow_unused=True)
    assert all(value is None or torch.isfinite(value).all() for value in values)
    return [None if value is None else value.detach().float().cpu() / SCALE for value in values]


def norm(values, indices):
    return math.sqrt(math.fsum(float(values[i].square().sum()) for i in indices if values[i] is not None))


def compare(a, b, indices):
    an, bn = norm(a, indices), norm(b, indices)
    # Raw auxiliary branches actually omit PM/PF; their cosine is undefined.
    dot = math.fsum(float((a[i] * b[i]).sum()) for i in indices if a[i] is not None and b[i] is not None)
    return dict(a_l2=an, b_l2=bn, dot=dot, cosine=None if an == 0 or bn == 0 else dot / (an * bn))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, output_path = Path(args.run_dir), Path(args.output)
    assert not output_path.exists()
    trained = json.loads((run / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['descriptor_dim'] == 5120
    assert trained['arguments']['dataset'] == 'RGBNT201' and trained['arguments']['variant'] == 'axis_shared'
    assert trained['amp_skipped_steps'] == trained['training_heldout_identities'] == 0
    assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
    assert json.loads((run / 'normal_cpu_audit.json').read_text())['status'] == 'PASS'
    with (run / 'batch_orders.jsonl').open() as handle:
        first = json.loads(next(handle))
    temperature = first['temperature']
    assert temperature in (.01, .05) and first['primary_full_metric'] == 'label_masked_smooth_AP_unit5120'
    arguments = argparse.Namespace(**trained['arguments'])
    cfg = configuration(arguments)
    assert cfg.MODEL.ID_LOSS_WEIGHT == .25 and cfg.MODEL.TRIPLET_LOSS_WEIGHT == 1
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    train, _, _, classes, cameras, manifest = full_records(arguments.data_root, 'RGBNT201')
    assert len(train) == 3951 and manifest == json.loads((run / 'official_split_manifest.json').read_text())
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    loss_fn, _ = make_loss(cfg, classes)
    xent = CrossEntropyLabelSmooth(classes)
    named = [(name, value) for name, value in model.named_parameters()
             if value.requires_grad and name.split('.')[0] in PREFIXES]
    parameters = [value for _, value in named]
    groups = {prefix: [i for i, (name, _) in enumerate(named) if name.split('.')[0] == prefix] for prefix in PREFIXES}
    assert all(groups.values())
    assert len(groups['residual_scale']) == 1
    scale_index = groups['residual_scale'][0]
    learned_scales = model.residual_scale.detach().float().cpu().tolist()
    versions = {name: value._version for name, value in model.named_parameters()}
    seed_all(43)
    selector = torch.Generator().manual_seed(1043)
    model.train()
    rows = []
    for batch_index, batch in enumerate(make_loader(train, cfg, True, 43), 1):
        images, labels, cam, scene, names = batch
        assert len(names) == len(set(names)) == 64
        images = {key: value.cuda() for key, value in images.items()}
        labels, cam, scene = labels.cuda(), cam.cuda(), scene.cuda()
        retained = PARTIAL_SETS[int(torch.randint(6, (1,), generator=selector))]
        assert torch.unique(labels, return_counts=True)[1].eq(8).all()
        with torch.autocast('cuda'):
            output = model(images, label=labels, cam_label=cam, view_label=scene)
            assert output[1].shape == (64, 5120) and len(output) % 2 == 1
            assert len(model.loss_weights) == (len(output) - 1) // 2
            components = dict(primary_CE=.25 * xent(output[0], labels),
                primary_AP=smooth_ap(output[1], labels, temperature=temperature),
                modality_aux=model.loss_weights[1] * loss_fn(output[2], output[3], labels, cam),
                frequency_aux=model.loss_weights[2] * loss_fn(output[4], output[5], labels, cam),
                contribution=output[-1],
                other_full=sum(model.loss_weights[i // 2] * loss_fn(output[i], output[i + 1], labels, cam)
                               for i in range(6, len(output) - 1, 2)))
            gallery = output[1].detach()
        assert all(torch.isfinite(value) for value in components.values())
        values = {key: float(value.detach()) for key, value in components.items()}
        gradients = {key: gradient(value, parameters, True) for key, value in components.items()}
        gradients['full_total'] = gradient(sum(components.values()), parameters, False)
        del components, output
        partial = {key: value if index in retained else torch.zeros_like(value)
                   for index, (key, value) in enumerate(images.items())}
        with torch.autocast('cuda'):
            score, feature = model(partial, label=labels, cam_label=cam, view_label=scene, partial=True)
            pce = .25 * xent(score, labels)
            ptriplet, _, _ = partial_gallery_triplet(feature, gallery, labels)
            ptriplet = .5 * ptriplet
        values.update(partial_CE=float(pce.detach()), partial_triplet=float(ptriplet.detach()))
        gradients['partial_CE'] = gradient(pce, parameters, True)
        gradients['partial_triplet'] = gradient(ptriplet, parameters, True)
        gradients['partial_total'] = gradient(pce + ptriplet, parameters, False)
        del score, feature, pce, ptriplet, gallery
        summaries = {}
        for role, indices in groups.items():
            summaries[role] = dict(
                gradient_l2={key: norm(g, indices) for key, g in gradients.items()},
                unused_parameter_tensors={key: sum(g[i] is None for i in indices) for key, g in gradients.items()},
                AP_vs_CE=compare(gradients['primary_AP'], gradients['primary_CE'], indices),
                full_vs_modality_aux=compare(gradients['full_total'], gradients['modality_aux'], indices),
                full_vs_frequency_aux=compare(gradients['full_total'], gradients['frequency_aux'], indices),
                full_vs_partial=compare(gradients['full_total'], gradients['partial_total'], indices))
        rows.append(dict(batch=batch_index, names=list(names), training_labels=labels.tolist(),
            partial_set=''.join('RNT'[i] for i in retained), temperature=temperature,
            losses=values, deploy_parameter_roles=summaries,
            residual_scale_gradients={key: None if g[scale_index] is None else g[scale_index].tolist()
                                      for key, g in gradients.items()}))
        assert_anchor_unchanged(model)
        assert all(value.grad is None for value in model.parameters())
        assert versions == {name: value._version for name, value in model.named_parameters()}
        if batch_index == BATCHES:
            break
    assert len(rows) == BATCHES
    result = dict(status='ACTUAL_SELECTED_TRAIN_ONLY_DEPLOY_PARAMETER_GRADIENTS',
        completed_at=datetime.now().isoformat(timespec='seconds'), source_run=str(run),
        selected_epoch=trained['best']['epoch'], temperature=temperature, observation_seed=43,
        training_records=3951, observed_training_identities=len({y for row in rows for y in row['training_labels']}),
        batches=BATCHES, neural_forwards=2 * BATCHES, optimizer_updates=0, checkpoint_writes=0,
        parameters_unchanged=True, frozen_identity_unchanged=True, gradient_scale=SCALE,
        learned_residual_scales_at_load=learned_scales, selected_loss_weights=list(model.loss_weights),
        selected_parameter_tensors=[name for name, _ in named], rows=rows,
        limits='Four training batches, same labels and partial coverage, selected mature weights. '
               'Weighted original losses and scale512 autograd derivatives, no optimizer or Adam preconditioning. '
               'Only named expert/projector/router/calibrator/public-projection/fused-head parameters, not all model or frozen CLIP. '
               'Training head BN buffers adapt privately and are never saved. Query/gallery images are not forwarded. '
               'A local gradient conflict is not proof of a whole-run conflict or a retrieval improvement. '
               'No synthetic replacement for actual gradients; unused branches explicitly report no tensor gradient.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    print('SELECTED_MATURE_TRAIN_ONLY_PARAMETER_GRADIENTS_COMPLETE', temperature, flush=True)


if __name__ == '__main__':
    main()
