"""Fresh50 with matched full supervision and partial-query/full-gallery updates."""
import argparse
import csv
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiment_data import make_loader, seed_all, split_records
from full_evaluation import full_metrics
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from run_mass_experiment import configuration, evaluate, write_json
from shared_identity_axis import KEYS, VARIANTS, SharedIdentityAxis, SharedIdentityDeMo, partial_gallery_triplet
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler


PARTIAL_SETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))


def build(args, cfg, classes, cameras):
    seed_all(args.seed)
    if args.variant == 'demo_shared':
        model = SharedIdentityDeMo(classes, cfg, cameras, args.seed)
    else:
        model = SharedIdentityAxis(classes, cfg, cameras, args.variant, args.seed)
    return model.float().cuda()


def step(model, batch, optimizer, scaler, loss_fn, xent, retained):
    images, target, cam, scene, names = batch
    images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
    target, cam, scene = target.cuda(), cam.cuda(), scene.cuda()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        output = model(images, label=target, cam_label=cam, view_label=scene)
        end = len(output) - len(output) % 2
        assert len(model.loss_weights) == end // 2
        full_loss = sum(model.loss_weights[i // 2] * loss_fn(output[i], output[i + 1], target, cam)
                        for i in range(0, end, 2))
        if len(output) % 2:
            full_loss = full_loss + output[-1]
        gallery = output[1].detach()
    assert torch.isfinite(full_loss)
    # Release the full-view activations before the second backbone pass on B64.
    scaler.scale(full_loss).backward()
    del output
    partial = {key: value if index in retained else torch.zeros_like(value)
               for index, (key, value) in enumerate(images.items())}
    with torch.autocast('cuda'):
        score, query = model(partial, label=target, cam_label=cam, view_label=scene, partial=True)
        partial_ce = xent(score, target)
        cross_triplet, pos, neg = partial_gallery_triplet(query, gallery, target)
        partial_loss = .25 * partial_ce + .5 * cross_triplet
    assert torch.isfinite(partial_loss)
    scaler.scale(partial_loss).backward()
    scaler.unscale_(optimizer)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    detail = {'loss': float(full_loss.detach() + partial_loss.detach()),
              'full_loss': float(full_loss.detach()), 'partial_ce': float(partial_ce.detach()),
              'cross_triplet': float(cross_triplet.detach()), 'partial_set': ''.join('RNT'[i] for i in retained),
              'positive_indices': pos.tolist(), 'negative_indices': neg.tolist(),
              'reference_requires_grad': gallery.requires_grad,
              'optimizer_updated': scaler.get_scale() >= previous_scale, 'amp_scale': scaler.get_scale(),
              'names': list(names)}
    assert not detail['reference_requires_grad']
    return detail


def metrics_from_arrays(path, dataset, output):
    arrays = np.load(path)
    query = arrays['query_indices']
    selector = arrays['scenes'] if dataset == 'MSVR310' else arrays['cameras']
    return full_metrics(arrays['distances'], arrays['ids'][query], arrays['ids'], selector[query], selector,
                        arrays['names'][query], arrays['cameras'][query], arrays['scenes'][query], output)


def main(model_builder=build,
         method_revision='shared_identity_v11: availability-aware original DeMo, shared raw-CLIP identity coordinates, full-view preservation and asymmetric partial-query/full-gallery training',
         metric='5632D: sqrt(.75) normalized private5120 + sqrt(.25) normalized common512; F modifies nonzero common identity coordinates'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['RGBNT201', 'RGBNT100', 'MSVR310'], required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['smoke', 'train'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args)
    fit, dev, queries, classes, cameras = split_records(args.data_root, args.dataset)
    model = model_builder(args, cfg, classes, cameras)
    info = {'arguments': vars(args), 'config': cfg.dump(), 'classes': classes, 'camera_embeddings': cameras,
            'fit_records': len(fit), 'dev_records': len(dev), 'dev_queries': len(queries),
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'descriptor_dim': 5632, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
            'started': time.time(), 'loss_weights': model.loss_weights,
            'pretraining': 'public CLIP ViT-B/16; fresh ReID and shared heads; no previous research checkpoints',
            'checkpoint_rule': 'highest development mAP; ties keep earliest epoch; official test not used',
            'evaluation_scope': 'identity-heldout development; not full official-train paper reproduction',
            'amp_policy': 'native fp16 GradScaler512; two backwards, one optimizer update; skips counted',
            'method_revision': method_revision,
            'metric': metric,
            'partial_training': 'one of six nonempty proper modality sets sampled uniformly per batch by independent seeded generator; same augmented images; full reference stopped; positives exclude self; GT IDs only',
            'partial_loss': {'label_smoothed_CE_weight': .25, 'cross_gallery_soft_triplet_weight': .5},
            'contribution_loss_weight': .05 if args.variant != 'demo_shared' else 0,
            'retention': 'only best.pth saved during training; smoke strict reload in memory; no last/initial/smoke weights'}
    write_json(out / 'run.json', info)
    loss_fn, center = make_loss(cfg, classes)
    xent = CrossEntropyLabelSmooth(classes)
    optimizer, _ = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    if args.mode == 'smoke':
        model.train()
        details = []
        for index, batch in enumerate(make_loader(fit, cfg, True, args.seed)):
            detail = step(model, batch, optimizer, scaler, loss_fn, xent, PARTIAL_SETS[index])
            assert detail['optimizer_updated'], 'smoke optimizer step skipped'
            details.append(detail)
            print('SMOKE_STEP', index + 1, json.dumps(detail), flush=True)
            if len(details) == 3:
                break
        gradients = {name: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
                     for name, p in model.named_parameters() if p.requires_grad}
        assert all(gradients.values()), [name for name, good in gradients.items() if not good]
        model.eval()
        image, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
        image = {key: value.cuda() for key, value in image.items()}
        with torch.no_grad():
            before = model(image, cam_label=cam.cuda(), view_label=scene.cuda())
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, map_location='cuda', weights_only=True), strict=True)
        with torch.no_grad():
            after = model(image, cam_label=cam.cuda(), view_label=scene.cuda())
        assert torch.equal(before, after)
        write_json(out / 'smoke.json', {**info, 'status': 'SMOKE_PASS', 'steps': 3, 'details': details,
                                      'gradients': gradients, 'strict_reload_equal': True,
                                      'peak_memory': torch.cuda.max_memory_allocated(), 'finished': time.time()})
        print('SMOKE_PASS', flush=True)
        return
    val_loader = make_loader(dev, cfg, False, args.seed)
    best, steps, optimizer_steps = {'mAP': -1}, 0, 0
    fields = ['epoch', 'steps', 'optimizer_steps', 'amp_skipped_steps', 'amp_scale', 'loss',
              'full_loss', 'partial_ce', 'cross_triplet', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'seconds']
    with (out / 'epochs.csv').open('w', newline='', encoding='utf-8') as table, (out / 'batch_orders.jsonl').open('w', encoding='utf-8') as orders:
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 51):
            start = time.time()
            seed_all(args.seed + epoch)
            generator = torch.Generator().manual_seed(args.seed + epoch + 1000)
            loader = make_loader(fit, cfg, True, args.seed + epoch)
            model.train()
            scheduler.step(epoch)
            details = []
            for batch in loader:
                retained = PARTIAL_SETS[int(torch.randint(6, (1,), generator=generator))]
                detail = step(model, batch, optimizer, scaler, loss_fn, xent, retained)
                details.append(detail)
                steps += 1
                optimizer_steps += int(detail['optimizer_updated'])
                orders.write(json.dumps({'epoch': epoch, 'step': steps, **detail}) + '\n')
                if not detail['optimizer_updated']:
                    print('AMP_SKIPPED', epoch, steps, detail['amp_scale'], flush=True)
            metrics = evaluate(model, val_loader, queries, args.dataset)
            row = {'epoch': epoch, 'steps': steps, 'optimizer_steps': optimizer_steps,
                   'amp_skipped_steps': steps - optimizer_steps, 'amp_scale': scaler.get_scale(),
                   'seconds': time.time() - start,
                   **{key: float(np.mean([detail[key] for detail in details])) for key in ('loss', 'full_loss', 'partial_ce', 'cross_triplet')},
                   **{key: metrics[key] for key in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')}}
            writer.writerow(row)
            table.flush()
            orders.flush()
            if metrics['mAP'] > best['mAP']:
                best = {**metrics, 'epoch': epoch}
                torch.save(model.state_dict(), out / 'best.pth')
                write_json(out / 'best.json', best)
            write_json(out / 'status.json', {'status': 'RUNNING', 'epoch': epoch, 'steps': steps, 'latest': row, 'best': best})
            print('EPOCH', json.dumps(row), 'BEST', json.dumps(best), flush=True)
    model.load_state_dict(torch.load(out / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    reloaded = evaluate(model, val_loader, queries, args.dataset, out / 'best_dev_arrays.npz')
    assert all(abs(reloaded[key] - best[key]) < 1e-8 for key in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
    complete_metrics = metrics_from_arrays(out / 'best_dev_arrays.npz', args.dataset, out / 'development_metrics')
    terminal = {**info, 'status': 'COMPLETE', 'epochs': 50, 'steps': steps, 'best': best, 'strict_reload': reloaded,
                'full_metrics': {key: complete_metrics[key] for key in ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')},
                'optimizer_steps': optimizer_steps, 'amp_skipped_steps': steps - optimizer_steps,
                'peak_memory': torch.cuda.max_memory_allocated(), 'finished': time.time()}
    write_json(out / 'result.json', terminal)
    write_json(out / 'status.json', terminal)
    print('TRAIN_COMPLETE', json.dumps(terminal), flush=True)


if __name__ == '__main__':
    main()
