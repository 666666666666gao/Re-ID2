"""Fresh50 on all official training records, with complete benchmark evaluation."""
import argparse
import csv
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiment_data import make_loader, seed_all
from full_evaluation import distance, full_metrics, extract
from gpu_thermal_execute import check_limits
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from official_training_data import full_records, metadata, name
from run_experiment import build as build_demo, configuration, step as demo_step, write_json
from run_shared_identity_experiment import build as build_shared, step as shared_step, PARTIAL_SETS
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler

import os


VARIANTS = ('demo', 'demo_shared')
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')


def build(args, cfg, classes, cameras):
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    return (build_demo if args.variant == 'demo' else build_shared)(args, cfg, classes, cameras)


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    if variant == 'demo':
        loss, names, updated, scale = demo_step(model, batch, optimizer, scaler, loss_fn)
        return dict(loss=loss, names=names, optimizer_updated=updated, amp_scale=scale)
    return shared_step(model, batch, optimizer, scaler, loss_fn, xent, retained)


@torch.no_grad()
def evaluate(model, records, cfg, args, output, arrays_path=None):
    query, gallery = records
    model.eval()
    # Query and gallery are distinct official lists, even when they share files.
    features, runtime = extract(model, query + gallery, cfg, args.seed)
    q, g = features[:len(query)], features[len(query):]
    arrays = metadata(query, 'query') | metadata(gallery, 'gallery')
    selector = 'scenes' if args.dataset == 'MSVR310' else 'cameras'
    distances = distance(q, g)
    values = full_metrics(distances, arrays['query_ids'], arrays['gallery_ids'],
        arrays['query_' + selector], arrays['gallery_' + selector], arrays['query_names'],
        arrays['query_cameras'], arrays['query_scenes'], output)
    if arrays_path is not None:
        np.savez_compressed(arrays_path, **arrays, query_features=q.numpy(), gallery_features=g.numpy(), distances=distances)
    return values, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('MSVR310', 'RGBNT201', 'RGBNT100'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    for key in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args)
    train, query, gallery, classes, cameras, manifest = full_records(args.data_root, args.dataset)
    write_json(out / 'official_split_manifest.json', manifest)
    model = build(args, cfg, classes, cameras)
    info = dict(arguments=vars(args), config=cfg.dump(), classes=classes, camera_embeddings=cameras,
        train_records=len(train), query_records=len(query), gallery_records=len(gallery),
        training_heldout_identities=0, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        descriptor_dim=5120 if args.variant == 'demo' else 5632, started=time.time(), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), pretraining='Public CLIP ViT-B/16, fresh identity heads, no historical research weights',
        checkpoint_rule='Maximum full official query/gallery mAP across 50 epochs; ties keep earliest epoch',
        evaluation_scope='Entire official training split and entire official query/gallery; no fit/dev holdout',
        selection_limitation='Official benchmark is used for checkpoint selection; this is not an untouched independent final test',
        partial_training='None; original complete DeMo' if args.variant == 'demo' else 'Matched available-source DeMo; one uniformly sampled proper modality set per batch; stopped full-view gallery, same shared identity interface and original partial CE/triplet weights',
        amp_policy='Native fp16 GradScaler512; actual optimizer updates and skipped steps recorded',
        retention='Only best.pth; smoke strict reload in memory; no initial, last, or smoke weight files')
    write_json(out / 'run.json', info)
    loss_fn, center = make_loss(cfg, classes)
    optimizer, _ = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    xent = CrossEntropyLabelSmooth(classes)
    if args.mode == 'smoke':
        seed_all(args.seed)
        model.train()
        details = []
        for i, batch in enumerate(make_loader(train, cfg, True, args.seed)):
            details.append(step(model, batch, optimizer, scaler, loss_fn, xent, PARTIAL_SETS[i % 6], args.variant))
            if sum(d['optimizer_updated'] for d in details) == 3:
                break
        assert sum(d['optimizer_updated'] for d in details) == 3
        gradients = {n: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
                     for n, p in model.named_parameters() if p.requires_grad}
        assert all(gradients.values()), [n for n, good in gradients.items() if not good]
        model.eval()
        images, _, cam, scene, _ = next(iter(make_loader(query[:8], cfg, False, args.seed)))
        images = {k: v.cuda() for k, v in images.items()}
        with torch.no_grad():
            before = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, map_location='cuda', weights_only=True), strict=True)
        with torch.no_grad():
            after = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        assert torch.equal(before, after)
        smoke = dict(**info, status='SMOKE_PASS', steps=3, attempts=len(details),
            amp_skipped_steps=len(details) - 3, gradients=gradients, strict_reload_equal=True,
            descriptor_shape=list(before.shape), details=details, peak_memory=torch.cuda.max_memory_allocated(), finished=time.time())
        write_json(out / 'smoke.json', smoke)
        print('FULL_OFFICIAL_SMOKE_PASS', args.dataset, args.variant, flush=True)
        return
    epoch_dir = out / 'epoch_metrics'
    epoch_dir.mkdir()
    best, steps, updates, visited = {'mAP': -1}, 0, 0, set()
    expected_names = {name(r) for r in train}
    fields = ['epoch', 'steps', 'optimizer_steps', 'amp_skipped_steps', 'amp_scale', 'loss',
              *METRICS, 'epoch_unique_train_records', 'cumulative_unique_train_records', 'seconds']
    with (out / 'epochs.csv').open('w', newline='', encoding='utf-8') as table, (out / 'batch_orders.jsonl').open('w', encoding='utf-8') as orders:
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 51):
            start = time.time()
            seed_all(args.seed + epoch)
            generator = torch.Generator().manual_seed(args.seed + epoch + 1000)
            model.train()
            scheduler.step(epoch)
            details, epoch_names = [], set()
            for batch in make_loader(train, cfg, True, args.seed + epoch):
                retained = PARTIAL_SETS[int(torch.randint(6, (1,), generator=generator))]
                detail = step(model, batch, optimizer, scaler, loss_fn, xent, retained, args.variant)
                details.append(detail)
                steps += 1
                updates += int(detail['optimizer_updated'])
                epoch_names.update(detail['names'])
                orders.write(json.dumps(dict(epoch=epoch, step=steps, **detail)) + '\n')
            visited.update(epoch_names)
            assert visited <= expected_names
            values, _ = evaluate(model, (query, gallery), cfg, args, epoch_dir / ('epoch_' + str(epoch)))
            row = dict(epoch=epoch, steps=steps, optimizer_steps=updates, amp_skipped_steps=steps - updates,
                amp_scale=scaler.get_scale(), loss=float(np.mean([d['loss'] for d in details])),
                epoch_unique_train_records=len(epoch_names), cumulative_unique_train_records=len(visited),
                seconds=time.time() - start, **{m: values[m] for m in METRICS})
            writer.writerow(row)
            table.flush()
            orders.flush()
            if values['mAP'] > best['mAP']:
                best = dict(epoch=epoch, **values)
                torch.save(model.state_dict(), out / 'best.pth')
                write_json(out / 'best.json', best)
            write_json(out / 'status.json', dict(status='RUNNING', epoch=epoch, latest=row, best=best))
            print('EPOCH', json.dumps(row), 'BEST', best['epoch'], best['mAP'], flush=True)
    model.load_state_dict(torch.load(out / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    reloaded, runtime = evaluate(model, (query, gallery), cfg, args, out / 'best_per_query', out / 'best_official_arrays.npz')
    assert all(abs(reloaded[m] - best[m]) < 1e-8 for m in METRICS)
    result = dict(**info, status='COMPLETE', epochs=50, steps=steps, optimizer_steps=updates,
        amp_skipped_steps=steps - updates, best=best, strict_reload=reloaded, full_metrics={m: reloaded[m] for m in METRICS},
        training_coverage=dict(eligible=len(train), visited=len(visited), unvisited=sorted(expected_names - visited)),
        runtime=runtime, peak_memory=torch.cuda.max_memory_allocated(), finished=time.time())
    write_json(out / 'result.json', result)
    write_json(out / 'status.json', result)
    print('FULL_OFFICIAL_TRAIN_COMPLETE', args.dataset, args.variant, flush=True)


if __name__ == '__main__':
    main()
