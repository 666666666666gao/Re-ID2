"""DeMo development comparison: 50 epochs, dev-only checkpoint selection."""
import argparse
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from config import cfg as default_cfg
from modeling.make_model import make_model
from dual_axis import AugmentedDeMo
from experiment_data import split_records, make_loader, seed_all
from layers.make_loss import make_loss
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.reid_evaluation import evaluate_reid


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def configuration(args):
    cfg = default_cfg.clone()
    cfg.merge_from_file(f'configs/{args.dataset}/DeMo.yml')
    cfg.MODEL.PRETRAIN_PATH_T = args.pretrained
    cfg.DATASETS.ROOT_DIR = args.data_root
    cfg.SOLVER.SEED = args.seed
    cfg.SOLVER.IMS_PER_BATCH = 64
    cfg.SOLVER.MAX_EPOCHS = 50
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.freeze()
    return cfg


def build(args, cfg, classes, cameras):
    seed_all(args.seed)
    if args.variant == 'demo':
        return make_model(cfg, classes, cameras).float().cuda()
    return AugmentedDeMo(classes, cfg, cameras, args.variant).float().cuda()


@torch.no_grad()
def evaluate(model, loader, query, dataset, path=None):
    model.eval()
    features, ids, cams, scenes, names = [], [], [], [], []
    for images, pid, cam, scene, name in loader:
        images = {k: v.cuda(non_blocking=True) for k, v in images.items()}
        # Fixed FP32 evaluation for original and augmented models, including reload.
        feature = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        features.append(F.normalize(feature.float(), dim=1).cpu())
        ids.extend(pid.tolist()); cams.extend(cam.tolist()); scenes.extend(scene.tolist()); names.extend(name)
    features = torch.cat(features)
    query = np.asarray(query)
    q = features[query]
    distances = (q.square().sum(1, keepdim=True) + features.square().sum(1)[None] - 2 * q @ features.T).numpy()
    ids, cams, scenes = np.asarray(ids), np.asarray(cams), np.asarray(scenes)
    selector = scenes if dataset == 'MSVR310' else cams
    cmc, map_ = evaluate_reid(distances, ids[query], ids, selector[query], selector)
    result = {'mAP': 100 * map_, 'Rank-1': 100 * float(cmc[0]), 'Rank-5': 100 * float(cmc[4]), 'Rank-10': 100 * float(cmc[9]),
              'query_count': len(query), 'gallery_count': len(ids), 'descriptor_dim': features.shape[1]}
    if path is not None:
        np.savez_compressed(path, features=features.numpy(), distances=distances, query_indices=query,
                            ids=ids, cameras=cams, scenes=scenes, names=np.asarray(names))
    return result


def step(model, batch, optimizer, scaler, loss_fn):
    images, target, cam, scene, names = batch
    images = {k: v.cuda(non_blocking=True) for k, v in images.items()}
    target, cam, scene = target.cuda(), cam.cuda(), scene.cuda()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        output = model(images, label=target, cam_label=cam, view_label=scene)
        end = len(output) - len(output) % 2
        loss = sum(loss_fn(output[i], output[i + 1], target, cam) for i in range(0, end, 2))
        if len(output) % 2:
            loss = loss + output[-1]
    assert torch.isfinite(loss), 'non-finite training loss'
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nonfinite = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not nonfinite, nonfinite
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), list(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['RGBNT201', 'RGBNT100', 'MSVR310'], required=True)
    parser.add_argument('--variant', choices=['demo', 'ordinary', 'dual'], required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['prepare', 'smoke', 'train'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = configuration(args)
    fit, dev, queries, classes, cameras = split_records(args.data_root, args.dataset)
    model = build(args, cfg, classes, cameras)
    info = {'arguments': vars(args), 'config': cfg.dump(), 'classes': classes, 'camera_embeddings': cameras,
            'fit_records': len(fit), 'dev_records': len(dev), 'dev_queries': len(queries),
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'descriptor_dim': 5120 if args.variant == 'demo' else 5632,
            'pretraining': 'public CLIP ViT-B/16; fresh ReID heads; no earlier research checkpoints',
            'checkpoint_rule': 'highest development mAP; ties keep earliest epoch; official test not used',
            'evaluation_scope': 'identity-heldout development only, not full-training paper reproduction',
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'started': time.time()}
    write_json(out / 'run.json', info)
    if args.mode == 'prepare':
        torch.save(model.state_dict(), out / 'initial.pth')
        print('PREPARE_PASS', json.dumps(info), flush=True)
        return
    loss_fn, center = make_loss(cfg, classes)
    optimizer, _ = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    steps = 0
    if args.mode == 'smoke':
        seed_all(args.seed)
        model.train()
        losses = []
        for batch in make_loader(fit, cfg, True, args.seed):
            loss, names = step(model, batch, optimizer, scaler, loss_fn)
            steps += 1; losses.append(loss)
            print('SMOKE_STEP', steps, loss, flush=True)
            if steps == 3:
                break
        gradients = {name: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
                     for name, p in model.named_parameters() if p.requires_grad}
        assert all(gradients.values()), [name for name, good in gradients.items() if not good]
        model.eval()
        image, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
        image = {k: v.cuda() for k, v in image.items()}
        with torch.no_grad():
            before = model(image, cam_label=cam.cuda(), view_label=scene.cuda())
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, out / 'smoke.pth')
        saved = torch.load(out / 'smoke.pth', map_location='cuda', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        with torch.no_grad():
            after = model(image, cam_label=cam.cuda(), view_label=scene.cuda())
        assert torch.equal(before, after)
        result = {**info, 'status': 'SMOKE_PASS', 'steps': steps, 'losses': losses, 'gradients': gradients,
                  'strict_reload_equal': True, 'peak_memory': torch.cuda.max_memory_allocated(), 'finished': time.time()}
        write_json(out / 'smoke.json', result)
        print('SMOKE_PASS', flush=True)
        return
    assert not (out / 'epochs.csv').exists(), 'use a new training output directory'
    val_loader = make_loader(dev, cfg, False, args.seed)
    best = {'mAP': -1}
    with (out / 'epochs.csv').open('w', newline='', encoding='utf-8') as table, (out / 'batch_orders.jsonl').open('w', encoding='utf-8') as orders:
        writer = csv.DictWriter(table, fieldnames=['epoch', 'steps', 'loss', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'seconds'])
        writer.writeheader()
        for epoch in range(1, 51):
            start = time.time()
            # Sampler and worker augmentation streams are matched across variants.
            seed_all(args.seed + epoch)
            loader = make_loader(fit, cfg, True, args.seed + epoch)
            model.train(); scheduler.step(epoch)
            losses = []
            for batch in loader:
                loss, names = step(model, batch, optimizer, scaler, loss_fn)
                steps += 1; losses.append(loss)
                orders.write(json.dumps({'epoch': epoch, 'step': steps, 'names': names}) + '\n')
            metrics = evaluate(model, val_loader, queries, args.dataset)
            row = {'epoch': epoch, 'steps': steps, 'loss': float(np.mean(losses)), 'seconds': time.time() - start,
                   **{k: metrics[k] for k in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')}}
            writer.writerow(row); table.flush(); orders.flush()
            if metrics['mAP'] > best['mAP']:
                best = {**metrics, 'epoch': epoch}
                torch.save(model.state_dict(), out / 'best.pth')
                write_json(out / 'best.json', best)
            write_json(out / 'status.json', {'status': 'RUNNING', 'epoch': epoch, 'steps': steps, 'latest': row, 'best': best})
            print('EPOCH', json.dumps(row), 'BEST', json.dumps(best), flush=True)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': 50, 'steps': steps}, out / 'last.pth')
    model.load_state_dict(torch.load(out / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    reloaded = evaluate(model, val_loader, queries, args.dataset, out / 'best_dev_arrays.npz')
    assert all(abs(reloaded[k] - best[k]) < 1e-8 for k in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
    terminal = {**info, 'status': 'COMPLETE', 'epochs': 50, 'steps': steps, 'best': best, 'strict_reload': reloaded,
                'peak_memory': torch.cuda.max_memory_allocated(), 'finished': time.time()}
    write_json(out / 'result.json', terminal)
    write_json(out / 'status.json', terminal)
    print('TRAIN_COMPLETE', json.dumps(terminal), flush=True)


if __name__ == '__main__':
    main()
