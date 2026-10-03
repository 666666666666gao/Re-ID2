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
from residual_dual_axis import ResidualAugmentedDeMo
from axis_collaboration import AxisCollaborationDeMo
from scaled_axis_collaboration import ScaledAxisCollaborationDeMo, VARIANTS as SCALED_VARIANTS
from projected_mass_axis_collaboration import ProjectedMassAxisCollaborationDeMo
from experiment_data import split_records, make_loader, seed_all
from layers.make_loss import make_loss
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.reid_evaluation import evaluate_reid


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


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
    assert args.variant == 'axis_mass_projected_fullref'
    model = ProjectedMassAxisCollaborationDeMo(classes, cfg, cameras).float().cuda()
    model.contribution_loss_weight = args.contribution_weight
    return model


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
        weights = model.loss_weights if isinstance(model, (ResidualAugmentedDeMo, AxisCollaborationDeMo)) else [1.] * (end // 2)
        assert len(weights) == end // 2
        loss = sum(weights[i // 2] * loss_fn(output[i], output[i + 1], target, cam) for i in range(0, end, 2))
        if len(output) % 2:
            loss = loss + output[-1]
    assert torch.isfinite(loss), 'non-finite training loss'
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= previous_scale
    return float(loss.detach()), list(names), updated, scaler.get_scale()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['RGBNT201', 'RGBNT100', 'MSVR310'], required=True)
    parser.add_argument('--variant', choices=['axis_mass_projected_fullref'], required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--contribution-weight', type=float, default=.05)
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
    info['amp_policy'] = 'native DeMo fp16 GradScaler; skipped updates counted separately from batch attempts'
    if isinstance(model, ResidualAugmentedDeMo):
        info['method_revision'] = 'residual_v2: full DeMo HDM anchor, rank64 frequency values/band adapter, small learned residual scales; no contribution loss or modality-dropout training'
        info['loss_weights'] = model.loss_weights
    if isinstance(model, AxisCollaborationDeMo):
        info['method_revision'] = 'axis_collaboration_v3: separate source/band constrained experts or parameter-matched unrestricted ordinary twins; full DeMo base; one simultaneous query-condition round; stopped-gradient four-state retrieval-contribution calibration'
        info['loss_weights'] = model.loss_weights
        info['contribution_loss_weight'] = model.contribution_loss_weight
        info['missing_protocol'] = 'exactly zero normalized inputs determine availability; structured relations require all their members available; no modality-dropout training in this stage'
    if isinstance(model, ScaledAxisCollaborationDeMo):
        info['method_revision'] = 'axis_collaboration_v4: independent residual-direction normalization and stopped full11 reference interventions; same V3 experts/parameters/loss weights'
        info['residual_direction_normalization'] = model.normalize_residual
        info['residual_anchor'] = 'stopped per-relation original DeMo norms for M/I; stopped whole-base norm for F; unchanged learned scales and independent gates' if model.normalize_residual else 'original raw V3 projection amplitudes'
        info['contribution_reference'] = 'stopped current-batch full11 gallery' if model.full_reference else 'stopped current-batch base00 gallery'
    info['method_revision'] = 'axis_collaboration_v6: same V5 model and routing; group identity losses supervise normalized PM/PF projection directions'
    info['auxiliary_identity_taps'] = 'M: mean of normalized PM pooled independent relation vectors; F: normalized PF pooled independent frequency vector; same512D heads, CE/Triplet weights unchanged'
    info['terminal_checkpoint_storage'] = 'After all50 epochs only: model, scheduler/scaler and counters; no Adam moment duplicate or optimizer resume from last. Smoke still saves and strictly restores complete model+optimizer. Best weights and all training/evaluation computation unchanged.'
    info['relation_routing'] = 'joint pi relation marginal times number of eligible relations; mean eligible weight one; M-only recomputes independent a_S, no full-state mass'
    info['unchanged_V5_factors'] = 'initial parameters/state, backbone, shared band experts, one-round conditioning, joint relation mass, normalized retrieval residuals/gates/scales, full11 stopped reference, all loss weights/optimizer/split/sampling/publicCLIP/50epochs; only two group auxiliary feature taps move after existing PM/PF'
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
            loss, names, updated, scale = step(model, batch, optimizer, scaler, loss_fn)
            assert updated, 'smoke optimizer step skipped'
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
        if isinstance(model, AxisCollaborationDeMo):
            result['contribution_sanity'] = model.contribution_audit
        write_json(out / 'smoke.json', result)
        print('SMOKE_PASS', flush=True)
        return
    assert not (out / 'epochs.csv').exists(), 'use a new training output directory'
    val_loader = make_loader(dev, cfg, False, args.seed)
    best = {'mAP': -1}
    optimizer_steps = 0
    with (out / 'epochs.csv').open('w', newline='', encoding='utf-8') as table, (out / 'batch_orders.jsonl').open('w', encoding='utf-8') as orders:
        writer = csv.DictWriter(table, fieldnames=['epoch', 'steps', 'optimizer_steps', 'amp_skipped_steps', 'amp_scale', 'loss', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'seconds'])
        writer.writeheader()
        for epoch in range(1, 51):
            start = time.time()
            # Sampler and worker augmentation streams are matched across variants.
            seed_all(args.seed + epoch)
            loader = make_loader(fit, cfg, True, args.seed + epoch)
            model.train(); scheduler.step(epoch)
            losses = []
            for batch in loader:
                loss, names, updated, scale = step(model, batch, optimizer, scaler, loss_fn)
                steps += 1; losses.append(loss)
                optimizer_steps += int(updated)
                orders.write(json.dumps({'epoch': epoch, 'step': steps, 'optimizer_updated': updated, 'amp_scale': scale, 'names': names}) + '\n')
                if not updated:
                    print('AMP_SKIPPED', epoch, steps, 'scale', scale, flush=True)
            metrics = evaluate(model, val_loader, queries, args.dataset)
            row = {'epoch': epoch, 'steps': steps, 'optimizer_steps': optimizer_steps,
                   'amp_skipped_steps': steps - optimizer_steps, 'amp_scale': scaler.get_scale(),
                   'loss': float(np.mean(losses)), 'seconds': time.time() - start,
                   **{k: metrics[k] for k in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')}}
            writer.writerow(row); table.flush(); orders.flush()
            if metrics['mAP'] > best['mAP']:
                best = {**metrics, 'epoch': epoch}
                torch.save(model.state_dict(), out / 'best.pth')
                write_json(out / 'best.json', best)
            write_json(out / 'status.json', {'status': 'RUNNING', 'epoch': epoch, 'steps': steps, 'latest': row, 'best': best})
            print('EPOCH', json.dumps(row), 'BEST', json.dumps(best), flush=True)
        torch.save({'model': model.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': 50, 'steps': steps, 'optimizer_steps': optimizer_steps}, out / 'last.pth')
    model.load_state_dict(torch.load(out / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    reloaded = evaluate(model, val_loader, queries, args.dataset, out / 'best_dev_arrays.npz')
    assert all(abs(reloaded[k] - best[k]) < 1e-8 for k in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
    terminal = {**info, 'status': 'COMPLETE', 'epochs': 50, 'steps': steps, 'best': best, 'strict_reload': reloaded,
                'optimizer_steps': optimizer_steps, 'amp_skipped_steps': steps - optimizer_steps,
                'peak_memory': torch.cuda.max_memory_allocated(), 'finished': time.time()}
    write_json(out / 'result.json', terminal)
    write_json(out / 'status.json', terminal)
    print('TRAIN_COMPLETE', json.dumps(terminal), flush=True)


if __name__ == '__main__':
    main()
