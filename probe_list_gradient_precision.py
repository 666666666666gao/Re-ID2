"""One actual full-view forward, zero optimizer updates; compare grad contexts."""
import argparse
import json
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all
from list_retrieval_objective import smooth_ap
from official_training_data import full_records
from run_experiment import configuration
from run_r201e_list_objective import build


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'anchor-run-dir', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    args.dataset, args.variant, args.seed = 'RGBNT201', 'frequency_shared', 42
    args.freeze_identity_encoder, args.mode = 1, 'smoke'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = configuration(args)
    train, _, _, classes, cameras, _ = full_records(args.data_root, args.dataset)
    model = build(args, cfg, classes, cameras)
    seed_all(args.seed)
    model.train()
    images, target, cam, scene, names = next(iter(make_loader(train, cfg, True, args.seed)))
    assert len(names) == len(set(names)) == 64
    images = {key: value.cuda() for key, value in images.items()}
    target, cam, scene = target.cuda(), cam.cuda(), scene.cuda()
    with torch.autocast('cuda'):
        output = model(images, label=target, cam_label=cam, view_label=scene)
        feature = output[1]
        loss = smooth_ap(feature, target)
        grad_amp = torch.autograd.grad(loss, feature, retain_graph=True)[0]
    with torch.autocast('cuda', enabled=False):
        grad_fp32 = torch.autograd.grad(loss, feature, retain_graph=True)[0]
        unit = torch.nn.functional.normalize(feature.float(), dim=1)
        scores = unit @ unit.T
        same = target[:, None].eq(target[None, :])
        eye = torch.eye(64, device=target.device, dtype=torch.bool)
        positive = same & ~eye
        negative = ~same
        margin = scores.masked_fill(~positive, torch.inf).min(1).values - scores.masked_fill(~negative, -torch.inf).max(1).values
    def gradient(values):
        return dict(finite=bool(torch.isfinite(values).all()), l1=float(values.abs().sum()),
                    maximum=float(values.abs().max()), dtype=str(values.dtype), nonzero=int(torch.count_nonzero(values)))
    result = dict(status='ACTUAL_ONE_FORWARD_AP_FEATURE_GRADIENT_CONTEXT_DIAGNOSIS',
        optimizer_updates=0, batch_size=64, identity_counts=torch.unique(target, return_counts=True)[1].tolist(),
        primary_metric_loss=float(loss.detach()), feature_dtype=str(feature.dtype),
        amp_context_gradient=gradient(grad_amp), fp32_context_gradient=gradient(grad_fp32),
        max_gradient_difference=float((grad_amp-grad_fp32).abs().max()),
        positive_counts=positive.sum(1).tolist(), margin=dict(minimum=float(margin.min()),mean=float(margin.mean()),maximum=float(margin.max())),
        names=list(names), new_formal_training=False,
        limits='One diagnostic forward of ordinary control; no optimizer or proof of retrieval gain. Same loss graph, two autograd contexts.')
    path = Path(args.output)
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
