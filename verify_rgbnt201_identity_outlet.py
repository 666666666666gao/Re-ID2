"""Reuse actual native-update gates; read all official5120D identity states."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import diagnose_rgbnt201_projection_probe as native
from evaluate_full_official49 import SETS
from experiment_data import make_loader
from full_evaluation import full_metrics, distance
from missing_evaluation import mask_images
from official_training_data import metadata
from rgbnt201_identity_outlet import build
from run_experiment import write_json
from shared_identity_axis import encode_available


def selected_builder(args, cfg, classes, cameras):
    args.freeze_identity_encoder = 1
    return build(args, cfg, classes, cameras)


@torch.no_grad()
def readout(model, query, gallery, cfg, seed, output):
    output.mkdir()
    model.eval()
    images, _, cam, scene, _ = next(iter(make_loader(query[:8], cfg, False, seed)))
    images = {key: value.cuda() for key, value in images.items()}
    cam, scene = cam.cuda(), scene.cuda()
    for available, missing in SETS.items():
        masked = mask_images(images, missing)
        private = F.normalize(encode_available(model, masked, cam, scene)[2][:, :5120].float(), dim=1)
        states = model(masked, cam_label=cam, view_label=scene, return_states=True)
        assert torch.equal(states['00'], private)
        assert all(list(value.shape) == [8,5120] and torch.isfinite(value).all() for value in states.values())
    initial = model(images, cam_label=cam, view_label=scene, return_states=True)
    saved = model.modality_expert.query.detach().clone()
    model.modality_expert.query.add_(.1)
    changed_m = model(images, cam_label=cam, view_label=scene, return_states=True)
    assert torch.equal(initial['01'], changed_m['01']) and torch.equal(initial['00'], changed_m['00'])
    assert not torch.equal(initial['10'], changed_m['10'])
    model.modality_expert.query.copy_(saved)
    saved = model.frequency_expert.query.detach().clone()
    model.frequency_expert.query.add_(.1)
    changed_f = model(images, cam_label=cam, view_label=scene, return_states=True)
    assert torch.equal(initial['10'], changed_f['10']) and torch.equal(initial['00'], changed_f['00'])
    assert not torch.equal(initial['01'], changed_f['01'])
    model.frequency_expert.query.copy_(saved)
    restored = model(images, cam_label=cam, view_label=scene, return_states=True)
    assert all(torch.equal(initial[key], restored[key]) for key in initial)
    collected = {name: [] for name in ('00','10','01','11','private')}
    for images, _, cam, scene, _ in make_loader(query + gallery, cfg, False, seed):
        images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
        cam, scene = cam.cuda(), scene.cuda()
        states = model(images, cam_label=cam, view_label=scene, return_states=True)
        private = F.normalize(encode_available(model, images, cam, scene)[2][:, :5120].float(), dim=1)
        assert torch.equal(states['00'], private)
        for name, value in dict(states, private=private).items():
            collected[name].append(F.normalize(value,dim=1).cpu())
    arrays = metadata(query,'query') | metadata(gallery,'gallery')
    metrics, raw = {}, dict(arrays)
    for name, parts in collected.items():
        feature = torch.cat(parts)
        assert list(feature.shape) == [1672,5120] and torch.isfinite(feature).all()
        q,g=feature[:len(query)],feature[len(query):]
        metrics[name]=full_metrics(distance(q,g), arrays['query_ids'],arrays['gallery_ids'],
            arrays['query_cameras'],arrays['gallery_cameras'],arrays['query_names'],
            arrays['query_cameras'],arrays['query_scenes'],output/name)
        raw[name+'_query_features'],raw[name+'_gallery_features']=q.numpy(),g.numpy()
    assert all(metrics['00'][key] == metrics['private'][key] for key in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20'))
    np.savez_compressed(output/'readout_arrays.npz',**raw)
    write_json(output/'result.json',dict(metrics=metrics,query_count=836,gallery_count=836,
        descriptor_dim=5120, all7_closed00_equals_available_private=True,
        M_perturbation_does_not_change_state01=True,F_perturbation_does_not_change_state10=True,
        limits='Three native updates per mode, no50/all49 claim; closed states do not receive disabled-expert messages.'))
    return metrics


def record(path, value):
    if value.get('status') == 'PASS_RGBNT201_PROJECTION_NATIVE3_PER_MODE_FULL_READOUT':
        value['status'] = 'PASS_RGBNT201_IDENTITY_OUTLET_NATIVE3_PER_MODE_FULL_READOUT'
        value['descriptor_dim'] = 5120
        value['limits'] = 'Paired3 nativeupdates per mode, all836 query/gallery, exact original full00/private six metrics, no formal50/all49.'
    write_json(path,value)


if __name__ == '__main__':
    native.build=selected_builder
    native.readout=readout
    native.write_json=record
    native.main()
