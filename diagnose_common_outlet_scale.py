"""Frozen V12 outlet scale and source-preserving projection utility, no training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np
import torch
from torch.nn import functional as F

from common_coordinate_axis import build
from diagnose_shared_identity_states import SETS
from experiment_data import make_loader, split_records
from full_evaluation import distance, full_metrics
from missing_evaluation import METRICS
from run_experiment import configuration, write_json
from shared_identity_axis import metric_descriptor


STAGES = ('deployed', 'base_common', 'M_pre', 'M_post', 'F_pre', 'F_post')


def attach_probe(model):
    captured = {}
    handles = []
    for name, module in (('M', model.modality_projection), ('F', model.frequency_projection),
                         ('I', model.interaction_projection)):
        def capture(module, inputs, output, name=name):
            assert name not in captured
            captured[name] = output.detach()
        handles.append(module.register_forward_hook(capture))
    original = model.fuse

    def inspected(self, base, modality, frequency, gates, use_m, use_f, eligible, relation_mass=None):
        captured.clear()
        result = original(base, modality, frequency, gates, use_m, use_f, eligible, relation_mass)
        assert use_m and use_f and set(captured) == {'M', 'F', 'I'}
        with torch.autocast('cuda', enabled=False):
            base, modality, frequency, gates = (value.float() for value in (base, modality, frequency, gates))
            batch, _, dim = modality.shape
            count = eligible.sum(1, keepdim=True)
            anchor = base[:, 1536:5120].reshape(batch, 7, dim).norm(dim=-1, keepdim=True)
            shared = base[:, 5120:]
            shared_norm = shared.norm(dim=1, keepdim=True)
            assert (shared_norm > 0).all()
            weights = relation_mass * count
            mpre_rows = F.normalize(modality, dim=-1) * anchor * eligible[..., None] * weights[..., None]
            mpost_rows = F.normalize(captured['M'].float(), dim=-1) * anchor * eligible[..., None] * weights[..., None]
            dm = self.residual_scale[0] * gates[:, :1] * mpost_rows.flatten(1)
            df = self.residual_scale[1] * gates[:, 1:2] * F.normalize(captured['F'].float(), dim=1) * shared_norm
            irows = F.normalize(captured['I'].float(), dim=1)[:, None] * anchor * eligible[..., None] * weights[..., None]
            di = self.residual_scale[2] * gates[:, 2:] * irows.flatten(1)
            mi = (dm + di).reshape(batch, 7, dim).mean(1)
            # Production adds df+mi before adding the base; preserve its association.
            reconstructed = metric_descriptor(torch.cat((base[:, :5120], shared + (df + mi)), 1))
            assert torch.equal(reconstructed, result), 'read-only reconstruction must equal the production fuse'
            assert torch.count_nonzero(anchor * ~eligible[..., None]) == 0
            assert torch.count_nonzero(modality * ~eligible[..., None]) == 0
            weighted_anchor = (anchor.squeeze(-1) * weights).sum(1)
            assert (weighted_anchor > 0).all()
            self.outlet_stages = dict(deployed=result, base_common=shared,
                M_pre=mpre_rows.mean(1), M_post=mpost_rows.mean(1), F_pre=frequency, F_post=captured['F'].float())
            self.outlet_scalars = dict(legal_relations=count[:, 0].float(), shared_norm=shared_norm[:, 0],
                private_norm=base[:, :5120].norm(dim=1), weighted_anchor=weighted_anchor,
                weighted_anchor_mean7=weighted_anchor / 7,
                weighted_anchor_relative_to_shared=weighted_anchor / (7 * shared_norm[:, 0]),
                M_common_increment_relative_to_shared=dm.reshape(batch, 7, dim).mean(1).norm(dim=1) / shared_norm[:, 0],
                F_common_increment_relative_to_shared=df.norm(dim=1) / shared_norm[:, 0],
                I_common_increment_relative_to_shared=di.reshape(batch, 7, dim).mean(1).norm(dim=1) / shared_norm[:, 0],
                MI_common_increment_relative_to_shared=mi.norm(dim=1) / shared_norm[:, 0],
                total_increment_relative_to_shared=(df + mi).norm(dim=1) / shared_norm[:, 0],
                M_cancellation_ratio=mpost_rows.sum(1).norm(dim=1) / weighted_anchor,
                common_cosine_change=1 - F.cosine_similarity(shared, shared + (df + mi), dim=1),
                legal_mean_amplitude_factor=7 / count[:, 0].float(),
                M_pre_norm=mpre_rows.mean(1).norm(dim=1), M_post_norm=mpost_rows.mean(1).norm(dim=1),
                F_pre_norm=frequency.norm(dim=1), F_post_norm=captured['F'].float().norm(dim=1),
                gate_M=gates[:, 0], gate_F=gates[:, 1], gate_I=gates[:, 2])
            for index in range(7):
                self.outlet_scalars['anchor_' + str(index)] = anchor[:, index, 0]
                self.outlet_scalars['relation_mass_' + str(index)] = relation_mass[:, index]
                self.outlet_scalars['eligible_' + str(index)] = eligible[:, index].float()
        return result

    model.fuse = types.MethodType(inspected, model)
    return handles


@torch.no_grad()
def extract(model, records, cfg, seed, retained):
    banks = {key: [] for key in STAGES}
    scalars = {key: [] for key in ('names',)}
    for images, _, cam, scene, names in make_loader(records, cfg, False, seed):
        images = {key: value.cuda(non_blocking=True) if index in retained else torch.zeros_like(value).cuda()
                  for index, (key, value) in enumerate(images.items())}
        model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        for key, value in model.outlet_stages.items():
            assert torch.isfinite(value).all()
            banks[key].append(F.normalize(value.float(), dim=1).cpu())
        scalars['names'].extend(names)
        for key, value in model.outlet_scalars.items():
            assert torch.isfinite(value).all()
            scalars.setdefault(key, []).extend(value.cpu().tolist())
    return {key: torch.cat(values) for key, values in banks.items()}, scalars


def main():
    parser = argparse.ArgumentParser()
    for name in ('run-dir', 'previous-frozen', 'output', 'data-root', 'pretrained'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run, output, previous = Path(args.run_dir), Path(args.output), Path(args.previous_frozen)
    terminal = json.loads((run / 'result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == 50
    arguments = argparse.Namespace(**terminal['arguments'])
    assert arguments.variant in ('axis_shared', 'frequency_shared', 'twins_shared') and arguments.dataset == 'MSVR310'
    arguments.data_root, arguments.pretrained = args.data_root, args.pretrained
    exit_file = run.parent / (run.name + '_exit.json')
    assert json.loads(exit_file.read_text())['exit_code'] == 0
    old = json.loads((previous / 'result.json').read_text())
    assert old['status'] == 'COMPLETE' and len(old['measurements']) == 49
    protected = (run / 'best.pth', run / 'best_dev_arrays.npz', run / 'result.json', exit_file, previous / 'result.json')
    inputs = {str(p): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in protected}
    cfg = configuration(arguments)
    _, dev, query, classes, cameras = split_records(args.data_root, arguments.dataset)
    with np.load(run / 'best_dev_arrays.npz') as saved:
        assert np.array_equal(saved['query_indices'], query)
        for key, values in (('ids', [r[1] for r in dev]), ('cameras', [r[2] for r in dev]), ('scenes', [r[3] for r in dev]),
                            ('names', [Path(r[0] if isinstance(r[0], str) else r[0][0]).name for r in dev])):
            assert np.array_equal(saved[key], values)
        saved = {key: saved[key] for key in saved.files}
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    assert all(not m.training for m in model.modules())
    versions = {name: value._version for name, value in model.state_dict().items()}
    handles = attach_probe(model)
    output.mkdir(exist_ok=False)
    started = time.time()
    records = dev[:64] if args.smoke else dev
    banks, scalars = {}, {}
    for name, retained in SETS.items():
        banks[name], scalars[name] = extract(model, records, cfg, arguments.seed, retained)
    for handle in handles:
        handle.remove()
    assert versions == {name: value._version for name, value in model.state_dict().items()}
    parity = float(np.abs(banks['RNT']['deployed'].numpy() - saved['features'][:len(records)]).max())
    assert parity == 0
    if args.smoke:
        write_json(output / 'smoke.json', dict(status='PASS', normal_feature_max_error=parity,
            original_fuse_reconstructed_exact=True, seven_banks=True, stages=list(STAGES),
            state_tensor_versions_unchanged=True, optimizer_updates=0, triplets=len(records)))
        return
    for name, values in scalars.items():
        assert len(values['names']) == len(dev)
        with (output / ('scale_' + name + '.csv')).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['row', 'availability'] + list(values))
            writer.writeheader()
            writer.writerows(dict(row=i, availability=name, **{key: value[i] for key, value in values.items()}) for i in range(len(dev)))
    query = np.asarray(query)
    ids, cams, scenes, names = (saved[key] for key in ('ids', 'cameras', 'scenes', 'names'))
    measurements, raw = {}, {}
    for qset in SETS:
        for gset in SETS:
            condition = 'q_' + qset + '_g_' + gset
            folder = output / condition
            folder.mkdir()
            values = {}
            for stage in STAGES:
                distances = saved['distances'] if stage == 'deployed' and qset == gset == 'RNT' else distance(banks[qset][stage][query], banks[gset][stage])
                raw[condition + '_' + stage] = distances
                values[stage] = full_metrics(distances, ids[query], ids, scenes[query], scenes,
                                            names[query], cams[query], scenes[query], folder / stage)
            assert all(values['deployed'][key] == old['measurements'][condition]['metrics'][key] for key in METRICS)
            with (previous / (condition + '.csv')).open(encoding='utf-8') as handle:
                old_rows = list(csv.DictReader(handle))
            with (folder / 'deployed.csv').open(encoding='utf-8') as handle:
                assert list(csv.DictReader(handle)) == old_rows
            measurements[condition] = values
            print('OUTLET_CONDITION', condition, json.dumps({s: {m: v[m] for m in METRICS} for s, v in values.items()}), flush=True)
    np.savez_compressed(output / 'raw_distances.npz', **raw, query_indices=query, ids=ids, cameras=cams, scenes=scenes, names=names)
    for path in protected:
        assert path.stat().st_size == inputs[str(path)]['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == inputs[str(path)]['sha256']
    write_json(output / 'result.json', dict(status='COMPLETE', dataset='MSVR310', variant=arguments.variant,
        selected_epoch=terminal['best']['epoch'], measurements=measurements, metric_count=49 * len(STAGES),
        normal_feature_max_error=parity, original_fuse_reconstructed_exact=True, previous_all49_deployed_exact=True,
        state_tensor_versions_unchanged=True, original_inputs=inputs, optimizer_updates=0, official_test_uses=0,
        residual_scale=model.residual_scale.detach().cpu().tolist(), seconds=time.time() - started,
        raw_distance_archive='raw_distances.npz',
        protocol='Frozen V12; all seven original availability banks. M_pre/M_post use identical route mass, anchor and mean7 before/after PM; F_pre/F_post use the actual routed F before/after PF. Independent normalized retrieval stages, no changed deployment or pooling intervention.',
        limits='Single dataset seed42; frozen stage diagnostics do not prove trained pooling superiority, shared semantics or causality. legal_mean_amplitude_factor is a mathematical diagnostic, not a validated recipe.'))


if __name__ == '__main__':
    main()
