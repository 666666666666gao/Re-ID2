"""P0: frozen routed retrieval directions versus independently supervised features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiment_data import make_loader, split_records
from full_evaluation import distance, full_metrics
from run_mass_experiment import build as build_v5, configuration, write_json
from run_projected_mass_experiment import build as build_v6


def load_checkpoint(run):
    exit_file = run.parent / (run.name + '_exit.json')
    terminal = json.loads((run / 'result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == 50
    assert json.loads(exit_file.read_text())['exit_code'] == 0
    args = argparse.Namespace(**terminal['arguments'])
    assert args.seed == 42
    inputs = (run/'best.pth', run/'best_dev_arrays.npz', run/'result.json', exit_file)
    proofs = {str(path): {'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs}
    cfg = configuration(args)
    fit, dev, query, classes, cameras = split_records(args.data_root, args.dataset)
    builder = {'axis_mass_fullref': build_v5, 'axis_mass_projected_fullref': build_v6}[args.variant]
    model = builder(args, cfg, classes, cameras)
    model.load_state_dict(torch.load(run/'best.pth', map_location='cuda', weights_only=True), strict=True)
    return model, args, cfg, fit, dev, np.asarray(query), classes, terminal, proofs


def unchanged_files(proofs):
    for filename, proof in proofs.items():
        path = Path(filename)
        assert path.stat().st_size == proof['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == proof['sha256']


@torch.no_grad()
def extract(model, dev, cfg, seed, variant):
    features = {key: [] for key in ('base00', 'full11', 'M_routed', 'F_routed', 'M_aux', 'F_aux')}
    capture, mreads, freads = {}, [], []
    read_m, read_f = model.modality_expert.read, model.frequency_expert.read

    def read_modality(*args, **kwargs):
        result = read_m(*args, **kwargs)
        mreads.append(result)
        return result

    def read_frequency(*args, **kwargs):
        result = read_f(*args, **kwargs)
        freads.append(result)
        return result

    def remember(name):
        def hook(module, inputs, result):
            # The full path calls each projection first; V6 AUX replay follows.
            capture.setdefault(name, result)
        return hook

    model.modality_expert.read = read_modality
    model.frequency_expert.read = read_frequency
    hooks = [model.calibrator.register_forward_pre_hook(
        lambda module, inputs: capture.update(base=inputs[0]))]
    for name, module in [('PM', model.modality_projection), ('PF', model.frequency_projection), ('PI', model.interaction_projection)]:
        hooks.append(module.register_forward_hook(remember(name)))
    max_reconstruction_error = 0.0
    for images, _, cam, scene, _ in make_loader(dev, cfg, False, seed):
        capture.clear(); mreads.clear(); freads.clear()
        images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
        assert all(value.flatten(1).ne(0).any(1).all() for value in images.values())
        full = model(images, cam_label=cam.cuda(), view_label=scene.cuda()).float()
        assert len(mreads) == len(freads) == 2 and set(capture) == {'base','PM','PF','PI'}
        base = capture['base'].float()
        batch, dim = len(base), 512
        anchor_m = base[:,3*dim:10*dim].reshape(batch,7,dim).norm(dim=-1,keepdim=True)
        weights = model.last_route.sum(2) * 7
        m = F.normalize(capture['PM'].float(), dim=-1) * anchor_m * weights[:,:,None]
        f = F.normalize(capture['PF'].float(), dim=1) * base.norm(dim=1,keepdim=True)
        interaction = F.normalize(capture['PI'].float(), dim=1)[:,None].expand(batch,7,dim)
        interaction = interaction * anchor_m * weights[:,:,None]
        gates = model.last_gates.float()
        delta_m = model.residual_scale[0] * gates[:,:1] * m.flatten(1)
        delta_m = delta_m + model.residual_scale[2] * gates[:,2:] * interaction.flatten(1)
        delta_f = model.residual_scale[1] * gates[:,1:2] * f
        rebuilt = base + torch.cat((torch.zeros_like(base[:,:3*dim]),delta_m,delta_f),-1)
        error = float((rebuilt-full).abs().max())
        max_reconstruction_error = max(max_reconstruction_error,error)
        assert error == 0, 'captured routed directions must reconstruct the existing fusion exactly'
        # AUX-only projections differ from the conditioned, jointly routed path.
        # Hooks retain the first (actual fusion) projection output.
        available = torch.ones((batch,3),dtype=torch.bool,device=base.device)
        eligible = torch.ones((batch,7),dtype=torch.bool,device=base.device)
        ma = model.modality_expert.pool(mreads[0].float(),eligible)
        fa = model.frequency_expert.pool(freads[0].float(),available,model.structured)
        if variant == 'axis_mass_projected_fullref':
            ma = F.normalize(model.modality_projection(ma),dim=-1).mean(1)
            fa = F.normalize(model.frequency_projection(fa),dim=1)
        else:
            ma = ma.mean(1)
        values = dict(base00=base, full11=full, M_routed=m.flatten(1), F_routed=f, M_aux=ma, F_aux=fa)
        for key, value in values.items():
            assert torch.isfinite(value).all() and (value.norm(dim=1)>0).all()
            features[key].append(F.normalize(value.float(),dim=1).cpu())
    for hook in hooks: hook.remove()
    model.modality_expert.read, model.frequency_expert.read = read_m, read_f
    return {key: torch.cat(value) for key,value in features.items()}, max_reconstruction_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    started = time.time()
    model, original, cfg, _, dev, query, _, terminal, proofs = load_checkpoint(Path(args.run_dir))
    saved = np.load(Path(args.run_dir)/'best_dev_arrays.npz')
    assert np.array_equal(saved['query_indices'],query)
    assert np.array_equal(saved['ids'],[row[1] for row in dev])
    assert np.array_equal(saved['cameras'],[row[2] for row in dev])
    assert np.array_equal(saved['scenes'],[row[3] for row in dev])
    assert np.array_equal(saved['names'],[Path(row[0] if isinstance(row[0],str) else row[0][0]).name for row in dev])
    model.eval()
    versions = {name:value._version for name,value in model.state_dict().items()}
    features, reconstruction = extract(model, dev[:64] if args.smoke else dev, cfg, original.seed, original.variant)
    parity = float(np.abs(features['full11'].numpy()-saved['features'][:len(features['full11'])]).max())
    assert parity == 0
    assert versions == {name:value._version for name,value in model.state_dict().items()}
    unchanged_files(proofs)
    report = dict(status='PASS_FROZEN_UTILITY_SMOKE' if args.smoke else 'PASS_FROZEN_RETRIEVAL_UTILITY',
        dataset=original.dataset, variant=original.variant, selected_epoch=terminal['best']['epoch'],
        optimizer_updates=0, official_test_uses=0, original_input_files=proofs,
        normal_feature_max_error=parity, fusion_reconstruction_max_error=reconstruction,
        state_tensor_versions_unchanged=True, descriptor_dimensions={key:value.shape[1] for key,value in features.items()},
        scope='Frozen clean development diagnostic. Routed M is exact normalized PM output with original relation mass and base relation anchors; routed F is exact normalized PF output. Scalar outer gates/scales cancel under standalone normalization. Independent auxiliary features follow the existing V5/V6 training taps.',
        limits='Standalone descriptors have different dimensions and no retraining. Standalone correctness or an oracle union is evidence of available information, not realized fusion gain or a capacity-matched method comparison.')
    if not args.smoke:
        ids, cams, scenes, names = (saved[key] for key in ('ids','cameras','scenes','names'))
        exclusion = scenes if original.dataset=='MSVR310' else cams
        metrics, rows = {}, {}
        for key,value in features.items():
            distances = saved['distances'] if key=='full11' else distance(value[query],value)
            metrics[key] = full_metrics(distances,ids[query],ids,exclusion[query],exclusion,
                names[query],cams[query],scenes[query],output/key)
            with (output/(key+'.csv')).open(encoding='utf-8',newline='') as handle: rows[key]=list(csv.DictReader(handle))
        assert all(abs(metrics['full11'][key]-terminal['best'][key])<1e-8 for key in ('mAP','Rank-1','Rank-5','Rank-10'))
        base_correct = np.array([int(row['Rank-1']) for row in rows['base00']])
        base_ap = np.array([float(row['AP']) for row in rows['base00']])
        complement = {}
        for key in ('M_routed','F_routed','M_aux','F_aux','full11'):
            correct = np.array([int(row['Rank-1']) for row in rows[key]])
            ap = np.array([float(row['AP']) for row in rows[key]])
            complement[key] = dict(base_wrong_expert_correct=int(((base_correct==0)&(correct==1)).sum()),
                base_correct_expert_wrong=int(((base_correct==1)&(correct==0)).sum()),
                both_correct=int(((base_correct==1)&(correct==1)).sum()),both_wrong=int(((base_correct==0)&(correct==0)).sum()),
                queries_AP_improved=int((ap>base_ap).sum()),queries_AP_worsened=int((ap<base_ap).sum()),queries_AP_equal=int((ap==base_ap).sum()),
                base_wrong_query_count=int((base_correct==0).sum()),
                base_wrong_AP_delta_sum=float((ap-base_ap)[base_correct==0].sum()))
        m = np.array([int(row['Rank-1']) for row in rows['M_routed']])
        f = np.array([int(row['Rank-1']) for row in rows['F_routed']])
        report.update(metrics=metrics,complementarity_to_base=complement,
            routed_M_F_correctness=dict(M_only=int(((m==1)&(f==0)).sum()),F_only=int(((m==0)&(f==1)).sum()),
                both=int(((m==1)&(f==1)).sum()),neither=int(((m==0)&(f==0)).sum())),
            base_wrong_routed_expert_oracle_union=int(((base_correct==0)&((m==1)|(f==1))).sum()))
        np.savez_compressed(output/'features.npz',**{key:value.numpy() for key,value in features.items()},query_indices=query,ids=ids,cameras=cams,scenes=scenes,names=names)
    report['wall_seconds'] = time.time()-started
    write_json(output/'result.json',report)
    print(report['status'], original.dataset, json.dumps(report.get('complementarity_to_base',{})), flush=True)


if __name__=='__main__': main()
