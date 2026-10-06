"""Read the two completed I201 selected feature banks; no NN or NumPy."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

from analyze_r201c_saved_geometry import floats
from analyze_identity_coordinate_three_normal import load, query_rows, table

PROJECT = Path(__file__).resolve().parent


def main():
    out = PROJECT / 'results/r201i_selected_identity_shift_20261007'
    assert not out.exists()
    receipt_path = PROJECT / 'results/preflight/r201i_primary_margin_RGBNT201_actual_session_20261006.json'
    receipt = load(receipt_path)
    assert receipt['exit_code'] == 0 and receipt['successful_updates'] == 5294
    previous_path = PROJECT / 'results/r201gh_selected_geometry_20261006/result.json'
    previous = load(previous_path)
    base_path = Path(previous['base_archive'])
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == previous['base_archive_sha256']
    original, _ = query_rows(PROJECT / 'results/full_official_baselines_20261004/training/RGBNT201_demo_s42', 'RGBNT201')
    sources = {}
    samples = []
    with zipfile.ZipFile(base_path) as base_archive:
        for variant in ('frequency_shared', 'axis_shared'):
            name = 'RGBNT201_r201i_' + variant + '_s42'
            entry = receipt['archives'][name]
            path = Path(entry['local'])
            assert path.stat().st_size == entry['file']['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['file']['sha256']
            result_dir = PROJECT / 'results/r201i_primary_margin_20261006/RGBNT201/training' / name
            result = load(result_dir / 'result.json')
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50 and result['amp_skipped_steps'] == 0
            current, _ = query_rows(result_dir, 'RGBNT201')
            model = 'I_' + variant
            sources[model] = dict(path=str(path), file=entry['file'], selected_epoch=result['best']['epoch'])
            with zipfile.ZipFile(path) as saved:
                for side in ('query', 'gallery'):
                    for key in ('ids', 'cameras', 'scenes', 'names'):
                        assert base_archive.read(side + '_' + key + '.npy') == saved.read(side + '_' + key + '.npy')
                    base = floats(base_archive, 'private_' + side + '_features')
                    fused = floats(saved, side + '_features')
                    for index in range(836):
                        b = base[index * 5120:(index + 1) * 5120]
                        z = fused[index * 5120:(index + 1) * 5120]
                        assert abs(math.fsum(x*x for x in b) - 1) < 1e-5
                        assert abs(math.fsum(x*x for x in z) - 1) < 1e-5
                        bg, zg = (math.fsum(x*x for x in v[:1536]) for v in (b, z))
                        assert bg > 0 and zg > 0
                        scale = math.sqrt(bg / zg)
                        residual = [scale*y - x for x, y in zip(b, z)]
                        global_error = max(abs(v) for v in residual[:1536])
                        assert global_error < 2e-6
                        energy = math.fsum(v*v for v in residual)
                        assert energy > 0
                        radial = math.fsum(x*y for x, y in zip(b, residual))
                        cosine = math.fsum(x*y for x, y in zip(b, z))
                        epsilon = math.sqrt(math.fsum((x-y)**2 for x, y in zip(b, z)))
                        outcome = ''
                        delta_AP = ''
                        if side == 'query':
                            before, after = original[index], current[index]
                            assert all(before[k] == after[k] for k in ('name', 'identity', 'camera', 'scene', 'query_index'))
                            a, c = int(before['Rank-1']), int(after['Rank-1'])
                            outcome = 'rescued' if c > a else 'harmed' if c < a else 'both_correct' if c else 'both_wrong'
                            delta_AP = float(after['AP']) - float(before['AP'])
                        samples.append(dict(model=model, selected_epoch=result['best']['epoch'], role=side, index=index,
                            rank1_outcome=outcome, delta_AP=delta_AP,
                            final_descriptor_shift_epsilon=epsilon,
                            relative_raw_correction_norm=math.sqrt(energy),
                            radial_energy_fraction=radial*radial/energy,
                            tangent_energy_fraction=1-radial*radial/energy,
                            direction_angle_degrees=math.degrees(math.acos(min(1, max(-1, cosine)))),
                            fused_to_base_raw_norm_ratio=scale,
                            unchanged_global_max_error=global_error))
    assert len(samples) == 3344
    fields = ('final_descriptor_shift_epsilon', 'relative_raw_correction_norm', 'radial_energy_fraction',
              'tangent_energy_fraction', 'direction_angle_degrees', 'fused_to_base_raw_norm_ratio', 'unchanged_global_max_error')
    groups = {}
    for model in sources:
        for side in ('query', 'gallery'):
            all_samples = [r for r in samples if r['model'] == model and r['role'] == side]
            subsets = {'all': all_samples}
            if side == 'query':
                for outcome in ('rescued', 'harmed', 'both_correct', 'both_wrong'):
                    subsets[outcome] = [r for r in all_samples if r['rank1_outcome'] == outcome]
            for label, subset in subsets.items():
                # All these cohorts are present in the already completed GT query table.
                assert subset
                groups[model + '/' + side + '/' + label] = dict(samples=len(subset), **{
                    key: dict(mean=statistics.mean(r[key] for r in subset),
                              median=statistics.median(r[key] for r in subset),
                              maximum=max(r[key] for r in subset)) for key in fields})
    assert groups['I_axis_shared/query/rescued']['samples'] == 46
    assert groups['I_axis_shared/query/harmed']['samples'] == 13
    assert groups['I_frequency_shared/query/rescued']['samples'] == 7
    assert groups['I_frequency_shared/query/harmed']['samples'] == 7
    comparisons = []
    for variant in ('frequency_shared', 'axis_shared'):
        for side in ('query', 'gallery'):
            label = side + '/all'
            i, g = groups['I_' + variant + '/' + label], previous['groups']['G_' + variant + '/' + label]
            comparisons.append(dict(variant=variant, role=side, samples=836,
                I_selected_epoch=sources['I_' + variant]['selected_epoch'],
                G_selected_epoch=previous['selected_feature_archives']['G_' + variant]['selected_epoch'],
                I_mean_angle_degrees=i['direction_angle_degrees']['mean'],
                G_mean_angle_degrees=g['direction_angle_degrees']['mean'],
                I_mean_relative_correction_norm=i['relative_raw_correction_norm']['mean'],
                G_mean_relative_correction_norm=g['relative_raw_correction_norm']['mean'],
                I_mean_final_shift_epsilon=i['final_descriptor_shift_epsilon']['mean']))
    fingerprint_sources = ('analyze_r201i_selected_identity_shift.py', 'analyze_r201c_saved_geometry.py',
                           'analyze_identity_coordinate_three_normal.py', 'rgbnt201_identity_outlet.py')
    result = dict(status='ACTUAL_SELECTED_I201_COMPLETE_FEATURE_IDENTITY_SHIFT_CPU',
        completed_at=datetime.now().isoformat(timespec='seconds'), dataset='RGBNT201', expert_seed=42, anchor_seed=42,
        base_archive=str(base_path), base_archive_sha256=previous['base_archive_sha256'],
        selected_feature_archives=sources, samples=3344, groups=groups, G_selected_comparisons=comparisons,
        sources_sha256={n: hashlib.sha256((PROJECT/n).read_bytes()).hexdigest() for n in fingerprint_sources},
        reference_results_sha256={p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (receipt_path, previous_path)},
        reconstruction='Unchanged global1536 coordinates recover R/||B|| = s*z-b, s=sqrt(||b_global||^2/||z_global||^2). Final epsilon is measured directly as ||z-b||. Unit norms and global reconstruction checked on all3344 feature occurrences.',
        limits='Selected normal mAP-best only, 836query+836gallery per model, not independent seeds. G/I each select their own checkpoint; outcome cohorts are observational, not causal. No late50 feature banks, branch-specific M/F/I residuals, new neural inference, new margin target or tuned correction cap. This does not show why training deteriorates, guarantee a useful epsilon bound, or establish MSVR/RGBNT100/missing/wholepipeline gains.',
        new_neural_calls=0, new_optimizer_updates=0, new_weights=0)
    out.mkdir()
    table(out/'per_sample.csv', samples)
    table(out/'selected_G_I_comparison.csv', comparisons)
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], samples=3344, comparisons=comparisons), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
