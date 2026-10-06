"""Read closed selected feature archives; reconstruct corrections using fixed globals."""
import csv
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
ARCHIVE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')


def main():
    out = PROJECT / 'results/r201gh_selected_geometry_20261006'
    assert not out.exists()
    base_path = ARCHIVE / 'rgbnt201_identity_outlet_native_20261005/axis_shared/bypass_0/readout_arrays.npz'
    normal_closed = load(PROJECT / 'results/preflight/r201g_normal_priority_actual_session_20261006.json')
    assert normal_closed['exit_code'] == 0 and normal_closed['successful_updates'] == 5294
    first_wave = load(PROJECT / 'results/preflight/r201h_protocol_list_first_wave_closed_actual_20261006.json')
    assert first_wave['first_wave_full_epochs'] == 100 and first_wave['first_wave_optimizer_updates'] == 5294
    verified_raw = {Path(value['local']): value['file'] for value in normal_closed['archives'].values()}
    verified_raw.update({ARCHIVE / row['path']: row for row in first_wave['label_raw_local_verified']})
    original_rows, original_summary = query_rows(PROJECT / 'results/full_official_baselines_20261004/training/RGBNT201_demo_s42', 'RGBNT201')
    models = {}
    for family in ('G', 'H_label'):
        for variant in ('frequency_shared', 'axis_shared'):
            name = 'RGBNT201_r201g_' + variant + '_s42' if family == 'G' else 'RGBNT201_r201h_label_' + variant + '_s42'
            tag = 'r201g_normal_priority_20261006' if family == 'G' else 'r201h_protocol_list_20261006'
            path = ARCHIVE / tag / 'training' / name / 'best_official_arrays.npz'
            proof = verified_raw[path]
            assert path.stat().st_size == proof['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == proof['sha256']
            rows = query_rows(PROJECT / 'results' / tag / 'training' / name, 'RGBNT201')[0] if family == 'G' else None
            selected = load(PROJECT / 'results' / tag / 'training' / name / 'result.json')['best']['epoch'] if family == 'G' else next(r['best']['epoch'] for r in first_wave['observation']['runs'] if r['name'] == name)
            models[family + '_' + variant] = dict(path=path, rows=rows, selected_epoch=selected)
    samples = []
    with zipfile.ZipFile(base_path) as base_archive:
        for model, entry in models.items():
            with zipfile.ZipFile(entry['path']) as saved:
                for side in ('query', 'gallery'):
                    for key in ('ids', 'cameras', 'scenes', 'names'):
                        assert base_archive.read(side + '_' + key + '.npy') == saved.read(side + '_' + key + '.npy')
                    base = floats(base_archive, 'private_' + side + '_features')
                    fused = floats(saved, side + '_features')
                    for index in range(836):
                        b = base[index*5120:(index+1)*5120]
                        z = fused[index*5120:(index+1)*5120]
                        assert abs(math.fsum(x*x for x in b) - 1) < 1e-5
                        assert abs(math.fsum(x*x for x in z) - 1) < 1e-5
                        bg, zg = (math.fsum(x*x for x in value[:1536]) for value in (b, z))
                        assert bg > 0 and zg > 0
                        scale = math.sqrt(bg / zg)
                        residual = [scale*y-x for x, y in zip(b, z)]
                        global_error = max(abs(value) for value in residual[:1536])
                        assert global_error < 2e-6
                        energy = math.fsum(value*value for value in residual)
                        radial = math.fsum(x*y for x, y in zip(b, residual))
                        cosine = math.fsum(x*y for x, y in zip(b, z))
                        assert energy > 0
                        outcome = ''
                        if side == 'query' and entry['rows'] is not None:
                            original, current = original_rows[index], entry['rows'][index]
                            assert all(original[key] == current[key] for key in ('name', 'identity', 'camera', 'scene', 'query_index'))
                            before, after = int(original['Rank-1']), int(current['Rank-1'])
                            outcome = 'rescued' if after > before else 'harmed' if after < before else 'both_correct' if after else 'both_wrong'
                        samples.append(dict(model=model, selected_epoch=entry['selected_epoch'], role=side, index=index,
                            rank1_outcome=outcome, relative_raw_correction_norm=math.sqrt(energy),
                            radial_energy_fraction=radial*radial/energy, tangent_energy_fraction=1-radial*radial/energy,
                            direction_angle_degrees=math.degrees(math.acos(min(1, max(-1, cosine)))),
                            fused_to_base_raw_norm_ratio=scale, unchanged_global_max_error=global_error))
    assert len(samples) == 6688
    fields = ('relative_raw_correction_norm', 'radial_energy_fraction', 'tangent_energy_fraction',
              'direction_angle_degrees', 'fused_to_base_raw_norm_ratio', 'unchanged_global_max_error')
    groups = {}
    for model in models:
        for side in ('query', 'gallery'):
            subsets = {'all': [r for r in samples if r['model'] == model and r['role'] == side]}
            if side == 'query' and model.startswith('G_'):
                for outcome in ('rescued', 'harmed', 'both_correct', 'both_wrong'):
                    subsets[outcome] = [r for r in subsets['all'] if r['rank1_outcome'] == outcome]
            for label, subset in subsets.items():
                assert subset
                groups[model + '/' + side + '/' + label] = dict(samples=len(subset), **{
                    key: dict(mean=statistics.mean(r[key] for r in subset), median=statistics.median(r[key] for r in subset),
                              maximum=max(r[key] for r in subset)) for key in fields})
    source_names = ('analyze_r201gh_saved_geometry.py', 'analyze_r201c_saved_geometry.py',
                    'rgbnt201_identity_outlet.py', 'identity_coordinate_three_dataset.py', 'original_identity_anchor.py')
    result = dict(status='ACTUAL_G_AND_CLOSED_H_LABEL_SELECTED_CORRECTION_GEOMETRY', completed_at=datetime.now().isoformat(timespec='seconds'),
        base_archive=str(base_path), base_archive_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        selected_feature_archives={key: dict(path=str(entry['path']), selected_epoch=entry['selected_epoch'], verified_file=verified_raw[entry['path']]) for key, entry in models.items()},
        sources_sha256={name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in source_names},
        samples=6688, groups=groups, new_neural_calls=0, new_optimizer_updates=0, new_weights=0,
        reconstruction='For b=B/||B|| and z=(B+R)/||B+R||, unchanged globals1536 give s=sqrt(||b_global||^2/||z_global||^2)=||B+R||/||B||. Thus s*z-b reconstructs R/||B||. Every sample checks unchanged global coordinates.',
        limits='Normal selected early mAP-best only. Four banks x query/gallery are repeated observations, not independent seeds. No late-E50 features or branch-specific raw residuals; cannot establish collapse cause, individual M/F/I geometry, projection bottleneck, causal utility, or missing-source improvements. H label geometry only; canonical H per-query outcomes remain with the original complete CPU queue. G outcome groups are observational associations.')
    out.mkdir()
    table(out / 'per_sample.csv', samples)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], samples=6688, groups={key: value for key, value in groups.items() if key.endswith('/all')}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
