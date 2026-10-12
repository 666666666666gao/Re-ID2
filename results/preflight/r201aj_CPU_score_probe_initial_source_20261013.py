"""Full cached RGBNT100 score mixtures; CPU only, no model or branch oracle."""
import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
import zipfile

from analyze_r201o_saved_normal_geometry import values

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / 'results/rgbnt100_common_distance_probe_CPU_20261013'
PLAN = PROJECT / 'results/preflight/r201aj_RGB_common_score_probe_plan_20261013.json'
WEIGHTS = (0., .05, .10, .25)
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def names(archive, key, count):
    data = archive.read(key + '.npy')
    assert data[:8] == b'\x93NUMPY\x01\x00'
    length = struct.unpack('<H', data[8:10])[0]
    assert ast.literal_eval(data[10:10 + length].decode('latin1')) == dict(
        descr='<U18', fortran_order=False, shape=(count,))
    text = data[10 + length:].decode('utf-32-le')
    assert len(text) == count * 18
    return [text[i * 18:(i + 1) * 18].rstrip(chr(0)) for i in range(count)]


def summarize(rows):
    result = dict(mAP=100 * sum(r['AP'] for r in rows) / len(rows),
                  mINP=100 * sum(r['INP'] for r in rows) / len(rows))
    result.update({key: 100 * sum(r[key] for r in rows) / len(rows)
                   for key in METRICS[2:]})
    return result


def recount(base, common, weight, queries, gallery):
    rows = []
    for i, query in enumerate(queries):
        offset = i * len(gallery)
        # Stable index ties match the existing installed-GT auditor's lexsort.
        order = sorted(range(len(gallery)), key=lambda j:
                       (1 - weight) * base[offset + j] + weight * common[offset + j])
        rank = 0
        positive_ranks = []
        for j in order:
            item = gallery[j]
            same = item['identity'] == query['identity']
            if same and item['camera'] == query['camera']:
                continue
            rank += 1
            if same:
                positive_ranks.append(rank)
        assert positive_ranks
        rows.append(dict(query_index=i, **query, valid=True,
            relevant_gallery=len(positive_ranks), kept_gallery=rank,
            AP=sum(k / r for k, r in enumerate(positive_ranks, 1)) / len(positive_ranks),
            INP=len(positive_ranks) / positive_ranks[-1],
            first_match=positive_ranks[0], last_match=positive_ranks[-1],
            **{f'Rank-{k}': int(positive_ranks[0] <= k) for k in (1, 5, 10, 20)}))
    return rows


def verify_endpoint(rows, folder, stage, reported):
    with (folder / (stage + '.csv')).open(encoding='utf-8', newline='') as stream:
        expected = list(csv.DictReader(stream))
    assert len(rows) == len(expected) == 1715
    assert all(abs(summarize(rows)[k] - reported[k]) < 1e-8 for k in METRICS)
    for actual, prior in zip(rows, expected):
        assert set(actual) == set(prior)
        for key, value in actual.items():
            if key in ('AP', 'INP'):
                assert abs(value - float(prior[key])) < 1e-12
            else:
                assert str(value) == prior[key]


def changes(before, after):
    return dict(rank1_rescued=sum(a['Rank-1'] == 0 and b['Rank-1'] == 1
                                for a, b in zip(before, after)),
                rank1_harmed=sum(a['Rank-1'] == 1 and b['Rank-1'] == 0
                                for a, b in zip(before, after)),
                AP_improved=sum(b['AP'] > a['AP'] for a, b in zip(before, after)),
                AP_worsened=sum(b['AP'] < a['AP'] for a, b in zip(before, after)))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert not OUTPUT.exists() and not PLAN.exists()
    intake = load(PROJECT / 'results/preflight/r201ai_AG_selected_outlets_20261012_complete_intake.json')
    assert intake['status'] == 'ACTUAL_AI_TWO_FIXED_AG_FULL_NORMAL_30_OUTLETS_D_ARCHIVED_GT_PARITY_COMPLETE'
    plan = dict(status='PREDECLARED_BEFORE_CPU_SCORE_EVALUATION',
        at=datetime.now().astimezone().isoformat(timespec='seconds'),
        controls=['frequency_shared', 'axis_shared'], weights=list(WEIGHTS),
        endpoint_common_weight=1., private_paths=['base00', 'normal11'],
        cases=18, query_records=1715, gallery_records=8575,
        score='(1-lambda)*distance_private + lambda*distance_common_context',
        selection='Axis normal11 highest benchmark mAP across 0/.05/.10/.25, earliest lower weight tie. Same selected weight for ordinary and base00 control.',
        new_neural_calls=0, optimizer_updates=0, new_weights=0, remote_queries=0,
        scope='Full symmetric complete RGBNT100 cached-score diagnostic. No missing/49/201 NN. Actual float32 descriptor replay required before promotion.',
        comparison='Existing fixed25 public-block training failed on RGBNT100; current AG public context is a different trained outlet. Do not call this an untested universal public-branch idea.',
        limitations='Official benchmark already selected AG epoch and now probes weight; not untouched test, independent seed result, branch oracle, or causal expert gain.')
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    OUTPUT.mkdir()
    summaries, groups, cmc, inputs, cases = [], [], [], {}, {}
    for variant in plan['controls']:
        name = 'RGBNT100_r201ag_' + variant + '_s42'
        folder = PROJECT / 'results/r201ai_AG_selected_outlets_20261012/RGBNT100/full' / name
        reported = load(folder / 'result.json')
        manifest_path = PROJECT / 'results/r201ag_RGB_coordinate_metric_20261012/RGBNT100/training' / name / 'official_split_manifest.json'
        manifest = load(manifest_path)
        queries, gallery = manifest['query'], manifest['gallery']
        assert len(manifest['train']) == 8675 and len(queries) == 1715 and len(gallery) == 8575
        saved = intake['archives'][name]
        path = Path(saved['local'])
        assert path.stat().st_size == saved['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == saved['sha256']
        inputs[name] = saved
        with zipfile.ZipFile(path) as archive:
            for side, records in (('query', queries), ('gallery', gallery)):
                assert names(archive, side + '_names', len(records)) == [r['name'] for r in records]
                for suffix, key in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene')):
                    assert list(values(archive, side + '_' + suffix, (len(records),), '<i8', 'q')) == [r[key] for r in records]
            distances = {stage: values(archive, 'distance_' + stage, (1715, 8575), '<f4', 'f')
                         for stage in ('base00', 'normal11', 'common_context')}
        with (folder / 'base00.csv').open(encoding='utf-8', newline='') as stream:
            base_rows = list(csv.DictReader(stream))
        for private, weight in [(s, w) for s in plan['private_paths'] for w in WEIGHTS] + [('common_context', 1.)]:
            rows = recount(distances[private], distances['common_context'], weight, queries, gallery)
            if weight in (0., 1.):
                verify_endpoint(rows, folder, private, reported['measurements'][private])
            label = variant + '_' + private + '_lambda' + str(weight)
            metrics = summarize(rows)
            comparison = changes([{**r, 'Rank-1': int(r['Rank-1']), 'AP': float(r['AP'])} for r in base_rows], rows)
            summaries.append(dict(variant=variant, selected_epoch=reported['selected_epoch'], private_path=private,
                                  common_metric_weight=weight, **metrics, **comparison))
            cases[(variant, private, weight)] = rows
            write_csv(OUTPUT / (label + '.csv'), rows)
            cmc.extend(dict(case=label, rank=k, accuracy=100 * sum(r['first_match'] <= k for r in rows) / len(rows))
                       for k in range(1, 51))
            for key in ('identity', 'camera', 'scene'):
                for value in sorted({r[key] for r in rows}):
                    members = [r for r in rows if r[key] == value]
                    groups.append(dict(case=label, group=key, value=value, queries=len(members), **summarize(members)))
            print(json.dumps(summaries[-1]), flush=True)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == saved['sha256']
    assert len(summaries) == 18
    chosen = max((r for r in summaries if r['variant'] == 'axis_shared' and r['private_path'] == 'normal11'), key=lambda r: r['mAP'])
    weight = chosen['common_metric_weight']
    controls = [r for r in summaries if r['common_metric_weight'] == weight and r['private_path'] in plan['private_paths']]
    comparisons = []
    for variant in plan['controls']:
        a = next(r for r in controls if r['variant'] == variant and r['private_path'] == 'base00')
        b = next(r for r in controls if r['variant'] == variant and r['private_path'] == 'normal11')
        comparisons.append(dict(variant=variant, common_metric_weight=weight,
            **{k: b[k] - a[k] for k in METRICS},
            **changes(cases[(variant, 'base00', weight)], cases[(variant, 'normal11', weight)])))
    write_csv(OUTPUT / 'six_metrics.csv', summaries)
    write_csv(OUTPUT / 'groups.csv', groups)
    write_csv(OUTPUT / 'cmc.csv', cmc)
    write_csv(OUTPUT / 'selected_weight_expert_vs_base.csv', comparisons)
    baseline = reported['measurements']['base00']
    result = dict(status='ACTUAL_FULL_RGBNT100_CACHED_COMMON_SCORE_PROBE_CPU_COMPLETE',
        at=datetime.now().astimezone().isoformat(timespec='seconds'), plan=plan, inputs=inputs,
        cases=18, query_rows=18 * 1715, complete_metadata_and_endpoint_parity=True,
        measurements=summaries, chosen=chosen, selected_weight_matched_controls=controls,
        selected_weight_expert_vs_base=comparisons,
        selected_axis_vs_original={k: chosen[k] - baseline[k] for k in METRICS},
        original_plus1_met=all(chosen[k] - baseline[k] >= 1. for k in ('mAP', 'Rank-1')),
        new_neural_calls=0, optimizer_updates=0, new_weights=0, remote_queries=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=plan['limitations'] + ' Float64 score arithmetic on cached float32 distances, not a recomputed descriptor or trained combined head.')
    (OUTPUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], chosen=chosen, original_plus1_met=result['original_plus1_met'])), flush=True)


if __name__ == '__main__':
    main()
