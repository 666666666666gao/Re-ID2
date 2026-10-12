"""Account for fixed AG positive/negative score margins using cached 5120D blocks."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import zipfile

from analyze_r201o_saved_normal_geometry import values
from analyze_rgbnt100_common_distance_probe import names

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / 'results/rgbnt100_AG_block_margins_CPU_20261013'
MODEL = 'RGBNT100_r201ag_axis_shared_s42'
BLOCKS = ('global_RGB', 'global_NIR', 'global_TIR', 'relation_R', 'relation_N',
          'relation_T', 'relation_RN', 'relation_RT', 'relation_NT', 'relation_RNT')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def role(before, after):
    if before['Rank-1'] == '0':
        return 'rescued' if after['Rank-1'] == '1' else 'remaining_original_error'
    return 'newly_harmed' if after['Rank-1'] == '0' else 'correct_preserved'


def main():
    assert not OUTPUT.exists()
    manifest = load(PROJECT / 'results/full_official_baselines_20261004/training/RGBNT100_demo_s42/official_split_manifest.json')
    query, gallery = manifest['query'], manifest['gallery']
    assert len(query) == 1715 and len(gallery) == 8575
    folder = PROJECT / 'results/r201ai_AG_selected_outlets_20261012/RGBNT100/full' / MODEL
    base, full = rows(folder / 'base00.csv'), rows(folder / 'normal11.csv')
    assert len(base) == len(full) == 1715
    raw = load(PROJECT / 'results/preflight/r201ag_RGB_coordinate_metric_20261012_complete_intake.json')['archives'][MODEL]
    file = Path(raw['local'])
    assert file.stat().st_size == raw['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest() == raw['sha256']
    with zipfile.ZipFile(file) as archive:
        for side, records in (('query', query), ('gallery', gallery)):
            assert names(archive, side + '_names', len(records)) == [r['name'] for r in records]
            for suffix, key in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene')):
                assert list(values(archive, side + '_' + suffix, (len(records),), '<i8', 'q')) == [r[key] for r in records]
        qfeatures = values(archive, 'query_features', (1715, 5120), '<f4', 'f')
        gfeatures = values(archive, 'gallery_features', (8575, 5120), '<f4', 'f')
        distance = values(archive, 'distances', (1715, 8575), '<f4', 'f')
    references = []
    for i, item in enumerate(query):
        offset = i * 8575
        positives = [j for j, candidate in enumerate(gallery)
            if candidate['identity'] == item['identity'] and candidate['camera'] != item['camera']]
        negatives = [j for j, candidate in enumerate(gallery) if candidate['identity'] != item['identity']]
        assert positives and negatives
        p = min(positives, key=lambda j: distance[offset + j])
        n = min(negatives, key=lambda j: distance[offset + j])
        assert int(distance[offset + p] < distance[offset + n]) == int(full[i]['Rank-1'])
        references.append((p, n))
    records, maximum_error = [], 0.
    for i, (p, n) in enumerate(references):
        item = query[i]
        assert all(str(base[i][k]) == str(full[i][k]) == str(item[k]) for k in ('name','identity','camera','scene'))
        block_margins = {}
        for index, name in enumerate(BLOCKS):
            start = index * 512
            q = qfeatures[i * 5120 + start:i * 5120 + start + 512]
            positive = gfeatures[p * 5120 + start:p * 5120 + start + 512]
            negative = gfeatures[n * 5120 + start:n * 5120 + start + 512]
            block_margins[name] = math.fsum(a * (b - c) for a, b, c in zip(q, positive, negative))
        total = math.fsum(block_margins.values())
        cache_margin = .5 * (distance[i * 8575 + n] - distance[i * 8575 + p])
        error = abs(total - cache_margin)
        assert error < 1e-6
        maximum_error = max(maximum_error, error)
        global_margin = math.fsum(block_margins[name] for name in BLOCKS[:3])
        relation_margin = math.fsum(block_margins[name] for name in BLOCKS[3:])
        records.append(dict(query_index=i, **item, role=role(base[i], full[i]),
            base_AP=float(base[i]['AP']), full_AP=float(full[i]['AP']),
            base_Rank1=int(base[i]['Rank-1']), full_Rank1=int(full[i]['Rank-1']),
            positive_name=gallery[p]['name'], negative_name=gallery[n]['name'],
            positive_camera=gallery[p]['camera'], negative_camera=gallery[n]['camera'],
            positive_identity=gallery[p]['identity'], negative_identity=gallery[n]['identity'],
            total_margin=total, cached_margin=cache_margin, global_margin=global_margin,
            relation_margin=relation_margin, **block_margins))
    summaries = {}
    for label in sorted({r['role'] for r in records}):
        members = [r for r in records if r['role'] == label]
        summaries[label] = dict(queries=len(members), identities=len({r['identity'] for r in members}),
            global_prefers_positive_relation_prefers_negative=sum(r['global_margin'] > 0 and r['relation_margin'] < 0 for r in members),
            global_prefers_negative_relation_prefers_positive=sum(r['global_margin'] < 0 and r['relation_margin'] > 0 for r in members),
            both_prefer_positive=sum(r['global_margin'] > 0 and r['relation_margin'] > 0 for r in members),
            both_prefer_negative=sum(r['global_margin'] < 0 and r['relation_margin'] < 0 for r in members),
            mean_global_margin=math.fsum(r['global_margin'] for r in members) / len(members),
            mean_relation_margin=math.fsum(r['relation_margin'] for r in members) / len(members))
    OUTPUT.mkdir()
    with (OUTPUT / 'query_block_margins.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = dict(status='ACTUAL_FULL1715_CACHED_AG_SCORE_BLOCK_MARGIN_ACCOUNTING',
        at=datetime.now().astimezone().isoformat(timespec='seconds'), model=MODEL,
        queries=1715, gallery=8575, blocks=list(BLOCKS), raw=raw,
        maximum_total_vs_cached_margin_error=maximum_error, summaries=summaries,
        references='For each query, closest valid positive and closest different-identity negative under final AG distance; same fixed references for all blocks.',
        new_neural_calls=0, optimizer_updates=0, new_remote_queries=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations='Additive score accounting at fixed final AG references, not independently ranked block mAP, a causal ablation, evidence of M-versus-F gradient conflict, or an AK outcome. Final normalization scales global and relation blocks together; no score weight is selected or changed.')
    (OUTPUT / 'result.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(dict(status=result['status'], maximum_error=maximum_error, summaries=summaries)), flush=True)


if __name__ == '__main__':
    main()
