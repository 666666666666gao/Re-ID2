"""Inspect completed AV frozen00/11 nearest matches; no new inference or fitting."""
import array
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct
import sys
import zipfile

from analyze_identity_coordinate_three_normal import load, query_rows, table

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / 'results/rgbnt100_saved_error_cases_CPU_20261015'


def read_array(archive, name):
    with archive.open(name + '.npy') as stream:
        assert stream.read(8) == b'\x93NUMPY\x01\x00'
        size = struct.unpack('<H', stream.read(2))[0]
        header = ast.literal_eval(stream.read(size).decode('latin1'))
        assert header['fortran_order'] is False and sys.byteorder == 'little'
        data = stream.read()
    kind = header['descr']
    if kind == '<f4':
        values = array.array('f')
        values.frombytes(data)
    elif kind == '<i8':
        values = array.array('q')
        values.frombytes(data)
    else:
        assert kind == '<U18'
        values = [data[offset:offset + 72].decode('utf-32-le').rstrip('\0') for offset in range(0, len(data), 72)]
    count = 1
    for dimension in header['shape']:
        count *= dimension
    assert len(values) == count
    return values, header['shape']


def margins(distances, index, positives, negatives, count):
    offset = index * count
    key = lambda j: (distances[offset + j], j)
    positive = min(positives, key=key)
    negative = min(negatives, key=key)
    first = min((positive, negative), key=key)
    first_rank = 1 + sum(key(j) < key(positive) for j in negatives)
    return dict(top1_index=first, positive_index=positive, negative_index=negative,
        positive_distance=float(distances[offset + positive]),
        negative_distance=float(distances[offset + negative]),
        margin=float(distances[offset + negative] - distances[offset + positive]),
        first_positive_rank=first_rank)


def distribution(values):
    ordered = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
        minimum=ordered[0], maximum=ordered[-1],
        p10=ordered[int((len(ordered) - 1) * .1)], p90=ordered[int((len(ordered) - 1) * .9)])


def main():
    assert not OUTPUT.exists()
    intake = load(PROJECT / 'results/preflight/r201av_RGB_frequency_value_decay_20261014_complete_intake.json')
    assert intake['status'] == 'ACTUAL_AV_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    archive = intake['archives']['RGBNT100_r201av_axis_shared_s42/states_raw']
    path = Path(archive['local'])
    assert path.stat().st_size == archive['bytes']
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == archive['sha256']
    original_dir = PROJECT / 'results/full_official_baselines_20261004/training/RGBNT100_demo_s42'
    av_dir = PROJECT / 'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training/RGBNT100_r201av_axis_shared_s42'
    original_rows, original_metrics = query_rows(original_dir, 'RGBNT100')
    av_rows, av_metrics = query_rows(av_dir, 'RGBNT100')
    with zipfile.ZipFile(path) as stored:
        qid, _ = read_array(stored, 'query_ids')
        qcam, _ = read_array(stored, 'query_cameras')
        qnames, _ = read_array(stored, 'query_names')
        gid, _ = read_array(stored, 'gallery_ids')
        gcam, _ = read_array(stored, 'gallery_cameras')
        gnames, _ = read_array(stored, 'gallery_names')
        base, base_shape = read_array(stored, 'distances_00')
        full, full_shape = read_array(stored, 'distances')
    assert base_shape == full_shape == (1715, 8575)
    assert len(qid) == len(qcam) == len(qnames) == 1715
    assert len(gid) == len(gcam) == len(gnames) == 8575 and min(gid) > 0
    assert len(set(gnames)) == len(gnames)
    for names, identities, cameras in ((qnames, qid, qcam), (gnames, gid, gcam)):
        for name, identity, camera in zip(names, identities, cameras):
            match = re.fullmatch(r'(\d+)_c(\d+)_(\d+)\.jpg', name)
            assert match and int(match[1]) == identity and int(match[2]) - 1 == camera
    rows = []
    details = []
    for index, (identity, camera, name) in enumerate(zip(qid, qcam, qnames)):
        assert all(row['name'] == name and int(row['identity']) == identity and int(row['camera']) == camera
                   for row in (original_rows[index], av_rows[index]))
        eligible = [j for j in range(len(gid)) if not (gid[j] == identity and gcam[j] == camera)]
        positives = [j for j in eligible if gid[j] == identity]
        negatives = [j for j in eligible if gid[j] != identity]
        before = margins(base, index, positives, negatives, len(gid))
        after = margins(full, index, positives, negatives, len(gid))
        for measured, saved in ((before, original_rows[index]), (after, av_rows[index])):
            assert len(eligible) == int(saved['kept_gallery']) and len(positives) == int(saved['relevant_gallery'])
            assert int(gid[measured['top1_index']] == identity) == int(saved['Rank-1']), (index, name, measured, saved)
            assert measured['first_positive_rank'] == int(saved['first_match']), (index, name, measured, saved)
        b, a = int(original_rows[index]['Rank-1']), int(av_rows[index]['Rank-1'])
        if b == 0 and a == 0:
            cohort = 'both_wrong'
        elif b == 0 and a == 1:
            cohort = 'rescued_by_AV'
        elif b == 1 and a == 0:
            cohort = 'harmed_by_AV'
        else:
            cohort = 'both_correct'
        row = dict(query_index=index, name=name, identity=identity, camera=camera,
            cohort=cohort, original_Rank1=b, AV_Rank1=a,
            original_first_match=before['first_positive_rank'], AV_first_match=after['first_positive_rank'],
            original_margin=before['margin'], AV_margin=after['margin'],
            margin_change=after['margin'] - before['margin'],
            original_AP=float(original_rows[index]['AP']), AV_AP=float(av_rows[index]['AP']))
        for label, measured in (('original', before), ('AV', after)):
            for role in ('top1', 'positive', 'negative'):
                j = measured[role + '_index']
                row[label + '_' + role + '_index'] = j
                row[label + '_' + role + '_name'] = gnames[j]
                row[label + '_' + role + '_identity'] = int(gid[j])
                row[label + '_' + role + '_camera'] = int(gcam[j])
            row[label + '_positive_distance'] = measured['positive_distance']
            row[label + '_negative_distance'] = measured['negative_distance']
        rows.append(row)
        details.append((before, after))
    cohorts = {}
    selected = []
    for cohort in ('both_wrong', 'rescued_by_AV', 'harmed_by_AV', 'both_correct'):
        members = [row for row in rows if row['cohort'] == cohort]
        cohorts[cohort] = dict(queries=len(members), identities=len({r['identity'] for r in members}),
            cameras=sorted({r['camera'] for r in members}),
            original_margin=distribution([r['original_margin'] for r in members]),
            AV_margin=distribution([r['AV_margin'] for r in members]),
            margin_change=distribution([r['margin_change'] for r in members]),
            margin_improved=sum(r['margin_change'] > 0 for r in members),
            AP_improved=sum(r['AV_AP'] > r['original_AP'] for r in members))
        if cohort != 'both_correct':
            for identity in sorted({r['identity'] for r in members}):
                row = next(r for r in members if r['identity'] == identity)
                before, after = details[row['query_index']]
                wrong = before['negative_index'] if cohort == 'rescued_by_AV' else after['negative_index']
                correct = before['positive_index'] if cohort == 'harmed_by_AV' else after['positive_index']
                selected.append(dict(case=len(selected) + 1, cohort=cohort, query_index=row['query_index'],
                    identity=identity, camera=row['camera'], query_name=row['name'],
                    wrong_name=gnames[wrong], wrong_identity=int(gid[wrong]), wrong_camera=int(gcam[wrong]),
                    correct_name=gnames[correct], correct_identity=int(gid[correct]), correct_camera=int(gcam[correct]),
                    original_margin=row['original_margin'], AV_margin=row['AV_margin'],
                    original_first_match=row['original_first_match'], AV_first_match=row['AV_first_match']))
    assert sum(row['original_Rank1'] == 0 for row in rows) == 58
    assert cohorts['rescued_by_AV']['queries'] == 9 and cohorts['harmed_by_AV']['queries'] == 6
    OUTPUT.mkdir()
    table(OUTPUT / 'all_query_nearest_margins.csv', rows)
    (OUTPUT / 'selected_visual_cases.json').write_text(json.dumps(dict(
        selection='One lowest-query-index example per identity within each non-correct cohort; no visual cherry-picking. Diagnostic only, not deployment selection.',
        cases=selected), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = dict(status='ACTUAL_SAVED_AV00_11_COMPLETE_NEAREST_MATCH_CPU_ANALYSIS_NO_NN',
        at=datetime.now().astimezone().isoformat(timespec='seconds'),
        dataset='RGBNT100', query_count=1715, gallery_count=8575, states=2,
        whole_cache_size_SHA_verified=archive, metadata_filename_GT_exact=True,
        first_positive_rank_and_Rank1_match_all_original_and_AV_CSV=True,
        original_six={k:original_metrics[k] for k in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')},
        AV_six={k:av_metrics[k] for k in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')},
        margin_definition='nearest_negative_squared_distance - nearest_positive_squared_distance; positive is favorable. Same-ID same-camera removed as official scorer. Ties use gallery index; all resulting first-positive ranks and Rank1 verified against saved CSV.',
        cohorts=cohorts, selected_visual_cases=len(selected),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_neural_calls=0, new_optimizer_updates=0, new_fitting=0, training_progress_queries=0,
        limits='Cached AV frozen00 represents the unchanged original teacher; use its verified retained state archive, not a retired original49 file. No new inference or AP/INP ranking recomputation. Saved six metrics are qualified historical benchmark-selected results. Label-selected nearest pairs and visual sample are offline diagnosis, not new method scores, causal evidence, a deployable oracle or proof that local frequency cues solve errors. Images not fetched or visually reviewed by this script.')
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','query_count','gallery_count','cohorts','selected_visual_cases','new_neural_calls')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
