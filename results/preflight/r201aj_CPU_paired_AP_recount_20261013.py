"""Recount paired AP signs from completed CSVs, without repeating any ranking."""
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from analyze_rgbnt100_common_distance_probe import OUTPUT, changes, write_csv

result_path = OUTPUT / 'result.json'
result = json.loads(result_path.read_text(encoding='utf-8'))
initial = json.loads((PROJECT / 'results/preflight/r201aj_CPU_score_probe_AP_numeric_initial_20261013.json').read_text())
assert hashlib.sha256(result_path.read_bytes()).hexdigest() == initial['result_sha256']


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        data = list(csv.DictReader(stream))
    assert len(data) == 1715
    return [{**r, 'Rank-1': int(r['Rank-1']), 'AP': float(r['AP'])} for r in data]


tables = {}
for item in result['measurements']:
    variant, private, weight = (item[k] for k in ('variant', 'private_path', 'common_metric_weight'))
    before = rows(PROJECT / 'results/r201ai_AG_selected_outlets_20261012/RGBNT100/full' /
                  ('RGBNT100_r201ag_' + variant + '_s42') / 'base00.csv')
    after = rows(OUTPUT / (variant + '_' + private + '_lambda' + str(weight) + '.csv'))
    recalculated = changes(before, after)
    assert recalculated['rank1_rescued'] == item['rank1_rescued']
    assert recalculated['rank1_harmed'] == item['rank1_harmed']
    item.update(recalculated)
    tables[(variant, private, weight)] = after
    if private == 'base00' and weight == 0.:
        assert item['AP_improved'] == item['AP_worsened'] == 0
weight = result['chosen']['common_metric_weight']
result['chosen'] = next(r for r in result['measurements'] if
    r['variant'] == 'axis_shared' and r['private_path'] == 'normal11' and r['common_metric_weight'] == weight)
result['selected_weight_matched_controls'] = [r for r in result['measurements'] if
    r['common_metric_weight'] == weight and r['private_path'] in ('base00', 'normal11')]
for item in result['selected_weight_expert_vs_base']:
    variant = item['variant']
    item.update(changes(tables[(variant, 'base00', weight)], tables[(variant, 'normal11', weight)]))
result.update(paired_AP_tolerance=1e-12, initial_source_sha256=result['source_sha256'],
    source_sha256=hashlib.sha256((PROJECT / 'analyze_rgbnt100_common_distance_probe.py').read_bytes()).hexdigest(),
    paired_AP_recount_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    paired_AP_recount='Completed CSV only; six metrics, all ranks, selected weight and distances unchanged. Initial float-sign noise/source/result preserved in preflight snapshot.')
write_csv(OUTPUT / 'six_metrics.csv', result['measurements'])
write_csv(OUTPUT / 'selected_weight_expert_vs_base.csv', result['selected_weight_expert_vs_base'])
result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status='ACTUAL_CPU_PAIRED_AP_SIGNS_RECOUNTED_ZERO_RERANK_OR_NN',
    query_rows=30870, chosen=result['chosen'], original_plus1_met=result['original_plus1_met'])))
