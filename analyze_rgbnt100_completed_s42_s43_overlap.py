"""Describe error overlap in the completed seed42/43 checkpoints, using saved GT scores."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from analyze_identity_coordinate_three_normal import METRICS, query_rows, table

PROJECT = Path(__file__).resolve().parent
PREFIX = PROJECT/'results/preflight/rgbnt100_N5_AV_s42_s43_completed_prefix_selection_20261015.json'
OUT = PROJECT/'results/rgbnt100_expert_N5_AV_20261015/completed_s42_s43_overlap'


def main():
    prefix = json.loads(PREFIX.read_text('utf-8'))
    assert prefix['completed_expert_seeds'] == [42, 43]
    assert prefix['remaining_expert_seeds'] == [44, 45, 46]
    folders = {'original': PROJECT/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42'}
    for record in prefix['metrics']:
        name = record['variant']+'_s'+str(record['seed'])
        assert name not in folders
        folders[name] = PROJECT/record['run_dir']
    assert len(folders) == 5
    inputs = [PREFIX, Path(__file__), PROJECT/'analyze_identity_coordinate_three_normal.py']
    reads, six = {}, []
    alignment = ('query_index', 'name', 'identity', 'camera', 'scene', 'valid', 'relevant_gallery', 'kept_gallery')
    for name, folder in folders.items():
        rows, summary = query_rows(folder, 'RGBNT100')
        if reads:
            assert all(tuple(a[key] for key in alignment) == tuple(b[key] for key in alignment)
                       for a, b in zip(rows, reads['original']))
        reads[name] = rows
        six.append(dict(model=name, **{key: summary[key] for key in METRICS}))
        inputs.extend([folder/'best_per_query.csv', folder/'best_per_query.json'])
    assert len(reads['original']) == 1715
    correct = {name: {i for i, row in enumerate(rows) if int(row['Rank-1'])} for name, rows in reads.items()}
    base = correct['original']
    assert len(base) == 1657
    wrong = set(range(1715))-base
    target_rank1 = 100*len(base)/1715+1
    required = math.ceil(target_rank1*1715/100-1e-10)-len(base)
    assert required == 18
    cohorts, changes = [], []
    for name in list(reads)[1:]:
        rescued = correct[name]-base
        harmed = base-correct[name]
        cohorts.append(dict(model=name, baseline_errors=len(wrong), rescued=len(rescued), harmed=len(harmed),
                            net_correct=len(rescued)-len(harmed), rescue_identities=len({reads['original'][i]['identity'] for i in rescued}),
                            harmed_identities=len({reads['original'][i]['identity'] for i in harmed}),
                            remaining_baseline_errors=len(wrong-correct[name])))
        for cohort, members in [('both_correct', base & correct[name]), ('both_wrong', wrong-correct[name]),
                                ('rescued', rescued), ('harmed', harmed)]:
            ap = [100*(float(reads[name][i]['AP'])-float(reads['original'][i]['AP'])) for i in sorted(members)]
            inp = [100*(float(reads[name][i]['INP'])-float(reads['original'][i]['INP'])) for i in sorted(members)]
            assert ap and inp
            changes.append(dict(model=name, cohort=cohort, queries=len(members),
                                AP_improved=sum(v > 1e-10 for v in ap), AP_worsened=sum(v < -1e-10 for v in ap),
                                mean_delta_AP_pp=math.fsum(ap)/len(ap), mean_delta_INP_pp=math.fsum(inp)/len(inp)))
    overlap = []
    for variant in ('axis_shared', 'frequency_shared'):
        a = correct[variant+'_s42']-base
        b = correct[variant+'_s43']-base
        union = a | b
        overlap.append(dict(variant=variant, seed42_rescues=len(a), seed43_rescues=len(b),
                            shared_rescues=len(a & b), unique_seed42_rescues=len(a-b), unique_seed43_rescues=len(b-a),
                            union_original_error_rescues=len(union), union_rescue_identities=len({reads['original'][i]['identity'] for i in union}),
                            label_oracle_with_baseline_correct=len(base | correct[variant+'_s42'] | correct[variant+'_s43']),
                            label_oracle_with_baseline_rank1=100*(len(base)+len(union))/1715,
                            additional_rescues_still_required_for_plus1=max(0, required-len(union))))
    rows = []
    for i, original in enumerate(reads['original']):
        row = {key: original[key] for key in alignment}
        for name, records in reads.items():
            row.update({name+'_'+key: records[i][key] for key in ('Rank-1', 'first_match', 'last_match', 'AP', 'INP')})
        rows.append(row)
    source_seals = {str(path.relative_to(PROJECT)).replace('\\', '/'): dict(bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in inputs}
    result = dict(status='ACTUAL_COMPLETED_S42_S43_SAVED_QUERY_OVERLAP_CPU_ONLY',
                  at=datetime.now().astimezone().isoformat(timespec='seconds'), query_count=1715, gallery_count=8575,
                  fixed_teacher_seed=42, completed_expert_seeds=[42, 43], actual_models=4,
                  original_correct=len(base), original_errors=len(wrong), plus1_rank1_target=target_rank1,
                  required_net_new_correct_queries=required, six_metrics=six, changes_from_original=cohorts,
                  AP_INP_changes_by_correctness_cohort=changes, same_variant_seed_overlap=overlap,
                  source_inputs=source_seals, new_neural_calls=0, new_optimizer_updates=0, remote_queries=0,
                  limits='Saved benchmark-GT query scores only; not a new full-distance recount, fresh independent audit, inference, new seed result, deployment selector, or five-seed completion. Label-oracle union is an offline upper bound ONLY for choosing among the original and the specified two saved same-variant checkpoints per query; it does not bound untrained seeds, other representations, ensembles, or retraining. It requires query GT and cannot be reported as model performance. Whole-checkpoint best-single rules and the active AV recipe remain unchanged. Missing and four-state inference are not evaluated here.')
    OUT.mkdir(exist_ok=False)
    table(OUT/'six_metrics.csv', six)
    table(OUT/'query_outcomes.csv', rows)
    table(OUT/'changes_from_original.csv', cohorts)
    table(OUT/'AP_INP_correctness_cohorts.csv', changes)
    table(OUT/'same_variant_seed_overlap.csv', overlap)
    (OUT/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    assert all((PROJECT/name).stat().st_size == info['bytes'] and
               hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() == info['sha256']
               for name, info in source_seals.items())
    print(json.dumps(dict(status=result['status'], same_variant_seed_overlap=overlap,
                         AP_INP_changes_by_correctness_cohort=changes, new_neural_calls=0), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
