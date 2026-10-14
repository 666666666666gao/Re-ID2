"""Compare saved MSVR310 evidence coverage; no fitting, reranking or new inference."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from analyze_identity_coordinate_three_normal import METRICS, six, table

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / 'results/msvr_saved_evidence_coverage_CPU_20261015'


def main():
    manifest_path = PROJECT / 'results/full_official_baselines_20261004/training/MSVR310_demo_s42/official_split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = [(r['name'], str(r['identity']), str(r['camera']), str(r['scene'])) for r in manifest['query']]
    assert len(expected) == 591 and len(manifest['gallery']) == 1055
    paths = {}
    dimensions = {}
    blocks = PROJECT / 'results/msvr_reference_blocks_p0_20261009'
    for name, dimension in [('original_DeMo', 5120), ('shared_private_only', 5120),
                            ('shared_public_only', 512), ('shared_fused', 5632)]:
        paths[name] = blocks / (name + '.csv')
        dimensions[name] = dimension
    ah = PROJECT / 'results/r201ah_MS_full_identity_20261012/MSVR310/states'
    for variant in ('frequency_shared', 'axis_shared'):
        for state in ('00', '10', '01', '11'):
            label = 'AH_' + variant + '_' + state
            paths[label] = ah / ('MSVR310_r201ah_' + variant + '_s42') / ('state_' + state + '.csv')
            dimensions[label] = 5632
    outlet_root = PROJECT / 'results/selected_expert_outlets_20261011'
    jobs_path = outlet_root / 'jobs.json'
    jobs = json.loads(jobs_path.read_text(encoding='utf-8'))['jobs']
    for variant in ('frequency_shared', 'axis_shared'):
        name = 'MSVR310_r201r_' + variant + '_s42'
        job = next(j for j in jobs if j['name'] == name)
        result_path = outlet_root / name / job['completion_file']
        record = json.loads(result_path.read_text(encoding='utf-8'))
        assert record['status'] == job['completion_status']
        assert record['query_records'] == 591 and record['optimizer_updates'] == record['new_weights'] == 0
        for stage in ('normal11', 'common_context', 'M_projection_input', 'F_projection_input',
                      'M_projection_output', 'F_projection_output'):
            label = 'R_' + variant + '_' + stage
            paths[label] = outlet_root / name / (stage + '.csv')
            dimensions[label] = record['dimensions'][stage]
    reads = {}
    metrics = []
    for name, path in paths.items():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 591 and all(r['valid'] == 'True' for r in rows)
        assert [(r['name'], r['identity'], r['camera'], r['scene']) for r in rows] == expected
        reads[name] = rows
        metrics.append(dict(model=name,dimension=dimensions[name],**six(rows)))
    for variant in ('frequency_shared', 'axis_shared'):
        assert reads['AH_' + variant + '_00'] == reads['shared_fused']
    correct = {name: {i for i,r in enumerate(rows) if int(r['Rank-1']) == 1} for name,rows in reads.items()}
    universe = set(range(591))
    strong = correct['shared_fused']
    original = correct['original_DeMo']
    private = correct['shared_private_only']
    public = correct['shared_public_only']
    cohorts = {
        'strong_remaining_errors': universe - strong,
        'strong_harms_original': original - strong,
        'strong_rescues_original': strong - original,
        'public_block_rescues_private': strong - private,
        'strong_correct_both_blocks_wrong': strong - private - public,
        'strong_errors_R_axis_common_correct': (universe - strong) & correct['R_axis_shared_common_context'],
        'R_axis_common_correct_final_wrong': correct['R_axis_shared_common_context'] - correct['R_axis_shared_normal11'],
        'strong_errors_AH_axis11_correct': (universe - strong) & correct['AH_axis_shared_11'],
        'strong_correct_AH_axis11_wrong': strong - correct['AH_axis_shared_11'],
    }
    coverage = []
    for cohort, indices in cohorts.items():
        assert indices
        for name, rows in reads.items():
            coverage.append(dict(cohort=cohort,model=name,queries=len(indices),
                rank1_correct=len(indices & correct[name]),
                mean_AP_pp=100*sum(float(rows[i]['AP']) for i in indices)/len(indices),
                mean_INP_pp=100*sum(float(rows[i]['INP']) for i in indices)/len(indices)))
    pairwise = []
    measured = {r['model']: r for r in metrics}
    for name in reads:
        if name == 'shared_fused':
            continue
        rescued = correct[name] - strong
        harmed = strong - correct[name]
        delta = {k: measured[name][k] - measured['shared_fused'][k] for k in METRICS}
        assert abs(delta['Rank-1'] - 100*(len(rescued)-len(harmed))/591) < 1e-8
        pairwise.append(dict(model=name,reference='shared_fused',**delta,
            rank1_rescued=len(rescued),rank1_harmed=len(harmed)))
    flags = []
    for i, (name, identity, camera, scene) in enumerate(expected):
        row = dict(query_index=i,name=name,identity=identity,camera=camera,scene=scene)
        row.update({label+'_correct': int(i in values) for label,values in correct.items()})
        row.update({label: int(i in values) for label,values in cohorts.items()})
        flags.append(row)
    OUTPUT.mkdir(exist_ok=False)
    for name,rows in [('six_metrics',metrics),('coverage',coverage),('vs_strong',pairwise),('query_flags',flags)]:
        table(OUTPUT/(name+'.csv'),rows)
    inputs = list(paths.values()) + [manifest_path, jobs_path]
    inputs.extend(outlet_root/('MSVR310_r201r_'+v+'_s42')/next(j for j in jobs if j['name']=='MSVR310_r201r_'+v+'_s42')['completion_file']
                  for v in ('frequency_shared','axis_shared'))
    result = dict(status='ACTUAL_SAVED_MSVR310_EVIDENCE_COVERAGE_CPU_COMPLETE',
        query_records=591,gallery_records=1055,model_outlets=len(paths),
        input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cohort_counts={name:len(indices) for name,indices in cohorts.items()},
        cohort_identity_counts={name:dict(Counter(expected[i][1] for i in sorted(indices))) for name,indices in cohorts.items()},
        metrics=metrics,vs_strong=pairwise,new_neural_calls=0,new_optimizer_updates=0,new_weights=0,
        selected_new_model=False,missing_evaluated=False,
        limits='Recounts saved per-query GT CSV only; no new raw-distance ranking. Labels define offline diagnostic cohorts, not an inference-time selector. R and AH have different training/teachers/interfaces; auxiliary evidence overlap does not prove that a trained decoder can recover it or that a module caused any gain. All outputs refer to existing fixed normal-best checkpoints; no epoch, seed or metric-weight selection here. Dimensions are explicit; these outlets are not interchangeable final model results.')
    (OUTPUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],model_outlets=len(paths),cohort_counts=result['cohort_counts'])),flush=True)


if __name__ == '__main__':
    main()
