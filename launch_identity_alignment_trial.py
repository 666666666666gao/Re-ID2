"""M2b matched controls; contract and all three smokes precede fresh50."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from gpu_thermal_execute import execute

SCHEDULE = {2: ('axis_shared', 'twins_shared'), 3: ('frequency_shared',)}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(exist_ok=False)
    preflight = output/'preflight'; preflight.mkdir()
    tensor = execute([sys.executable,'-u','verify_identity_alignment_axis.py','--data-root',args.data_root,
        '--pretrained',args.pretrained,'--output',str(preflight/'tensor')],preflight,'tensor',2)
    assert json.loads((preflight/'tensor/result.json').read_text())['status'] == 'PASS_IDENTITY_ALIGNMENT_CONTRACT'

    def command(variant):
        return [sys.executable,'-u','run_identity_alignment_experiment.py','--dataset','MSVR310','--variant',variant,
            '--relation-weight','0.1','--alignment-weight','0.1','--alignment-temperature','0.07',
            '--seed','42','--data-root',args.data_root,'--pretrained',args.pretrained]

    def smokes(gpu):
        rows = []
        for variant in SCHEDULE[gpu]:
            row = execute(command(variant)+['--mode','smoke','--output',str(preflight/variant)],preflight,variant,gpu)
            result = json.loads((preflight/variant/'smoke.json').read_text())
            assert result['status'] == 'SMOKE_PASS' and result['steps'] == 3 and result['strict_reload_equal']
            assert all(d['frequency_relation_target_requires_grad'] is False and d['frequency_relation_weight'] == .1
                and d['identity_alignment_reference_requires_grad'] is False and d['identity_alignment_weight'] == .1
                and d['identity_alignment_temperature'] == .07 and d['identity_alignment_valid_anchors'] > 0 for d in result['details'])
            rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(smokes,g) for g in SCHEDULE]
        smokes = [row for future in futures for row in future.result()]
    assert len(smokes) == 3
    save(output/'preflight_result.json',dict(status='PASS',tensor=tensor,smokes=smokes))
    root = output/'original_mean'; root.mkdir()
    for name in ('development','frozen49','controlled_states','utility'):
        (root/name).mkdir()
    cross_root = output/'cross_coordinates'; cross_root.mkdir()

    def jobs(gpu):
        rows = []
        for variant in SCHEDULE[gpu]:
            name = 'MSVR310_'+variant+'_s42'; run = root/'development'/name
            train = execute(command(variant)+['--mode','train','--output',str(run)],root/'development',name,gpu)
            trained = json.loads((run/'result.json').read_text())
            assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
            assert trained['arguments']['pooling'] == 'original_mean' and trained['arguments']['relation_weight'] == .1
            assert trained['arguments']['alignment_weight'] == .1 and trained['arguments']['alignment_temperature'] == .07
            frozen = root/'frozen49'/name
            evaluation = execute([sys.executable,'-u','missing_common_outlet_development.py','--run-dir',str(run),
                '--output',str(frozen),'--data-root',args.data_root,'--pretrained',args.pretrained],root/'frozen49',name,gpu)
            result = json.loads((frozen/'result.json').read_text())
            assert result['status'] == 'COMPLETE' and len(result['measurements']) == 49 and result['normal_feature_max_error'] == 0
            state = root/'controlled_states'/name; state.mkdir()
            state_command = [sys.executable,'-u','diagnose_common_outlet_states.py','--run-dir',str(run),
                '--previous-frozen',str(frozen),'--data-root',args.data_root,'--pretrained',args.pretrained]
            smoke = execute(state_command+['--output',str(state/'smoke'),'--smoke'],state,'smoke',gpu)
            assert json.loads((state/'smoke/smoke.json').read_text())['status'] == 'PASS'
            full = execute(state_command+['--output',str(state/'full')],state,'full',gpu)
            result = json.loads((state/'full/result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['metric_count'] == 294 and result['previous_all49_full11_sixmetric_and_perquery_exact']
            utility = root/'utility'/name; utility.mkdir()
            utility_command = [sys.executable,'-u','diagnose_identity_alignment_utility.py','--run-dir',str(run),
                '--previous-frozen',str(frozen),'--data-root',args.data_root,'--pretrained',args.pretrained]
            usmoke = execute(utility_command+['--output',str(utility/'smoke'),'--smoke'],utility,'smoke',gpu)
            assert json.loads((utility/'smoke/smoke.json').read_text())['status'] == 'PASS'
            ufull = execute(utility_command+['--output',str(utility/'full')],utility,'full',gpu)
            result = json.loads((utility/'full/result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['metric_count'] == 392 and result['previous_all49_deployed_exact']
            cross = cross_root/variant; cross.mkdir()
            cross_command = [sys.executable,'-u','diagnose_identity_alignment_coordinates.py','--run-dir',str(run),
                '--previous-utility',str(utility/'full'),'--data-root',args.data_root,'--pretrained',args.pretrained]
            csmoke = execute(cross_command+['--output',str(cross/'smoke'),'--smoke'],cross,'smoke',gpu)
            assert json.loads((cross/'smoke/smoke.json').read_text())['status'] == 'PASS'
            cfull = execute(cross_command+['--output',str(cross/'full')],cross,'full',gpu)
            assert json.loads((cross/'full/result.json').read_text())['metric_count'] == 343
            rows.append(dict(pooling='original_mean',variant=variant,gpu=gpu,train=train,frozen49=evaluation,
                state_smoke=smoke,state_full=full,utility_smoke=usmoke,utility_full=ufull,cross_smoke=csmoke,cross_full=cfull))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jobs,g) for g in SCHEDULE]
        rows = [row for future in futures for row in future.result()]
    assert len(rows) == 3
    save(cross_root/'controller_result.json',dict(status='COMPLETE',runs=[dict(variant=r['variant'],**r['cross_full']) for r in rows]))
    for folder in (output,root,root/'controlled_states'):
        save(folder/'controller_result.json',dict(status='COMPLETE',runs=rows,temperature_power_control=False))


if __name__ == '__main__':
    main()
