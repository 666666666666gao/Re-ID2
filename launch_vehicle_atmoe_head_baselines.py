"""Finite vehicle identity configurations after the F128 experiment is audited."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute

CASES = (
    dict(dataset='RGBNT100', variant='demo', heads=32, gpu=2, audit_script='audit_full_official_normal.py',
         original_parameters=97952256, parameters=97780224,
         train=8675, query=1715, gallery=8575, dimension=5120, target=(87.2, 98.6)),
    dict(dataset='MSVR310', variant='demo_shared', heads=16, gpu=3, audit_script='audit_msvr_full_identity_normal.py',
         original_parameters=99982336, parameters=99867648,
         train=1032, query=591, gallery=1055, dimension=5632, target=(50.2, 60.8)),
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    for name in ('previous-intake', 'previous-audit', 'data-root', 'pretrained', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    previous = json.loads(Path(args.previous_intake).read_text())
    audited = json.loads(Path(args.previous_audit).read_text())
    assert previous['status'] == 'ACTUAL_F_RANK128_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    assert previous['formal_updates'] == 13126 and previous['state_cases'] == 8
    assert previous['cleared']['matching_owned_duplicates_cleared'] == 7
    assert audited['status'] == 'FINAL' and audited['verdict'] in ('PASS', 'WARN')
    assert not audited['blocking_findings']
    assert audited['request'].endswith('rgbnt100_F_rank_result_integrity_request_20261016.json')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('parity', 'native', 'training', 'audit'):
        (root/phase).mkdir()
    save(root/'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        cases=CASES, physical_gpus=[2, 3], maximum_NN=2, new_module=False,
        scope='Two fresh identity baselines with existing ATMoE counts; not Ours or a seed search. '
              'Historical head8 references are not freshly retrained controls. No RGBNT201 or missing evaluation.'))

    def parity(case):
        name = case['dataset']+'_atmoe8_default_parity'
        out = root/'parity'/name
        ended = execute([sys.executable, '-u', 'check_vehicle_atmoe_head_default_parity.py',
            '--dataset', case['dataset'], '--variant', case['variant'],
            '--data-root', args.data_root, '--pretrained', args.pretrained, '--output', str(out)],
            root/'parity', name, case['gpu'])
        data = json.loads((out/'result.json').read_text())
        assert data['status'] == 'ACTUAL_VEHICLE_ATMOE_HEAD8_DEFAULT_CONFIG_STATE_RNG_FP32_AMP_PARITY_PASS'
        assert all(data[key] for key in ('config_dump_exact', 'state_tensors_exact', 'CPU_CUDA_RNG_exact',
                                        'fp32_output_exact', 'autocast_output_exact'))
        assert data['parameters'] == case['original_parameters'] and data['descriptor_dim'] == case['dimension']
        assert data['forward_record_executions'] == 32 and data['optimizer_updates'] == data['checkpoint_writes'] == 0
        return dict(result=data, exit=ended)

    with ThreadPoolExecutor(max_workers=2) as pool:
        parities = list(pool.map(parity, CASES))

    def run(case, mode):
        name = case['dataset']+'_atmoe'+str(case['heads'])+'_'+case['variant']+'_s42'+('_smoke' if mode == 'smoke' else '')
        phase = 'native' if mode == 'smoke' else 'training'
        out = root/phase/name
        ended = execute([sys.executable, '-u', 'run_vehicle_atmoe_head_baseline.py',
            '--atmoe-heads', str(case['heads']), '--dataset', case['dataset'],
            '--variant', case['variant'], '--seed', '42', '--mode', mode,
            '--data-root', args.data_root, '--pretrained', args.pretrained, '--output', str(out)],
            root/phase, name, case['gpu'])
        return name, out, ended

    def verify_counts(case, data):
        assert data['arguments']['atmoe_heads'] == case['heads']
        assert data['parameters'] == case['parameters'] and data['descriptor_dim'] == case['dimension']
        assert (data['train_records'], data['query_records'], data['gallery_records']) == (case['train'], case['query'], case['gallery'])
        assert data['training_heldout_identities'] == 0

    def native(case):
        _, out, ended = run(case, 'smoke')
        data = json.loads((out/'smoke.json').read_text())
        assert data['status'] == 'SMOKE_PASS' and data['steps'] == data['attempts'] == 3
        assert data['amp_skipped_steps'] == 0 and data['strict_reload_equal'] and all(data['gradients'].values())
        assert data['descriptor_shape'] == [8, case['dimension']] and not list(out.glob('*.pth'))
        verify_counts(case, data)
        return dict(result=data, exit=ended)

    with ThreadPoolExecutor(max_workers=2) as pool:
        natives = list(pool.map(native, CASES))
    save(root/'native_acceptance.json', dict(status='PASS', actual_updates=6,
        parity_forward_record_executions=64, parities=parities, natives=natives,
        strict_reload=True, all_active_gradients=True, physical_gpus=[2, 3], maximum_NN=2))
    print('VEHICLE_ATMOE_HEAD_NATIVE_PASS', flush=True)

    def train(case):
        name, out, ended = run(case, 'train')
        data = json.loads((out/'result.json').read_text())
        assert data['status'] == 'COMPLETE' and data['epochs'] == 50
        assert data['steps'] == data['optimizer_steps'] > 0 and data['amp_skipped_steps'] == 0
        verify_counts(case, data)
        assert data['training_coverage'] == dict(eligible=case['train'], visited=case['train'], unvisited=[])
        audit = execute([sys.executable, '-u', case['audit_script'], '--run-dir', str(out)],
                        root/'audit', name, case['gpu'])
        assert json.loads((out/'normal_cpu_audit.json').read_text())['status'] == 'PASS'
        files = {}
        for key, filename in (('raw', 'best_official_arrays.npz'), ('best', 'best.pth')):
            file = out/filename
            files[key] = dict(path=str(file), bytes=file.stat().st_size,
                              sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        result = dict(name=name, dataset=case['dataset'], variant=case['variant'], heads=case['heads'],
            gpu=case['gpu'], train=ended, audit=audit, selected_epoch=data['best']['epoch'],
            successful_updates=data['steps'], full_metrics=data['full_metrics'], **files,
            paper_target=dict(mAP=case['target'][0], Rank_1=case['target'][1]),
            paper_both_plus1=data['full_metrics']['mAP'] >= case['target'][0] and data['full_metrics']['Rank-1'] >= case['target'][1],
            interpretation='Identity baseline only; not dual-axis contribution or a missing-modality result.')
        save(root/(case['dataset']+'_component_complete.json'), result)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, CASES))
    save(root/'controller_result.json', dict(status='COMPLETE', cases=2, runs=trained,
        native_updates=6, parity_forward_record_executions=64,
        formal_epochs=100, formal_updates=sum(item['successful_updates'] for item in trained),
        full_training_coverage=True, normal_GT_audited=True, physical_gpus=[2, 3], maximum_NN=2,
        missing_evaluated=False, raw_preserved_pending_local_archive=True,
        limits='Fresh public-CLIP identity references only; historical8 references, no expert contribution claim. '
               'All six metrics share one earliest mAP-best checkpoint; official benchmark participates in selection.'))
    print('VEHICLE_ATMOE_HEAD_BASELINES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
