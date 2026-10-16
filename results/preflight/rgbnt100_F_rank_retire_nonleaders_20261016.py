"""Retire only the two failed F128 bests after the fresh audit and next admission."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import shutil

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF = P/'results/preflight'
BASE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')
TAG = 'rgbnt100_F_rank128_AV_20261016'


def load(path):
    return json.loads(path.read_text('utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    output = PF/'rgbnt100_F_rank_nonleader_weights_retired_actual_20261016.json'
    assert not output.exists()
    audit_path = PF/'rgbnt100_F_rank_result_integrity_audit_20261016.json'
    audit = load(audit_path)
    assert audit['status'] == 'FINAL' and audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
    request_path = PF/'rgbnt100_F_rank_result_integrity_request_20261016.json'
    assert audit['request'] == str(request_path)
    request = load(request_path)
    assert request['input_count'] == 757
    for row in request['inputs']:
        path = Path(row['path'])
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    next_admission = load(PF/'vehicle_atmoe_head_actual_session_20261016.json')
    assert next_admission['F_previous_full_intake_and_fresh_audit_final'] and next_admission['F_inputs_revalidated'] == 757
    intake = load(PF/'rgbnt100_F_rank_complete_intake_20261016.json')
    assert intake['status'] == 'ACTUAL_F_RANK128_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    assert intake['terminal']['all_children_and_controller_absent']
    final = load(P/'results/rgbnt100_expert_N5_AV_20261015/five_normal_analysis/result.json')
    assert final['status'] == 'ALL_FIVE_COMPLETE_SELECTED'
    kept = load(PF/'rgbnt100_band_ranges_nonleader_weights_retired_actual_20261016.json')['preserved_archives'].copy()
    assert len(kept) == 23
    cpu = load(PF/'msvr_public_metric_weight_CPU_actual_20261016.json')
    kept.update({'MS_public_weights_'+name: row for name, row in cpu['archived'].items()})
    assert len(kept) == 26
    kept.update({'F_rank_'+name: row for name, row in intake['archives'].items() if not name.endswith('/best')})
    assert len(kept) == 30
    for row in kept.values():
        path = Path(row['local'])
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    targets = []
    for run in intake['terminal']['controller']['runs']:
        variant = run['variant']
        metric, leader = run['full_metrics'], final['main_best_single'][variant]
        assert metric['mAP'] < leader['mAP'] and metric['Rank-1'] <= leader['Rank-1']
        row = intake['archives'][run['name']+'/best']
        path = Path(row['local'])
        expected = BASE/TAG/run['name']/'best/best.pth'
        assert path.resolve() == expected.resolve() and path.resolve().is_relative_to(BASE.resolve())
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        targets.append(dict(path=str(path.resolve()), bytes=row['bytes'], sha256=row['sha256'],
            variant=variant, selected_epoch=run['selected_epoch'], six=metric))
    assert len(targets) == 2
    source = load(PF/'rgbnt100_F_rank_postprocess_source_review_request_20261016.json')
    seals = source['sources_sha256'] | source['directly_reused_sources_sha256']
    assert len(seals) == 376 and all(sha(P/n) == h for n, h in seals.items())
    before = shutil.disk_usage(BASE).free
    for row in targets:
        path = Path(row['path'])
        assert sha(path) == row['sha256']
        path.unlink()
        assert not path.exists()
    for row in kept.values():
        path = Path(row['local'])
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    assert all(sha(P/n) == h for n, h in seals.items())
    result = dict(status='ACTUAL_AUDITED_F_RANK128_TWO_NONLEADER_WEIGHTS_RETIRED_ONLY',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        retired=targets, bytes_retired=sum(row['bytes'] for row in targets), preserved_archives=kept,
        D_free_before=before, D_free_after=shutil.disk_usage(BASE).free,
        audit_sha256=sha(audit_path), sources_unchanged=len(seals), input_count_revalidated=757,
        new_NN=0, remote_queries=0, directories_deleted=0,
        reason='Both F128 bests have lower mAP and no higher Rank-1 than their corresponding preserved BestSingle. The next head baseline builds from public CLIP and never consumes these failed weights. Keep all30 leader/raw archives and1042 offloaded closed N5 text files/junction.',
        limits='All757 inputs were revalidated after fresh FINAL and after next deployment already validated them. Historical request retains the two before-delete paths. K201, MSstrong, original teachers, data, build dependencies and other projects are outside deletion targets. No removal of unique raw evidence.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps({key: result[key] for key in ('status','bytes_retired','D_free_before','D_free_after')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
