"""Retire ten closed unpromoted local checkpoints; keep all twenty raw archives."""
from datetime import datetime
from pathlib import Path
import json
import shutil

from retire_closed_BA_failed_weights_20261015 import PROJECT, PF, load, sha


def main():
    output = PF / 'r201al_am_ao_aq_as_failed_weights_local_retired_actual_20261015.json'
    assert not output.exists()
    source_review = load(PF / 'rgbnt100_expert_N5_source_review_20261015.json')
    assert source_review['status'] == 'PASS' and not source_review['blocking_findings']
    seals = source_review['sources_sha256'] | source_review['directly_reused_sources_sha256']
    assert all(sha(PROJECT / name) == digest for name, digest in seals.items())
    av = load(PROJECT / 'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training/RGBNT100_r201av_axis_shared_s42/result.json')['full_metrics']
    ar = load(PROJECT / 'results/r201ar_AQ_spatial_attention_20261014/controller_result.json')
    at = load(PROJECT / 'results/r201at_AS_spatial_attention_20261014/controller_result.json')
    au = load(PROJECT / 'results/r201au_AS_frequency_value_gradients_20261014/controller_result.json')
    assert ar['status'] == at['status'] == au['status'] == 'COMPLETE'
    assert all(row['exit']['exit_code'] == 0 for row in ar['stages'].values())
    assert all(row['exit']['exit_code'] == 0 for rows in at['stages'].values() for row in rows)
    assert all(row['exited']['exit_code'] == 0 for rows in [au['native'], au['full']] for row in rows)
    assert ar['optimizer_updates'] == at['optimizer_updates'] == au['optimizer_updates'] == 0
    audit = load(PF / 'r201as_closed_normal_integrity_audit_20261014.json')
    assert audit['review_status'] == 'COMPLETED' and audit['computation_integrity'] == 'PASS'
    assert not audit['substantive_findings']['blocking_integrity_defects']
    root = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts').resolve()
    families = [
        ('AL', 'r201al_RGB_route_temperature_20261013', 'r201al_closed_best_weights_D_only_actual_20261013.json', 'r201al_closed_best_weights_20261013'),
        ('AM', 'r201am_RGB_correction_budget_20261013', 'r201am_closed_best_weights_D_only_actual_20261013.json', 'r201am_closed_best_weights_20261013'),
        ('AO', 'r201ao_RGB_hdm_query_guidance_20261013', 'r201ao_closed_best_weights_D_only_actual_20261013.json', 'r201ao_closed_best_weights_20261013'),
        ('AQ', 'r201aq_RGB_attention_decay_20261013', 'r201aq_closed_best_weights_D_archived_actual_20261014.json', 'r201aq_closed_best_weights_20261014'),
        ('AS', 'r201as_RGB_query_normalization_20261014', 'r201as_closed_best_weights_D_only_actual_20261014.json', 'r201as_closed_best_weights_20261014'),
    ]
    deleted, retained, metrics, closure = {}, {}, {}, {}
    for short, tag, receipt_name, weight_dir in families:
        intake = load(PF / (tag + '_complete_intake.json'))
        assert intake['status'] == 'ACTUAL_' + short + '_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
        archived = load(PF / receipt_name)
        assert len(archived['files']) == 2 and len(intake['archives']) == 4
        closure[short] = dict(intake_status=intake['status'], weight_receipt=receipt_name)
        for name, row in archived['files'].items():
            path = Path(row['local']).resolve()
            assert path == root / weight_dir / name / 'best.pth' and path.is_relative_to(root)
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
            result = load(PROJECT / 'results' / tag / 'RGBNT100/training' / name / 'result.json')
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50
            value = result['full_metrics']
            assert value['mAP'] < av['mAP'] and value['Rank-1'] < av['Rank-1']
            deleted[short + '/' + name] = row
            metrics[short + '/' + name] = value
        for name, row in intake['archives'].items():
            path = Path(row['local']).resolve()
            assert path.is_relative_to(root / tag) and path.suffix == '.npz'
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
            retained[short + '/' + name] = row
    assert len(deleted) == 10 and len(retained) == 20
    assert len({str(Path(row['local']).resolve()) for row in deleted.values()}) == 10
    before = shutil.disk_usage(root).free
    for row in deleted.values():
        Path(row['local']).unlink()
    assert all(not Path(row['local']).exists() for row in deleted.values())
    assert all(Path(row['local']).stat().st_size == row['bytes'] for row in retained.values())
    assert all(sha(PROJECT / name) == digest for name, digest in seals.items())
    record = dict(status='ACTUAL_CLOSED_FIVE_FAMILY_TEN_FAILED_WEIGHTS_RETIRED_LOCAL_ONLY',
        at=datetime.now().astimezone().isoformat(timespec='seconds'), deleted=deleted,
        deleted_bytes=sum(row['bytes'] for row in deleted.values()), retained_raw_size_SHA_verified=retained,
        D_free_before=before, D_free_after=shutil.disk_usage(root).free,
        failed_six_metrics=metrics, preserved_AV_axis_six_metrics=av, closed_family_evidence=closure,
        diagnostic_consumers=dict(AR=ar['status'], AT=at['status'], AU=au['status']),
        active_BC_and_new_N5_sources_unchanged=True, new_remote_queries=0, new_neural_calls=0,
        active_or_promoted_weight_deleted=False,
        reason='All five families have completed full50/eight-state intakes. All ten whole checkpoint rows are below retained AV on both mAP and Rank-1. AR/AT/AU diagnostic consumers are complete. Active BC and prepared N5 build fresh selected modules and load original DeMo teacher42, not these trained candidates. Only ten exact owned .pth files are removed; all twenty raw/state caches, logs, failures and metrics remain. K201/AV/original teachers/MS strong reference are outside these exact paths. Historical weight receipts remain point-in-time archival evidence, not current existence guarantees.')
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in ('status', 'deleted_bytes', 'D_free_before', 'D_free_after', 'new_neural_calls')}), flush=True)


if __name__ == '__main__':
    main()
