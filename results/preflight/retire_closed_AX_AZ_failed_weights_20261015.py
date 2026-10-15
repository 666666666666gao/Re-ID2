"""Retire four unpromoted AX/AZ checkpoints after closed audits; retain raw evidence."""
from datetime import datetime
import json
from pathlib import Path
import shutil

from retire_closed_BA_failed_weights_20261015 import PROJECT, PF, load, sha


def main():
    proof_path = PF / 'r201ax_az_failed_weights_local_retired_actual_20261015.json'
    assert not proof_path.exists()
    sealed = load(PF / 'r201bc_RGB_reference_interaction_source_review_20261015.json')
    assert sealed['status'] == 'PASS' and not sealed['blocking_findings']
    sources = sealed['sources_sha256'] | sealed['directly_reused_sources_sha256']
    assert all(sha(PROJECT / name) == expected for name,expected in sources.items())
    av_path = PROJECT / 'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training/RGBNT100_r201av_axis_shared_s42/result.json'
    av = load(av_path)['full_metrics']
    deleted, retained, metrics = {}, {}, {}
    archive_root = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts').resolve()
    for short,tag,audit_key in [
        ('AX','r201ax_RGB_frequency_query_normalization_20261014','overall_verdict'),
        ('AZ','r201az_RGB_corrected_state_metric_20261014','verdict'),
    ]:
        intake = load(PF / (tag + '_complete_intake.json'))
        assert intake['status'] == 'ACTUAL_' + short + '_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
        audit = load(PF / ('r201' + short.lower() + '_closed_normal_integrity_audit_20261014.json'))
        assert audit[audit_key] == 'WARN' and not audit['blocking_findings']
        root = archive_root / tag
        for name,row in intake['archives'].items():
            path = Path(row['local']).resolve()
            assert path.is_relative_to(root) and path.is_file()
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
            if name.endswith('/best'):
                assert path == root / name / 'best.pth'
                deleted[short + '/' + name] = row
            else:
                retained[short + '/' + name] = row
        name = 'RGBNT100_r201' + short.lower() + '_axis_shared_s42'
        metrics[short] = load(PROJECT / 'results' / tag / 'RGBNT100/training' / name / 'result.json')['full_metrics']
        assert metrics[short]['mAP'] < av['mAP'] and metrics[short]['Rank-1'] < av['Rank-1']
    assert len(deleted) == 4 and len(retained) == 8
    before = shutil.disk_usage(archive_root).free
    for row in deleted.values():
        Path(row['local']).unlink()
    assert all(not Path(row['local']).exists() for row in deleted.values())
    record = dict(status='ACTUAL_CLOSED_UNPROMOTED_AX_AZ_FOUR_WEIGHTS_RETIRED_LOCAL_ONLY',
        at=datetime.now().astimezone().isoformat(timespec='seconds'),deleted=deleted,
        deleted_bytes=sum(row['bytes'] for row in deleted.values()),retained_raw_size_SHA_verified=retained,
        D_free_before=before,D_free_after=shutil.disk_usage(archive_root).free,
        failed_axis_six_metrics=metrics,preserved_AV_axis_six_metrics=av,
        current_BC_sealed_sources_unchanged=True,new_remote_queries=0,new_neural_calls=0,
        active_model_or_promoted_weight_deleted=False,
        reason='Both full50 families and fresh integrity audits are closed; neither axis candidate was promoted over AV. Current BC is fresh AV-derived and references original DeMo, not AX/AZ trained bests. Only the exact four owned .pth files are retired; preserve all eight raw/state archives, full text evidence, and K201/AV/original teachers/MS strong reference. Historical intake remains a point-in-time receipt rather than current checkpoint-existence proof.')
    proof_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],deleted_bytes=record['deleted_bytes'],D_free_after=record['D_free_after'])),flush=True)


if __name__ == '__main__':
    main()
