"""Retire exactly two closed, unpromoted BA checkpoints; retain their raw evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF = PROJECT / 'results/preflight'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    proof_path = PF / 'r201ba_failed_weights_local_retired_actual_20261015.json'
    assert not proof_path.exists()
    intake = load(PF / 'r201ba_RGB_teacher_head_20261015_complete_intake.json')
    assert intake['status'] == 'ACTUAL_BA_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    audit = load(PF / 'r201ba_closed_normal_integrity_audit_20261015.json')
    assert audit['verdict'] == 'WARN' and not audit['blocking_findings']
    bb = load(PF / 'r201bb_parallel_jobs_20261015_complete_intake.json')
    assert bb['status'] == 'ACTUAL_BB_SERIAL_PARALLEL_THREE_REPEATS_COLLECTED'
    assert intake['cleared']['protected_bests_exact'] and bb['cleanup']['protected_four_exact']
    review = load(PF / 'r201bc_RGB_reference_interaction_source_review_20261015.json')
    assert review['status'] == 'PASS' and not review['blocking_findings']
    seals = review['sources_sha256'] | review['directly_reused_sources_sha256']
    assert all(sha(PROJECT / name) == expected for name, expected in seals.items())
    root = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201ba_RGB_teacher_head_20261015').resolve()
    selected = {name: row for name,row in intake['archives'].items() if name.endswith('/best')}
    assert set(selected) == {'RGBNT100_r201ba_frequency_shared_s42/best', 'RGBNT100_r201ba_axis_shared_s42/best'}
    retained = {name: row for name,row in intake['archives'].items() if not name.endswith('/best')}
    assert len(retained) == 4
    for name,row in selected.items():
        path = Path(row['local']).resolve()
        assert path == root / name / 'best.pth' and path.is_relative_to(root)
        assert path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    for row in retained.values():
        path = Path(row['local'])
        assert path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    before = shutil.disk_usage(root).free
    for row in selected.values():
        Path(row['local']).unlink()
    assert all(not Path(row['local']).exists() for row in selected.values())
    record = dict(status='ACTUAL_CLOSED_UNPROMOTED_BA_TWO_WEIGHTS_RETIRED_LOCAL_ONLY',
        at=datetime.now().astimezone().isoformat(timespec='seconds'),deleted=selected,
        deleted_bytes=sum(row['bytes'] for row in selected.values()),retained_raw_size_SHA_verified=retained,
        D_free_before=before,D_free_after=shutil.disk_usage(root).free,
        current_BC_sealed_sources_unchanged=True,new_remote_queries=0,new_neural_calls=0,
        active_model_or_promoted_weight_deleted=False,
        reason='BA full50 and fresh result audit are closed and BA was not promoted over AV. All BB consumers are closed; BB constructed fresh BA models from the original teacher and TRAIN cache, never these trained bests. Active BC is a fresh AV-derived model anchored to original DeMo, not BA bests. Keep BA raw distances/states and complete metrics/logs; only these two owned checkpoint files are retired. K201, AV, original teachers and MS strong reference are outside the exact deletion scope. Historical intake remains a point-in-time receipt, not a current-existence guarantee.')
    proof_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],deleted_bytes=record['deleted_bytes'],D_free_after=record['D_free_after'])),flush=True)


if __name__ == '__main__':
    main()
