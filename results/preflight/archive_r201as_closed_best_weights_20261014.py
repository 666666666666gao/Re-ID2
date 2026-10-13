"""Run once after the AS terminal intake; keep both selected bests on D."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shlex
import shutil
import subprocess
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command

TAG = 'r201as_RGB_query_normalization_20261014'
PF = PROJECT / 'results/preflight'
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
ROOT = REMOTE + '/runs/' + TAG
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ARCHIVE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201as_closed_best_weights_20261014')
load = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    intake = load(PF / (TAG + '_complete_intake.json'))
    assert intake['status'] == 'ACTUAL_AS_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    started = load(PF / (TAG + '_actual_session.json'))
    owner = started['remote_owner']['pid']
    protected = started['protected_weights']
    assert intake['terminal']['all_children_and_controller_absent']
    assert not ARCHIVE.exists()
    code = f'''import hashlib,json
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});owner={owner}
assert not Path('/proc/'+str(owner)).exists()
d=json.loads((r/'RGBNT100/controller_result.json').read_text())
assert d=={intake['terminal']['controller']!r}
assert d['status']=='COMPLETE' and d['states_cases']==8 and d['successful_updates']==13126
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
files={{}}
for row in d['runs']:
 f=r/'RGBNT100/training'/row['name']/'best.pth'
 assert f.resolve().is_relative_to(r.resolve())
 selected=json.loads(f.with_name('result.json').read_text())['best']['epoch']
 files[row['name']]=dict(remote=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),selected_epoch=selected)
print(json.dumps(files))'''
    files = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code))
    assert len(files) == 2
    assert shutil.disk_usage(ARCHIVE.parent).free > sum(row['bytes'] for row in files.values()) + 500_000_000
    ARCHIVE.mkdir()
    for name, row in files.items():
        folder = ARCHIVE / name
        folder.mkdir()
        destination = folder / 'best.pth'
        subprocess.run(['scp', *OPTIONS, '2026:' + row['remote'], str(destination)],
            check=True, capture_output=True, text=True, encoding='utf-8', timeout=900)
        assert destination.stat().st_size == row['bytes'] and sha(destination) == row['sha256']
        row['local'] = str(destination)
    assert all(Path(row['local']).stat().st_size == row['bytes'] and sha(Path(row['local'])) == row['sha256'] for row in files.values())
    code = f'''import hashlib,json,shutil
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});files={files!r};protected={protected!r}
assert not Path('/proc/{owner}').exists()
assert json.loads((r/'RGBNT100/controller_result.json').read_text())=={intake['terminal']['controller']!r}
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in protected.items())
for row in files.values():
 f=Path(row['remote']);assert f.resolve().is_relative_to(r.resolve()) and f.name=='best.pth'
 assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
for row in files.values():Path(row['remote']).unlink()
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in protected.items())
print(json.dumps(dict(remote_copies_removed=len(files),bytes=sum(row['bytes'] for row in files.values()),protected_four_exact_before_after=True,remote_free_bytes=shutil.disk_usage(p).free)))'''
    removed = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code))
    proof = dict(status='ACTUAL_AS_SELECTED_BESTS_D_ONLY_VERIFIED',
        at=datetime.now().astimezone().isoformat(timespec='seconds'), files=files, removed=removed,
        D_free_bytes=shutil.disk_usage(ARCHIVE).free, unique_weights_lost=0, new_neural_calls=0,
        reason='Full50, selected normal, eight states and raw archives are closed. Preserve the one selected best per control exactly on D; remove only the identical remote copies. No result-dependent seed selection.')
    (PF / 'r201as_closed_best_weights_D_only_actual_20261014.json').write_text(
        json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=proof['status'], removed=removed, D_free_bytes=proof['D_free_bytes'])), flush=True)


if __name__ == '__main__':
    main()
