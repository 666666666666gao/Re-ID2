"""Restore closed AS selected weights, sanity/full diagnostic once, then clear exact restored duplicates."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(P))
from collect_results import OPTIONS, command

REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG = 'r201au_AS_frequency_value_gradients_20261014'
PF = P / 'results/preflight'
ROOT = REMOTE + '/runs/' + TAG
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
remote = lambda code: json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code))


def main():
    review = load(PF / 'r201au_AS_frequency_value_gradients_source_review_20261014.json')
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources, reuse = review['sources_sha256'], review['directly_reused_sources_sha256']
    assert all(sha(P/n) == value for n, value in (sources | reuse).items())
    intake = load(PF / 'r201as_RGB_query_normalization_20261014_complete_intake.json')
    assert intake['formal_updates'] == 13126 and intake['state_cases'] == 8
    archive = load(PF / 'r201as_closed_best_weights_D_only_actual_20261014.json')
    assert archive['unique_weights_lost'] == 0 and len(archive['files']) == 2
    weights = archive['files']
    assert all(Path(r['local']).stat().st_size == r['bytes'] and sha(Path(r['local'])) == r['sha256'] for r in weights.values())
    protected = load(PF / 'r201as_RGB_query_normalization_20261014_actual_session.json')['protected_weights']
    admission = remote(f'''import hashlib,json,shutil,subprocess
from pathlib import Path
r=Path({REMOTE!r});out=Path({ROOT!r});weights={weights!r}
assert not out.exists() and all(not Path(v['remote']).exists() for v in weights.values())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==v for n,v in {protected!r}.items())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==v for n,v in {reuse!r}.items() if not n.startswith('results/'))
assert shutil.disk_usage(r).free>sum(v['bytes'] for v in weights.values())+300_000_000
gpu=[]
for i in (2,3):
 value=subprocess.check_output(['nvidia-smi','-i',str(i),'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert int(value[0])==i and float(value[1])<500
 gpu.append(dict(gpu=i,memory_used_mib=float(value[1])))
print(json.dumps(dict(physical_gpus=gpu,free_bytes=shutil.disk_usage(r).free)))''')
    for row in weights.values():
        subprocess.run(['scp', *OPTIONS, row['local'], '2026:' + row['remote']], check=True,
            capture_output=True, text=True, encoding='utf-8', timeout=900)
    for name in sources:
        command(['scp', *OPTIONS, str(P/name), '2026:' + REMOTE + '/' + name])
    remote(f'''import hashlib,json
from pathlib import Path
r=Path({REMOTE!r});sources={sources!r};weights={weights!r}
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==v for n,v in sources.items())
assert all(Path(v['remote']).stat().st_size==v['bytes'] and hashlib.sha256(Path(v['remote']).read_bytes()).hexdigest()==v['sha256'] for v in weights.values())
print(json.dumps(dict(restored_bests=2,sources_exact=True)))''')
    receipt = PF / (TAG + '_actual_session.json')
    assert not receipt.exists()
    state = dict(status='ACTUAL_AU_F_VALUE_TRAIN_ONLY_DIAGNOSTIC_STARTING',
        at=datetime.now().astimezone().isoformat(timespec='seconds'), admission=admission,
        restored_weights=weights, physical_gpus=[2, 3], maximum_NN=2, optimizer_updates=0,
        checkpoint_writes=0, official_query_gallery_neural_uses=0)
    receipt.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    code = f'''import subprocess
subprocess.run([{PYTHON!r},'-u','launch_r201au_AS_frequency_value_gradients.py','--training-root',
 {REMOTE!r}+'/runs/r201as_RGB_query_normalization_20261014/RGBNT100/training','--output',{ROOT!r}],cwd={REMOTE!r},check=True)'''
    finished = subprocess.run(['ssh', *OPTIONS, '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
        '2026', shlex.quote(PYTHON) + ' -'], input=code, check=True, capture_output=True,
        text=True, encoding='utf-8', timeout=None)
    print(finished.stdout, end='', flush=True)
    terminal = remote(f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({ROOT!r});c=json.loads((r/'controller_result.json').read_text());assert c['status']=='COMPLETE' and c['neural_forwards']==10
assert not Path('/proc/'+str(json.loads((r/'controller_launch.json').read_text())['pid'])).exists()
for phase in ('native','full'):
 for f in (r/phase).glob('*_launch.json'):
  assert not Path('/proc/'+str(json.loads(f.read_text())['pid'])).exists()
  assert json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text())['exit_code']==0
manifest={{str(f.relative_to(r)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file()}}
pkg=r/'text_intake.tar.gz';assert not pkg.exists()
with tarfile.open(pkg,'w:gz') as tar:
 for n in manifest:tar.add(r/n,arcname=n)
print(json.dumps(dict(controller=c,manifest=manifest,package=dict(bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()),original_controller_children_absent=True)))''')
    local = P / 'results' / TAG
    local.mkdir(exist_ok=False)
    package = local / 'text_intake.tar.gz'
    command(['scp', *OPTIONS, '2026:' + ROOT + '/text_intake.tar.gz', str(package)])
    assert package.stat().st_size == terminal['package']['bytes'] and sha(package) == terminal['package']['sha256']
    with tarfile.open(package) as tar:
        tar.extractall(local, filter='data')
    assert all((local/n).stat().st_size == v['bytes'] and sha(local/n) == v['sha256'] for n, v in terminal['manifest'].items())
    subprocess.run([sys.executable, '-X', 'utf8', '-B', '-S', str(P/'analyze_r201au_AS_frequency_value_gradients.py')], cwd=P, check=True)
    assert all(sha(Path(v['local'])) == v['sha256'] for v in weights.values())
    cleared = remote(f'''import hashlib,json,shutil
from pathlib import Path
r=Path({REMOTE!r});weights={weights!r};out=Path({ROOT!r})
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==v for n,v in {protected!r}.items())
assert all(Path(v['remote']).stat().st_size==v['bytes'] and hashlib.sha256(Path(v['remote']).read_bytes()).hexdigest()==v['sha256'] for v in weights.values())
for v in weights.values():
 p=Path(v['remote']);assert p.resolve().is_relative_to((r/'runs/r201as_RGB_query_normalization_20261014').resolve()) and p.name=='best.pth'
 p.unlink()
(out/'text_intake.tar.gz').unlink()
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==v for n,v in {protected!r}.items())
print(json.dumps(dict(restored_duplicate_bests_cleared=2,bytes=sum(v['bytes'] for v in weights.values()),protected_four_exact=True,remote_free_bytes=shutil.disk_usage(r).free)))''')
    package.unlink()
    state.update(status='ACTUAL_AU_AS_SELECTED_F_VALUE_TASK_GRADIENTS_COMPLETE',
        completed_at=datetime.now().astimezone().isoformat(timespec='seconds'), terminal=terminal,
        cleared=cleared, native_forwards=2, full_forwards=8, neural_forwards=10,
        unique_weights_lost=0, new_missing_evaluation=False, new49=False,
        limits='No training or new retrieval result; four sampled training batches per selected AS control. Same record/label order, not sustained or causal conflict.')
    receipt.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: state[k] for k in ('status','neural_forwards','optimizer_updates','checkpoint_writes','cleared')}), flush=True)


if __name__ == '__main__':
    main()
