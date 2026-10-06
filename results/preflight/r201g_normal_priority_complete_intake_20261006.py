"""Collect closed R201G text once; no neural job, weight, or raw-array transfer."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command

PF = PROJECT / 'results/preflight'
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
ROOT = REMOTE + '/runs/r201g_normal_priority_20261006'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'


def main():
    actual = json.loads((PF / 'r201g_normal_priority_actual_session_20261006.json').read_text(encoding='utf-8'))
    assert actual['status'] == 'ACTUAL_R201G_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW'
    assert actual['exit_code'] == 0 and actual['successful_updates'] == 5294 and actual['additional_epochs'] == 100
    assert len(actual['archives']) == 2
    review = json.loads((PF / 'r201g_normal_priority_source_review_20261006.json').read_text(encoding='utf-8'))
    sources = review['sources_sha256'] | review['directly_reused_sources_sha256']
    assert review['status'] == 'PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in sources.items())
    package_name = '/tmp/r201g_normal_priority_terminal_text_20261006.tar.gz'
    code = f'''import hashlib,json,tarfile
from pathlib import Path
project=Path({REMOTE!r}); root=Path({ROOT!r}); package=Path({package_name!r})
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items() if not name.startswith('results/'))
controller=json.loads((root/'controller_result.json').read_text())
assert controller['status']=='COMPLETE' and controller['controls']==2
assert controller['additional_epochs']==100 and controller['successful_updates']==5294 and controller['native_updates']==6
assert controller['paired_sampling_exact'] and controller['normal_archives_local_verified']==2
for job in controller['runs']:
 name=job['name']; run=root/'training'/name
 data=json.loads((run/'result.json').read_text())
 assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==2647
 assert data['amp_skipped_steps']==0 and data['descriptor_dim']==5120 and data['training_heldout_identities']==0
 assert data['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
 assert [p.name for p in run.glob('*.pth')]==['best.pth']
 assert json.loads((root/'training'/(name+'_exit.json')).read_text())['exit_code']==0
 audit=json.loads((run/'normal_cpu_audit.json').read_text())
 assert audit['status']=='PASS' and audit['installed_split_counts']==dict(train=3951,query=836,gallery=836)
 assert audit['CMC50_and_per_query_and_groups'] and audit['max_metric_error']<1e-8
 assert json.loads((run/'normal_local_archive.json').read_text())['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
 assert not (run/'best_official_arrays.npz').exists()
files={{}}; assert not package.exists()
with tarfile.open(package,'w:gz') as archive:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   relative=path.relative_to(root).as_posix(); data=path.read_bytes()
   files[relative]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
   archive.add(path,arcname=relative,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code))
    package = PF / 'r201g_normal_priority_terminal_text_20261006.tar.gz'
    local = PROJECT / 'results/r201g_normal_priority_20261006'
    assert not package.exists() and not local.exists()
    command(['scp', *OPTIONS, '2026:' + package_name, str(package)])
    assert package.stat().st_size == inventory['package']['bytes']
    assert hashlib.sha256(package.read_bytes()).hexdigest() == inventory['package']['sha256']
    local.mkdir()
    with tarfile.open(package, 'r:gz') as archive:
        assert set(archive.getnames()) == set(inventory['files'])
        archive.extractall(local, filter='data')
    assert all((local / name).stat().st_size == row['bytes'] and hashlib.sha256((local / name).read_bytes()).hexdigest() == row['sha256']
               for name, row in inventory['files'].items())
    command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input='from pathlib import Path\nPath(' + repr(package_name) + ').unlink()\n')
    package.unlink()
    result = dict(status='ACTUAL_R201G_TWO50_NORMAL_INSTALLED_GT_TERMINAL_TEXT_VERIFIED',
                  verified_at=datetime.now().isoformat(timespec='seconds'), files=len(inventory['files']),
                  remote_root=ROOT, local_root=str(local), manifest=inventory['files'],
                  raw_arrays_already_archived=2, retained_best_weights=2, new_neural_calls=0, new_optimizer_updates=0)
    (PF / 'r201g_normal_priority_terminal_text_intake_20261006.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'verified_at', 'files', 'new_neural_calls', 'new_optimizer_updates')}), flush=True)


if __name__ == '__main__':
    main()
