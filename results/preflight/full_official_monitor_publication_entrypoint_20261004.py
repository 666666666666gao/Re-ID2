"""Publish the monitor correction while retaining the unchanged full-data jobs."""
from datetime import datetime
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
pf = PROJECT / 'results/preflight'
proof = pf / 'full_official_observer_correction_publication_20261004.json'
assert not proof.exists()
review = json.loads((pf / 'full_official_primary_observer_v2_review_20261004.json').read_text(encoding='utf-8'))
assert review['verdict'] == 'PASS' and review['blocking_issues'] == []
names = ['full_official_observer_failure_20261004.json', 'full_official_primary_observer_review_20261004.json',
    'full_official_primary_observer_v2_review_20261004.json', 'full_official_observer_initial_20261004.py',
    'full_official_primary_observer_initial_20261004.py', 'full_official_primary_observer_v2_20261004.py',
    'full_official_baselines_primary_observer_latest_20261004.json', 'research_goal_optimized_20261003.json']
intake = pf / 'full_official_intake_prepared_20261004.py'
assert not intake.exists()
intake.write_bytes(Path('C:/Users/gb/.codex_tmp/demo_full_official_collect_20261004.py').read_bytes())
names.append(intake.name)
entry = pf / 'full_official_monitor_publication_entrypoint_20261004.py'
assert not entry.exists()
entry.write_bytes(Path(__file__).read_bytes())
names.append(entry.name)
archive = Path(__file__).with_suffix('.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tar:
    for name in names:
        tar.add(pf / name, arcname='results/preflight/' + name)
for host, (root, _) in HOSTS.items():
    destination = root + '/full_official_monitor_correction_20261004.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + destination])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(destination) + ');tarfile.open(p).extractall(' + repr(root) + ',filter="data");p.unlink()')
digest = sync_handoff()
row = dict(status='MONITOR_V2_SOURCE_PASS_AND_EXACT_HANDOFF_MIRRORED', published_at=datetime.now().isoformat(timespec='seconds'),
    handoff_sha256=digest, observer_session=95924, old_observer_sessions=dict(failed=61934,replaced=74069),
    neural_controller_unchanged=2173034, neural_training_PIDs_unchanged=[2175872,2175873], goal_complete=False)
proof.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
for host, (root, _) in HOSTS.items():
    command(['scp', *OPTIONS, str(proof), host + ':' + root + '/results/preflight/' + proof.name])
files = ['results/preflight/' + name for name in names] + ['docs/实验交接.md', 'results/preflight/' + proof.name]
command(['git', 'add', '--', *files], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Track full-data neural jobs by launch PID and preserve monitor diagnostics'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
assert command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0] == head
print('FULL_OFFICIAL_MONITOR_CORRECTION_PUBLISHED', json.dumps(dict(head=head, doc_sha256=digest, observer_session=95924)), flush=True)
