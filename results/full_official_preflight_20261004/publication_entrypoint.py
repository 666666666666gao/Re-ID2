"""Publish actual six-smoke PASS and current full-data launch, not final scores."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
pf = PROJECT / 'results/preflight'
proof = pf / 'full_official_baselines_start_publication_20261004.json'
assert not proof.exists()
launch = json.loads((pf / 'full_official_baselines_launch_20261004.json').read_text(encoding='utf-8'))
root = launch['launch']['output']
code = f'''import json,tarfile
from pathlib import Path
root=Path({root!r})
assert json.loads((root/'preflight_result.json').read_text())['status']=='PASS'
assert not list((root/'preflight').rglob('*.pth'))
files=[p for p in (root/'preflight').rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.log')]
files.append(root/'preflight_result.json')
archive=root.parent/'full_official_actual_preflight_20261004.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
 for path in files:out.add(path,arcname=str(path.relative_to(root)))
print(json.dumps(dict(archive=str(archive),files=len(files))))
'''
packed = json.loads(remote_python('2026', code))
archive = Path(__file__).with_suffix('.tar.gz')
assert not archive.exists()
command(['scp', *OPTIONS, '2026:' + packed['archive'], str(archive)])
destination = PROJECT / 'results/full_official_preflight_20261004'
destination.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:
    tar.extractall(destination, filter='data')
rows = []
for dataset in ('MSVR310', 'RGBNT201', 'RGBNT100'):
    for variant in ('demo', 'demo_shared'):
        name = dataset + '_' + variant + '_s42'
        run = json.loads((destination / 'preflight' / name / 'smoke.json').read_text())
        assert run['status'] == 'SMOKE_PASS' and run['steps'] == 3 and run['strict_reload_equal']
        assert all(run['gradients'].values()) and run['training_heldout_identities'] == 0
        assert json.loads((destination / 'preflight' / (name + '_exit.json')).read_text())['exit_code'] == 0
        rows.append({k: run[k] for k in ('train_records', 'query_records', 'gallery_records', 'classes',
            'parameters', 'trainable_parameters', 'descriptor_dim', 'steps', 'attempts', 'amp_skipped_steps',
            'strict_reload_equal', 'peak_memory')} | dict(dataset=dataset, variant=variant, all_active_gradients=True))
snapshot = json.loads((pf / 'full_official_baselines_observer_20261004.jsonl').read_text().splitlines()[-1])
assert snapshot['preflight_pass'] and snapshot['controller_alive'] and not snapshot['failures']
write = lambda p,x: p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
write(pf / 'full_official_baselines_actual_preflight_20261004.json', dict(status='SIX_ACTUAL_SMOKES_PASS', runs=rows, no_weight_files_saved=True))
write(pf / 'full_official_baselines_start_snapshot_20261004.json', snapshot)
now = datetime.now().isoformat(timespec='seconds')
doc = PROJECT / 'docs/实验交接.md'
s = doc.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
assert s.count(marker) == 1
status = f'''\n\n### 全数据实测启动状态（{now}）\n\n独立源码审查PASS（same-family/provisional，阻断项0）；六组真实CUDA预检全部PASS：各3次实际optimizer更新、全部可训练参数有限非零梯度、严格内存重载逐元素相同，退出码全部0，未保存smoke/initial/last权重。\n\n'''
status += '| 数据集/模型 | train/query/gallery | 类别数 | 参数/可训练参数 | 输出维度 | 更新/AMP跳过 | 峰值显存MiB |\n|---|---:|---:|---:|---:|---:|---:|\n'
for r in rows:
    status += f'| {r["dataset"]}/{r["variant"]} | {r["train_records"]}/{r["query_records"]}/{r["gallery_records"]} | {r["classes"]} | {r["parameters"]}/{r["trainable_parameters"]} | {r["descriptor_dim"]} | {r["steps"]}/{r["amp_skipped_steps"]} | {r["peak_memory"]/2**20:.2f} |\n'
status += f'''\n唯一远端控制器PID{launch['launch']['pid']}存活，初次实测两组MSVR310 full50已启动，物理GPU2/3分别一个NN进程；后续RGBNT201与RGBNT100各卡串行。全部官方数据、全部query/gallery、正常条件每轮六指标，固定最高mAP权重再完整49组合和独立GT审计。当前无50轮终态，不能把短训练或中间峰值当最终成绩。观察器本地session61934每240秒只读原控制器，不重启训练。\n\n实测证据results/full_official_preflight_20261004及results/preflight/full_official_baselines_actual_preflight_20261004.json；实际启动与进程快照results/preflight/full_official_baselines_start_snapshot_20261004.json。新正式协议+2阈值尚待三个完整DeMo完成后建立；Goal仍ACTIVE/UNMET。\n'''
doc.write_bytes(s.replace(marker, marker + status, 1).encode('utf-8'))
gpath = pf / 'research_goal_optimized_20261003.json'
g = json.loads(gpath.read_text(encoding='utf-8'))
g['updated_at'] = now
g['next_action']['status'] = 'SIX_REAL_PREFLIGHT_PASS_FULL50_RUNNING'
g['current_evidence']['full_official_protocol'].update(status='SIX_REAL_PREFLIGHT_PASS_FULL50_RUNNING', real_smokes=6,
    real_smoke_optimizer_updates=18, actual_preflight='results/preflight/full_official_baselines_actual_preflight_20261004.json',
    observer_session=61934, pending_full_training=6, pending_all49=6, pending_independent_GT_audits=6)
write(gpath, g)
(destination / 'publication_entrypoint.py').write_bytes(Path(__file__).read_bytes())
files = [p.relative_to(PROJECT).as_posix() for p in destination.rglob('*') if p.is_file()]
files += ['official_training_data.py', 'run_full_official_experiment.py', 'evaluate_full_official49.py', 'audit_full_official49.py',
    'launch_full_official_baselines.py', 'AGENTS.md', 'results/preflight/full_official_baseline_plan_20261004.json',
    'results/preflight/full_official_inventory_20261004.json', 'results/preflight/full_official_PK_coverage_20261004.json',
    'results/preflight/full_official_baselines_review_20261004.json', 'results/preflight/full_official_baselines_launch_20261004.json',
    'results/preflight/full_official_baselines_actual_preflight_20261004.json',
    'results/preflight/full_official_baselines_start_snapshot_20261004.json', 'results/preflight/research_goal_optimized_20261003.json']
mirror_archive = Path(__file__).with_name('demo_full_official_start_mirror_20261004.tar.gz')
assert not mirror_archive.exists()
with tarfile.open(mirror_archive, 'w:gz') as tar:
    for relative in files:
        tar.add(PROJECT / relative, arcname=relative)
for host, (remote_root, _) in HOSTS.items():
    remote_archive = remote_root + '/full_official_start_mirror_20261004.tar.gz'
    command(['scp', *OPTIONS, str(mirror_archive), host + ':' + remote_archive])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(remote_archive) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    print('FULL_OFFICIAL_START_TEXT_MIRROR', host, len(files), flush=True)
digest = sync_handoff()
publication = dict(status='SOURCE_AND_SIX_REAL_PREFLIGHT_PUBLISHED_FULL50_RUNNING', published_at=now,
    handoff_sha256=digest, source_and_text_files=len(files), controller_pid=launch['launch']['pid'],
    observer_session=61934, full50_completed_at_start=0, all49_completed_at_start=0, goal_complete=False)
write(proof, publication)
for host, (remote_root, _) in HOSTS.items():
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/' + proof.relative_to(PROJECT).as_posix()])
git_files = files + ['docs/实验交接.md', proof.relative_to(PROJECT).as_posix()]
command(['git', 'add', '--', *git_files], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Use complete official training and query/gallery for six DeMo baselines'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
remote_head = command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0]
assert remote_head == head
print('FULL_OFFICIAL_START_PUBLISHED', json.dumps(dict(head=head, handoff_sha256=digest, files=len(git_files), goal_complete=False)), flush=True)
