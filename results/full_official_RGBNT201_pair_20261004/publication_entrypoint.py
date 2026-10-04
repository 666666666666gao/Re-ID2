"""Text-only intake of the completed full-data RGBNT201 baseline pair."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, DESKTOP, HOSTS, OPTIONS, command, remote_python

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
root, _ = HOSTS['2026']
campaign = root + '/runs/full_official_baselines_20261004'
destination = PROJECT / 'results/full_official_RGBNT201_pair_20261004'
assert not destination.exists()
code = f'''import json,tarfile
from pathlib import Path
root=Path({campaign!r});files=[];paired=[]
for variant in ('demo','demo_shared'):
 name='RGBNT201_'+variant+'_s42';run=root/'training'/name
 trained=json.loads((run/'result.json').read_text());audit=json.loads((root/'frozen49'/name/'independent_cpu_audit.json').read_text())
 assert trained['status']=='COMPLETE' and trained['epochs']==50 and audit['status']=='PASS' and audit['cases']==49
 assert trained['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
 with (run/'batch_orders.jsonl').open() as handle:paired.append([(r['epoch'],r['step'],r['names']) for r in map(json.loads,handle)])
 for phase in ('training','frozen49','audit'):
  assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
  files.extend(p for p in (root/phase/name).rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.log'))
  files.extend(root/phase/(name+suffix) for suffix in ('_launch.json','_exit.json','.log'))
assert paired[0]==paired[1] and len(paired[0])==2647
archive=root.parent/'full_official_RGBNT201_pair_20261004_text.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
 for path in files:out.add(path,arcname=str(path.relative_to(root)))
print(json.dumps(dict(archive=str(archive),files=len(files),paired_sampling_exact=True)))
'''
packed = json.loads(remote_python('2026', code))
archive = Path(__file__).with_suffix('.tar.gz')
assert not archive.exists()
command(['scp', *OPTIONS, '2026:' + packed['archive'], str(archive)])
destination.mkdir()
with tarfile.open(archive) as tar:
    tar.extractall(destination, filter='data')
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
references = {}
for variant in ('demo', 'demo_shared'):
    name = 'RGBNT201_' + variant + '_s42'
    run = json.loads((destination / 'training' / name / 'result.json').read_text())
    audit = json.loads((destination / 'frozen49' / name / 'independent_cpu_audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['cases'] == 49 and audit['perquery_count'] == 40964
    assert run['optimizer_steps'] == 2647 and run['amp_skipped_steps'] == 0
    references[variant] = dict(selected_epoch=run['best']['epoch'], normal=audit['normal'],
        all49_equal_condition_mean=audit['equal_condition_mean'], conditions=audit['conditions'],
        optimizer_steps=run['optimizer_steps'], amp_skips=0, training_coverage=run['training_coverage'])
delta = {m: references['demo_shared']['normal'][m] - references['demo']['normal'][m] for m in metrics}
groups = {g: [] for g in ('same_availability', 'overlap_mismatch', 'source_disjoint', 'partial_query_full_gallery', 'both_partial')}
condition_deltas = {}
for condition, original in references['demo']['conditions'].items():
    _, q, _, g = condition.split('_')
    change = {m: references['demo_shared']['conditions'][condition][m] - original[m] for m in metrics}
    condition_deltas[condition] = change
    group = 'same_availability' if q == g else 'overlap_mismatch' if set(q) & set(g) else 'source_disjoint'
    groups[group].append(change)
    if q != 'RNT' and g == 'RNT': groups['partial_query_full_gallery'].append(change)
    if q != 'RNT' and g != 'RNT': groups['both_partial'].append(change)
group_deltas = {g: dict(conditions=len(rows), mean_delta={m: sum(r[m] for r in rows) / len(rows) for m in metrics}) for g, rows in groups.items()}
with (destination / 'frozen49/RGBNT201_demo_s42/q_RNT_g_RNT.csv').open() as handle:
    original_rows = list(csv.DictReader(handle))
with (destination / 'frozen49/RGBNT201_demo_shared_s42/q_RNT_g_RNT.csv').open() as handle:
    augmented_rows = list(csv.DictReader(handle))
assert len(original_rows) == len(augmented_rows) == 836
for original, augmented in zip(original_rows, augmented_rows):
    assert all(original[k] == augmented[k] for k in ('query_index', 'name', 'identity', 'camera', 'scene'))
query_change = dict(harmed=sum(int(a['first_match']) == 1 and int(b['first_match']) != 1 for a, b in zip(original_rows, augmented_rows)),
    rescued=sum(int(a['first_match']) != 1 and int(b['first_match']) == 1 for a, b in zip(original_rows, augmented_rows)),
    AP_improved=sum(float(b['AP']) > float(a['AP']) for a, b in zip(original_rows, augmented_rows)),
    AP_worse=sum(float(b['AP']) < float(a['AP']) for a, b in zip(original_rows, augmented_rows)))
report = dict(status='TWO_FULL_OFFICIAL_RGBNT201_REFERENCES_COMPLETE_ALL98_GT_CPU_PASS',
    collected_at=datetime.now().isoformat(timespec='seconds'), references=references, cases=98,
    repeated_condition_query_rows=81928, train_records=3951, query_records=836, gallery_records=836,
    paired_identity_sampling_exact=True, augmented_minus_original_normal_pp=delta,
    condition_deltas=condition_deltas, group_deltas=group_deltas, normal_query_change=query_change,
    both_metrics_plus2_conditions=sum(all(v[m] >= 2 for m in ('mAP', 'Rank-1')) for v in condition_deltas.values()),
    full_protocol_original_DeMo_plus2_thresholds={m: references['demo']['normal'][m] + 2 for m in ('mAP', 'Rank-1')},
    goal_complete=False, limits='Availability/shared-identity/missing-augmentation baseline only, with no added dual-axis experts. Full official benchmark-selected best weights, one seed; all49 means/group deltas are diagnostics, not official aggregate mAP.')
(destination / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(destination / 'publication_entrypoint.py').write_bytes(Path(__file__).read_bytes())
pf = PROJECT / 'results/preflight'
goal_path = pf / 'research_goal_optimized_20261003.json'
goal = json.loads(goal_path.read_text(encoding='utf-8'))
goal['updated_at'] = report['collected_at']
goal['current_evidence']['full_official_protocol'].update(status='FOUR_FULL_OFFICIAL_REFERENCES_AUDITED_BOTH_RGBNT100_RUNNING',
    completed_full_training=4, completed_all49=4, completed_independent_GT_audits=4,
    pending_full_training=2, pending_all49=2, pending_independent_GT_audits=2,
    RGBNT201_pair='results/full_official_RGBNT201_pair_20261004/summary.json')
goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
assert text.count(marker) == 1 and '### 第二组完整官方数据终态：RGBNT201' not in text
addition = f'''\n\n### 第二组完整官方数据终态：RGBNT201（{report['collected_at']}）\n\n两模型完整50轮、3951训练记录全覆盖、2647次真实更新/AMP跳过0、逐batch身份采样顺序一致。全部836查询/836图库、固定最佳重载、49组合与原始安装文件GT独立CPU复算均PASS，共98条件/81928重复query行。完整六指标：\n\n| 模型 | epoch | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'''
for variant, ref in references.items():
    addition += '| ' + variant + ' | ' + str(ref['selected_epoch']) + ' | ' + ' | '.join(f'{ref["normal"][m]:.6f}' for m in metrics) + ' |\n'
addition += '| 同增强−原始 | — | ' + ' | '.join(f'{delta[m]:+.6f}' for m in metrics) + ' |\n'
addition += f'''\n该通用增强在MSVR310正常指标提升，但RGBNT201正常mAP{delta['mAP']:+.6f}/R1{delta['Rank-1']:+.6f}，首位误伤{query_change['harmed']}、救回{query_change['rescued']}、AP改善{query_change['AP_improved']}、下降{query_change['AP_worse']}，没有形成跨数据集一致优势。这里没有新M/F双轴专家，不能把正向或负向都解释成双轴协作作用。缺失条件双指标都+2的数量为{report['both_metrics_plus2_conditions']}/49，完整原值/逐query/分组/CMC与复算证据在results/full_official_RGBNT201_pair_20261004；原始距离NPZ和唯一最佳权重仍在26，未上传GitHub。\n\n预登记组的同增强−原始诊断均值（组可能重叠，不能相加为独立样本；也不能把旧基线缺失来源污染的修复当新颖性）：\n\n| 组 | 条件数 | ΔmAP | ΔR1 |\n|---|---:|---:|---:|\n'''
for group, values in group_deltas.items():
    addition += f'| {group} | {values["conditions"]} | {values["mean_delta"]["mAP"]:+.6f} | {values["mean_delta"]["Rank-1"]:+.6f} |\n'
addition += '\n当前4/6全数据基线完成；2026 GPU2/3已分别运行原始/同增强RGBNT100进程2346313/2384027。两张卡没有预抢或重启；M3b源码及部署器已审查但未执行，等待基线整体完成和真实预检。Goal ACTIVE/UNMET。2025仍因已证实I/O错误待镜像，主项目/Desktop/26/27本次内容逐字节核验。\n'
document.write_bytes(text.replace(marker, marker + addition, 1).encode('utf-8'))
DESKTOP.write_bytes(document.read_bytes())
digest = hashlib.sha256(document.read_bytes()).hexdigest()
files = [p.relative_to(PROJECT).as_posix() for p in destination.rglob('*') if p.is_file()]
files.append('results/preflight/research_goal_optimized_20261003.json')
mirror = Path(__file__).with_name('demo_full_official_rgbnt201_pair_mirror_20261004.tar.gz')
assert not mirror.exists()
with tarfile.open(mirror, 'w:gz') as tar:
    for relative in files: tar.add(PROJECT / relative, arcname=relative)
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    target = remote_root + '/full_official_rgbnt201_pair_text_20261004.tar.gz'
    command(['scp', *OPTIONS, str(mirror), host + ':' + target])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    command(['scp', *OPTIONS, str(document), host + ':' + remote_root + '/docs/实验交接.md'])
    digest_code = 'import hashlib;from pathlib import Path;print(hashlib.sha256(Path(' + ascii(remote_root + '/docs/实验交接.md') + ').read_bytes()).hexdigest())'
    assert remote_python(host, digest_code).strip() == digest
assert hashlib.sha256(DESKTOP.read_bytes()).hexdigest() == digest
proof = pf / 'full_official_RGBNT201_pair_publication_20261004.json'
assert not proof.exists()
proof.write_text(json.dumps(dict(status='PAIR_COMPLETE_PUBLISHED_NOT_FINAL_METHOD', document_sha256=digest,
    cases=98, repeated_query_rows=81928, verified_locations=['project','Desktop','2026','2027'],
    pending_mirror='2025_IO', goal_complete=False), indent=2) + '\n', encoding='utf-8')
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/results/preflight/' + proof.name])
files += ['docs/实验交接.md', proof.relative_to(PROJECT).as_posix()]
pathspec = Path(__file__).with_suffix('.pathspec')
pathspec.write_bytes(b'\0'.join(p.encode('utf-8') for p in files) + b'\0')
command(['git', 'add', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Report full RGBNT201 baseline pair and complete missing-condition negative evidence'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
assert command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0] == head
print('FULL_RGBNT201_PAIR_PUBLISHED', json.dumps(dict(head=head, document_sha256=digest,
    normal_delta=delta, query_change=query_change, cases=98, goal_complete=False)), flush=True)
