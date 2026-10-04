"""Collect and publish two completed MSVR full-data references without rerunning NN."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
root, _ = HOSTS['2026']
campaign = root + '/runs/full_official_baselines_20261004'
destination = PROJECT / 'results/full_official_MSVR310_first_pair_20261004'
assert not destination.exists()
code = f'''import json,tarfile
from pathlib import Path
root=Path({campaign!r});files=[];paired=[]
for variant in ('demo','demo_shared'):
 name='MSVR310_'+variant+'_s42';run=root/'training'/name
 trained=json.loads((run/'result.json').read_text());audit=json.loads((root/'frozen49'/name/'independent_cpu_audit.json').read_text())
 assert trained['status']=='COMPLETE' and trained['epochs']==50 and audit['status']=='PASS' and audit['cases']==49
 with (run/'batch_orders.jsonl').open() as handle:paired.append([(r['epoch'],r['step'],r['names']) for r in map(json.loads,handle)])
 for phase in ('training','frozen49','audit'):
  assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
  folder=root/phase/name
  files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.log'))
  files.extend(root/phase/(name+suffix) for suffix in ('_launch.json','_exit.json','.log'))
assert paired[0]==paired[1] and len(paired[0])==705
archive=root.parent/'full_official_MSVR310_first_pair_20261004.tar.gz';assert not archive.exists()
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
    name = 'MSVR310_' + variant + '_s42'
    run = json.loads((destination / 'training' / name / 'result.json').read_text())
    audit = json.loads((destination / 'frozen49' / name / 'independent_cpu_audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['perquery_count'] == 49 * 591
    assert run['optimizer_steps'] == 705 and run['amp_skipped_steps'] == 0 and run['training_coverage']['unvisited'] == []
    references[variant] = dict(selected_epoch=run['best']['epoch'], normal=audit['normal'],
        all49_equal_condition_mean=audit['equal_condition_mean'], optimizer_steps=705, amp_skips=0,
        training_coverage=run['training_coverage'], conditions=audit['conditions'])
delta = {m: references['demo_shared']['normal'][m] - references['demo']['normal'][m] for m in metrics}
thresholds = {m: references['demo']['normal'][m] + 2 for m in ('mAP', 'Rank-1')}
report = dict(status='TWO_FULL_OFFICIAL_REFERENCES_COMPLETE_ALL98_GT_CPU_PASS',
    collected_at=datetime.now().isoformat(timespec='seconds'), cases=98, repeated_condition_query_rows=57918,
    train_records=1032, query_records=591, gallery_records=1055, paired_identity_sampling_exact=True,
    references=references, augmented_minus_original_normal_pp=delta,
    full_protocol_original_DeMo_plus2_thresholds=thresholds, goal_complete=False,
    limits='This is the shared identity/availability/missing-augmentation baseline, with no added dual-axis M/F experts. It is not new dual-axis method evidence. One seed and benchmark-selected best weights; all49 mean is diagnostic.')
(destination / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(destination / 'publication_entrypoint.py').write_bytes(Path(__file__).read_bytes())
pf = PROJECT / 'results/preflight'
gpath = pf / 'research_goal_optimized_20261003.json'
g = json.loads(gpath.read_text(encoding='utf-8'))
g['updated_at'] = report['collected_at']
g['current_evidence']['full_official_protocol'].update(status='MSVR_TWO_FULL50_FULL49_AUDITED_RGBNT201_RUNNING',
    completed_full_training=2, completed_all49=2, completed_independent_GT_audits=2,
    pending_full_training=4, pending_all49=4, pending_independent_GT_audits=4,
    MSVR310_full_original_DeMo_plus2_thresholds=thresholds,
    first_pair='results/full_official_MSVR310_first_pair_20261004/summary.json')
gpath.write_text(json.dumps(g, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
doc = PROJECT / 'docs/实验交接.md'
s = doc.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
assert s.count(marker) == 1
status = f'''\n\n### 首个完整官方数据集终态：MSVR310（{report['collected_at']}）\n\n两类基线均完整50轮、全部1032训练记录实际访问、705次真实optimizer更新/AMP跳过0；逐batch epoch/step/文件名顺序完全一致。全部官方591查询和1055图库，固定最佳重载后49组合全部完成，从实际安装文件名重建GT的独立CPU复算PASS共98条件、57918条重复条件-query记录（不是57918条独立查询）。六指标、CMC1..50、逐query AP/INP/首末匹配及身份/相机/scene分组均完整。\n\n| 完整官方协议/模型 | 选中epoch | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'''
for variant, ref in references.items():
    status += '| ' + variant + ' | ' + str(ref['selected_epoch']) + ' | ' + ' | '.join(f'{ref["normal"][m]:.6f}' for m in metrics) + ' |\n'
status += '| 同增强DeMo−原DeMo | — | ' + ' | '.join(f'{delta[m]:+.6f}' for m in metrics) + ' |\n'
status += '\n全部49等权诊断均值（不是官方统一mAP）：\n\n| 模型 | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---:|---:|---:|---:|---:|---:|\n'
for variant, ref in references.items():
    status += '| ' + variant + ' | ' + ' | '.join(f'{ref["all49_equal_condition_mean"][m]:.6f}' for m in metrics) + ' |\n'
status += f'''\n本次同公共身份接口/可用性修正/缺失增强DeMo正常条件相对原完整DeMo达到mAP+{delta['mAP']:.6f}、R1+{delta['Rank-1']:.6f}。它没有新增M/F双轴专家，不能归为双轴协作的贡献，也不能用旧基线缺失来源污染造成的大幅差值主张新颖性。后续统一双轴方法必须同时比较这一同增强基线与同参数普通专家。新MSVR正式+2阈值为mAP{thresholds['mAP']:.6f}、R1{thresholds['Rank-1']:.6f}；不能继续沿用旧fit/dev的49.607792/61.047619。\n\nRGBNT201两组已在2026物理GPU2/3运行，RGBNT100随后各卡串行。总计2/6组完整基线完成，统一双轴候选、公平普通专家和三种子尚未完成，Goal ACTIVE/UNMET。全部49逐条件原值和完整文字证据位于results/full_official_MSVR310_first_pair_20261004；raw距离NPZ/二进制best权重留在26，未上传GitHub。\n'''
doc.write_bytes(s.replace(marker, marker + status, 1).encode('utf-8'))
files = [p.relative_to(PROJECT).as_posix() for p in destination.rglob('*') if p.is_file()]
files.append('results/preflight/research_goal_optimized_20261003.json')
mirror = Path(__file__).with_name('demo_full_official_msvr_pair_mirror_20261004.tar.gz')
assert not mirror.exists()
with tarfile.open(mirror, 'w:gz') as tar:
    for relative in files:
        tar.add(PROJECT / relative, arcname=relative)
for host, (remote_root, _) in HOSTS.items():
    target = remote_root + '/full_official_msvr_pair_text_20261004.tar.gz'
    command(['scp', *OPTIONS, str(mirror), host + ':' + target])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    print('MSVR_FULL_OFFICIAL_TEXT_MIRROR', host, len(files), flush=True)
digest = sync_handoff()
proof = pf / 'full_official_MSVR310_pair_publication_20261004.json'
assert not proof.exists()
proof.write_text(json.dumps(dict(status='PAIR_COMPLETE_PUBLISHED_NOT_FINAL_METHOD', handoff_sha256=digest,
    files=len(files), cases=98, repeated_query_rows=57918, goal_complete=False), indent=2) + '\n', encoding='utf-8')
for host, (remote_root, _) in HOSTS.items():
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/results/preflight/' + proof.name])
files += ['docs/实验交接.md', 'results/preflight/' + proof.name]
command(['git', 'add', '--', *files], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Report full-data MSVR310 DeMo baselines and all49 ground-truth metrics'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
assert command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0] == head
print('MSVR_FULL_OFFICIAL_PAIR_PUBLISHED', json.dumps(dict(head=head, doc_sha256=digest, cases=98, goal_complete=False)), flush=True)
