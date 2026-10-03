from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
plan = json.loads((PROJECT / 'results/preflight/common_coordinate_plan.json').read_text())
review = json.loads((PROJECT / 'results/preflight/common_coordinate_review.json').read_text())
launch = json.loads((PROJECT / 'results/preflight/common_coordinate_2026_launch.json').read_text())
assert review['status'] == 'PASS' and not review['blockers']
assert plan['selected_gpus'] == launch['selected_gpus'] == [2, 3]
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in plan['sources'].items())
now = datetime.now().isoformat(timespec='seconds')
phase = f'''### V12 同一公共身份坐标的检索增量试验：2026 GPU2/3 两卡队列（{now}）

用户最新资源指令为只使用2026服务器的GPU2、GPU3，覆盖之前四卡请求。实际部署receipt为results/preflight/common_coordinate_2026_launch.json，controller PID{launch['pid']}，项目/data/gaob/Re-ID/DeMo-DualAxis，运行目录{launch['output']}，日志{launch['log']}。只调度物理GPU2/3；GPU2先axis_shared后twins_shared，GPU3先frequency_shared后demo_shared。先GPU2执行28可用性张量检查，再两卡分别串行完成两组3步短训练；四组均通过后才开始fresh50。每卡每次仅一个任务，不抢占其他任务，不使用2025/2027训练。预计30–35分钟，后台每240秒观察真实PID及子任务退出/epoch；观察窗口到期不代表训练失败，须继续核查原进程，不凭超时重启。

单因素改动：保留V11的完整DeMo私有身份路径、公共raw-CLIP投影、FFT/三频带、模态与频域专家的信息访问范围、一轮条件更新、联合路由、PM/PF/PI、门控/残差尺度、loss和缺失训练。只将原先计算好的七关系M/I增量取mean后写入公共512D，这是V11普通频域已经使用的相同接口；现在双轴、普通双专家也使用此相同检索几何。输出仍5632D、.75/.25度量权重固定，无新专家、无新增参数/损失/可学习频带/权重搜索，private5120不接受新增增量。普通频域和DeMo属于不变的数学对照，本轮仍fresh50重跑全部四组，以获取同一轮训练与评测证据；不混用不同版本最优成绩。不能据源码就宣称修复了V11关闭专家后已有的缺失基础表征差距。

可复用训练/评测函数仅增加显式model_builder传入，使V12训练、缺失、四状态、张量检查均实例化同一CommonCoordinateAxis，而原V11 CLI继续使用原工厂。具体实现为common_coordinate_axis.py、run_common_coordinate_experiment.py、verify_common_coordinate_axis.py、missing_common_coordinate_development.py、diagnose_common_coordinate_states.py、launch_common_coordinate_trial.py；原shared_identity_axis.py及step/loss/采样保持原行为。普通频域初始四状态在全部7可用集合必须和旧V11逐元素相等；所有模型四状态的private5120必须与00逐元素相等；三个增强对照初始state/参数一致；所有trainable参数在真实3步中均需非零有限梯度，严格内存checkpoint重载一致后才能正式训练。

配方：MSVR310固定fit/dev身份隔离，seed42、公开CLIP重新初始化、batch64、实际每身份K4、原Adam/nativeFP16 GradScaler512、50轮；同epoch身份采样和独立seeded缺失集合序列。每batch先完整目标backward，再部分查询对停止梯度完整图库的CE.25/soft-triplet.5 backward，一次optimizer更新，AMP skips真实记录。每组按开发mAP最高且最早并列选择best.pth，完整六指标/CMC1..50/逐查询/分组严格重载；每个best再完成full49模态矩阵，随后seven-bank四状态smoke和49×6=294状态诊断并保留私有原始距离。全部四组共200轮、196完整协作可用性条件、1176状态案例计划，实际完成数以终态/退出/原始CSV为准，当前不宣称这些计划已经完成。

fresh Astra/max同家族provisional源码审查已PASS：44份source SHA、40份Python AST、10个队列情境、6个远端helper模拟，结果见common_coordinate_review.json，运行时各门槛另验。审查发现并修正两处真实记录错误：共用诊断结果protocol硬编码V11，改为版本中性的指定模型dev-best；计划误记K8，实际MSVR配置NUM_INSTANCE=4且训练不覆盖，计划纠正K4。原计划/源码快照和发现保留trace，未因此改sampler或训练协议。先前V11的1176真实诊断原始source/数据仍保留于Git343be26及实际receipt，不用新源码冒充历史as-launched bytes。

尚无V12有效终态指标或最终方法结论。原验收三个数据集mAP/Rank-1各超过同协议DeMo≥2点并全面报告缺失模态仍ACTIVE_UNMET；MSVR单种子开发试验不能代表全部数据集/多种子/正式测试。先检查完整/缺失双轴与同接口普通频域/普通双专家的公平差异、11相对00/10/01的检索价值与误伤，独立GT原始距离复算后再决定跨数据集扩展。每组只保存best.pth，无initial/last/smoke权重；公平基线与对照best保留。唯一交接文档继续本文件，Desktop/document和2025/26/27对应本文件字节相同。

'''
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
assert '### V12 同一公共身份坐标的检索增量试验' not in text
start, end = text.index('## 当前状态（'), text.index('\n', text.index('## 当前状态（'))
text = text[:start] + f'## 当前状态（{now}）' + text[end:]
point = text.index('### V11 固定最佳权重的四状态与公共/私有坐标诊断')
document.write_bytes((text[:point] + phase + text[point:]).encode('utf-8'))
changed = [name for name, sha in plan['sources'].items() if plan['previous_remote_sources'].get(name) != sha]
assert all('/' not in name for name in changed)
for host in ('2025','2027'):
    remote_root, _ = HOSTS[host]
    command(['scp', *OPTIONS, *[str(PROJECT / name) for name in changed], host + ':' + remote_root + '/'])
for host in HOSTS:
    remote_root, _ = HOSTS[host]
    code = 'import hashlib,json;from pathlib import Path;print(json.dumps({name:hashlib.sha256((Path(' + repr(remote_root) + ')/name).read_bytes()).hexdigest() for name in ' + repr(changed) + '}))'
    assert json.loads(remote_python(host, code)) == {name: plan['sources'][name] for name in changed}
digest = sync_handoff()
sync = PROJECT / 'results/preflight/common_coordinate_start_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],selected_gpus=[2,3],goal='ACTIVE_UNMET'),indent=2) + '\n')
helpers = ('prepare','deploy','observe','publish_start')
for short in helpers:
    name = f'demo_common_coordinate_{short}_20261003.py'
    (PROJECT / 'results/preflight' / name).write_bytes((Path('C:/Users/gb/.codex_tmp') / name).read_bytes())
stage = changed + ['docs/实验交接.md','results/preflight/common_coordinate_plan.json',
    'results/preflight/common_coordinate_review.json','results/preflight/common_coordinate_2026_launch.json',
    'results/preflight/common_coordinate_start_handoff_sync.json']
stage += [f'results/preflight/demo_common_coordinate_{short}_20261003.py' for short in helpers]
command(['git','add',*stage], cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'], cwd=PROJECT)
command(['git','commit','-m','Start matched common-coordinate50 trial exclusively on2026 GPUs2 and3'], cwd=PROJECT)
command(['git','push','origin','main'], cwd=PROJECT)
head = command(['git','rev-parse','HEAD'], cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'], cwd=PROJECT).split()[0] == head
with Path('C:/Users/gb/memory/2026-10-03.md').open('a', encoding='utf-8') as handle:
    handle.write(f'\nDeMo {now} V12SOURCEPASS then actual controller{launch["pid"]} started on2026GPU2/3ONLY; nativeobserver24294live initial20:15:26; globaltwoqueues/no0/1/25/27 NN. Parent V11frozen1176original343be26 remains. V12 changes only M/I mean mapping intoexistingFcommon coords, all3augmatchedinterface; all4fresh50 plus49+294states each plannednotyetterminal. ActualK4 metadata repaired no samplingchange, originaltracepreserved. Git{head};ONEdocfiveSHA{digest}; source10 changedfilesexact25/26/27. GoalACTIVE_UNMET; observerpoll240 expectednear20:50, specificPID only, no timeoutrestart.\n')
print('COMMON_COORDINATE_LAUNCH_PUBLISHED', json.dumps(dict(head=head,sha256=digest,pid=launch['pid'],selected_gpus=[2,3],goal='ACTIVE_UNMET')), flush=True)
