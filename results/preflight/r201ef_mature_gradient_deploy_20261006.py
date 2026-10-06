"""One pair of train-only mature probes, after F and normal state closure."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command

REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT = REMOTE + '/runs/r201ef_mature_gradient_20261006'


def main():
    pf = PROJECT / 'results/preflight'
    proof = pf / 'r201ef_mature_gradient_actual_20261006.json'
    assert not proof.exists()
    states = json.loads((pf / 'r201f_normal_states_actual_20261006.json').read_text())
    assert states['status'] == 'ACTUAL_R201F_TWO_NORMAL_FOURSTATE_GT_RAW_COMPLETE' and states['state_cases'] == 8
    review = json.loads((pf / 'r201ef_mature_gradient_source_review_20261006.json').read_text())
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources, reused = review['sources_sha256'], review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in (sources | reused).items())
    state_root = PROJECT / 'results/r201f_normal_states_20261006'
    pids = [json.loads((state_root / 'controller_launch.json').read_text())['pid']]
    pids += [json.loads((state_root / ('RGBNT201_r201f_' + variant + '_s42_launch.json')).read_text())['pid']
             for variant in ('frequency_shared', 'axis_shared')]
    assert json.loads((state_root / 'controller_result.json').read_text())['status'] == 'COMPLETE'
    command(['scp', *OPTIONS, str(PROJECT / 'probe_mature_identity_gradient.py'), '2026:' + REMOTE + '/probe_mature_identity_gradient.py'])
    source_map = {name: sha for name, sha in (sources | reused).items() if not name.startswith('results/')}
    code = f'''from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
from gpu_thermal_execute import execute
project=Path({REMOTE!r});root=Path({ROOT!r})
assert not root.exists()
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {source_map!r}.items())
for pid in {tuple(pids+[3810565,3811689,3811690])!r}:
 assert subprocess.run(['ps','-p',str(pid)],stdout=subprocess.DEVNULL).returncode==1
root.mkdir()
controller=dict(pid=os.getpid(),started=time.time(),gpus=[2,3],maximum_NN=2,temperature_power_control=False)
(root/'controller_launch.json').write_text(json.dumps(controller,indent=2)+'\\n')
runs=[('E',2,project/'runs/r201e_list_objective_fp32check_20261006/training/RGBNT201_r201e_axis_shared_s42'),
      ('F',3,project/'runs/r201f_list_temperature_20261006/training/RGBNT201_r201f_axis_shared_s42')]
def probe(item):
 name,gpu,run=item;weights=run/'best.pth';before=hashlib.sha256(weights.read_bytes()).hexdigest()
 execute([sys.executable,'-u','probe_mature_identity_gradient.py','--run-dir',str(run),'--output',str(root/(name+'.json'))],root,name,gpu)
 assert hashlib.sha256(weights.read_bytes()).hexdigest()==before
 data=json.loads((root/(name+'.json')).read_text())
 assert data['status']=='ACTUAL_SELECTED_TRAIN_ONLY_DEPLOY_PARAMETER_GRADIENTS' and data['neural_forwards']==8
 assert data['optimizer_updates']==data['checkpoint_writes']==0 and data['parameters_unchanged'] and data['frozen_identity_unchanged']
 return name,data,dict(bytes=weights.stat().st_size,sha256=before,unchanged=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(probe,r) for r in runs];observed=[f.result() for f in futures]
rows={{name:data for name,data,_ in observed}}
assert [(r['names'],r['training_labels'],r['partial_set']) for r in rows['E']['rows']]==[(r['names'],r['training_labels'],r['partial_set']) for r in rows['F']['rows']]
assert rows['E']['temperature']==.01 and rows['F']['temperature']==.05
result=dict(status='COMPLETE',controller=controller,rows=rows,checkpoint_receipts={{name:receipt for name,_,receipt in observed}},
 neural_forwards=16,optimizer_updates=0,checkpoint_writes=0,paired_training_observations=True)
(root/'controller_result.json').write_text(json.dumps(result,indent=2)+'\\n')
print('MATURE_GRADIENT_RECEIPT '+json.dumps(result),flush=True)'''
    process = subprocess.Popen(['ssh', *OPTIONS, '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
        '2026', 'cd ' + shlex.quote(REMOTE) + ' && ' + shlex.quote(PYTHON) + ' -'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    text, _ = process.communicate(input=code)
    for line in text.splitlines():
        if not line.startswith('MATURE_GRADIENT_RECEIPT '):
            print(line, flush=True)
    assert process.returncode == 0
    receipts = [line[len('MATURE_GRADIENT_RECEIPT '):] for line in text.splitlines() if line.startswith('MATURE_GRADIENT_RECEIPT ')]
    assert len(receipts) == 1
    result = json.loads(receipts[0])
    assert result['status'] == 'COMPLETE' and result['neural_forwards'] == 16 and result['optimizer_updates'] == 0
    result.update(status='ACTUAL_E_F_MATURE_TRAIN_ONLY_DEPLOY_GRADIENT_PAIR_COMPLETE',
        verified_at=datetime.now().isoformat(timespec='seconds'),
        source_review='SOURCE_ONLY same-family/provisional; actual runtime model/backend/effort unattested',
        limits='Four paired training batches from two mature selected checkpoints; no query/gallery forwards or optimizer. '
               'Named parameter derivative norms/directions, not Adam updates or whole-run/whole-model gradient conclusions. '
               'Does not prove test gains, independent routing necessity or calibrated collaboration.')
    proof.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('ACTUAL_MATURE_DEPLOY_GRADIENT_PAIR_CLOSED', flush=True)


if __name__ == '__main__':
    main()
