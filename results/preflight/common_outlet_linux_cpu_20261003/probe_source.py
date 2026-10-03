from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

root = Path('/data/gaob/Re-ID/DeMo-DualAxis')
sys.path.insert(0, str(root))
import gpu_thermal_execute as thermal

plan = json.loads((root / 'results/preflight/common_outlet_m1_plan.json').read_text())
expected_sources = {**plan['previous_sources'], **plan['sources']}
def source_hashes():
    return {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected_sources}
assert source_hashes() == expected_sources
actual_before = [thermal.telemetry(gpu) for gpu in (2, 3)]
out = root / 'results/preflight/common_outlet_linux_cpu_20261003'
out.mkdir(exist_ok=False)
child_program = '''import json,os,random,sys,time
rng=random.Random(20261003)
with open(sys.argv[1],'x') as output:
 for i in range(80):
  output.write(json.dumps(dict(index=i,value=rng.random(),pid=os.getpid(),pgid=os.getpgrp(),time=time.time()))+'\\n')
  output.flush()
  time.sleep(.02)
'''
child_file = out / 'child.py'
child_file.write_text(child_program)
phase = []

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def wait_rows(path):
    deadline = time.monotonic() + 5
    while not path.is_file() or len(path.read_text().splitlines()) < 10:
        assert time.monotonic() < deadline, 'CPU child did not reach ten rows'
        time.sleep(.01)
    return read_rows(path)

def fake_sample(gpu, temperature):
    return dict(gpu=gpu,temperature_c=temperature,power_draw_w=50.,power_limit_w=250.,memory_used_mib=0.,observed_at=time.time(),sensor='SCRIPTED_CPU_TEST_NOT_GPU_TELEMETRY')

heartbeat = out / 'normal_rows.jsonl'
calls = 0
paused_count = None
def normal_sensor(gpu):
    global calls, paused_count
    calls += 1
    if calls == 1:
        return fake_sample(gpu, 60)
    rows = wait_rows(heartbeat)
    pid = rows[0]['pid']
    assert pid == rows[0]['pgid'] and pid != os.getpgrp()
    if calls == 2:
        phase.append(dict(phase='before_pause',pid=pid,rows=len(rows)))
        return fake_sample(gpu, 75)
    assert calls in (3,4)
    process_state = next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('State:'))
    assert 'T (stopped)' in process_state, process_state
    if calls == 3:
        paused_count = len(rows)
    else:
        assert len(rows) == paused_count, 'CPU child advanced while SIGSTOPped'
    phase.append(dict(phase='paused_verified',pid=pid,rows=len(rows),state=process_state))
    return fake_sample(gpu, 72 if calls == 3 else 70)

thermal.telemetry = normal_sensor
started = time.monotonic()
normal_exit = thermal.execute([sys.executable,'-u',str(child_file),str(heartbeat)],out,'normal',2)
assert normal_exit['exit_code'] == 0 and not normal_exit['forced_own_group_cleanup']
rows = read_rows(heartbeat)
assert len(rows) == 80 and [r['index'] for r in rows] == list(range(80))
rng = random.Random(20261003)
assert [r['value'] for r in rows] == [rng.random() for _ in rows]
assert len({(r['pid'],r['pgid']) for r in rows}) == 1
events = read_rows(out / 'normal_thermal.jsonl')
assert [r['event'] for r in events] == ['pause_own_group','sample','resume_same_group']
assert max(rows[i+1]['time']-rows[i]['time'] for i in range(79)) >= 10
assert not Path(f'/proc/{rows[0]["pid"]}').exists()
normal = dict(status='PASS_REAL_LINUX_OWN_GROUP_PAUSE_RESUME',elapsed_seconds=time.monotonic()-started,child_pid=rows[0]['pid'],rows=80,random_sequence_exact=True,progress_during_pause=0,events=[r['event'] for r in events],phases=phase,exit=normal_exit)

errors = []
for paused in (False, True):
    name = 'failure_paused' if paused else 'failure_running'
    heartbeat = out / (name + '_rows.jsonl')
    calls = 0
    original_error = subprocess.TimeoutExpired('scripted CPU sensor failure',15)
    child_pid = None
    def failure_sensor(gpu):
        global calls, child_pid
        calls += 1
        if calls == 1:
            return fake_sample(gpu,60)
        observed = wait_rows(heartbeat)
        child_pid = observed[0]['pid']
        assert child_pid == observed[0]['pgid'] and child_pid != os.getpgrp()
        if paused and calls == 2:
            return fake_sample(gpu,75)
        if paused:
            state = Path(f'/proc/{child_pid}/status').read_text()
            assert 'T (stopped)' in state
        raise original_error
    thermal.telemetry = failure_sensor
    received = None
    try:
        thermal.execute([sys.executable,'-u',str(child_file),str(heartbeat)],out,name,2)
    except subprocess.TimeoutExpired as error:
        received = error
    assert received is original_error
    receipt = json.loads((out / (name+'_exit.json')).read_text())
    assert receipt['exit_code'] == -9 and receipt['forced_own_group_cleanup']
    assert not Path(f'/proc/{child_pid}').exists()
    errors.append(dict(status='PASS_REAL_LINUX_OWN_GROUP_FAILURE_REAP',case=name,pid=child_pid,exception_identity_preserved=True,exit=receipt))

assert source_hashes() == expected_sources
result = dict(status='PASS_LINUX_CPU_PROCESS_LIFECYCLE',observed_at=datetime.now(timezone.utc).isoformat(),normal=normal,failures=errors,source57_unchanged=True,actual_gpu_before=actual_before,neural_jobs=0,CUDA_models=0,physical_GPU_thermal_validation='NOT_RUN',limitation='Real Linux CPU Popen/process-group/signals and CPU loop state; all runtime temperature/power values in these tests were scripted, not a GPU load or optimizer-state validation.')
(out / 'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
