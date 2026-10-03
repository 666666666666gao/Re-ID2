"""Run our GPU2/3 process groups; the user removed power and temperature control."""
import json
import os
import signal
import subprocess
import time


def telemetry(gpu):
    assert gpu in (2, 3)
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                                  '--query-gpu=index,memory.used',
                                  '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
    index, memory = [float(value) for value in raw]
    assert int(index) == gpu
    return dict(gpu=gpu, memory_used_mib=memory, observed_at=time.time())


def check_limits(gpu):
    """Retain physical GPU2/3 restriction, without power or temperature gates."""
    return telemetry(gpu)


def execute(command, output, name, gpu):
    sample = check_limits(gpu)
    while sample['memory_used_mib'] >= 500:
        print('WAIT_SELECTED_GPU', gpu, sample['memory_used_mib'], flush=True)
        time.sleep(240)
        sample = check_limits(gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with (output / (name + '.log')).open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            assert os.getpgid(child.pid) == child.pid
            launch = dict(pid=child.pid, process_group=child.pid, gpu=gpu, started=time.time(),
                          command=command, initial_resources=sample, temperature_power_control=False)
            (output / (name + '_launch.json')).write_text(json.dumps(launch, indent=2) + '\n')
            child.wait()
        finally:
            forced_cleanup = child.poll() is None
            if forced_cleanup:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = child.returncode
            row = dict(name=name, exit_code=code, finished=time.time(), gpu=gpu,
                       temperature_power_control=False, forced_own_group_cleanup=forced_cleanup)
            (output / (name + '_exit.json')).write_text(json.dumps(row, indent=2) + '\n')
    assert code == 0, name
    return row
