"""Run only our GPU2/3 process groups, with verified power caps and temperature pauses."""
import json
import os
import signal
import subprocess
import time


def telemetry(gpu):
    assert gpu in (2, 3)
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                                  '--query-gpu=index,temperature.gpu,power.draw,power.limit,memory.used',
                                  '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
    index, temperature, draw, limit, memory = [float(value) for value in raw]
    assert int(index) == gpu
    return dict(gpu=gpu, temperature_c=temperature, power_draw_w=draw,
                power_limit_w=limit, memory_used_mib=memory, observed_at=time.time())


def check_limits(gpu):
    sample = telemetry(gpu)
    assert sample['power_limit_w'] <= 250, f'GPU{gpu} power cap is {sample["power_limit_w"]}W; administrator must set <=250W before launch'
    assert sample['temperature_c'] < 75, f'GPU{gpu} temperature is {sample["temperature_c"]}C; cool before launch'
    return sample


def execute(command, output, name, gpu):
    sample = check_limits(gpu)
    while sample['memory_used_mib'] >= 500:
        print('WAIT_SELECTED_GPU', gpu, sample['memory_used_mib'], flush=True)
        time.sleep(240)
        sample = check_limits(gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with (output / (name + '.log')).open('x') as log, (output / (name + '_thermal.jsonl')).open('x') as monitor:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            assert os.getpgid(child.pid) == child.pid
            launch = dict(pid=child.pid, process_group=child.pid, gpu=gpu, started=time.time(),
                          command=command, initial_telemetry=sample, power_limit_w=250,
                          temperature_ceiling_c=75, resume_temperature_c=70)
            (output / (name + '_launch.json')).write_text(json.dumps(launch, indent=2) + '\n')
            paused = False
            while child.poll() is None:
                sample = telemetry(gpu)
                stop = sample['temperature_c'] >= 75 or sample['power_limit_w'] > 250 or sample['power_draw_w'] > 250
                resume = sample['temperature_c'] <= 70 and sample['power_limit_w'] <= 250 and sample['power_draw_w'] <= 250
                event = 'sample'
                if stop and not paused:
                    os.killpg(child.pid, signal.SIGSTOP)
                    paused, event = True, 'pause_own_group'
                elif paused and resume:
                    os.killpg(child.pid, signal.SIGCONT)
                    paused, event = False, 'resume_same_group'
                monitor.write(json.dumps(dict(**sample, event=event, paused=paused, process_group=child.pid)) + '\n')
                monitor.flush()
                # Local thermal response; remote progress observations remain milestone/240s.
                time.sleep(5)
        finally:
            forced_cleanup = child.poll() is None
            if forced_cleanup:
                os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
            row = dict(name=name, exit_code=code, finished=time.time(), gpu=gpu,
                       thermal_log=name + '_thermal.jsonl', forced_own_group_cleanup=forced_cleanup)
            (output / (name + '_exit.json')).write_text(json.dumps(row, indent=2) + '\n')
    assert code == 0, name
    return row
