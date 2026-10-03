"""CPU-only checks of the user-requested launch refusal and owned-group thermal control."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import Mock, patch

import gpu_thermal_execute as guard


def sample(temperature=60, draw=200, limit=250):
    return dict(gpu=2, temperature_c=temperature, power_draw_w=draw,
                power_limit_w=limit, memory_used_mib=0, observed_at=0)


class ThermalControlTests(unittest.TestCase):
    def test_uncapped_gpu_refuses_before_starting_a_child(self):
        with TemporaryDirectory() as folder, patch.object(guard, 'telemetry', return_value=sample(limit=350)), \
                patch.object(guard.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(AssertionError, 'administrator must set <=250W'):
                guard.execute(['python', 'train.py'], Path(folder), 'uncapped', 2)
            popen.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_temperature_pause_and_resume_preserve_the_same_owned_group(self):
        child = Mock(pid=4242)
        child.poll.side_effect = [None, None, None, None, 0, 0]
        child.wait.return_value = 0
        samples = [sample(), sample(73), sample(75), sample(72), sample(70)]
        with TemporaryDirectory() as folder, patch.object(guard, 'telemetry', side_effect=samples), \
                patch.object(guard.subprocess, 'Popen', return_value=child) as popen, \
                patch.object(guard.os, 'getpgid', return_value=4242, create=True), \
                patch.object(guard.os, 'killpg', create=True) as killpg, \
                patch.object(guard, 'signal', SimpleNamespace(SIGSTOP=19, SIGCONT=18)), \
                patch.object(guard.time, 'sleep'):
            row = guard.execute(['python', 'train.py'], Path(folder), 'thermal', 2)
            self.assertEqual(row['exit_code'], 0)
            self.assertEqual([item.args for item in killpg.call_args_list], [(4242, 19), (4242, 18)])
            self.assertEqual(popen.call_count, 1)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            entries = [json.loads(line) for line in (Path(folder) / 'thermal_thermal.jsonl').read_text().splitlines()]
            self.assertEqual([row['event'] for row in entries], ['sample', 'pause_own_group', 'sample', 'resume_same_group'])
            self.assertTrue(entries[2]['paused'])

    def test_excess_power_pauses_only_our_group(self):
        child = Mock(pid=4343)
        child.poll.side_effect = [None, None, 0, 0]
        child.wait.return_value = 0
        with TemporaryDirectory() as folder, patch.object(guard, 'telemetry', side_effect=[sample(), sample(draw=260), sample()]), \
                patch.object(guard.subprocess, 'Popen', return_value=child), \
                patch.object(guard.os, 'getpgid', return_value=4343, create=True), \
                patch.object(guard.os, 'killpg', create=True) as killpg, \
                patch.object(guard, 'signal', SimpleNamespace(SIGSTOP=19, SIGCONT=18)), \
                patch.object(guard.time, 'sleep'):
            guard.execute(['python', 'train.py'], Path(folder), 'power', 3)
            self.assertEqual([item.args for item in killpg.call_args_list], [(4343, 19), (4343, 18)])

    def test_other_gpus_are_rejected(self):
        with patch.object(guard.subprocess, 'check_output') as query:
            for gpu in (0, 1):
                with self.assertRaises(AssertionError):
                    guard.telemetry(gpu)
            query.assert_not_called()

    def test_sensor_failure_reaps_our_running_group(self):
        child = Mock(pid=4444)
        child.poll.return_value = None
        child.wait.return_value = -9
        error = subprocess.TimeoutExpired('nvidia-smi', 15)
        with TemporaryDirectory() as folder, patch.object(guard, 'telemetry', side_effect=[sample(), error]), \
                patch.object(guard.subprocess, 'Popen', return_value=child), \
                patch.object(guard.os, 'getpgid', return_value=4444, create=True), \
                patch.object(guard.os, 'killpg', create=True) as killpg, \
                patch.object(guard, 'signal', SimpleNamespace(SIGSTOP=19, SIGCONT=18, SIGKILL=9)):
            with self.assertRaises(subprocess.TimeoutExpired):
                guard.execute(['python', 'train.py'], Path(folder), 'sensor_failure', 2)
            killpg.assert_called_once_with(4444, 9)
            child.wait.assert_called_once()
            receipt = json.loads((Path(folder) / 'sensor_failure_exit.json').read_text())
            self.assertEqual(receipt['exit_code'], -9)
            self.assertTrue(receipt['forced_own_group_cleanup'])

    def test_sensor_failure_reaps_our_paused_group(self):
        child = Mock(pid=4545)
        child.poll.return_value = None
        child.wait.return_value = -9
        error = subprocess.TimeoutExpired('nvidia-smi', 15)
        with TemporaryDirectory() as folder, patch.object(guard, 'telemetry', side_effect=[sample(), sample(75), error]), \
                patch.object(guard.subprocess, 'Popen', return_value=child), \
                patch.object(guard.os, 'getpgid', return_value=4545, create=True), \
                patch.object(guard.os, 'killpg', create=True) as killpg, \
                patch.object(guard, 'signal', SimpleNamespace(SIGSTOP=19, SIGCONT=18, SIGKILL=9)), \
                patch.object(guard.time, 'sleep'):
            with self.assertRaises(subprocess.TimeoutExpired):
                guard.execute(['python', 'train.py'], Path(folder), 'paused_sensor_failure', 2)
            self.assertEqual([item.args for item in killpg.call_args_list], [(4545, 19), (4545, 9)])
            child.wait.assert_called_once()
            receipt = json.loads((Path(folder) / 'paused_sensor_failure_exit.json').read_text())
            self.assertEqual(receipt['exit_code'], -9)
            self.assertTrue(receipt['forced_own_group_cleanup'])


if __name__ == '__main__':
    unittest.main()
