"""CPU checks of GPU scope, capacity and owned-child cleanup after the user's waiver."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import Mock, patch

import gpu_thermal_execute as guard


def sample(memory=0):
    return dict(gpu=2, memory_used_mib=memory, observed_at=0,
                temperature_c=80, power_draw_w=330, power_limit_w=350)


class GpuExecutionTests(unittest.TestCase):
    def test_uncapped_hot_gpu_no_longer_blocks_or_pauses(self):
        child = Mock(pid=4242, returncode=0)
        child.poll.return_value = 0
        child.wait.return_value = 0
        with TemporaryDirectory() as folder, patch.object(guard, 'telemetry', return_value=sample()) as query, \
                patch.object(guard.subprocess, 'Popen', return_value=child) as popen, \
                patch.object(guard.os, 'getpgid', return_value=4242, create=True), \
                patch.object(guard.os, 'killpg', create=True) as killpg:
            row = guard.execute(['python', 'train.py'], Path(folder), 'unrestricted', 2)
            self.assertEqual(row['exit_code'], 0)
            self.assertFalse(row['temperature_power_control'])
            self.assertFalse(row['forced_own_group_cleanup'])
            query.assert_called_once_with(2)
            killpg.assert_not_called()
            child.wait.assert_called_once()
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            self.assertFalse((Path(folder) / 'unrestricted_thermal.jsonl').exists())

    def test_busy_selected_gpu_waits_240_seconds_without_preemption(self):
        child = Mock(pid=4343, returncode=0)
        child.poll.return_value = 0
        child.wait.return_value = 0
        with TemporaryDirectory() as folder, patch.object(guard, 'telemetry', side_effect=[sample(700), sample()]) as query, \
                patch.object(guard.time, 'sleep') as sleep, \
                patch.object(guard.subprocess, 'Popen', return_value=child), \
                patch.object(guard.os, 'getpgid', return_value=4343, create=True), \
                patch.object(guard.os, 'killpg', create=True) as killpg:
            guard.execute(['python', 'train.py'], Path(folder), 'capacity', 3)
            sleep.assert_called_once_with(240)
            self.assertEqual([call.args for call in query.call_args_list], [(3,), (3,)])
            killpg.assert_not_called()

    def test_other_gpus_are_rejected(self):
        with patch.object(guard.subprocess, 'check_output') as query:
            for gpu in (0, 1):
                with self.assertRaises(AssertionError):
                    guard.telemetry(gpu)
            query.assert_not_called()

    def test_wait_failure_reaps_only_our_running_group(self):
        child = Mock(pid=4444, returncode=None)
        child.poll.return_value = None
        error = subprocess.TimeoutExpired('owned child wait', 1)
        def wait():
            if child.wait.call_count == 1:
                raise error
            child.returncode = -9
            return -9
        child.wait.side_effect = wait
        with TemporaryDirectory() as folder, patch.object(guard, 'telemetry', return_value=sample()), \
                patch.object(guard.subprocess, 'Popen', return_value=child), \
                patch.object(guard.os, 'getpgid', return_value=4444, create=True), \
                patch.object(guard.os, 'killpg', create=True) as killpg, \
                patch.object(guard, 'signal', SimpleNamespace(SIGKILL=9)):
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                guard.execute(['python', 'train.py'], Path(folder), 'wait_failure', 2)
            self.assertIs(caught.exception, error)
            killpg.assert_called_once_with(4444, 9)
            self.assertEqual(child.wait.call_count, 2)
            receipt = json.loads((Path(folder) / 'wait_failure_exit.json').read_text())
            self.assertEqual(receipt['exit_code'], -9)
            self.assertTrue(receipt['forced_own_group_cleanup'])


if __name__ == '__main__':
    unittest.main()
