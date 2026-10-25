"""No GPU workloads are launched by these tests."""
import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from wait_gpu_feature_smoke import IdleTracker, is_idle, lock_file, query_gpus, run_smoke, smoke_commands


def gpu(busy=False, memory=0, utilization=0):
    return dict(uuid='GPU-test', has_compute_process=busy, memory_mib=memory, utilization=utilization)


class WaitTests(unittest.TestCase):
    def test_requires_all_idle_conditions(self):
        self.assertTrue(is_idle(gpu(), 512, 5))
        for value in (gpu(busy=True), gpu(memory=513), gpu(utilization=6)):
            self.assertFalse(is_idle(value, 512, 5))

    def test_continuous_idle_and_identity(self):
        tracker = IdleTracker(120)
        def step(now, value):
            return tracker.update({0: value}, [0], now, 512, 5)
        self.assertEqual(step(0, gpu()), [])
        self.assertEqual(step(119, gpu()), [])
        self.assertEqual(step(120, gpu()), [0])
        self.assertEqual(step(121, gpu(busy=True)), [])
        self.assertEqual(step(200, gpu()), [])
        self.assertEqual(step(320, gpu()), [0])
        self.assertEqual(step(321, dict(gpu(), uuid='GPU-other')), [])

    def test_query_fail_closed_and_compute_process(self):
        with patch('wait_gpu_feature_smoke.subprocess.run', side_effect=[
            Mock(stdout='0, GPU-test, 0, 0\n'), Mock(stdout='GPU-test, 123\n')]):
            self.assertTrue(query_gpus()[0]['has_compute_process'])
        with patch('wait_gpu_feature_smoke.subprocess.run', side_effect=[
            Mock(stdout='0, GPU-test, N/A, 0\n'), Mock(stdout='')]):
            with self.assertRaises(ValueError):
                query_gpus()

    def test_lock_prevents_same_launcher_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'lock'
            first = lock_file(path)
            self.assertIsNone(lock_file(path))
            first.close()
            second = lock_file(path)
            self.assertIsNotNone(second)
            second.close()

    def test_only_smoke_and_resume_commands(self):
        commands = smoke_commands(Path('/venv/bin/python'), Path('/tmp/smoke'))
        self.assertEqual(len(commands), 2)
        self.assertTrue(all('--smoke' in cmd for cmd in commands))
        self.assertEqual(commands[1], commands[0]+['--resume'])

    def test_smoke_failure_does_not_retry_or_create_child_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'smoke'
            child = Mock()
            child.wait.return_value = 1
            with patch('wait_gpu_feature_smoke.subprocess.Popen', return_value=child) as spawn:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    run_smoke(Path('/python'), output, 'GPU-test')
                self.assertEqual(spawn.call_count, 1)
                self.assertFalse(output.exists())
                self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-test')


if __name__ == '__main__':
    unittest.main()
