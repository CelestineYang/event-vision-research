from pathlib import Path
import tempfile
import unittest

import torch

from archive_ucf101_a_checkpoints import archive_latest


class CheckpointArchiverTests(unittest.TestCase):
    def test_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(archive_latest(tmp))

    def test_interval_atomic_replacement_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            signature = {'stage': 'fusion', 'epochs': 150}
            def save(epoch):
                state = {'epoch': epoch, 'signature': signature, 'models': {'A': {'w': torch.tensor(epoch)}},
                         'optimizers': {'A': {}}, 'complete': epoch == 150}
                torch.save(state, directory / 'last.pt.tmp')
                (directory / 'last.pt.tmp').replace(directory / 'last.pt')
            save(4)
            self.assertFalse(archive_latest(directory)['archived'])
            save(5)
            self.assertTrue(archive_latest(directory)['archived'])
            self.assertTrue(archive_latest(directory)['already_saved'])
            self.assertEqual((directory / 'last.pt').stat().st_ino,
                             (directory / 'epoch_005.pt').stat().st_ino)
            save(10)
            self.assertTrue(archive_latest(directory)['archived'])
            previous = torch.load(directory / 'epoch_005.pt', weights_only=False)
            self.assertEqual(previous['epoch'], 5)
            self.assertEqual(previous['models']['A']['w'].item(), 5)
            self.assertNotEqual((directory / 'last.pt').stat().st_ino,
                                (directory / 'epoch_005.pt').stat().st_ino)
            save(150)
            result = archive_latest(directory)
            self.assertTrue(result['archived'] and result['complete'])

    def test_refuses_wrong_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.save({'epoch': 5, 'signature': {'stage': 'images'}, 'models': {}, 'optimizers': {}},
                       Path(tmp) / 'last.pt')
            with self.assertRaises(ValueError):
                archive_latest(tmp)


if __name__ == '__main__':
    unittest.main()
