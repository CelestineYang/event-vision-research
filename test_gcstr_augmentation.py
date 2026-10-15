"""Checks that event-level augmentation preserves gCSTR input invariants."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gcstr_augmentation import AUGMENTATIONS, augment_events
from gcstr_dataset import DailyDVS200GCSTRDataset
from gcstr_representations import GCSTRConfig


class EventAugmentationTests(unittest.TestCase):
    def test_modes_preserve_event_invariants_without_mutating_input(self):
        config = GCSTRConfig()
        t = np.arange(1000, 2000, 2, dtype=np.int64)
        x = np.arange(len(t), dtype=np.int64) % config.sensor_width
        y = np.arange(len(t), dtype=np.int64) % config.sensor_height
        p = np.arange(len(t)) % 2 == 0
        original = [array.copy() for array in (t, x, y, p)]
        for mode in AUGMENTATIONS:
            with self.subTest(mode=mode):
                result = augment_events(
                    t, x, y, p, mode=mode,
                    rng=np.random.default_rng(42), config=config,
                )
                at, ax, ay, ap = result
                self.assertGreater(len(at), 0)
                self.assertEqual(len(at), len(ax))
                self.assertEqual(len(at), len(ay))
                self.assertEqual(len(at), len(ap))
                self.assertTrue(np.all(np.diff(at) >= 0))
                self.assertTrue(np.all((ax >= 0) & (ax < config.sensor_width)))
                self.assertTrue(np.all((ay >= 0) & (ay < config.sensor_height)))
                for current, before in zip((t, x, y, p), original):
                    np.testing.assert_array_equal(current, before)

    def test_train_cache_is_separate_and_validation_rejects_augmentation(self):
        manifest = Path(__file__).with_name("gcstr_splits_seed0.csv")
        if not manifest.exists():
            self.skipTest("DailyDVS-200 manifest unavailable")
        with tempfile.TemporaryDirectory() as directory:
            dataset = DailyDVS200GCSTRDataset(
                manifest, "train", views=("ty_p",), cache_dir=directory,
                augmentation="spatial_temporal", augmentation_count=1,
            )
            first = dataset[0]["ty_p"]
            second = dataset[0]["ty_p"]
            self.assertEqual(first.shape, (3, 224, 224))
            self.assertTrue(np.isfinite(first.numpy()).all())
            self.assertTrue(np.array_equal(first.numpy(), second.numpy()))
            cache_path = dataset._cache_path(dataset.rows[0], "ty_p", 0)
            self.assertTrue(cache_path.is_file())
            self.assertFalse(dataset._cache_path(dataset.rows[0], "ty_p").exists())
            with self.assertRaises(ValueError):
                DailyDVS200GCSTRDataset(manifest, "val", augmentation="temporal")


if __name__ == "__main__":
    unittest.main()
