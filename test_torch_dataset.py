import unittest
from pathlib import Path

import torch

from daily_dvs200_dataset import DailyDVS200Dataset, create_dataloader


MANIFEST = Path(__file__).with_name("dataset_inventory.csv")
DATASET_ROOT = Path(
    "/remote-home/datasets/Joind-dataset for Action Recognition/DailyDvs-200"
)


@unittest.skipUnless(MANIFEST.exists() and DATASET_ROOT.exists(), "dataset unavailable")
class TorchDatasetTests(unittest.TestCase):
    def test_evaluation_sample_and_batch_shapes(self):
        dataset = DailyDVS200Dataset(MANIFEST, training=False, max_samples=2)
        sample = dataset[0]
        self.assertEqual(sample["frame"].shape, (2, 240, 320))
        self.assertEqual(sample["voxel"].shape, (2, 8, 240, 320))
        self.assertEqual(sample["points"].shape, (2048, 4))
        self.assertEqual(sample["point_mask"].shape, (2048,))
        self.assertEqual(sample["point_bin"].shape, (2048,))
        self.assertEqual(sample["frame"].dtype, torch.float32)
        self.assertEqual(sample["voxel"].dtype, torch.float32)
        self.assertEqual(sample["points"].dtype, torch.float32)
        self.assertEqual(sample["label"].item(), 0)

        loader = create_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["frame"].shape, (2, 2, 240, 320))
        self.assertEqual(batch["voxel"].shape, (2, 2, 8, 240, 320))
        self.assertEqual(batch["points"].shape, (2, 2048, 4))
        self.assertEqual(batch["point_mask"].shape, (2, 2048))
        self.assertEqual(batch["label"].shape, (2,))
        self.assertEqual(len(batch["path"]), 2)

    def test_training_sampling_changes_by_epoch_and_is_reproducible(self):
        first = DailyDVS200Dataset(MANIFEST, training=True, seed=11, max_samples=1)
        second = DailyDVS200Dataset(MANIFEST, training=True, seed=11, max_samples=1)
        points_epoch_zero = first[0]["points"]
        self.assertTrue(torch.equal(points_epoch_zero, second[0]["points"]))
        first.set_epoch(1)
        points_epoch_one = first[0]["points"]
        self.assertFalse(torch.equal(points_epoch_zero, points_epoch_one))

    def test_fixed_sampling_can_be_cached(self):
        dataset = DailyDVS200Dataset(
            MANIFEST,
            training=True,
            deterministic_points=True,
            cache_in_memory=True,
            max_samples=1,
        )
        first = dataset[0]
        second = dataset[0]
        self.assertIs(first, second)
        self.assertEqual(len(dataset._cache), 1)

    def test_random_sampling_cannot_be_cached(self):
        with self.assertRaisesRegex(ValueError, "deterministic"):
            DailyDVS200Dataset(
                MANIFEST,
                training=True,
                deterministic_points=False,
                cache_in_memory=True,
                max_samples=1,
            )

    def test_person_filter(self):
        dataset = DailyDVS200Dataset(
            MANIFEST, training=False, person_ids={2}, max_samples=3
        )
        self.assertEqual(len(dataset), 3)
        self.assertTrue(all(int(record["person_id"]) == 2 for record in dataset.records))


if __name__ == "__main__":
    unittest.main()
