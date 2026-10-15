import tempfile
import unittest
from pathlib import Path

import torch

from baseline_models import build_baseline
from create_subject_splits import allocate_people
from daily_dvs200_dataset import DailyDVS200Dataset, create_dataloader
from train_baseline import run_epoch, save_checkpoint


MANIFEST = Path(__file__).with_name("dataset_splits.csv")


class SplitTests(unittest.TestCase):
    def test_people_are_disjoint_and_assignment_is_reproducible(self):
        people = list(range(2, 49))
        first = allocate_people(people, 0.70, 0.15, seed=124)
        second = allocate_people(people, 0.70, 0.15, seed=124)
        self.assertEqual(first, second)
        groups = {
            split: {person for person, assigned in first.items() if assigned == split}
            for split in ("train", "val", "test")
        }
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        self.assertEqual(set.union(*groups.values()), set(people))


@unittest.skipUnless(MANIFEST.exists(), "split manifest unavailable")
class TrainingPipelineTests(unittest.TestCase):
    def test_all_model_variants_process_a_real_batch(self):
        dataset = DailyDVS200Dataset(
            MANIFEST, split="val", training=False, max_samples=2
        )
        batch = next(iter(create_dataloader(dataset, batch_size=2, shuffle=False)))
        for name in ("frame", "voxel", "points", "concat"):
            model = build_baseline(name, num_classes=200).eval()
            with torch.no_grad():
                logits = model.forward_batch(batch)
            self.assertEqual(logits.shape, (2, 200), name)

    def test_epoch_metrics_and_atomic_checkpoint(self):
        dataset = DailyDVS200Dataset(
            MANIFEST, split="train", training=True, seed=3, max_samples=2
        )
        loader = create_dataloader(dataset, batch_size=2, shuffle=False)
        model = build_baseline("points", num_classes=200)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = run_epoch(
            model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), optimizer, 1
        )
        self.assertEqual(metrics["samples"], 2.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 1, "points", metrics)
            checkpoint = torch.load(path, map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["model_name"], "points")


if __name__ == "__main__":
    unittest.main()
