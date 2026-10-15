import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from secnet_encoder import SECNetEncoder
from secnet_only import SECNetOnly
from secnet_only_dataset import SECNetOnlyDataset, official_rows
from create_gcstr_splits import DEFAULT_ROOT
from train_secnet_only import atomic_save, run_epoch, seed_all


class SECNetOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        # This environment's PyTorch 2.0.1 CPU oneDNN backward produced NaNs;
        # native CPU kernels and the intended CUDA path are checked separately.
        torch.backends.mkldnn.enabled = False
        seed_all(9)

    def model(self):
        return SECNetOnly(encoder=SECNetEncoder(embed_dim=8, groups=(16, 8, 4), neighbors=4))

    def test_head_and_all_encoder_gradients(self):
        model = self.model()
        logits = model(torch.rand(2, 4, 32))
        self.assertEqual(tuple(logits.shape), (2, 200))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 199])).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_epoch_metrics_and_checkpoint_roundtrip(self):
        model = self.model()
        data = [{"events": torch.rand(4, 32), "label": torch.tensor(i)} for i in range(4)]
        loader = DataLoader(data, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150)
        metrics = run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer, emit=lambda _: None)
        scheduler.step()
        self.assertEqual(metrics["samples"], 4)
        self.assertLessEqual(metrics["top1"], metrics["top5"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state()}, path)
            state = torch.load(path, weights_only=False)
            restored = self.model()
            restored.load_state_dict(state["model"])
            opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 150)
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            self.assertEqual(sched.state_dict(), scheduler.state_dict())
            self.assertEqual(opt.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
            torch.set_rng_state(state["rng"])
            run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer, emit=lambda _: None)
            torch.set_rng_state(state["rng"])
            run_epoch(restored, loader, torch.device("cpu"), optimizer=opt, emit=lambda _: None)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_event_only_determinism_and_train_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "sample.npz", t=np.arange(1000),
                     x=np.arange(1000) % 320, y=np.arange(1000) % 240, p=np.arange(1000) % 2)
            row = {"sample_id": "sample", "source_split": "train", "npz_relpath": "sample.npz", "label_id": "3"}
            with patch("gcstr_fusion_dataset.build_gcstr_views", side_effect=AssertionError("image generated")):
                data = SECNetOnlyDataset([row], dataset_root=directory)
                first, second = data[0], data[0]
                self.assertEqual(set(first), {"events", "label", "sample_id"})
                self.assertEqual(tuple(first["events"].shape), (4, 8192))
                self.assertTrue(torch.equal(first["events"], second["events"]))
                training = SECNetOnlyDataset([row], dataset_root=directory, training=True)
                a = training[0]["events"]
                self.assertTrue(torch.equal(a, training[0]["events"]))
                training.epoch = 1
                self.assertFalse(torch.equal(a, training[0]["events"]))
            with self.assertRaises(ValueError):
                SECNetOnlyDataset([{**row, "source_split": "test"}], training=True)

    def test_actual_official_manifests(self):
        # Metadata only: this test never opens official Test event files.
        rows = official_rows(DEFAULT_ROOT)
        self.assertEqual(len(rows["train"]), 14773)
        self.assertEqual(len(rows["test"]), 4093)


if __name__ == "__main__":
    unittest.main()
