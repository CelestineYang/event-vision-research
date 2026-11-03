"""Protocol tests; GPU integration uses the separate --smoke entry point."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import h5py
import numpy as np
import torch

from eventmamba_paper_data import (PaperPoints, augmented_windows, build_cache,
    crop_rng, load_config, normalize_sample, prepare_sample, sha, window_bounds)
from eventmamba_v2_upstream.dataprocess.uti import normaliztion, random_crop, shuffle_downsample
from eventmamba_v2_metrics import SampleVotes, smoothed_loss
from train_eventmamba_paper import evaluation_rng, optimizer_for, restore_rng, rng_state


class PaperProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = load_config()
        self.spec = self.config["data"]

    def test_boundaries_no_short_or_tail_fallback(self):
        t = np.arange(0, 3000001, 250000, dtype=np.int64) + 1700000000000000
        self.assertEqual(window_bounds(t, self.spec),
                         [(0, 0, 6), (500000, 2, 8), (1000000, 4, 10)])
        self.assertEqual(window_bounds(t[:3], self.spec), [])
        self.assertEqual(window_bounds(t[:7], self.spec), [])
        with self.assertRaises(ValueError):
            window_bounds(t[::-1], self.spec)

    def test_sampling_and_normalization_against_published_functions(self):
        events = np.column_stack((np.arange(25000) * 35, np.arange(25000) % 320, np.arange(25000) % 240))
        for seed in (0, 93):
            np.random.seed(seed)
            sampled = shuffle_downsample(events, 2048)
            np.testing.assert_allclose(normalize_sample(events, seed, self.spec),
                                       normaliztion(sampled, 320, 240), atol=1e-7, rtol=1e-6)
            original = sampled.astype(np.float64)
            original[:, 0] //= 1000
            original[:, 0] -= original[:, 0].min()
            original[:, 0] /= original[:, 0].max()
            original[:, 1] /= 320
            original[:, 2] /= 240
            np.testing.assert_array_equal(normalize_sample(events, seed, self.spec, original=True), original.astype("f4"))
        degenerate = np.array([[1700000000000000, 319, 239]], dtype=np.int64)
        actual = normalize_sample(degenerate, 0, self.spec)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(actual[:, 0].max(), 0)

    def test_crop_matches_official_and_preserves_caller_rng(self):
        events = np.column_stack((np.arange(25000) * 35, np.arange(25000) % 320, np.arange(25000) % 240))
        # Test both the official spatial and temporal crop paths across seeds.
        for seed in range(5):
            py, pt = random.getstate(), torch.get_rng_state()
            with crop_rng(seed):
                expected = random_crop(events.copy(), 320, 240)
            variants = list(augmented_windows(events, seed, self.spec))
            np.testing.assert_array_equal(variants[0][1], expected)
            self.assertEqual(py, random.getstate())
            self.assertTrue(torch.equal(pt, torch.get_rng_state()))
            # Full event population is augmented, not the 2048-point sample.
            self.assertEqual(len(variants[1][1]), len(events))
            self.assertTrue((np.diff(variants[1][1][:, 0]) >= 0).all())
            np.testing.assert_array_equal(variants[2][1][:, 2], 239 - events[:, 2])
            np.testing.assert_array_equal(variants[3][1][:, 1], 319 - events[:, 1])

    def fixture(self, root, index=0, count=30000, duration=2500001):
        events = np.arange(count)
        path = root / f"{index}.npz"
        t = np.linspace(0, duration, count, dtype=np.int64) + 1700000000000000
        np.savez(path, t=t, x=(events % 320).astype("u2"), y=(events % 240).astype("u2"))
        return {"sample_id": f"train_{index}", "split": "train", "npz_relpath": path.name,
                "label_id": str(index), "sha256": sha(path), "event_count": str(count),
                "start_time_usec": str(t[0]), "end_time_usec": str(t[-1])}

    def test_augmented_population_and_checksum_enforcement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = self.fixture(root)
            points, meta, stats = prepare_sample((root, row, True, 0, self.spec))
            self.assertEqual(len(points), 15)
            self.assertTrue((meta[:, 5] > 2048).all())
            self.assertTrue((points[:, 0, 0] == 0).all())
            self.assertTrue((points[:, -1, 0] == 1).all())
            self.assertEqual(stats["zero_window_recordings"], 0)
            self.assertEqual(set(meta[:, 1]), set(range(5)))
            test_points, test_meta, _ = prepare_sample((root, row, False, 0, self.spec))
            self.assertEqual(len(test_points), 3)
            self.assertEqual(set(test_meta[:, 1]), {0})
            row["sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "Source changed"):
                prepare_sample((root, row, True, 0, self.spec))

    def test_threshold_applies_to_original_not_training_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = self.fixture(root, count=5000)
            points, meta, _ = prepare_sample((root, row, True, 0, self.spec))
            self.assertEqual(len(points), 12)
            self.assertNotIn(0, meta[:, 1])
            points, _, stats = prepare_sample((root, row, False, 0, self.spec))
            self.assertEqual(len(points), 0)
            self.assertEqual(stats["zero_window_recordings"], 1)

    def test_cache_interrupt_resume_and_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [self.fixture(root, i) for i in range(2)]
            path = root / "resume.h5"
            def interrupt(record):
                raise InterruptedError("simulated interruption after committed recording")
            with self.assertRaises(InterruptedError):
                build_cache(path, root, rows, training=True, config=self.config, workers=1, scope="smoke", emit=interrupt)
            self.assertFalse(path.exists())
            build_cache(path, root, rows, training=True, config=self.config, workers=1, scope="smoke", emit=lambda _: None)
            data = PaperPoints(path)
            self.assertEqual(len(data), 30)
            self.assertEqual(data.sample_counts.tolist(), [15, 15])
            with h5py.File(path, "r") as handle:
                for i, row in enumerate(rows):
                    expected, _, _ = prepare_sample((root, row, True, 0, self.spec))
                    np.testing.assert_array_equal(handle["points"][i * 15:(i + 1) * 15], expected)
            changed = copy.deepcopy(self.config)
            changed["data"]["stride_us"] = 1000000
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                build_cache(path, root, rows, training=True, config=changed, workers=1, scope="smoke")

    def test_loss_and_final_video_votes(self):
        logits, targets = torch.randn(4, 200), torch.tensor([0, 10, 125, 199])
        soft = torch.full_like(logits, .2 / 199)
        soft.scatter_(1, targets[:, None], .8)
        torch.testing.assert_close(smoothed_loss(logits, targets), -(soft * logits.log_softmax(1)).sum(1).mean())
        votes = SampleVotes([2, 199], classes=200)
        votes.update([0, 0, 1], [0, 1, 2], np.eye(200)[[2, 3, 199]], [2, 2, 199])
        result, _ = votes.result()
        self.assertEqual(result["top1"], 1)
        self.assertEqual(result["samples"], 2)
        with self.assertRaisesRegex(ValueError, "Missing predictions"):
            SampleVotes([0, 1]).result()

    def test_explicit_exclusion_remaps_video_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [self.fixture(root, 0, duration=1000000), self.fixture(root, 1)]
            path = root / "filtered.h5"
            build_cache(path, root, rows, training=False, config=self.config, workers=1, scope="smoke", emit=lambda _: None)
            with self.assertRaisesRegex(ValueError, "no eligible window"):
                PaperPoints(path)
            data = PaperPoints(path, exclude_empty=True)
            self.assertEqual(data.excluded_ids, ["train_0"])
            self.assertEqual(data.original_recordings, 2)
            self.assertEqual(data.sample_ids, ["train_1"])
            self.assertEqual(data[0]["sample_index"], 0)
            self.assertEqual(data[0]["label"], 1)
            data.close()

    def test_paper_optimizer_scheduler_and_eval_rng(self):
        model = torch.nn.Linear(4, 200)
        optimizer, scheduler = optimizer_for(model, self.config)
        self.assertIs(type(optimizer), torch.optim.Adam)
        self.assertEqual(scheduler.T_max, 350)
        self.assertEqual(optimizer.param_groups[0]["lr"], .001)
        self.assertEqual(self.config["training"]["batch_size"], 32)
        before = rng_state()
        with evaluation_rng(100000):
            a = torch.rand(5)
        after = rng_state()
        self.assertEqual(before["python"], after["python"])
        np.testing.assert_equal(before["numpy"], after["numpy"])
        self.assertTrue(torch.equal(before["torch"], after["torch"]))
        with evaluation_rng(100000):
            self.assertTrue(torch.equal(a, torch.rand(5)))

    def test_optimizer_rng_resume_matches_uninterrupted(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 200), torch.nn.Dropout(.5))
        optimizer, scheduler = optimizer_for(model, self.config)
        def step():
            optimizer.zero_grad()
            smoothed_loss(model(torch.rand(8, 4)), torch.arange(8)).backward()
            optimizer.step()
            scheduler.step()
        step()
        saved = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), rng_state()))
        step()
        expected = copy.deepcopy(model.state_dict())
        model.load_state_dict(saved[0])
        optimizer.load_state_dict(saved[1])
        scheduler.load_state_dict(saved[2])
        restore_rng(saved[3])
        step()
        for key, value in expected.items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
