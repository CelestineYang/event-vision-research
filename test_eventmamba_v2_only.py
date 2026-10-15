import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from eventmamba_v2_data import window_bounds, normalize_sample, prepare_sample
from eventmamba_v2_layers import LocalGrouper, square_distance, index_points
from eventmamba_v2_metrics import smoothed_loss, SampleVotes


class DataAndMetricsTests(unittest.TestCase):
    def test_window_boundaries_and_short(self):
        t = np.arange(0, 3000001, 250000) + 1700000000000000
        bounds = window_bounds(t)
        self.assertEqual(bounds, [(0, 6), (2, 8), (4, 10), (6, 12)])
        self.assertEqual(window_bounds(t[:3]), [(0, 3)])
        with self.assertRaises(ValueError):
            window_bounds(t[:0])

    def test_sampling_normalization(self):
        events = np.array([[0, 0, 0], [1000, 319, 239], [2000, 30, 20]])
        a = normalize_sample(events, 1, milliseconds=True)
        self.assertEqual(a.shape, (2048, 3))
        self.assertTrue(np.array_equal(a, normalize_sample(events, 1, milliseconds=True)))
        self.assertTrue(np.all(a[1:, 0] >= a[:-1, 0]))
        self.assertTrue(np.isfinite(a).all())
        self.assertLess(a[:, 1:].max(), 1)
        self.assertTrue(np.isfinite(normalize_sample(events[:1], 0)).all())

    def test_official_smoothing(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 200)
        labels = torch.tensor([0, 5, 88, 199])
        targets = torch.full_like(logits, 0.2 / 199)
        targets.scatter_(1, labels[:, None], 0.8)
        expected = -(targets * logits.log_softmax(-1)).sum(-1).mean()
        torch.testing.assert_close(smoothed_loss(logits, labels), expected)

    def test_vote_sample_boundaries_and_ties(self):
        votes = SampleVotes([2, 1], classes=6)
        # Sample 0 tie: class 2 appeared first, even though class 1 is smaller.
        probs = np.eye(6)[[2, 1, 1]] * 0.9 + 0.1 / 6
        votes.update([0, 0, 1], [0, 1, 2], probs, [2, 2, 1])
        result, ranks = votes.result()
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["top1"], 1)
        self.assertEqual(ranks[0, 0], 2)
        self.assertTrue(np.isfinite(result["loss"]))
        with self.assertRaises(ValueError):
            SampleVotes([1]).result()
        single = SampleVotes([0], classes=6)
        single.update([0], [0], [np.eye(6)[5]], [0])
        self.assertEqual(single.result()[0]["top5"], 0)

    def test_knn_uses_features_not_coordinates(self):
        xyz = torch.tensor([[[0., 0, 0], [1., 0, 0], [2., 0, 0], [3., 0, 0]]])
        features = torch.tensor([[[0.], [100.], [1.], [200.]]])
        group = LocalGrouper(1, 1, 2, use_xyz=False, normalize=None)
        with patch("eventmamba_v2_layers.furthest_point_sample", return_value=torch.tensor([[0]])):
            _, out = group(xyz, features)
        self.assertEqual(out[0, 0, :, 0].tolist(), [0., 1.])


@unittest.skipUnless(torch.cuda.is_available(), "Real BiMamba requires CUDA")
class CUDAParityTests(unittest.TestCase):
    def test_full_official_v2_forward_backward_parity(self):
        # Import untouched official network under independent module names.
        # Only documented README feature-KNN switch and unused dependency shim.
        root = Path(__file__).parent / "eventmamba_v2_upstream/models"
        source = (root / "modules.py").read_text().replace(
            "        dists = square_distance(new_xyz, xyz)",
            "        dists = square_distance(new_points, points)")
        reference_modules = types.ModuleType("modules")
        exec(compile(source, str(root / "modules.py"), "exec"), reference_modules.__dict__)
        timm = types.ModuleType("timm.models.layers")
        timm.DropPath = nn.Identity  # Default drop_path=0 never constructs this.
        fused = types.ModuleType("mamba_ssm.ops.triton.layernorm")
        fused.RMSNorm = fused.layer_norm_fn = fused.rms_norm_fn = None
        reference_mamba = types.ModuleType("mamba_layer")
        # fused_add_norm=False: these version-specific imports are never used.
        with patch.dict(sys.modules, {"timm.models.layers": timm,
                                     "mamba_ssm.ops.triton.layernorm": fused}):
            exec(compile((root / "mamba_layer.py").read_text(), str(root / "mamba_layer.py"), "exec"), reference_mamba.__dict__)
        spec = importlib.util.spec_from_file_location("reference_eventmamba", root / "eventmamba_v2.py")
        reference = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modules": reference_modules, "mamba_layer": reference_mamba}):
            spec.loader.exec_module(reference)
        from eventmamba_v2_model import EventMamba
        torch.manual_seed(7)
        # Compare in eval mode to avoid tiny-batch BN amplifying CUDA atomic
        # reduction differences. The trainer smoke separately tests train mode.
        model = EventMamba(200).cuda().eval()
        official = reference.EventMamba(200).cuda().eval()
        official.load_state_dict(model.state_dict(), strict=True)
        x = torch.rand(2, 3, 2048, device="cuda")
        x[:, 0] = x[:, 0].sort(1).values
        labels = torch.tensor([0, 199], device="cuda")
        torch.manual_seed(77)
        a = model(x)
        smoothed_loss(a, labels).backward()
        torch.manual_seed(77)
        b = official(x)
        smoothed_loss(b, labels).backward()
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(parameter.grad, dict(official.named_parameters())[name].grad, atol=3e-4, rtol=3e-3)
        self.assertGreater(model.mamba_list[0].mixer.conv1d_b.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
