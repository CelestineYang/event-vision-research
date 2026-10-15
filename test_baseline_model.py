import unittest

import torch
from torch import nn

from baseline_models import PointEncoder, ThreeBranchConcatClassifier


class BaselineModelTests(unittest.TestCase):
    def test_forward_features_and_backward(self):
        torch.manual_seed(0)
        model = ThreeBranchConcatClassifier(num_classes=10, branch_dim=16, hidden_dim=24)
        frame = torch.rand(2, 2, 32, 40)
        voxel = torch.rand(2, 2, 4, 32, 40)
        points = torch.rand(2, 32, 4)
        mask = torch.ones(2, 32, dtype=torch.bool)
        labels = torch.tensor([1, 2])

        logits, features = model(
            frame, voxel, points, mask, return_features=True
        )
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(features["frame"].shape, (2, 16))
        self.assertEqual(features["voxel"].shape, (2, 16))
        self.assertEqual(features["points"].shape, (2, 16))

        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(torch.isfinite(loss))

    def test_point_padding_is_ignored(self):
        torch.manual_seed(1)
        encoder = PointEncoder(output_dim=8).eval()
        points = torch.rand(1, 6, 4)
        mask = torch.tensor([[True, True, True, False, False, False]])
        changed_padding = points.clone()
        changed_padding[:, 3:] = 1e6
        with torch.no_grad():
            first = encoder(points, mask)
            second = encoder(changed_padding, mask)
        self.assertTrue(torch.equal(first, second))

    def test_empty_point_set_returns_zero_feature(self):
        encoder = PointEncoder(output_dim=8).eval()
        points = torch.zeros(2, 4, 4)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        with torch.no_grad():
            output = encoder(points, mask)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))


if __name__ == "__main__":
    unittest.main()
