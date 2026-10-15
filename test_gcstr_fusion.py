import unittest
import numpy as np
import torch

from secnet_encoder import SECNetEncoder
from gcstr_secnet_fusion import GCSTRSECNetFusion, VIEWS
from gcstr_fusion_dataset import sample_event_cloud
from gcstr_representations import GCSTRConfig


class FusionTests(unittest.TestCase):
    def test_shapes_gradients_and_freeze(self):
        torch.manual_seed(7)
        encoder = SECNetEncoder(embed_dim=8, groups=(16, 8, 4), neighbors=4)
        model = GCSTRSECNetFusion(num_classes=5, point_encoder=encoder,
                                 dim=16, prototypes=3, heads=4).train()
        events = torch.rand(2, 4, 32)
        features, coords = encoder(events)
        self.assertEqual([tuple(f.shape) for f in features],
                         [(2, 16, 20), (2, 8, 44), (2, 4, 92)])
        self.assertEqual([c.shape[1] for c in coords], [16, 8, 4])
        images = {v: torch.rand(2, 3, 64, 64) for v in VIEWS}
        before = model.image_encoders['cstr3'].backbone.bn1.running_mean.clone()
        logits = model(events, images)
        self.assertEqual(tuple(logits.shape), (2, 5))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(torch.equal(before, model.image_encoders['cstr3'].backbone.bn1.running_mean))
        self.assertFalse(model.image_encoders.training)
        for module in model.interactions:
            self.assertGreater(module.prototypes.grad.abs().sum().item(), 0)
            for attention in module.view_attention.values():
                self.assertGreater(attention.attention.in_proj_weight.grad.abs().sum().item(), 0)

    def test_sampling_order_repeat_and_empty(self):
        config = GCSTRConfig()
        t = np.array([100, 200, 300])
        x, y, p = np.array([0, 1, 2]), np.array([2, 1, 0]), np.array([0, 1, 0])
        first = sample_event_cloud(t, x, y, p, config, 12, np.random.default_rng(1))
        second = sample_event_cloud(t, x, y, p, config, 12, np.random.default_rng(1))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (4, 12))
        self.assertTrue((first[2, 1:] >= first[2, :-1]).all())
        self.assertTrue(set(first[3].tolist()) <= {-1, 1})
        with self.assertRaises(ValueError):
            sample_event_cloud(t[:0], x[:0], y[:0], p[:0], config, 12, np.random.default_rng(1))

    def test_invalid_point_input(self):
        encoder = SECNetEncoder(embed_dim=8, groups=(16, 8, 4), neighbors=4)
        with self.assertRaises(ValueError):
            encoder(torch.rand(2, 32, 4))
        with self.assertRaises(ValueError):
            encoder(torch.rand(2, 4, 8))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
