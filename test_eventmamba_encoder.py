"""CPU interface tests use an explicit test mixer, NOT a Mamba substitute for training."""
import unittest
from unittest import mock
import types

import torch
from torch import nn

from eventmamba_encoder import EventMambaEncoder, make_bimamba
from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from gcstr_secnet_fusion import VIEWS


def test_mixer(channels, layer_idx):
    return nn.Linear(channels, channels)


def cloud():
    events = torch.rand(2, 4, 32)
    events[:, 2] = events[:, 2].sort(dim=-1).values
    return events


class EventMambaTests(unittest.TestCase):
    def test_vanilla_mamba_rejected(self):
        module = types.ModuleType('mamba_ssm.modules.mamba_simple')
        module.Mamba = nn.Linear
        with mock.patch.dict('sys.modules', {'mamba_ssm.modules.mamba_simple': module}):
            with self.assertRaisesRegex(RuntimeError, 'lacks bimamba_type'):
                make_bimamba(64, 0)

    def encoder(self):
        return EventMambaEncoder(groups=(16, 8, 4), neighbors=4, mixer_factory=test_mixer)

    def test_shapes_order_and_gradients(self):
        encoder = self.encoder()
        features, coords = encoder(cloud())
        self.assertEqual([tuple(f.shape) for f in features],
                         [(2, 16, 64), (2, 8, 128), (2, 4, 256)])
        for c in coords:
            self.assertEqual(c.shape[-1], 3)
            self.assertTrue((c[:, 1:, 0] >= c[:, :-1, 0]).all())
        sum(f.square().mean() for f in features).backward()
        for name, p in encoder.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_polarity_ignored_and_coordinates_preserved(self):
        encoder = self.encoder().eval()
        events = cloud()
        torch.manual_seed(4)
        first, coords = encoder(events)
        changed = events.clone()
        changed[:, 3] *= -1
        torch.manual_seed(4)
        second, _ = encoder(changed)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
        original = events[:, [2, 0, 1]].transpose(1, 2)
        for centers in coords:
            self.assertTrue((centers[:, :, None] == original[:, None]).all(-1).any(-1).all())

    def test_input_validation(self):
        encoder = self.encoder()
        with self.assertRaises(ValueError):
            encoder(cloud()[:, :3])
        with self.assertRaises(ValueError):
            encoder(cloud().flip(-1))
        with self.assertRaises(ValueError):
            encoder(cloud()[:, :, :8])
        events = cloud()
        events[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            encoder(events)

    def test_fusion_and_frozen_images(self):
        model = GCSTREventMambaFusion(num_classes=5, dim=16, prototypes=3,
                                     point_encoder=self.encoder()).train()
        before = model.image_encoders['cstr3'].backbone.bn1.running_mean.clone()
        output = model(cloud(), {v: torch.rand(2, 3, 64, 64) for v in VIEWS})
        self.assertEqual(tuple(output.shape), (2, 5))
        nn.functional.cross_entropy(output, torch.tensor([0, 3])).backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            else:
                self.assertIsNone(p.grad, name)
        self.assertTrue(torch.equal(before, model.image_encoders['cstr3'].backbone.bn1.running_mean))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
