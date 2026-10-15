"""Tests for pure-point supervision and actual gCSTR projection conventions."""
import contextlib
import io
import unittest

import numpy as np
import torch
from torch import nn

from eventmamba_encoder import EventMambaEncoder
from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from gcstr_local_fusion import GCSTRLocalEventMambaFusion, projected_pixels, sample_local_patches
from gcstr_representations import GCSTRConfig, _resize_official
from gcstr_secnet_fusion import VIEWS
from train_gcstr_fusion import run_epoch


def encoder():
    return EventMambaEncoder(groups=(16, 8, 4), neighbors=4,
                             mixer_factory=lambda c, i: nn.Linear(c, c))


def inputs():
    events = torch.rand(2, 4, 32)
    events[:, 2] = events[:, 2].sort(-1).values
    return events, {v: torch.rand(2, 3, 64, 64) for v in VIEWS}


class FusionImprovementTests(unittest.TestCase):
    def make_model(self, local):
        if local:
            return GCSTRLocalEventMambaFusion(num_classes=5, dim=16, point_encoder=encoder(),
                point_aux=True, config=GCSTRConfig(image_size=64))
        return GCSTREventMambaFusion(num_classes=5, dim=16, prototypes=3,
                                    point_encoder=encoder(), point_aux=True)

    def test_auxiliary_head_only_uses_points(self):
        for local in (False, True):
            with self.subTest(local=local):
                model = self.make_model(local).eval()
                events, views = inputs()
                torch.manual_seed(5)
                first = model.forward_outputs(events, views)['point_logits']
                torch.manual_seed(5)
                second = model.forward_outputs(events, {v: -x for v, x in views.items()})['point_logits']
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                nn.functional.cross_entropy(first, torch.tensor([0, 3])).backward()
                for name, p in model.named_parameters():
                    if name.startswith(('point_encoder.', 'point_head.')):
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    else:
                        self.assertIsNone(p.grad, name)

    def test_combined_gradients_and_frozen_bn(self):
        for local in (False, True):
            with self.subTest(local=local):
                model = self.make_model(local).train()
                before = {n: b.clone() for n, b in model.image_encoders.named_buffers()}
                outputs = model.forward_outputs(*inputs())
                labels = torch.tensor([0, 3])
                loss = nn.functional.cross_entropy(outputs['logits'], labels)
                loss += 0.5 * nn.functional.cross_entropy(outputs['point_logits'], labels)
                loss.backward()
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    else:
                        self.assertIsNone(p.grad, name)
                for name, b in model.image_encoders.named_buffers():
                    self.assertTrue(torch.equal(before[name], b), name)

    def test_projection_axes_zoom_and_histogram_bins(self):
        config = GCSTRConfig()
        coords = torch.tensor([[[0., 0., 0.], [0.25, 80/320, 180/240],
                                [1., 319/320, 239/240]]])
        source_indices = {'cstr3': ((0, 0), (80, 180), (319, 239)),
                          'xt_p': ((0, 0), (80, 56), (319, 223)),
                          'ty_p': ((0, 0), (56, 180), (223, 239))}
        shapes = {'cstr3': (320, 240), 'xt_p': (320, 224), 'ty_p': (224, 240)}
        for view in VIEWS:
            rows, cols = np.indices(shapes[view])
            # Linear ramp makes interpolated values analytically checkable after scipy zoom.
            source = (rows * 10. + cols)[None]
            image = torch.from_numpy(_resize_official(source, 224))[None]
            pixels = projected_pixels(coords, view, config)
            sampled = sample_local_patches(image, pixels, radius=0, stride=1)[0, :, 0, 0]
            expected = torch.tensor([r * 10. + c for r, c in source_indices[view]])
            torch.testing.assert_close(sampled, expected, atol=0.001, rtol=1e-6)

    def test_resnet_stride_and_border(self):
        rows, cols = torch.meshgrid(torch.arange(14.), torch.arange(14.), indexing='ij')
        feature = (rows * 1000 + cols)[None, None]
        pixels = torch.tensor([[[32., 16.], [223., 223.]]])
        sampled = sample_local_patches(feature, pixels, radius=0)[0, :, 0, 0]
        torch.testing.assert_close(sampled, torch.tensor([2001., 13013.]), atol=0.01, rtol=1e-6)

    def test_selection_loss_excludes_auxiliary_loss(self):
        class Toy(nn.Module):
            def forward_batch_outputs(self, batch):
                return {'logits': torch.tensor([[2., 0.], [2., 0.]]),
                        'point_logits': torch.tensor([[2., 0.], [0., 2.]])}
        batch = {'label': torch.tensor([0, 1])}
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = run_epoch(Toy(), [batch], torch.device('cpu'), 1, None, None, 1, 0.5)
        self.assertAlmostEqual(metrics['objective_loss'], metrics['loss'] + 0.5 * metrics['point_loss'], places=6)
        self.assertEqual(metrics['top1'], 0.5)
        self.assertEqual(metrics['point_top1'], 1.0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
