import unittest
import torch
from gcstr_image_query import ImageQueryClassifier
from train_gcstr_image_query_ddp import ExactTrainBatches
from gcstr_secnet_fusion import VIEWS


class ImageQueryTests(unittest.TestCase):
    def test_forward_backward_frozen_backbones(self):
        torch.set_num_threads(2)
        model = ImageQueryClassifier(dim=16, prototypes=3).train()
        self.assertFalse(hasattr(model, 'point_encoder'))
        before = {k: v.clone() for k, v in model.image_encoders.named_buffers()}
        out = model({v: torch.rand(2, 3, 64, 64) for v in VIEWS})
        self.assertEqual(out.shape, (2, 200))
        torch.nn.functional.cross_entropy(out, torch.tensor([2, 3])).backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
            else:
                self.assertIsNone(param.grad, name)
        for name, buffer in model.image_encoders.named_buffers():
            self.assertTrue(torch.equal(before[name], buffer))

    def test_exact_training_shards(self):
        for size in (11818, 35):
            shards = [list(ExactTrainBatches(size, 16, r, 2, 4)) for r in range(2)]
            self.assertEqual(len(shards[0]), len(shards[1]))
            flattened = [i for shard in shards for batch in shard for i in batch]
            self.assertEqual(sorted(flattened), list(range(size)))
            self.assertEqual(len(set(flattened)), size)

    def test_uneven_validation_shards(self):
        shards = [list(range(r, 2955, 2)) for r in range(2)]
        self.assertEqual(sorted(shards[0] + shards[1]), list(range(2955)))
        self.assertEqual(len(set(shards[0]) & set(shards[1])), 0)


if __name__ == '__main__':
    unittest.main()
