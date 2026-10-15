import unittest

import torch

from gcstr_models import GCSTRBranchClassifier, PaSNetProbabilityEnsemble


class GCSTRModelTests(unittest.TestCase):
    def test_branch_forward_and_backward(self):
        model = GCSTRBranchClassifier("cstr3", num_classes=10, pretrained=False)
        batch = {
            "cstr3": torch.rand(2, 3, 64, 64),
            "label": torch.tensor([1, 2]),
        }
        logits = model.forward_batch(batch)
        self.assertEqual(logits.shape, (2, 10))
        torch.nn.CrossEntropyLoss()(logits, batch["label"]).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_probability_ensemble_is_frozen(self):
        branches = {
            view: GCSTRBranchClassifier(view, num_classes=10, pretrained=False)
            for view in ("cstr3", "xt_p", "ty_p")
        }
        ensemble = PaSNetProbabilityEnsemble(branches).eval()
        batch = {view: torch.rand(2, 3, 64, 64) for view in branches}
        with torch.no_grad():
            scores = ensemble.forward_batch(batch)
        self.assertEqual(scores.shape, (2, 10))
        self.assertTrue(all(not parameter.requires_grad for parameter in ensemble.parameters()))


if __name__ == "__main__":
    unittest.main()

