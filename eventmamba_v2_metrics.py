"""Explicit window CE and sample majority-vote metrics (200-class adaptation)."""
import numpy as np
import torch
from torch.nn import functional as F


def smoothed_loss(logits, labels, reduction="mean"):
    # Official smoothing: true=0.8, each OTHER class=0.2/(C-1).
    # This differs slightly from torch.cross_entropy(label_smoothing=0.2).
    logp = F.log_softmax(logits, dim=-1)
    correct = logp.gather(1, labels[:, None]).squeeze(1)
    losses = -0.8 * correct - 0.2 / (logits.shape[1] - 1) * (logp.sum(1) - correct)
    return losses.mean() if reduction == "mean" else losses


class SampleVotes:
    def __init__(self, labels, classes=200):
        self.labels = np.asarray(labels, dtype=np.int64)
        shape = (len(labels), classes)
        self.votes = np.zeros(shape, dtype=np.int64)
        self.probs = np.zeros(shape, dtype=np.float64)
        self.first = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
        self.counts = np.zeros(len(labels), dtype=np.int64)

    def update(self, sample_indices, window_indices, probabilities, labels):
        for sample, window, probs, label in zip(sample_indices, window_indices, probabilities, labels):
            if self.labels[sample] != label:
                raise ValueError("Sample label mismatch")
            predicted = int(np.argmax(probs))
            self.votes[sample, predicted] += 1
            self.first[sample, predicted] = min(self.first[sample, predicted], window)
            self.probs[sample] += probs
            self.counts[sample] += 1

    def result(self):
        if np.any(self.counts == 0):
            raise ValueError("Missing predictions for official samples")
        classes = np.arange(self.votes.shape[1])
        # Majority ties: earliest window voting for that class, like Counter.
        # Remaining zero-vote ties: ascending class index, fixed before testing.
        ranks = np.stack([np.lexsort((classes, first, -votes))
                          for votes, first in zip(self.votes, self.first)])
        # Hard-vote Top-5 uses up to five classes that actually received votes.
        # Do not give arbitrary zero-vote classes credit when <5 classes voted.
        top5 = ranks[:, :5].copy()
        top5[np.take_along_axis(self.votes, top5, axis=1) == 0] = -1
        mean_probs = self.probs / self.counts[:, None]
        logp = np.log(np.maximum(mean_probs, 1e-30))
        true_logp = logp[np.arange(len(self.labels)), self.labels]
        return {"samples": len(self.labels), "top1": float(np.mean(ranks[:, 0] == self.labels)),
                "top5": float(np.mean((top5 == self.labels[:, None]).any(1))),
                "loss": float(-true_logp.mean()),
                "smoothed_loss": float((-0.8 * true_logp - 0.2 / (len(classes) - 1)
                                          * (logp.sum(1) - true_logp)).mean()),
                "mean_probability_top1": float(np.mean(mean_probs.argmax(1) == self.labels)),
                "min_windows": int(self.counts.min()), "max_windows": int(self.counts.max()),
                "top1_tied_samples": int(((self.votes == self.votes.max(1)[:, None]).sum(1) > 1).sum()),
                "fewer_than_5_voted_classes": int(((self.votes > 0).sum(1) < 5).sum()),
                "prediction_rule": "hard_vote_rank; ties earliest_window_then_class_id; top5_positive_votes_only",
                "loss_rule": "CE_of_mean_window_probabilities; not_loss_of_hard_vote"}, top5
