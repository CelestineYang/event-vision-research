"""Standalone classifier; the shared SECNet encoder is deliberately unchanged."""
import torch
from torch import nn
from secnet_encoder import SECNetEncoder


class SECNetOnly(nn.Module):
    def __init__(self, num_classes=200, encoder=None):
        super().__init__()
        self.encoder = SECNetEncoder() if encoder is None else encoder
        # Match the author's classification head (models/FECNet.py).
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(2 * self.encoder.stage_channels[-1], 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, events):
        features, _ = self.encoder.forward_features(events)
        last = features[-1]
        return self.classifier(torch.cat((last.mean(1), last.max(1).values), dim=-1))
