"""Feature-only EventMamba V1 adaptation (not the author's training pipeline).

Source: rhwxmx/EventMamba, commit 7f61fd397cf69f10f0f92a182109d36cd7b6fdbb,
models/eventmamba_v1.py, modules.py and mamba_layer.py (MIT).
Defaults use the upstream commented larger grouping configuration, for our
8192-point experiment. Input is our [B,4,N] x,y,t,p; polarity is discarded.
Requires a real BiMamba-v2 implementation; never silently substitutes a mixer.
"""
import inspect

import torch
from torch import nn

from secnet_encoder import fps, index_points

UPSTREAM_COMMIT = "7f61fd397cf69f10f0f92a182109d36cd7b6fdbb"


def make_bimamba(channels, layer_idx):
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "EventMamba requires a compatible BiMamba installation. "
            "See EVENTMAMBA_RUNBOOK.md; SECNet does not require this dependency."
        ) from exc
    if "bimamba_type" not in inspect.signature(Mamba.__init__).parameters:
        raise RuntimeError("Installed Mamba lacks bimamba_type; vanilla Mamba is not BiMamba-v2")
    return Mamba(channels, layer_idx=layer_idx, bimamba_type="v2")


class CoordinateGroup(nn.Module):
    """Upstream coordinate KNN + anchor normalization + center concatenation."""
    def __init__(self, channels, groups, neighbors):
        super().__init__()
        self.groups, self.neighbors = groups, neighbors
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, coords, features):
        if max(self.groups, self.neighbors) > coords.shape[1]:
            raise ValueError("stage input has fewer points than groups/neighbors")
        indices = fps(coords, self.groups)
        centers = index_points(coords, indices)
        anchors = index_points(features, indices)
        distance = (-2 * centers @ coords.transpose(1, 2)
                    + centers.square().sum(-1, keepdim=True)
                    + coords.square().sum(-1)[:, None])
        neighbors = distance.argsort(dim=-1)[..., :self.neighbors].sort(dim=-1).values
        grouped = index_points(features, neighbors) - anchors[:, :, None]
        std = grouped.reshape(coords.shape[0], -1).std(dim=-1).view(-1, 1, 1, 1)
        grouped = self.affine_alpha * (grouped / (std + 1e-5)) + self.affine_beta
        return centers, torch.cat((grouped, anchors[:, :, None].expand(
            -1, -1, self.neighbors, -1)), dim=-1)


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1), nn.BatchNorm1d(channels // 2),
            nn.ReLU(), nn.Conv1d(channels // 2, channels, 1), nn.BatchNorm1d(channels))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class EventMambaStage(nn.Module):
    def __init__(self, in_channels, channels, groups, neighbors, layer_idx, mixer_factory):
        super().__init__()
        self.group = CoordinateGroup(in_channels, groups, neighbors)
        self.embedding = (nn.Sequential(nn.Conv1d(6, channels, 1),
                          nn.BatchNorm1d(channels), nn.ReLU())
                          if layer_idx == 0 else nn.Identity())
        self.local = ResidualConv(channels)
        self.attention = nn.Linear(channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.mixer = mixer_factory(channels, layer_idx)
        self.global_conv = ResidualConv(channels)

    def forward(self, coords, features):
        coords, grouped = self.group(coords, features)
        batch, points, neighbors, channels = grouped.shape
        x = grouped.permute(0, 1, 3, 2).reshape(batch * points, channels, neighbors)
        x = self.local(self.embedding(x)).transpose(1, 2)
        x = (self.attention(x).softmax(dim=1) * x).sum(1).reshape(batch, points, -1)
        # Upstream MambaBlock defaults: LayerNorm, no DropPath/fused norm.
        # Its returned residual is discarded by EventMamba, NOT added here.
        x = self.mixer(self.norm(x))
        return coords, self.global_conv(x.transpose(1, 2)).transpose(1, 2)


class EventMambaEncoder(nn.Module):
    stage_channels = (64, 128, 256)
    coordinate_dim = 3

    def __init__(self, groups=(1024, 512, 256), neighbors=24, *, mixer_factory=None):
        super().__init__()
        if len(groups) != 3 or min(groups) < 1 or neighbors < 1:
            raise ValueError("require three positive group counts and neighbors")
        if any(groups[i+1] > groups[i] or neighbors > groups[i] for i in range(2)):
            raise ValueError("invalid downstream groups/neighbors")
        self.groups, self.neighbors = tuple(groups), neighbors
        # Injection is for interface unit tests only; no CLI fallback exists.
        factory = make_bimamba if mixer_factory is None else mixer_factory
        self.stages = nn.ModuleList(EventMambaStage(cin, cout, n, neighbors, i, factory)
            for i, (cin, cout, n) in enumerate(zip((3, 64, 128), self.stage_channels, groups)))

    def forward_features(self, events):
        if events.ndim != 3 or events.shape[1] != 4:
            raise ValueError("events must be [B,4,N] in x,y,t,p order")
        if not torch.isfinite(events).all():
            raise ValueError("non-finite events")
        if (events[:, 2, 1:] < events[:, 2, :-1]).any():
            raise ValueError("EventMamba requires events sorted by time")
        coords = events[:, [2, 0, 1]].transpose(1, 2).contiguous()  # t,x,y
        features = coords
        outputs, coordinates = [], []
        for stage in self.stages:
            coords, features = stage(coords, features)
            outputs.append(features)
            coordinates.append(coords)
        return outputs, coordinates

    def forward(self, events):
        return self.forward_features(events)
