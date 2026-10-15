"""Feature-only adaptation of rhwxmx/SECNet_ICML (0638bb18776f5cdb44912770f8f117c96e59b684).

Preserves feature-space KNN, first-three-coordinate FPS, coordinate evolution,
channel/sequence FFT and residual blocks. Not an official release. Unused
upstream parameters and classification head are omitted. Explicit irFFT length
also supports odd test configurations. Input convention here is [x,y,t,p].
"""
import torch
from torch import nn
from torch.nn import functional as F


def index_points(points, indices):
    return points.gather(1, indices.reshape(indices.shape[0], -1)[..., None].expand(
        -1, -1, points.shape[-1])).reshape(*indices.shape, points.shape[-1])


@torch.no_grad()
def fps(xyz, count):
    batch, size, _ = xyz.shape
    selected = torch.empty(batch, count, dtype=torch.long, device=xyz.device)
    distance = xyz.new_full((batch, size), 1e10)
    # Preserve the official random start. Seed torch before reproducible evaluation.
    farthest = torch.randint(size, (batch,), device=xyz.device)
    rows = torch.arange(batch, device=xyz.device)
    for i in range(count):
        selected[:, i] = farthest
        center = xyz[rows, farthest][:, None]
        distance = torch.minimum(distance, ((xyz - center) ** 2).sum(-1))
        farthest = distance.max(-1).indices
    return selected.sort(dim=1).values


class LocalGroup(nn.Module):
    def __init__(self, groups, neighbors):
        super().__init__()
        self.groups, self.neighbors = groups, neighbors
        self.scale = nn.Parameter(torch.ones(1, 1, 4))

    def forward(self, coords, features):
        if max(self.groups, self.neighbors) > coords.shape[1]:
            raise ValueError("stage input has fewer points than groups/neighbors")
        coords = coords * self.scale
        centers = index_points(features, fps(coords[..., :3], self.groups))
        distances = (-2 * centers @ features.transpose(1, 2)
                     + centers.square().sum(-1, keepdim=True)
                     + features.square().sum(-1)[:, None])
        indices = distances.argsort(dim=-1)[..., :self.neighbors].sort(dim=-1).values
        grouped_coords = index_points(coords, indices)
        grouped = torch.cat((index_points(features, indices), grouped_coords), -1)
        centered = grouped - grouped.mean(dim=2, keepdim=True)
        std = centered.reshape(coords.shape[0], -1).std(dim=-1).view(-1, 1, 1, 1)
        grouped = centered / (std + 1e-5)
        return grouped_coords.mean(2), torch.cat((grouped, centers[:, :, None].expand(
            -1, -1, self.neighbors, -1)), -1)


class FrequencyFilter(nn.Module):
    def __init__(self, length, channels):
        super().__init__()
        weight = torch.zeros(length // 2 + 1, channels, 2)
        weight[..., 0] = 1
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        # FFT is deliberately full precision; half precision lengths are restricted.
        spectrum = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        return torch.fft.irfft(spectrum * torch.view_as_complex(self.weight.float()),
                               n=x.shape[1], dim=1, norm="ortho")


class EventStage(nn.Module):
    def __init__(self, channels, groups, neighbors):
        super().__init__()
        self.group = LocalGroup(groups, neighbors)
        self.spatial = FrequencyFilter(channels, neighbors)
        self.aggregate = nn.Linear(channels, 1)
        self.temporal = FrequencyFilter(groups, channels)
        self.residual = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1), nn.BatchNorm1d(channels // 2), nn.GELU(),
            nn.Conv1d(channels // 2, channels, 1), nn.BatchNorm1d(channels))

    def forward(self, coords, features):
        coords, grouped = self.group(coords, features)
        b, n, k, c = grouped.shape
        grouped = grouped.permute(0, 1, 3, 2).reshape(b * n, c, k)
        grouped = F.gelu(self.spatial(grouped)).transpose(1, 2)
        weights = self.aggregate(grouped).softmax(dim=1)
        pooled = (weights * grouped).sum(1).reshape(b, n, c)
        features = F.gelu(self.temporal(pooled).transpose(1, 2))
        features = F.gelu(features + self.residual(features)).transpose(1, 2)
        return coords, features


class SECNetEncoder(nn.Module):
    def __init__(self, embed_dim=64, groups=(1024, 512, 256), neighbors=24):
        super().__init__()
        if len(groups) != 3 or min(groups) < 1 or embed_dim < 1 or neighbors < 1:
            raise ValueError("require three positive group counts, width and neighbors")
        if any(groups[i + 1] > groups[i] or neighbors > groups[i] for i in range(2)):
            raise ValueError("invalid downstream groups/neighbors")
        channels = [embed_dim]
        for _ in groups:
            channels.append(2 * channels[-1] + 4)
        self.stage_channels = tuple(channels[1:])
        self.embedding = nn.Sequential(nn.Conv1d(4, embed_dim, 1),
                                       nn.BatchNorm1d(embed_dim), nn.GELU())
        self.stages = nn.ModuleList(EventStage(c, n, neighbors)
                                    for c, n in zip(self.stage_channels, groups))

    def forward_features(self, events):
        if events.ndim != 3 or events.shape[1] != 4:
            raise ValueError("events must be [B,4,N] in x,y,t,p order")
        coords = events.transpose(1, 2)
        features = self.embedding(events).transpose(1, 2)
        outputs, coordinates = [], []
        for stage in self.stages:
            coords, features = stage(coords, features)
            outputs.append(features)
            coordinates.append(coords)
        return outputs, coordinates

    def forward(self, events):
        return self.forward_features(events)
