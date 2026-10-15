"""Minimal three-branch baselines for DailyDVS-200 classification."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ConvNormActivation2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConvNormActivation3d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class FrameEncoder(nn.Module):
    """Encode the two-channel accumulated event frame."""

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormActivation2d(2, 16, stride=2),
            ConvNormActivation2d(16, 32, stride=2),
            ConvNormActivation2d(32, output_dim, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(frame)).flatten(1)


class VoxelEncoder(nn.Module):
    """Encode polarity-separated voxel counts while retaining time initially."""

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormActivation3d(2, 8, stride=(1, 2, 2)),
            ConvNormActivation3d(8, 16, stride=(1, 2, 2)),
            ConvNormActivation3d(16, 32, stride=(2, 2, 2)),
            nn.Conv3d(32, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(voxel)).flatten(1)


class PointEncoder(nn.Module):
    """Encode normalized (x, y, t, polarity) points with masked max pooling."""

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 4:
            raise ValueError("points must have shape [batch, num_points, 4]")
        if point_mask.shape != points.shape[:2]:
            raise ValueError("point_mask must match the first two point dimensions")
        features = self.point_mlp(points)
        minimum = torch.finfo(features.dtype).min
        features = features.masked_fill(~point_mask.unsqueeze(-1), minimum)
        pooled = features.amax(dim=1)
        has_points = point_mask.any(dim=1, keepdim=True)
        return torch.where(has_points, pooled, torch.zeros_like(pooled))


class ThreeBranchConcatClassifier(nn.Module):
    """Baseline A: independent encoders followed by late feature concatenation."""

    def __init__(
        self,
        num_classes: int = 200,
        branch_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.frame_encoder = FrameEncoder(branch_dim)
        self.voxel_encoder = VoxelEncoder(branch_dim)
        self.point_encoder = PointEncoder(branch_dim)
        self.classifier = nn.Sequential(
            nn.Linear(branch_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        frame: torch.Tensor,
        voxel: torch.Tensor,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch_features = {
            "frame": self.frame_encoder(frame),
            "voxel": self.voxel_encoder(voxel),
            "points": self.point_encoder(points, point_mask),
        }
        fused = torch.cat(
            [branch_features["frame"], branch_features["voxel"], branch_features["points"]],
            dim=1,
        )
        logits = self.classifier(fused)
        if return_features:
            return logits, branch_features
        return logits

    def forward_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self(
            batch["frame"],
            batch["voxel"],
            batch["points"],
            batch["point_mask"],
            return_features=return_features,
        )


class SingleBranchClassifier(nn.Module):
    """A capacity-matched classification head for F, V, or P baselines."""

    def __init__(
        self,
        branch: str,
        num_classes: int = 200,
        branch_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        encoders = {
            "frame": FrameEncoder,
            "voxel": VoxelEncoder,
            "points": PointEncoder,
        }
        if branch not in encoders:
            raise ValueError(f"unknown branch: {branch}")
        self.branch = branch
        self.encoder = encoders[branch](branch_dim)
        self.classifier = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.branch == "points":
            features = self.encoder(batch["points"], batch["point_mask"])
        else:
            features = self.encoder(batch[self.branch])
        return self.classifier(features)


def build_baseline(
    name: str, num_classes: int = 200, dropout: float = 0.2
) -> nn.Module:
    """Build one of the four initial representation baselines."""
    if name == "concat":
        return ThreeBranchConcatClassifier(num_classes=num_classes, dropout=dropout)
    if name in {"frame", "voxel", "points"}:
        return SingleBranchClassifier(name, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"unknown baseline: {name}")
