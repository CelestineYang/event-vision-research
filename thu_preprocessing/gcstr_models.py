"""ResNet-18 branches and PaSNet ensemble for the gCSTR reproduction."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_gcstr_resnet18(num_classes: int = 200, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class GCSTRBranchClassifier(nn.Module):
    def __init__(
        self, view: str, num_classes: int = 200, pretrained: bool = True
    ) -> None:
        super().__init__()
        self.view = view
        self.backbone = build_gcstr_resnet18(num_classes, pretrained)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self(batch[self.view])


class PaSNetProbabilityEnsemble(nn.Module):
    """Frozen equal-weight product ensemble, evaluated in log space."""

    def __init__(self, branches: Mapping[str, nn.Module]) -> None:
        super().__init__()
        if not branches:
            raise ValueError("at least one branch is required")
        self.branches = nn.ModuleDict(dict(branches))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        log_probabilities = [
            torch.log_softmax(model(batch[view]), dim=1)
            for view, model in self.branches.items()
        ]
        return torch.stack(log_probabilities, dim=0).sum(dim=0)
