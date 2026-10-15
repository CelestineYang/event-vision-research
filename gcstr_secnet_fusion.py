"""Custom prototype fusion, NOT an official EV-CGNet implementation.

Pre-normalized Q/K/V multihead attention, no internal residual or extra FFN.
Only stage residual: H = A + MLP(cat(Z_xy, Z_xt, Z_ty)). No point feedback.
Position embeddings provide axis information, not explicit geometric alignment.
"""
import torch
from torch import nn

from gcstr_models import GCSTRBranchClassifier
from secnet_encoder import SECNetEncoder

VIEWS = ("cstr3", "xt_p", "ty_p")


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)

    def forward(self, query, source):
        source = self.source_norm(source)
        return self.attention(self.query_norm(query), source, source, need_weights=False)[0]


class PrototypeStage(nn.Module):
    def __init__(self, channels, dim=128, prototypes=16, heads=4):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(prototypes, dim) * 0.02)
        self.adapter = nn.Linear(channels, dim)
        self.position = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.point_attention = CrossAttention(dim, heads)
        self.view_attention = nn.ModuleDict({v: CrossAttention(dim, heads) for v in VIEWS})
        self.fusion = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(),
                                    nn.Linear(2 * dim, dim))

    def forward(self, features, coords, views):
        source = self.adapter(features) + self.position(coords)
        queries = self.prototypes[None].expand(features.shape[0], -1, -1)
        point_prototypes = self.point_attention(queries, source)
        projections = [self.view_attention[v](point_prototypes, views[v]) for v in VIEWS]
        return (point_prototypes + self.fusion(torch.cat(projections, -1))).mean(1)


class GCSTRSECNetFusion(nn.Module):
    def __init__(self, num_classes=200, dim=128, prototypes=16, heads=4,
                 point_encoder=None, freeze_images=True):
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads or prototypes < 1:
            raise ValueError("positive dimensions required; dim must be divisible by heads")
        self.image_encoders = nn.ModuleDict({v: GCSTRBranchClassifier(
            v, num_classes, pretrained=False) for v in VIEWS})
        # No network downloads: load the existing task-trained checkpoints explicitly.
        self.view_adapters = nn.ModuleDict({v: nn.Linear(512, dim) for v in VIEWS})
        # Independent per-view position networks distinguish row/column axis semantics:
        # cstr3=(x,y), xt_p=(x,t), ty_p=(t,y), matching gcstr_representations.py.
        self.view_positions = nn.ModuleDict({v: nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim)) for v in VIEWS})
        self.point_encoder = point_encoder if point_encoder is not None else SECNetEncoder()
        self.interactions = nn.ModuleList(PrototypeStage(c, dim, prototypes, heads)
                                         for c in self.point_encoder.stage_channels)
        self.classifier = nn.Linear(3 * dim, num_classes)
        self.freeze_images = freeze_images
        for parameter in self.image_encoders.parameters():
            parameter.requires_grad_(not freeze_images)
        # Unused legacy image classification heads are never optimized.
        for branch in self.image_encoders.values():
            branch.backbone.fc.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_images:
            self.image_encoders.eval()
        return self

    def load_image_checkpoints(self, paths):
        if set(paths) != set(VIEWS):
            raise ValueError("supply exactly cstr3, xt_p and ty_p checkpoints")
        for view in VIEWS:
            checkpoint = torch.load(paths[view], map_location="cpu", weights_only=False)
            if checkpoint.get("view", view) != view:
                raise ValueError(f"checkpoint view mismatch: {view}")
            self.image_encoders[view].load_state_dict(checkpoint["model_state"], strict=True)

    def image_tokens(self, view, image):
        net = self.image_encoders[view].backbone
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_images):
            x = net.maxpool(net.relu(net.bn1(net.conv1(image))))
            x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        height, width = x.shape[-2:]
        rows, cols = torch.meshgrid(
            (torch.arange(height, device=x.device, dtype=x.dtype) + 0.5) / height,
            (torch.arange(width, device=x.device, dtype=x.dtype) + 0.5) / width,
            indexing="ij")
        position = torch.stack((rows, cols), -1).reshape(1, height * width, 2)
        return self.view_adapters[view](x.flatten(2).transpose(1, 2)) + self.view_positions[view](position)

    def forward_batch(self, batch):
        return self(batch["events"], {v: batch[v] for v in VIEWS})

    def forward(self, events, views):
        tokens = {v: self.image_tokens(v, views[v]) for v in VIEWS}
        features, coords = self.point_encoder.forward_features(events)
        vectors = [module(f, c, tokens) for module, f, c in zip(self.interactions, features, coords)]
        return self.classifier(torch.cat(vectors, -1))
