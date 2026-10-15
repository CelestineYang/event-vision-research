"""Experimental coordinate-aligned EventMamba/gCSTR fusion (experiment B).

Use last-stage points and frozen ResNet layer3 maps. Event coordinates are t,x,y;
gCSTR array axes are (x,y), (x,t), (t,y). SciPy zoom(grid_mode=False) preserves
array endpoints; ResNet layer3 has stride 16 with feature centers at input pixel
0,16,... . These conventions are encoded explicitly below. This is a local
correspondence prior, not an assertion of one-event/one-feature correspondence.
"""
import torch
from torch import nn
from torch.nn import functional as F

from eventmamba_encoder import EventMambaEncoder
from gcstr_models import GCSTRBranchClassifier
from gcstr_representations import GCSTRConfig
from gcstr_secnet_fusion import CrossAttention, GCSTRSECNetFusion, VIEWS


def projected_pixels(coords, view, config):
    """Return resized-image (row,column) pixels for normalized [t,x,y] points."""
    if coords.shape[-1] != 3:
        raise ValueError('Expected t,x,y coordinates')
    t, x, y = coords.unbind(-1)
    # Undo sample_event_cloud's division by W/H (not W-1/H-1).
    x = (x * config.sensor_width).round().clamp(0, config.sensor_width - 1)
    y = (y * config.sensor_height).round().clamp(0, config.sensor_height - 1)
    t = (t * config.time_bins).floor().clamp(0, config.time_bins - 1)
    axis = {'x': x / max(config.sensor_width - 1, 1),
            'y': y / max(config.sensor_height - 1, 1),
            't': t / max(config.time_bins - 1, 1)}
    axes = {'cstr3': ('x', 'y'), 'xt_p': ('x', 't'), 'ty_p': ('t', 'y')}
    if view not in axes:
        raise ValueError(f'Unsupported view: {view}')
    row, col = axes[view]
    return torch.stack((axis[row], axis[col]), -1) * (config.image_size - 1)


def sample_local_patches(feature_map, pixels, radius=1, stride=16):
    """[B,C,H,W] -> [B,N,(2r+1)^2,C]; offsets are feature-map cells."""
    _, _, height, width = feature_map.shape
    center = pixels / stride
    row = center[..., 0].clamp(0, height - 1)
    col = center[..., 1].clamp(0, width - 1)
    offsets = torch.arange(-radius, radius + 1, device=pixels.device, dtype=pixels.dtype)
    dr, dc = torch.meshgrid(offsets, offsets, indexing='ij')
    row = row[..., None] + dr.flatten()
    col = col[..., None] + dc.flatten()
    # grid_sample expects (column,row), unlike gCSTR's semantic coordinate axes.
    grid = torch.stack((2 * col / max(width - 1, 1) - 1,
                        2 * row / max(height - 1, 1) - 1), -1)
    sampled = F.grid_sample(feature_map, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
    return sampled.permute(0, 2, 3, 1)


class GCSTRLocalEventMambaFusion(GCSTRSECNetFusion):
    def __init__(self, num_classes=200, dim=128, heads=4, point_encoder=None,
                 point_aux=False, config=GCSTRConfig()):
        # Reuse checkpoint loading and frozen-image train()/forward_batch() only.
        nn.Module.__init__(self)
        if dim < 1 or heads < 1 or dim % heads:
            raise ValueError('dim must be positive and divisible by heads')
        self.config = config
        self.point_encoder = EventMambaEncoder() if point_encoder is None else point_encoder
        self.image_encoders = nn.ModuleDict({v: GCSTRBranchClassifier(v, num_classes, pretrained=False)
                                            for v in VIEWS})
        self.freeze_images = True
        self.image_encoders.requires_grad_(False)
        self.view_adapters = nn.ModuleDict({v: nn.Conv2d(256, dim, 1) for v in VIEWS})
        self.point_adapter = nn.Linear(self.point_encoder.stage_channels[-1], dim)
        self.point_position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.local_attention = nn.ModuleDict({v: CrossAttention(dim, heads) for v in VIEWS})
        self.offset_positions = nn.ModuleDict({v: nn.Linear(2, dim) for v in VIEWS})
        offsets = torch.tensor([(r, c) for r in (-1., 0., 1.) for c in (-1., 0., 1.)])
        self.register_buffer('patch_offsets', offsets)
        self.classifier = nn.Linear(dim, num_classes)
        self.point_head = (nn.Sequential(nn.LayerNorm(self.point_encoder.stage_channels[-1]),
                                        nn.Linear(self.point_encoder.stage_channels[-1], num_classes))
                           if point_aux else None)
        self.train(self.training)

    def encode_views(self, views):
        outputs = {}
        for v in VIEWS:
            image = views[v]
            if image.shape[-2:] != (self.config.image_size, self.config.image_size):
                raise ValueError('Image size does not match projection config')
            net = self.image_encoders[v].backbone
            with torch.no_grad():
                x = net.maxpool(net.relu(net.bn1(net.conv1(image))))
                x = net.layer3(net.layer2(net.layer1(x)))
            outputs[v] = self.view_adapters[v](x)
        return outputs

    def fuse_features(self, features, coords, views):
        p = self.point_adapter(features[-1])
        centers = coords[-1]
        batch, points, channels = p.shape
        query = (p + self.point_position(centers)).reshape(batch * points, 1, channels)
        updates = []
        for v in VIEWS:
            patches = sample_local_patches(views[v], projected_pixels(centers, v, self.config))
            patches = patches + self.offset_positions[v](self.patch_offsets)[None, None]
            updates.append(self.local_attention[v](query, patches.reshape(batch * points, 9, channels))
                           .reshape(batch, points, channels))
        # Fixed equal view weights and fixed residual scale in the first ablation.
        h = p + 0.1 * torch.stack(updates).mean(0)
        return self.classifier(h.mean(1))

    def forward_outputs(self, events, views):
        maps = self.encode_views(views)
        features, coords = self.point_encoder.forward_features(events)
        return {'logits': self.fuse_features(features, coords, maps),
                'point_logits': None if self.point_head is None else self.point_head(features[-1].mean(1))}

    def forward_batch_outputs(self, batch):
        return self.forward_outputs(batch['events'], {v: batch[v] for v in VIEWS})

    def forward(self, events, views):
        return self.forward_outputs(events, views)['logits']
