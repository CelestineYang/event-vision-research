"""Feature-level B/G/GM adapters. Historical Fusion A sources remain untouched."""
import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

from gcstr_secnet_fusion import CrossAttention, VIEWS

ARCHITECTURE = 'feature_gate_memory_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def original_tokens(stage, features, coords, views):
    source = stage.adapter(features) + stage.position(coords)
    queries = stage.prototypes[None].expand(features.shape[0], -1, -1)
    point = stage.point_attention(queries, source)
    projections = [stage.view_attention[v](point, views[v]) for v in VIEWS]
    return point, point + stage.fusion(torch.cat(projections, -1))


class FrameGate(nn.Module):
    def __init__(self, stage):
        super().__init__()
        dim = stage.prototypes.shape[-1]
        # Copy values, NOT module references. Q remains registered only in base.
        self.attention = copy.deepcopy(stage.view_attention)
        self.merge = copy.deepcopy(stage.fusion)
        self.point_norm = nn.LayerNorm(dim)
        self.frame_norm = nn.LayerNorm(dim)
        self.history_norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(3 * dim + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        self.delta = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        for layer in (self.gate[-1], self.delta[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def frame(self, queries, views):
        return self.merge(torch.cat([self.attention[v](queries, views[v]) for v in VIEWS], -1))

    def forward(self, point, frame, history, history_valid, dt):
        p, f = self.point_norm(point), self.frame_norm(frame)
        r = self.history_norm(history) * history_valid[:, None, None]
        g = self.gate(torch.cat((p.mean(1), f.mean(1), r.mean(1), dt), -1)).sigmoid()
        return self.delta((1 - g[:, :, None]) * p + g[:, :, None] * f), g, r


class FeatureMemory(nn.Module):
    def __init__(self, stages, slots, dim, heads):
        super().__init__()
        self.read = nn.ModuleList(CrossAttention(dim, heads) for _ in range(stages))
        self.output = nn.ModuleList(nn.Linear(dim, dim) for _ in range(stages))
        for layer in self.output:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.query = nn.Parameter(torch.randn(slots, dim) * .02)
        self.stage_embedding = nn.Parameter(torch.randn(stages, dim) * .02)
        self.state_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.candidate_norm = nn.LayerNorm(dim)
        self.write = CrossAttention(dim, heads)
        self.write_gate = nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.Sigmoid())

    def read_stage(self, index, h0, state):
        # Never execute all-masked attention; zero-history rows bypass it entirely.
        idx = state['history_valid'].nonzero(as_tuple=True)[0]
        result = torch.zeros_like(h0)
        if idx.numel():
            result = result.index_copy(0, idx, self.read[index](h0[idx], state['memory'][idx]))
        return result

    def write_state(self, tokens, state, dt):
        old = state['memory']
        normalized = self.state_norm(old)
        candidates = torch.cat([self.token_norm(h) + self.stage_embedding[i]
                                for i, h in enumerate(tokens)], 1)
        c = self.write(self.query[None] + normalized, candidates)
        u = self.write_gate(torch.cat((normalized, self.candidate_norm(c),
                                      dt[:, None].expand(-1, old.shape[1], -1)), -1))
        return {'memory': (1 - u) * old + u * c,
                'history_valid': torch.ones_like(state['history_valid'])}, u


def detach_state(state):
    return {k: v.detach() for k, v in state.items()}


class FeatureMemoryFusion(nn.Module):
    def __init__(self, base, mode='GM', seed=0):
        super().__init__()
        if mode not in ('B', 'G', 'GM'):
            raise ValueError('Expected B/G/GM')
        self.base, self.mode = base, mode
        self.slots, self.dim = base.interactions[0].prototypes.shape
        # Independent RNG streams preserve common initialization AND caller RNG.
        if mode != 'B':
            with torch.random.fork_rng(devices=[]):
                # CPU-only initialization: torch.manual_seed would also reseed CUDA,
                # which fork_rng(devices=[]) intentionally does not snapshot.
                torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed + 41001).get_state())
                self.extra = nn.ModuleList(FrameGate(stage) for stage in base.interactions)
        if mode == 'GM':
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed + 41002).get_state())
                heads = base.interactions[0].point_attention.attention.num_heads
                self.memory = FeatureMemory(len(base.interactions), self.slots, self.dim, heads)
        base.freeze_images = True
        base.image_encoders.requires_grad_(False)
        base.point_encoder.requires_grad_(True)
        if hasattr(base, 'freeze_points'):
            base.freeze_points = False
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'base'):
            self.base.image_encoders.eval()
            for module in self.base.point_encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()  # affine remains trainable, including singleton tails
        return self

    def initial_state(self, batch_size, device):
        return {'memory': torch.zeros(batch_size, self.slots, self.dim, device=device),
                'history_valid': torch.zeros(batch_size, dtype=torch.bool, device=device)}

    def forward_window(self, batch, state=None):
        n = len(batch['events'])
        device = batch['events'].device
        if state is None:
            state = self.initial_state(n, device)
        valid = batch.get('valid', torch.ones(n, dtype=torch.bool, device=device))
        idx = valid.nonzero(as_tuple=True)[0]
        classes = self.base.classifier.out_features
        empty = batch['events'].new_zeros(n, classes)
        if not idx.numel():
            return {'logits': empty, 'point_logits': empty.clone(), 'diagnostics': {}}, state
        # Padding never reaches image/point encoders, BatchNorm, or attention.
        tokens = self.base.encode_views({v: batch[v][idx] for v in VIEWS})
        features, coords = self.base.point_encoder.forward_features(batch['events'][idx])
        point_logits = self.base.point_encoder.classify_features(features)
        local = {k: v[idx] for k, v in state.items()}
        dt_raw = batch['gap_seconds'][idx]
        if not torch.isfinite(dt_raw).all() or (dt_raw < 0).any():
            raise ValueError('Nonfinite/negative real time gap')
        dt = dt_raw.log1p().reshape(-1, 1)
        hs, diagnostics = [], {}
        for i, (stage, feat, xyz) in enumerate(zip(self.base.interactions, features, coords)):
            p, h0 = original_tokens(stage, feat, xyz, tokens)
            h = h0
            if self.mode != 'B':
                frame = self.extra[i].frame(stage.prototypes[None].expand(len(idx), -1, -1), tokens)
                r = self.memory.read_stage(i, h0, local) if self.mode == 'GM' else torch.zeros_like(h0)
                history = local['history_valid'] if self.mode == 'GM' else torch.zeros_like(local['history_valid'])
                delta, g, rnorm = self.extra[i](p, frame, r, history, dt)
                memory_delta = (self.memory.output[i](rnorm) * history[:, None, None]
                                if self.mode == 'GM' else torch.zeros_like(h0))
                h = h0 + delta + memory_delta
                diagnostics[f'stage{i}/gate'] = g.detach().flatten()
                diagnostics[f'stage{i}/residual_ratio'] = ((h-h0).flatten(1).norm(dim=1) /
                    h0.flatten(1).norm(dim=1).clamp_min(1e-12)).detach()
                diagnostics[f'stage{i}/read_norm'] = r.detach().flatten(1).norm(dim=1)
                diagnostics[f'stage{i}/no_history_read_norm'] = r.detach()[~history].flatten(1).norm(dim=1)
            hs.append(h)
        logits = self.base.classifier(torch.cat([h.mean(1) for h in hs], -1))
        next_state = state
        if self.mode == 'GM':
            updated, write_gate = self.memory.write_state(hs, local, dt)
            next_state = {k: state[k].index_copy(0, idx, value) for k, value in updated.items()}
            diagnostics['memory/write_gate'] = write_gate.detach().flatten()
            diagnostics['memory/state_norm'] = updated['memory'].detach().flatten(1).norm(dim=1)
        return {'logits': empty.index_copy(0, idx, logits),
                'point_logits': empty.index_copy(0, idx, point_logits),
                'diagnostics': diagnostics}, next_state

    def parameter_groups(self, config):
        groups = {'point': [], 'existing_fusion': [], 'new_modules': []}
        names = {key: [] for key in groups}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            group = ('point' if name.startswith('base.point_encoder.') else
                     'existing_fusion' if name.startswith('base.') else 'new_modules')
            groups[group].append(parameter)
            names[group].append(name)
        result = [dict(params=params, lr=config['lr_' + key], name=key)
                  for key, params in groups.items() if params]
        ids = [id(p) for group in result for p in group['params']]
        if len(ids) != len(set(ids)):
            raise RuntimeError('Duplicate optimizer parameter')
        return result, names


def load_model(config):
    # Lazy CUDA-extension imports: synthetic CPU tests do not substitute these.
    from eventmamba_v2_pretrained import PretrainedEventMambaV2
    from ucf101_pretrained_fusion import PretrainedWindowFusion
    path = Path(config['source_checkpoint'])
    if sha256(path) != config['source_sha256']:
        raise ValueError('Source checkpoint hash mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != 40:
        raise ValueError('Expected full Fusion A epoch40')
    for filename, expected in checkpoint['config']['code'].items():
        if sha256(Path(__file__).parent / filename) != expected:
            raise ValueError(f'Protected source changed: {filename}')
    base = PretrainedWindowFusion(PretrainedEventMambaV2(num_classes=config['num_classes']))
    if config['num_classes'] != 101:
        base.classifier = nn.Linear(base.classifier.in_features, config['num_classes'])
    state = {}
    for key, value in checkpoint['model_state'].items():
        name = key.removeprefix('module.').removeprefix('fusion.')
        if name in state:
            raise ValueError('Checkpoint prefix collision')
        state[name] = value
    base.load_state_dict(state, strict=True)
    # Verify copies exactly, not approximate logits or permissive missing keys.
    for key, value in base.state_dict().items():
        if not torch.equal(value, state[key]):
            raise RuntimeError(f'Weight restoration failed: {key}')
    model = FeatureMemoryFusion(base, config['mode'], config['seed'])
    model.source_config = checkpoint['config']
    return model
