"""M: H = H0 + memory residual; no independent frame reader or PF Gate."""
import torch
from torch import nn

from gcstr_feature_memory_fusion import FeatureMemoryFusion, FeatureMemory, original_tokens
from gcstr_secnet_fusion import VIEWS

ARCHITECTURE = 'feature_memory_only_v1'


class MemoryOnlyFusion(FeatureMemoryFusion):
    def __init__(self, base, seed=0):
        # B initializes no FrameGate. Keep the same base freezing policy.
        super().__init__(base, mode='B', seed=seed)
        self.mode = 'M'
        self.ablation_id = 'memory_only'
        self.history_norm = nn.ModuleList(nn.LayerNorm(self.dim) for _ in base.interactions)
        # Identical memory initialization to GM, without consuming caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed + 41002).get_state())
            heads = base.interactions[0].point_attention.attention.num_heads
            self.memory = FeatureMemory(len(base.interactions), self.slots, self.dim, heads)

    def forward_window(self, batch, state=None):
        n, device = len(batch['events']), batch['events'].device
        if state is None:
            state = self.initial_state(n, device)
        valid = batch.get('valid', torch.ones(n, dtype=torch.bool, device=device))
        idx = valid.nonzero(as_tuple=True)[0]
        empty = batch['events'].new_zeros(n, self.base.classifier.out_features)
        if not idx.numel():
            return dict(logits=empty, point_logits=empty.clone(), diagnostics={}), state
        views = self.base.encode_views({v: batch[v][idx] for v in VIEWS})
        features, coords = self.base.point_encoder.forward_features(batch['events'][idx])
        point_logits = self.base.point_encoder.classify_features(features)
        local = {k: v[idx] for k, v in state.items()}
        gap = batch['gap_seconds'][idx]
        if not torch.isfinite(gap).all() or (gap < 0).any():
            raise ValueError('Nonfinite/negative time gap')
        dt = gap.log1p().reshape(-1, 1)
        history = local['history_valid']
        hs, diagnostics = [], {}
        for i, (stage, feat, xyz) in enumerate(zip(self.base.interactions, features, coords)):
            _, h0 = original_tokens(stage, feat, xyz, views)
            read = self.memory.read_stage(i, h0, local)
            rnorm = self.history_norm[i](read) * history[:, None, None]
            # Mask after projection as well: its learned bias must not leak into
            # the first window of a recording.
            delta = self.memory.output[i](rnorm) * history[:, None, None]
            h = h0 + delta
            hs.append(h)
            ratio = delta.flatten(1).norm(dim=1) / h0.flatten(1).norm(dim=1).clamp_min(1e-12)
            diagnostics[f'stage{i}/memory_residual_ratio'] = ratio.detach()
            diagnostics[f'stage{i}/residual_ratio'] = ratio.detach()
            diagnostics[f'stage{i}/read_norm'] = read.detach().flatten(1).norm(dim=1)
            diagnostics[f'stage{i}/no_history_read_norm'] = read.detach()[~history].flatten(1).norm(dim=1)
        logits = self.base.classifier(torch.cat([h.mean(1) for h in hs], -1))
        # Write the ablated H, after forming the current prediction. Never use
        # the removed PF path to construct future memory candidates.
        updated, u = self.memory.write_state(hs, local, dt)
        next_state = {k: state[k].index_copy(0, idx, v) for k, v in updated.items()}
        diagnostics['memory/write_gate'] = u.detach().flatten()
        diagnostics['memory/state_norm'] = updated['memory'].detach().flatten(1).norm(dim=1)
        return dict(logits=empty.index_copy(0, idx, logits),
                    point_logits=empty.index_copy(0, idx, point_logits),
                    diagnostics=diagnostics), next_state
