"""Isolated feature ablations; historical B/G/GM implementation stays immutable."""
import torch
from torch import nn

from gcstr_feature_memory_fusion import FeatureMemoryFusion, original_tokens, load_model as load_original
from gcstr_secnet_fusion import VIEWS

ARCHITECTURE = 'feature_gate_memory_ablation_v1'
SCHEMA = 1
INTERVENTIONS = ('normal', 'reset_each_window', 'disable_history_to_gate',
                 'disable_direct_memory_residual', 'disable_both_history_outputs')


def policy(mode, ablation_id='base'):
    if mode not in ('B', 'G', 'GM'):
        raise ValueError('Expected B/G/GM')
    result = dict(frame_query_source='learned_q', gate_policy='learned',
                  history_to_gate=mode == 'GM', history_direct_residual=mode == 'GM',
                  memory_update='learned_gate' if mode == 'GM' else 'none')
    changes = {
        'base': ({'B', 'G', 'GM'}, {}),
        'fixed_gate': ({'G'}, dict(gate_policy='fixed_half')),
        'global_gate': ({'G'}, dict(gate_policy='global_scalar')),
        'point_query': ({'G'}, dict(frame_query_source='point_prototype')),
        'gate_only': ({'GM'}, dict(history_direct_residual=False)),
        'feature_only': ({'GM'}, dict(history_to_gate=False)),
        'no_write_gate': ({'GM'}, dict(memory_update='overwrite')),
        'fixed_update': ({'GM'}, dict(memory_update='fixed_half')),
        'point_complement': ({'G'}, dict(gate_policy='point_only')),
        'frame_complement': ({'G'}, dict(gate_policy='frame_only')),
    }
    if ablation_id not in changes or mode not in changes[ablation_id][0]:
        raise ValueError('Invalid base_mode/ablation_id combination')
    result.update(changes[ablation_id][1])
    return result


def validate_policy(config):
    expected = policy(config['base_mode'], config['ablation_id'])
    if config.get('mode') != config['base_mode']:
        raise ValueError('mode must equal base_mode')
    for key, value in expected.items():
        if key not in config or type(config[key]) is not type(value) or config[key] != value:
            raise ValueError(f'Contradictory policy: {key}')
    return expected


class AblationFusion(FeatureMemoryFusion):
    def __init__(self, base, mode='GM', seed=0, ablation_id='base'):
        settings = policy(mode, ablation_id)
        # Construct in the original order/independent RNG streams FIRST, so
        # disabling modules never shifts initialization of surviving parameters.
        super().__init__(base, mode, seed)
        self.settings, self.ablation_id = settings, ablation_id
        self.intervention = 'normal'
        self.observer = None
        if mode != 'B' and settings['gate_policy'] != 'learned':
            for extra in self.extra:
                extra.gate.requires_grad_(False)
                extra.history_norm.requires_grad_(False)
                if settings['gate_policy'] == 'global_scalar':
                    extra.global_gate_logit = nn.Parameter(torch.zeros(()))
                if settings['gate_policy'] == 'point_only':
                    extra.attention.requires_grad_(False)
                    extra.merge.requires_grad_(False)
                    extra.frame_norm.requires_grad_(False)
                if settings['gate_policy'] == 'frame_only':
                    extra.point_norm.requires_grad_(False)
        if mode == 'GM':
            if not settings['history_direct_residual']:
                self.memory.output.requires_grad_(False)
            if settings['memory_update'] != 'learned_gate':
                self.memory.write_gate.requires_grad_(False)
                self.memory.candidate_norm.requires_grad_(False)

    def set_intervention(self, mode):
        if mode not in INTERVENTIONS or (mode != 'normal' and
                (self.mode != 'GM' or self.ablation_id != 'base')):
            raise ValueError('History interventions require an unmodified GM checkpoint')
        self.intervention = mode

    def write_state(self, tokens, state, dt):
        if self.settings['memory_update'] == 'learned_gate':
            return self.memory.write_state(tokens, state, dt)
        memory = self.memory
        old = state['memory']
        normalized = memory.state_norm(old)
        candidates = torch.cat([memory.token_norm(h) + memory.stage_embedding[i]
                                for i, h in enumerate(tokens)], 1)
        c = memory.write(memory.query[None] + normalized, candidates)
        u = torch.full_like(old, 1. if self.settings['memory_update'] == 'overwrite' else .5)
        return dict(memory=(1-u)*old+u*c,
                    history_valid=torch.ones_like(state['history_valid'])), u

    def forward_window(self, batch, state=None):
        if self.training and self.intervention != 'normal':
            raise ValueError('Inference intervention cannot be used for training')
        n, device = len(batch['events']), batch['events'].device
        if state is None:
            state = self.initial_state(n, device)
        valid = batch.get('valid', torch.ones(n, dtype=torch.bool, device=device))
        idx = valid.nonzero(as_tuple=True)[0]
        empty = batch['events'].new_zeros(n, self.base.classifier.out_features)
        if not idx.numel():
            return dict(logits=empty, point_logits=empty.clone(), diagnostics={}), state
        tokens = self.base.encode_views({v: batch[v][idx] for v in VIEWS})
        features, coords = self.base.point_encoder.forward_features(batch['events'][idx])
        point_logits = self.base.point_encoder.classify_features(features)
        local = {k: v[idx] for k, v in state.items()}
        if self.intervention == 'reset_each_window':
            local = self.initial_state(len(idx), device)
        raw_dt = batch['gap_seconds'][idx]
        if not torch.isfinite(raw_dt).all() or (raw_dt < 0).any():
            raise ValueError('Nonfinite/negative time gap')
        dt = raw_dt.log1p().reshape(-1, 1)
        use_gate = self.settings['history_to_gate'] and self.intervention not in (
            'disable_history_to_gate', 'disable_both_history_outputs')
        use_residual = self.settings['history_direct_residual'] and self.intervention not in (
            'disable_direct_memory_residual', 'disable_both_history_outputs')
        hs, diagnostics = [], {}
        for i, (stage, feat, xyz) in enumerate(zip(self.base.interactions, features, coords)):
            point, h0 = original_tokens(stage, feat, xyz, tokens)
            h = h0
            if self.mode != 'B':
                extra = self.extra[i]
                query = (point if self.settings['frame_query_source'] == 'point_prototype'
                         else stage.prototypes[None].expand(len(idx), -1, -1))
                gate_policy = self.settings['gate_policy']
                frame = extra.frame(query, tokens) if gate_policy != 'point_only' else torch.zeros_like(point)
                p = extra.point_norm(point) if gate_policy != 'frame_only' else torch.zeros_like(point)
                f = extra.frame_norm(frame) if gate_policy != 'point_only' else torch.zeros_like(frame)
                history = local['history_valid'] if self.mode == 'GM' else torch.zeros_like(local['history_valid'])
                r = self.memory.read_stage(i, h0, local) if self.mode == 'GM' else torch.zeros_like(h0)
                # Keep normalized history for the residual; mask only the separate
                # Gate observation when disabling history-to-Gate.
                rnorm = extra.history_norm(r) * history[:, None, None]
                # G/base must retain the original zero-gradient graph (not None):
                # AdamW treats these differently. Only GM interventions sever it.
                observation = rnorm.mean(1) if self.mode != 'GM' or use_gate else torch.zeros_like(rnorm.mean(1))
                if gate_policy == 'learned':
                    g = extra.gate(torch.cat((p.mean(1), f.mean(1), observation, dt), -1)).sigmoid()
                elif gate_policy == 'global_scalar':
                    g = extra.global_gate_logit.sigmoid().expand(len(idx), 1)
                else:
                    g = p.new_full((len(idx), 1), {'fixed_half': .5, 'point_only': 0., 'frame_only': 1.}[gate_policy])
                delta = extra.delta((1-g[:, :, None])*p+g[:, :, None]*f)
                memory_delta = (self.memory.output[i](rnorm)*history[:, None, None]
                                if use_residual else torch.zeros_like(h0))
                h = h0 + delta + memory_delta
                norm = h0.flatten(1).norm(dim=1).clamp_min(1e-12)
                diagnostics[f'stage{i}/gate'] = g.detach().flatten()
                diagnostics[f'stage{i}/pf_residual_ratio'] = (delta.flatten(1).norm(dim=1)/norm).detach()
                diagnostics[f'stage{i}/residual_ratio'] = ((h-h0).flatten(1).norm(dim=1)/norm).detach()
                if self.mode == 'GM':
                    diagnostics[f'stage{i}/memory_residual_ratio'] = (memory_delta.flatten(1).norm(dim=1)/norm).detach()
                    diagnostics[f'stage{i}/gate_history_norm'] = observation.detach().norm(dim=1)
                    diagnostics[f'stage{i}/read_norm'] = r.detach().flatten(1).norm(dim=1)
                    diagnostics[f'stage{i}/no_history_read_norm'] = r.detach()[~history].flatten(1).norm(dim=1)
            hs.append(h)
        logits = self.base.classifier(torch.cat([h.mean(1) for h in hs], -1))
        next_state = state
        if self.mode == 'GM':
            updated, u = self.write_state(hs, local, dt)
            next_state = {k: state[k].index_copy(0, idx, v) for k, v in updated.items()}
            diagnostics['memory/write_gate'] = u.detach().flatten()
            diagnostics['memory/state_norm'] = updated['memory'].detach().flatten(1).norm(dim=1)
        output = dict(logits=empty.index_copy(0, idx, logits),
                      point_logits=empty.index_copy(0, idx, point_logits), diagnostics=diagnostics)
        if self.observer is not None:
            self.observer(batch, output)
        return output, next_state


def load_model(config):
    validate_policy(config)
    original = load_original(dict(config, mode=config['base_mode']))
    model = AblationFusion(original.base, config['base_mode'], config['seed'], config['ablation_id'])
    model.source_config = original.source_config
    return model
