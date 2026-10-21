"""Probability-space routing on five frozen experts; state is explicit and causal."""
import math

import torch
from torch import nn

from routing_data import OBSERVATIONS

FIXED = {
    'fusion': (0.5, 0.0),
    'frame': (1.0, 1.0),
    'frame_point': (0.5, 1.0),
    'fusion_frame': (1.0, 0.5),
    'three_way': (0.5, 2.0/3.0),
}


def log_weight(x):
    return x.clamp_min(torch.finfo(x.dtype).tiny).log().masked_fill(x == 0, -torch.inf)


def mix_log_probs(logits, alpha, eta):
    """Arithmetic probability mixture, returned in stable log space, T=1.

    Logits shape [...,5,C], alpha/eta shape [...] or scalar. Three image
    probabilities are averaged, NOT their logits. Supports exact endpoints.
    """
    if logits.shape[-2] != 5 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite [...,5,C] logits')
    alpha, eta = [torch.as_tensor(x, dtype=logits.dtype, device=logits.device) for x in (alpha, eta)]
    if any(not torch.isfinite(x).all() or ((x<0)|(x>1)).any() for x in (alpha, eta)):
        raise ValueError('Mixture weights must lie in [0,1]')
    lp = logits.log_softmax(-1)
    frame = torch.logsumexp(lp[..., 2:, :], dim=-2)-math.log(3)
    branch = torch.logaddexp(log_weight(alpha)[..., None]+frame,
                            log_weight(1-alpha)[..., None]+lp[..., 1, :])
    result = torch.logaddexp(log_weight(1-eta)[..., None]+lp[..., 0, :],
                            log_weight(eta)[..., None]+branch)
    return result-result.logsumexp(-1, keepdim=True)


def inverse_sigmoid(value):
    value = min(max(float(value), 1e-6), 1-1e-6)
    return math.log(value/(1-value))


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.mode = config.get('mode', 'fixed')
        self.dimensions = len(OBSERVATIONS)
        self.hidden = int(config.get('hidden', 64))
        self.alpha0, self.eta0 = float(config.get('alpha0', .5)), float(config.get('eta', .5))
        if self.mode not in ('fixed', 'static', 'current', 'ema', 'gru', 'conditional'):
            raise ValueError('Unknown router mode')
        if self.hidden<1 or not 0<=self.alpha0<=1 or not 0<=self.eta0<=1:
            raise ValueError('Invalid hidden size/initial mixture')
        self.restricted = bool(config.get('fusion_frame_only', False))
        self.register_buffer('base_logit', torch.tensor(inverse_sigmoid(self.alpha0)))
        self.register_buffer('fixed_eta', torch.tensor(self.eta0))
        self.register_buffer('fixed_alpha', torch.tensor(self.alpha0))
        if self.mode == 'fixed':
            self.alpha0, self.eta0 = FIXED[config.get('fixed', 'fusion')]
            self.fixed_alpha.fill_(self.alpha0)
            self.fixed_eta.fill_(self.eta0)
        elif self.mode == 'static':
            if not self.restricted:
                self.alpha_logit = nn.Parameter(torch.tensor(inverse_sigmoid(self.alpha0)))
            self.eta_logit = nn.Parameter(torch.tensor(inverse_sigmoid(self.eta0)))
        else:
            if self.restricted:
                raise ValueError('fusion_frame_only is a static restriction')
            self.current = nn.Sequential(nn.Linear(self.dimensions, self.hidden), nn.Tanh(), nn.Linear(self.hidden, 1))
            nn.init.zeros_(self.current[-1].weight)
            nn.init.zeros_(self.current[-1].bias)
            if self.mode == 'gru':
                self.cell = nn.GRUCell(self.dimensions, self.hidden)
                self.read = nn.Linear(self.hidden, 1)
            elif self.mode == 'conditional':
                self.read = nn.Sequential(nn.Linear(self.dimensions+self.hidden, self.hidden), nn.Tanh(), nn.Linear(self.hidden, 1))
                # At hidden=64, GRU and conditional variants have similar total
                # parameter counts; both use exactly the same observations.
                self.write = nn.Sequential(nn.Linear(self.dimensions+self.hidden+1, self.hidden),
                                           nn.Tanh(), nn.Linear(self.hidden, self.hidden))
                self.candidate = nn.Sequential(nn.Linear(self.dimensions, self.hidden), nn.Tanh())
            if self.mode in ('gru', 'conditional'):
                last = self.read if isinstance(self.read, nn.Linear) else self.read[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    @property
    def sequence(self):
        return self.mode in ('ema', 'gru', 'conditional')

    def initial_state(self, batch, reference):
        if self.mode == 'ema':
            return reference.new_full((batch, 1), self.alpha0)
        if self.mode in ('gru', 'conditional'):
            return reference.new_zeros((batch, self.hidden))
        return None

    def weights(self):
        if self.mode == 'static':
            alpha = self.fixed_alpha.new_tensor(1.) if self.restricted else self.alpha_logit.sigmoid()
            return alpha, self.eta_logit.sigmoid()
        return self.fixed_alpha, self.fixed_eta

    def forward(self, logits, observation, mask, dt, state=None, *, ema_decay=None):
        if logits.ndim != 4 or observation.shape != (*logits.shape[:2], self.dimensions):
            raise ValueError('Expected [B,T,5,C] and [B,T,D]')
        if mask.dtype != torch.bool or mask.shape != logits.shape[:2] or dt.shape != mask.shape:
            raise ValueError('Invalid sequence mask/time gaps')
        if not torch.isfinite(observation).all() or not torch.isfinite(dt).all() or (dt<0).any():
            raise ValueError('Invalid observations/time gaps')
        if self.mode in ('fixed', 'static'):
            alpha, eta = self.weights()
            alpha = alpha.expand_as(mask)
        else:
            use_ema = self.mode == 'ema' or ema_decay is not None
            if use_ema and self.mode not in ('ema', 'current'):
                raise ValueError('EMA is only an evaluation option for current Gate')
            decay = float(self.config.get('ema_decay', .9) if ema_decay is None else ema_decay)
            if use_ema and not 0<=decay<1:
                raise ValueError('EMA decay must be fixed in [0,1)')
            if use_ema:
                if state is None:
                    state = observation.new_full((len(logits), 1), self.alpha0)
            elif state is None:
                state = self.initial_state(len(logits), observation)
            alpha_values = []
            for t in range(logits.shape[1]):
                obs = observation[:, t]
                delta = self.current(obs).squeeze(-1)
                proposed = state
                if self.mode == 'gru':
                    # GRU processes only current/past observations, never future windows.
                    proposed = self.cell(obs, state)
                    delta = delta+self.read(proposed).squeeze(-1)
                elif self.mode == 'conditional':
                    # READ OLD state first; current write affects only later windows.
                    delta = delta+self.read(torch.cat((obs, state), -1)).squeeze(-1)
                    time_gap = torch.log1p(dt[:, t:t+1])
                    update = self.write(torch.cat((obs, state, time_gap), -1)).sigmoid()
                    proposed = (1-update)*state+update*self.candidate(obs)
                a = (self.base_logit+delta).sigmoid()
                # Exact boundary alpha has no routing freedom; keep it exact and
                # keep a zero-gradient graph rather than secretly perturbing it.
                if self.alpha0 in (0., 1.):
                    a = a*0+self.alpha0
                if use_ema:
                    proposed = decay*state+(1-decay)*a[:, None]
                    a = proposed.squeeze(-1)
                if proposed is not None:
                    state = torch.where(mask[:, t:t+1], proposed, state)
                alpha_values.append(torch.where(mask[:, t], a, self.fixed_alpha))
            alpha = torch.stack(alpha_values, 1)
            eta = self.fixed_eta
        return dict(log_probs=mix_log_probs(logits, alpha, eta), alpha=alpha,
                    eta=eta.expand_as(alpha), state=state)


def window_nll(log_probs, labels, mask):
    if not mask.any():
        raise ValueError('No valid windows')
    selected = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return selected[mask].mean()


def degeneracies(router):
    result = []
    if router.eta0 < .01:
        result.append('eta<0.01: alpha has little/no influence on final probabilities')
    if router.alpha0 in (0., 1.) and router.mode not in ('fixed', 'static'):
        result.append('boundary alpha0: dynamic correction is inactive to preserve exact initialization')
    if router.mode == 'static':
        result.append('static endpoint initializations are clamped to [1e-6,1-1e-6] before logit')
    return result
