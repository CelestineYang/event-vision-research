"""Stage-3 current-window alpha/eta gates; cache schema remains unchanged."""
import torch
from torch import nn

from fusion_router import mix_log_probs
from routing_data import OBSERVATIONS

ARCHITECTURE = 'current_alpha_outer_eta_v3'
MODEL_SCHEMA = 3
LOSSES = ('window_nll', 'video_nll', 'video_balanced_window_nll')


def mlp(seed, hidden):
    # Construction never consumes the caller's RNG or changes the common alpha.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(len(OBSERVATIONS), hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


class RouterV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.eta_mode = config['eta_mode']
        if self.eta_mode not in ('fixed', 'global', 'current'):
            raise ValueError('Unknown eta_mode')
        hidden, seed = int(config.get('hidden', 64)), int(config.get('seed', 0))
        if hidden < 1: raise ValueError('Invalid hidden dimension')
        self.alpha_net = mlp(seed, hidden)
        if self.eta_mode == 'current': self.eta_net = mlp(seed + 100003, hidden)
        self.register_buffer('b_alpha', torch.tensor(0.))
        self.register_buffer('alpha_reference', torch.tensor(.5))
        self.register_buffer('alpha_exact_endpoint', torch.tensor(False))
        self.register_buffer('fixed_eta', torch.tensor(.5))
        if self.eta_mode == 'fixed': self.register_buffer('b_eta', torch.tensor(0.))
        else: self.b_eta = nn.Parameter(torch.tensor(0.))

    @torch.no_grad()
    def initialize_static(self, state):
        a, e = state['alpha_logit'], state['eta_logit']
        if a.numel()!=1 or e.numel()!=1 or a.dtype!=e.dtype or not torch.isfinite(a).all() or not torch.isfinite(e).all():
            raise ValueError('Expected finite raw Static logits')
        self.to(dtype=a.dtype)
        self.b_alpha.copy_(a); self.b_eta.copy_(e)
        self.alpha_reference.copy_(a.sigmoid()); self.fixed_eta.copy_(e.sigmoid())
        # A finite saturated sigmoid is NOT an explicit frozen endpoint.
        self.alpha_exact_endpoint.fill_(False)
        return self

    @torch.no_grad()
    def copy_legacy_current(self, legacy):
        if self.eta_mode!='fixed' or legacy.mode!='current':
            raise ValueError('Regression conversion only: old Current -> fixed eta')
        self.to(dtype=legacy.base_logit.dtype)
        self.alpha_net.load_state_dict(legacy.current.state_dict(), strict=True)
        self.b_alpha.copy_(legacy.base_logit)
        self.alpha_reference.copy_(legacy.fixed_alpha)
        self.alpha_exact_endpoint.copy_(legacy.exact_alpha_endpoint)
        self.fixed_eta.copy_(legacy.fixed_eta)
        return self

    def forward(self, logits, observations, mask, dt, state=None):
        if state is not None: raise ValueError('V3 has no history state')
        if logits.ndim!=4 or observations.shape!=(*logits.shape[:2],len(OBSERVATIONS)):
            raise ValueError('Expected logits[B,T,5,C] and observations[B,T,17]')
        if mask.dtype!=torch.bool or mask.shape!=logits.shape[:2] or dt.shape!=mask.shape:
            raise ValueError('Invalid mask/dt')
        if not torch.isfinite(observations).all() or not torch.isfinite(dt).all() or (dt<0).any():
            raise ValueError('Invalid observations/gaps')
        if observations.shape[1]==0:raise ValueError('Empty sequence')
        # Match legacy Current's per-time GEMM layout as well as its equations.
        # Flattening B*T can introduce FP32 differences near saturated gates.
        da=torch.stack([self.alpha_net(observations[:,t]).squeeze(-1) for t in range(observations.shape[1])],1)
        de=(torch.stack([self.eta_net(observations[:,t]).squeeze(-1) for t in range(observations.shape[1])],1)
            if self.eta_mode=='current' else torch.zeros_like(da))
        alpha=torch.stack([(self.b_alpha+da[:,t]).sigmoid() for t in range(da.shape[1])],1)
        alpha=torch.where(self.alpha_exact_endpoint,self.alpha_reference,alpha)
        alpha=torch.where(mask,alpha,self.alpha_reference)
        eta=self.fixed_eta.expand_as(alpha) if self.eta_mode=='fixed' else (self.b_eta+de).sigmoid()
        eta=torch.where(mask,eta,self.fixed_eta)
        weights=dict(w_A=1-eta,w_F=eta*alpha,w_P=eta*(1-alpha))
        stacked=torch.stack(list(weights.values()),-1)
        if (stacked<0).any() or not torch.allclose(stacked.sum(-1),torch.ones_like(alpha),atol=1e-6,rtol=0):
            raise ValueError('Invalid explicit path weights')
        mix_eta=self.fixed_eta if self.eta_mode=='fixed' else eta
        return dict(log_probs=mix_log_probs(logits,alpha,mix_eta),alpha=alpha,eta=eta,
                    delta_alpha=da,delta_eta=de,state=None,**weights)


def loss_terms(log_probs, labels, mask, mode):
    """Return additive loss terms: windows or videos, with gradients intact."""
    if mode not in LOSSES: raise ValueError('Unknown loss_mode')
    if log_probs.ndim!=3 or log_probs.shape[0]==0 or log_probs.shape[1]==0 or mask.dtype!=torch.bool or labels.shape!=mask.shape or mask.shape!=log_probs.shape[:2]:
        raise ValueError('Loss shape/mask mismatch')
    count=mask.sum(1)
    if (count==0).any(): raise ValueError('Empty valid video')
    first=mask.long().argmax(1)
    target=labels.gather(1,first[:,None]).squeeze(1)
    if (labels[mask]<0).any() or (labels[mask]>=log_probs.shape[-1]).any():raise ValueError('Invalid label')
    if ((labels!=target[:,None]) & mask).any():raise ValueError('Inconsistent video labels')
    if not torch.isfinite(log_probs[mask]).all():raise ValueError('Nonfinite valid log probabilities')
    safe_labels=labels.masked_fill(~mask,0)
    nll=-log_probs.gather(-1,safe_labels[...,None]).squeeze(-1)
    if mode=='window_nll':return nll[mask]
    if mode=='video_balanced_window_nll':return nll.masked_fill(~mask,0).sum(1)/count
    video_lp=torch.logsumexp(log_probs.masked_fill(~mask[...,None],-torch.inf),dim=1)-count.to(log_probs.dtype).log()[:,None]
    return -video_lp.gather(1,target[:,None]).squeeze(1)


def training_objective(log_probs, labels, mask, mode, epoch_windows, epoch_videos, epoch_steps):
    denominator=epoch_windows if mode=='window_nll' else epoch_videos
    if min(denominator,epoch_steps)<1:raise ValueError('Invalid epoch scale')
    return loss_terms(log_probs,labels,mask,mode).sum()*epoch_steps/denominator
