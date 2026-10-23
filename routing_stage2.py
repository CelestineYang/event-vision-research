"""Read-only artifact validation and observation/intervention adapters for v2."""
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from fusion_router import Router
from routing_data import (EvidenceCache, ObservationNormalizer, RoutingDataset,
                          digest, validate_training_roles)
from train_fusion_routing import load_checkpoint

MODES = ('static', 'current', 'gru', 'conditional')
INTERVENTIONS = ('normal', 'reset_each_window', 'zero_memory_readout')


def verified_artifacts(sources_path):
    """Validate real server artifacts, not just Git snapshot file names."""
    sources_path = Path(sources_path)
    sources = json.loads(sources_path.read_text())
    paths = {}
    for name, entry in sources.items():
        path = Path(entry['source'])
        if not path.is_file() or digest(path) != entry['sha256']:
            raise ValueError('Missing/changed source: '+name)
        paths[name] = path
    train = EvidenceCache(paths['evidence_train'].parent)
    val = EvidenceCache(paths['evidence_test'].parent, train.identity)
    checkpoints = {}
    for mode in MODES:
        saved = load_checkpoint(paths[f'{mode}/best_val_nll.pt'], train.identity)
        if saved['config']['mode'] != mode:
            raise ValueError('Checkpoint mode mismatch')
        if saved['train_cache'] != train.fingerprint or saved['val_cache'] != val.fingerprint:
            raise ValueError('Checkpoint cache manifest fingerprint mismatch')
        if saved['train_role'] != 'gate_train' or saved['val_role'] != 'gate_val':
            raise ValueError('Unexpected saved training/selection roles')
        if saved['config'] != json.loads(paths[f'{mode}/config.json'].read_text()):
            raise ValueError('Saved config mismatch')
        history = [json.loads(line) for line in paths[f'{mode}/metrics.jsonl'].read_text().splitlines()]
        selected = min(history, key=lambda r:r['validation']['video_nll'])
        if saved['epoch'] != selected['epoch'] or saved['metrics'] != selected:
            raise ValueError('Checkpoint does not reproduce recorded selection')
        checkpoints[mode] = saved
    validate_training_roles(RoutingDataset(train,'gate_train'), RoutingDataset(val,'gate_val'),
                            selection_protocol=train.identity['selection_protocol'])
    static = checkpoints['static']
    for mode in MODES[1:]:
        saved = checkpoints[mode]
        if saved['config']['static_init_sha256'] != sources['static/best_val_nll.pt']['sha256']:
            raise ValueError('Dynamic static initializer changed')
        for key in ('mean','scale'):
            torch.testing.assert_close(saved['normalizer_state'][key],static['normalizer_state'][key],rtol=0,atol=0)
        for key,expected in (
                ('base_logit',static['router_state']['alpha_logit']),
                ('fixed_alpha',static['router_state']['alpha_logit'].sigmoid()),
                ('fixed_eta',static['router_state']['eta_logit'].sigmoid())):
            torch.testing.assert_close(saved['router_state'][key],expected,rtol=0,atol=0)
    report = dict(sources_sha256=digest(sources_path), cache_fingerprints=dict(train=train.fingerprint,test=val.fingerprint),
                  dataset_id=train.identity['dataset_id'], class_to_idx=train.identity['class_to_idx'],
                  observations=val.manifest['observations'], expert_provenance=train.identity['expert_provenance'],
                  selection_protocol=train.identity['selection_protocol'], independent_test=False,
                  windows=dict(train=len(train.rows),test=len(val.rows)), models={})
    report['verified_sources']={name:dict(path=str(paths[name]),sha256=entry['sha256'])
                                for name,entry in sources.items()}
    report['videos']={name:len({(row['dataset_id'],row['source_split'],row['video_id']) for row in cache.rows})
                      for name,cache in [('train',train),('test',val)]}
    for mode,saved in checkpoints.items():
        model, _ = restored(saved)
        alpha,eta = model.weights()
        report['models'][mode] = dict(epoch=saved['epoch'],checkpoint=str(paths[f'{mode}/best_val_nll.pt']),
            sha256=sources[f'{mode}/best_val_nll.pt']['sha256'],
            alpha=float(alpha),eta=float(eta),
            base_logit=float(model.base_logit) if mode!='static' else None,
            alpha_meaning='static value' if mode=='static' else 'initial/reference alpha; dynamic alpha varies')
    return train, val, checkpoints, report


def restored(saved, device='cpu'):
    router = Router(saved['config']).to(dtype=saved['router_state']['base_logit'].dtype)
    router.load_state_dict(saved['router_state'],strict=True)
    normalizer = ObservationNormalizer().to(dtype=saved['normalizer_state']['mean'].dtype)
    normalizer.load_state_dict(saved['normalizer_state'],strict=True)
    return router.to(device).requires_grad_(False).eval(), normalizer.to(device).eval()


class PrefixDataset(Dataset):
    def __init__(self, data, ratio=1.):
        if not data.sequence or not 0 < ratio <= 1:
            raise ValueError('Prefix requires sequences and 0<ratio<=1')
        self.data,self.ratio,self.cache = data,ratio,data.cache

    def __len__(self): return len(self.data)

    def __getitem__(self,index):
        item = self.data[index]
        count = max(1,math.ceil(self.ratio*len(item['rows'])))
        return {key:value[:count] for key,value in item.items()}


@contextmanager
def observers(router, capture, zero_read=False):
    """Temporary hooks; only read-output replacement is an intervention."""
    handles = []
    def record(name, replace=False):
        def hook(module, args, output):
            capture[name] = output.detach()
            return torch.zeros_like(output) if replace else None
        return hook
    try:
        if hasattr(router,'current'):
            handles.append(router.current.register_forward_hook(record('current')))
        if hasattr(router,'read'):
            handles.append(router.read.register_forward_hook(record('memory',zero_read)))
        if hasattr(router,'write'):
            handles.append(router.write.register_forward_hook(record('write')))
        yield
    finally:
        for handle in handles: handle.remove()


def _quantiles(x):
    return dict(write_mean=x.mean(-1),write_p05=torch.quantile(x,.05,dim=-1),
                write_p50=torch.quantile(x,.5,dim=-1),write_p95=torch.quantile(x,.95,dim=-1))


@torch.no_grad()
def forward_diagnostic(router, logits, observation, mask, dt, *, intervention='normal',
                       ema_decay=None, trace=False):
    """Step the unmodified router, preserving original observations and gaps."""
    if intervention not in INTERVENTIONS: raise ValueError('Unknown intervention')
    if intervention=='normal' and not trace:
        return router(logits,observation,mask,dt,state=None,ema_decay=ema_decay), {}
    # reset controls hidden memory only; Current/Static remain negative controls.
    memory = router.mode in ('gru','conditional')
    state=None; outputs=[]; traces=[]
    capture={}
    with observers(router,capture,intervention=='zero_memory_readout' and memory):
        for t in range(logits.shape[1]):
            if intervention=='reset_each_window' and memory: state=None
            old_state=state
            capture.clear()
            out=router(logits[:,t:t+1],observation[:,t:t+1],mask[:,t:t+1],dt[:,t:t+1],
                       state=state,ema_decay=ema_decay)
            state=out['state'];outputs.append(out)
            if trace:
                a,e=out['alpha'][:,0],out['eta'][:,0]
                zeros=torch.zeros_like(a)
                current=capture.get('current',zeros[:,None]).squeeze(-1)
                raw=capture.get('memory',zeros[:,None]).squeeze(-1)
                applied=torch.zeros_like(raw) if intervention=='zero_memory_readout' and memory else raw
                entry=dict(alpha=a,eta=e,w_A=1-e,w_F=e*a,w_P=e*(1-a),
                           delta_current=current,delta_memory=applied,delta_memory_unmasked=raw,
                           state_before_norm=zeros if old_state is None else old_state.norm(dim=-1),
                           state_after_norm=zeros if state is None else state.norm(dim=-1))
                if 'write' in capture:entry.update(_quantiles(capture['write'].sigmoid()))
                traces.append(entry)
    result={key:torch.cat([o[key] for o in outputs],dim=1) for key in ('log_probs','alpha','eta')}
    result['state']=state
    trace_values={key:torch.stack([r[key] for r in traces],dim=1) for key in traces[0]} if traces else {}
    if traces:
        weights=torch.stack([trace_values[k] for k in ('w_A','w_F','w_P')],-1)
        if (weights < 0).any():raise ValueError('Negative mixture weight')
        torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights[...,0]),atol=1e-6,rtol=0)
    return result,trace_values
