"""Real CUDA BiMamba forward/backward test; no dataset/checkpoint writes."""
import argparse
import inspect
import json

import torch

from eventmamba_encoder import EventMambaEncoder
from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from gcstr_secnet_fusion import VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-size', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    from mamba_ssm.modules.mamba_simple import Mamba
    print(json.dumps({'mamba_source': inspect.getfile(Mamba), 'torch': torch.__version__}), flush=True)
    n = 8192 if args.full_size else 32
    groups = (1024, 512, 256) if args.full_size else (16, 8, 4)
    encoder = EventMambaEncoder(groups=groups, neighbors=24 if args.full_size else 4)
    model = GCSTREventMambaFusion(point_encoder=encoder).cuda().train()
    events = torch.rand(2, 4, n, device='cuda')
    events[:, 2] = events[:, 2].sort(-1).values
    size = 224 if args.full_size else 64
    views = {v: torch.rand(2, 3, size, size, device='cuda') for v in VIEWS}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    logits = model(events, views)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1], device='cuda'))
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
            raise RuntimeError(f'Invalid gradient: {name}')
    optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps({'status': 'PASS', 'full_size': args.full_size,
                      'logits': list(logits.shape), 'loss': loss.item(),
                      'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20}), flush=True)


if __name__ == '__main__':
    main()
