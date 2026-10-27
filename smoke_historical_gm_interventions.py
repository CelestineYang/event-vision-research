"""Real-checkpoint CUDA intervention checks on three Train videos only."""
import argparse
from pathlib import Path

import torch

from gcstr_feature_memory_ablation import INTERVENTIONS, load_model
from gcstr_video_sequence_dataset import VideoSequenceDataset
from feature_ablation_runtime import configuration, verify_historical_gm, run_pass, paired
from train_gcstr_feature_memory import seed_all, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--record', required=True, type=Path)
    p.add_argument('--ddp-source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = configuration(args.config)
    seed_all(cfg['seed'])
    model = load_model(cfg)
    data = VideoSequenceDataset(cfg, 'train', 0, smoke=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    provenance = verify_historical_gm(saved, args.checkpoint, args.record, args.ddp_source, cfg, data, model)
    del saved
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model.to(device).eval()
    before = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    traces, diagnostics = {}, {}
    torch.cuda.reset_peak_memory_stats(device)
    for mode in INTERVENTIONS:
        model.set_intervention(mode)
        seed_all(cfg['seed']+100000)
        metrics, _, traces[mode] = run_pass(model, data, device, cfg, capture=True)
        diagnostics[mode] = dict(windows=metrics['main']['windows'], videos=metrics['main']['videos'],
                                 diagnostics=metrics['diagnostics'], gate_quantiles=metrics['gate_quantiles'])
    a, b = traces['reset_each_window'], traces['disable_both_history_outputs']
    paired(a, b)  # Also verifies order/labels/point input hashes/real gaps.
    torch.testing.assert_close(torch.from_numpy(a['logits']), torch.from_numpy(b['logits']), rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(torch.from_numpy(a['logits']).softmax(-1),
                               torch.from_numpy(b['logits']).softmax(-1), rtol=1e-5, atol=1e-6)
    for k, t in model.state_dict().items():
        torch.testing.assert_close(t.cpu(), before[k], rtol=0, atol=0)
    differences = {}
    for mode in INTERVENTIONS[1:]:
        result = paired(traces['normal'], traces[mode])
        differences[mode] = {k: result[k] for k in ('window_logits_max_abs', 'window_probability_max_abs')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(status='passed', scope='three_train_videos_only_engineering_check',
        test_evaluated=False, optimizer_steps=0, provenance=provenance,
        diagnostics=diagnostics, differences_vs_normal=differences,
        reset_vs_both_logits_max_abs=paired(a, b)['window_logits_max_abs'],
        parameters_buffers_unchanged=True, peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20))
    print('Historical GM interventions: five modes passed on Train-only real CUDA inputs.', flush=True)


if __name__ == '__main__':
    main()
