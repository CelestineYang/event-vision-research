"""Fixed-checkpoint evaluation and sustained causal history interventions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gcstr_feature_memory_ablation import INTERVENTIONS, load_model
from gcstr_feature_memory_fusion import sha256
from gcstr_video_sequence_dataset import VideoSequenceDataset
from feature_ablation_runtime import (configuration, run_pass, paired, verify_evaluation_checkpoint,
                                     verify_historical_gm, environment)
from train_gcstr_feature_memory import seed_all, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--interventions', action='store_true', help='All five modes on the SAME GM checkpoint')
    parser.add_argument('--reference', type=Path, help='Normal evaluation directory from another trained variant')
    parser.add_argument('--historical-record', type=Path, help='Explicit exploratory reuse of the staged legacy GM')
    parser.add_argument('--historical-ddp-source', type=Path, help='Exact archived DDP trainer, verified by SHA')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing existing evaluation output')
    config = configuration(args.config)
    if args.interventions and (config['base_mode'], config['ablation_id']) != ('GM', 'base'):
        parser.error('Interventions require original GM')
    seed_all(config['seed'])
    model = load_model(config)
    data = VideoSequenceDataset(config, 'test')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    historical = None
    if args.historical_record:
        if args.historical_ddp_source is None or args.reference:
            parser.error('Historical reuse needs --historical-ddp-source and cannot be a matched --reference comparison')
        historical = verify_historical_gm(saved, args.checkpoint, args.historical_record,
                                         args.historical_ddp_source, config, data, model)
    else:
        if args.historical_ddp_source:
            parser.error('--historical-ddp-source requires --historical-record')
        verify_evaluation_checkpoint(saved, config, data, model)
    device = torch.device(args.device)
    if device.type != 'cuda':
        parser.error('Real EventMamba requires CUDA')
    torch.cuda.set_device(device)
    model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', dict(config=config, epoch=saved['epoch'],
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        checkpoint_identity=saved['identity'], historical_reuse=historical, environment=environment(),
        intervention_semantics='sustained whole-video intervention; future states change with H',
        selection_protocol='original_test_as_val', independent_test=False))
    del saved
    modes = INTERVENTIONS if args.interventions else ('normal',)
    traces = {}
    for mode in modes:
        model.set_intervention(mode)
        # Reset Point FPS RNG, retain exact dataset/schedule/batch layout and gaps.
        seed_all(config['seed']+100000)
        torch.cuda.reset_peak_memory_stats(device)
        metrics, predictions, trace = run_pass(model, data, device, config, capture=True)
        metrics['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/2**20
        traces[mode] = trace
        folder = args.output/mode
        folder.mkdir()
        write_json(folder/'metrics.json', metrics)
        write_json(folder/'predictions.json', predictions)
        write_json(folder/'window_index.json', trace['rows'])
        np.save(folder/'logits.npy', trace['logits'], allow_pickle=False)
        if mode != 'normal':
            write_json(folder/'vs_normal.json', paired(traces['normal'], trace))
    if args.interventions:
        a, b = traces['reset_each_window'], traces['disable_both_history_outputs']
        if a['rows'] != b['rows']:
            raise AssertionError('Reset and both-disabled used different inputs')
        torch.testing.assert_close(torch.from_numpy(a['logits']), torch.from_numpy(b['logits']), rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(torch.from_numpy(a['logits']).softmax(-1),
                                   torch.from_numpy(b['logits']).softmax(-1), rtol=1e-5, atol=1e-6)
        write_json(args.output/'reset_equivalence.json', paired(a, b))
    if args.reference:
        provenance = json.loads((args.reference/'config.json').read_text())
        if provenance.get('historical_reuse') is not None:
            raise ValueError('Historical reference cannot be labeled as a matched comparison')
        reference_config = provenance['config']
        structural = {'mode', 'base_mode', 'ablation_id', 'frame_query_source', 'gate_policy',
                      'history_to_gate', 'history_direct_residual', 'memory_update',
                      'experiment', 'store', 'source_checkpoint'}
        if {k: v for k, v in reference_config.items() if k not in structural} != {
                k: v for k, v in config.items() if k not in structural}:
            raise ValueError('Reference training protocol differs')
        reference = dict(rows=json.loads((args.reference/'normal/window_index.json').read_text()),
                         logits=np.load(args.reference/'normal/logits.npy', allow_pickle=False))
        write_json(args.output/'vs_reference.json', paired(reference, traces['normal']))
    write_json(args.output/'status.json', dict(status='complete', modes=list(modes),
        windows=data.windows, videos=len(data.groups), independent_test=False))


if __name__ == '__main__':
    main()
