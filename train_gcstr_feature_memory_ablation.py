"""One explicit training entry for every feature ablation; no automatic queues."""
import argparse
import fcntl
import json
import os
from pathlib import Path

import torch

from gcstr_feature_memory_ablation import load_model
from gcstr_video_sequence_dataset import VideoSequenceDataset
from feature_ablation_runtime import configuration, identity, run_pass, environment, parameter_report
from train_gcstr_feature_memory import (seed_all, rng_state, restore_rng, restore_checkpoint,
    initial_regression, atomic_save, write_json, emit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true', help='Three whole Train videos; one epoch per invocation')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    config = configuration(args.config)
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        parser.error('Single GPU only')
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error('stop-after-epoch must be positive')
    device = torch.device(args.device)
    if device.type != 'cuda':
        parser.error('Real EventMamba requires CUDA; use unit tests for synthetic CPU training')
    if args.smoke and 'smoke' not in args.output.name:
        parser.error('Dedicated smoke output required')
    torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and any(p.name != 'run.lock' for p in args.output.iterdir()):
        raise FileExistsError('Fresh training refuses nonempty output')
    seed_all(config['seed'])
    model = load_model(config).to(device)
    train_data = VideoSequenceDataset(config, 'train', 1, args.smoke)
    eval_data = VideoSequenceDataset(config, 'train' if args.smoke else 'test', 0, args.smoke)
    run_identity = identity(config, train_data, model)
    groups, _ = model.parameter_groups(config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    start, best, total_steps = 1, dict(nll=float('inf'), epoch=0), 0
    if args.resume:
        saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
        start, best, total_steps = restore_checkpoint(saved, model, optimizer, run_identity, args.smoke)
        del saved
    else:
        write_json(args.output/'config.json', dict(identity=run_identity, smoke=args.smoke,
            parameters=parameter_report(model, config), environment=environment(),
            train_windows=train_data.windows, eval_windows=eval_data.windows, first_epoch_steps=len(train_data)))
        # Every variant must recover Fusion A before ANY optimization, including
        # formal runs. This only uses the first two Train windows in the schedule.
        regression_data = VideoSequenceDataset(config, 'train', 0, True)
        write_json(args.output/'initial_regression.json', initial_regression(model, regression_data, device))
        previous = rng_state()
        seed_all(config['seed']+100000)
        baseline, predictions, _ = run_pass(model, eval_data, device, config)
        restore_rng(previous)
        write_json(args.output/'initial_evaluation.json', dict(metrics=baseline, predictions=predictions))
    log = args.output/'metrics.jsonl'
    if args.resume and log.exists():
        rows = [json.loads(s) for s in log.read_text().splitlines() if s.strip()]
        log.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch'] < start))
    limit = min(config['epochs'], args.stop_after_epoch or config['epochs'])
    if args.smoke:
        limit = min(limit, start)
    completed = start-1
    for epoch in range(start, limit+1):
        train_data = VideoSequenceDataset(config, 'train', epoch, args.smoke)
        torch.cuda.reset_peak_memory_stats(device)
        train, _, _ = run_pass(model, train_data, device, config, optimizer)
        total_steps += train['steps']
        previous = rng_state()
        seed_all(config['seed']+100000)
        evaluation, predictions, _ = run_pass(model, eval_data, device, config)
        restore_rng(previous)
        score = evaluation['main']['video_nll']
        improved = score < best['nll']
        if improved:
            best = dict(nll=score, epoch=epoch)
        item = dict(epoch=epoch, train=train, evaluation=evaluation, best=best, total_steps=total_steps,
            learning_rates={g['name']: g['lr'] for g in optimizer.param_groups},
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
        checkpoint = dict(identity=run_identity, smoke=args.smoke, epoch=epoch, data_epoch=epoch,
            model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
            rng_state=rng_state(), best=best, total_steps=total_steps, metrics=item)
        atomic_save(args.output/'last.pt', checkpoint)
        if improved:
            atomic_save(args.output/'best_video_nll.pt', checkpoint)
            write_json(args.output/'predictions_best.json', predictions)
        write_json(args.output/'predictions_last.json', predictions)
        with log.open('a') as stream:
            stream.write(json.dumps(item, allow_nan=False)+'\n')
        completed = epoch
        write_json(args.output/'status.json', dict(completed_epoch=epoch, total_epochs=config['epochs'],
            best=best, smoke=args.smoke, ablation_id=config['ablation_id'],
            status='complete' if epoch == config['epochs'] else 'paused' if epoch == limit else 'running'))
        emit('epoch', **item)
        if args.smoke:
            if train['continued_chunks'] < 1 or train['singleton_windows'] < 1:
                raise RuntimeError('Smoke must cover cross-chunk state and singleton tails')
            saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved['model_state'][key], rtol=0, atol=0)
            if not saved['optimizer_state']['state']:
                raise RuntimeError('Missing optimizer state')
            del saved
    emit('finished', completed_epoch=completed, smoke=args.smoke)


if __name__ == '__main__':
    main()
