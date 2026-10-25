"""Single-GPU full-input feature training, video TBPTT and epoch-boundary resume."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from gcstr_feature_memory_fusion import ARCHITECTURE, detach_state, load_model, sha256
from gcstr_video_sequence_dataset import VideoSequenceDataset

SOURCE_FILES = ('gcstr_feature_memory_fusion.py', 'gcstr_video_sequence_dataset.py',
                'train_gcstr_feature_memory.py', 'evaluate_gcstr_feature_memory.py')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def emit(event, **payload):
    print(json.dumps(dict(event=event, **payload), allow_nan=False), flush=True)


class PredictionMetrics:
    """Float64 probability means, 15 equal-width confidence bins, multiclass Brier sum."""
    def __init__(self):
        self.windows = self.ce = self.correct = self.top5 = 0
        self.videos = {}

    def update(self, logits, labels, ids):
        logp = logits.detach().double().log_softmax(-1).cpu()
        labels = labels.cpu()
        if not torch.isfinite(logp).all():
            raise FloatingPointError('Nonfinite predictions')
        self.windows += len(labels)
        self.ce += float(-logp[torch.arange(len(labels)), labels].sum())
        self.correct += int((logp.argmax(-1) == labels).sum())
        self.top5 += int((logp.topk(min(5, logp.shape[-1])).indices == labels[:, None]).any(1).sum())
        for video, label, p in zip(ids, labels.tolist(), logp.exp().numpy()):
            if video not in self.videos:
                self.videos[video] = [label, 0, np.zeros_like(p)]
            item = self.videos[video]
            if item[0] != label:
                raise ValueError('Video label mismatch')
            item[1] += 1
            item[2] += p

    def finish(self):
        if not self.windows:
            raise ValueError('No valid windows')
        rows, nlls, briers, hits, top5, confidences = [], [], [], [], [], []
        for video, (label, n, sums) in sorted(self.videos.items()):
            p = sums/n
            order = np.argsort(-p, kind='stable')
            nll = -float(np.log(max(p[label], np.finfo(float).tiny)))
            hits.append(float(order[0] == label))
            top5.append(float(label in order[:5]))
            confidences.append(float(p[order[0]]))
            nlls.append(nll)
            briers.append(float((p*p).sum()-2*p[label]+1))
            rows.append(dict(video_id=video, label=label, windows=n, prediction=int(order[0]),
                             top5=order[:5].tolist(), nll=nll, confidence=confidences[-1]))
        confidence, correct = np.asarray(confidences), np.asarray(hits)
        bins = np.minimum((confidence*15).astype(int), 14)
        ece = sum(float((bins == i).mean())*abs(float(confidence[bins == i].mean()-correct[bins == i].mean()))
                  for i in range(15) if (bins == i).any())
        return dict(windows=self.windows, videos=len(rows), window_ce=self.ce/self.windows,
                    window_top1=self.correct/self.windows, window_top5=self.top5/self.windows,
                    video_top1=float(np.mean(hits)), video_top5=float(np.mean(top5)),
                    video_nll=float(np.mean(nlls)), video_brier=float(np.mean(briers)),
                    video_ece_15=ece), rows


class Diagnostics:
    def __init__(self):
        self.values = {}

    def update(self, values):
        for key, tensor in values.items():
            tensor = torch.as_tensor(tensor).detach().double()
            if not tensor.numel():
                continue
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f'Nonfinite diagnostic: {key}')
            n, total, squares, low, high = self.values.get(key, (0, 0., 0., float('inf'), -float('inf')))
            self.values[key] = (n+tensor.numel(), total+float(tensor.sum()), squares+float(tensor.square().sum()),
                                min(low, float(tensor.min())), max(high, float(tensor.max())))

    def finish(self):
        return {key: dict(n=n, mean=s/n, std=max(0., ss/n-(s/n)**2)**.5, min=lo, max=hi)
                for key, (n, s, ss, lo, hi) in self.values.items()}


def objective_sum(output, labels, aux_weight):
    return (nn.functional.cross_entropy(output['logits'], labels, reduction='sum') +
            aux_weight * nn.functional.cross_entropy(output['point_logits'], labels, reduction='sum'))


def run_pass(model, data, device, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    generator = torch.Generator().manual_seed(config['seed'] + data.epoch)
    batches = DataLoader(data, batch_size=None, shuffle=False, num_workers=config['workers'],
                         pin_memory=device.type == 'cuda', generator=generator)
    metrics = {key: PredictionMetrics() for key in ('main', 'point')}
    diagnostics = Diagnostics()
    state, seen, steps, fallbacks = None, set(), 0, 0
    continued_chunks, singleton_windows = 0, 0
    started = time.perf_counter()
    coefficient = len(data)/data.windows  # Equal window weight despite ragged tails.
    for chunk_index, chunk in enumerate(batches):
        if chunk['group_start']:
            state = None
        elif state is None:
            raise RuntimeError('Lost cross-chunk state')
        else:
            continued_chunks += 1
        if training:
            optimizer.zero_grad(set_to_none=True)
        objective = None
        with torch.set_grad_enabled(training):
            for cpu_batch in chunk['batches']:
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in cpu_batch.items()}
                valid = batch['valid']
                if not valid.any():
                    continue
                singleton_windows += int(valid.sum() == 1)
                output, state = model.forward_window(batch, state)
                local = {k: output[k][valid] for k in ('logits', 'point_logits')}
                labels = batch['label'][valid]
                loss = objective_sum(local, labels, config['point_aux_weight'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                if training:
                    objective = loss if objective is None else objective + loss
                mask = valid.cpu().tolist()
                ids = [v for v, keep in zip(batch['video_id'], mask) if keep]
                wins = [v for v, keep in zip(batch['window_id'], mask) if keep]
                for identity in zip(ids, wins):
                    if identity in seen:
                        raise RuntimeError(f'Duplicate target: {identity}')
                    seen.add(identity)
                metrics['main'].update(local['logits'], labels, ids)
                metrics['point'].update(local['point_logits'], labels, ids)
                diagnostics.update(output['diagnostics'])
                fallbacks += int(batch['augmentation_fallbacks'][valid].sum())
                # Finished lanes are not refilled until group end, but release their
                # numeric memory immediately; current predictions/loss are untouched.
                if 'video_end' in batch:
                    ended = batch['video_end'] & valid
                    state = {key: value.masked_fill(ended.reshape(-1, *([1]*(value.ndim-1))), 0)
                             for key, value in state.items()}
            if training:
                if objective is None:
                    raise RuntimeError('Empty scheduled chunk')
                (objective * coefficient).backward()
                for group in optimizer.param_groups:
                    norm = torch.stack([p.grad.detach().norm().square() for p in group['params'] if p.grad is not None]).sum().sqrt()
                    diagnostics.update({f'gradient/{group["name"]}': norm})
                nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                steps += 1
        if state is not None:
            state = detach_state(state)
        if chunk['group_end']:
            state = None
        if chunk_index == 0 or (chunk_index+1) % 100 == 0 or chunk_index+1 == len(data):
            emit('progress', phase='train_online' if training else 'fixed_eval', epoch=data.epoch,
                 chunks=chunk_index+1, total_chunks=len(data), windows=len(seen), seconds=time.perf_counter()-started)
    expected = {(data.rows[i]['video_id'], str(data.rows[i]['window_id']))
                for indices in data.groups.values() for i in indices}
    if seen != expected:
        raise RuntimeError('Coverage mismatch')
    results, predictions = {}, {}
    for key, metric in metrics.items():
        results[key], predictions[key] = metric.finish()
    elapsed = time.perf_counter()-started
    results.update(steps=steps, seconds=elapsed, windows_per_second=data.windows/elapsed,
                   continued_chunks=continued_chunks, singleton_windows=singleton_windows,
                   objective_loss=results['main']['window_ce']+config['point_aux_weight']*results['point']['window_ce'],
                   diagnostics=diagnostics.finish(), augmentation_fallbacks=fallbacks,
                   mode='online_changing_parameters' if training else 'fixed_parameter_evaluation')
    return results, predictions


def configuration(path):
    config = json.loads(Path(path).read_text())
    if (config['architecture'] != ARCHITECTURE or config['mode'] not in ('B', 'G', 'GM') or
            config['precision'] != 'FP32' or config['optimizer'] != 'AdamW' or
            config['scheduler'] is not None or config['early_stopping'] or
            config['selection_protocol'] != 'original_test_as_val' or config['independent_test'] or
            config['point_count'] != 2048 or not config['freeze_image_backbones'] or
            not config['freeze_point_bn_running_stats']):
        raise ValueError('Unsupported protocol/configuration')
    if min(config['epochs'], config['video_batch_size_per_gpu'], config['tbptt_steps']) < 1 or config['workers'] < 0:
        raise ValueError('Invalid budget')
    return config


def identity(config, data, model):
    source = model.source_config
    if source['manifest_sha256'] != data.audit['manifest_sha256'] or source['video_splits_sha256'] != data.audit['video_splits_sha256']:
        raise ValueError('Source model trained with another data split')
    return dict(architecture=ARCHITECTURE, config=config,
                code={f: sha256(Path(__file__).parent/f) for f in SOURCE_FILES},
                manifest_sha256=data.audit['manifest_sha256'], video_splits_sha256=data.audit['video_splits_sha256'],
                store_summary_sha256=sha256(Path(config['store'])/'summary.json'),
                original_source_code=source['code'])


def restore_checkpoint(saved, model, optimizer, expected_identity, smoke):
    if saved['identity'] != expected_identity or saved['smoke'] != smoke:
        raise ValueError('Resume architecture/config/source/data/code identity mismatch')
    if saved.get('data_epoch', saved['epoch']) != saved['epoch']:
        raise ValueError('Not an epoch-boundary checkpoint')
    model.load_state_dict(saved['model_state'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state'])
    restore_rng(saved['rng_state'])
    return saved['epoch']+1, saved['best'], saved['total_steps']


def initial_regression(model, data, device):
    """Real input + CUDA path; fixed legacy FP32 tolerance, never fit to error."""
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    previous_rng = rng_state()
    model.eval()
    state, reports = None, []
    with torch.no_grad():
        for cpu_batch in data[0]['batches'][:2]:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cpu_batch.items()}
            if not batch['valid'].all():
                raise ValueError('Initial regression needs two unpadded windows')
            common_rng = rng_state()
            reference = model.base.forward_batch_outputs(batch)
            restore_rng(common_rng)
            repeat = model.base.forward_batch_outputs(batch)
            restore_rng(common_rng)
            actual, state = model.forward_window(batch, state)
            report = {}
            for name, output in (('repeat', repeat), ('feature', actual)):
                for head in ('logits', 'point_logits'):
                    torch.testing.assert_close(output[head], reference[head], rtol=1e-6, atol=1e-5)
                    torch.testing.assert_close(output[head].softmax(-1), reference[head].softmax(-1), rtol=1e-5, atol=1e-6)
                    if not torch.equal(output[head].argmax(-1), reference[head].argmax(-1)):
                        raise AssertionError('Initial regression changed top1')
                    report[f'{name}/{head}/max_abs'] = float((output[head]-reference[head]).abs().max())
            reports.append(report)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), before[key], rtol=0, atol=0)
    restore_rng(previous_rng)
    return dict(windows=reports, logits_rtol=1e-6, logits_atol=1e-5,
                probability_rtol=1e-5, probability_atol=1e-6, parameters_buffers_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Three complete Train videos only; not method evaluation')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    config = configuration(args.config)
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('DDP not supported; run separate experiments on separate GPUs')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Real EventMamba training requires CUDA; CPU synthetic tests have a separate entry')
    torch.cuda.set_device(device)
    if args.smoke and 'smoke' not in args.output.name:
        raise ValueError('Smoke requires a dedicated smoke output')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and any(p.name != 'run.lock' for p in args.output.iterdir()):
        raise FileExistsError('Refusing to overwrite nonempty output')
    seed_all(config['seed'])
    model = load_model(config).to(device)
    train_data = VideoSequenceDataset(config, 'train', 1, args.smoke)
    eval_data = VideoSequenceDataset(config, 'train' if args.smoke else 'test', 0, args.smoke)
    run_identity = identity(config, train_data, model)
    groups, names = model.parameter_groups(config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    start, best, total_steps = 1, dict(nll=float('inf'), epoch=0), 0
    if args.resume:
        saved = torch.load(args.output/'last.pt', map_location=device, weights_only=False)
        # RNG tensors must be CPU ByteTensors even when optimizer tensors are CUDA.
        saved['rng_state']['torch'] = saved['rng_state']['torch'].cpu()
        saved['rng_state']['cuda'] = [s.cpu() for s in saved['rng_state']['cuda']]
        start, best, total_steps = restore_checkpoint(saved, model, optimizer, run_identity, args.smoke)
    else:
        write_json(args.output/'config.json', dict(identity=run_identity, smoke=args.smoke, parameter_names=names,
            parameters={g['name']: sum(p.numel() for p in g['params']) for g in groups},
            train_windows=train_data.windows, eval_windows=eval_data.windows, first_epoch_steps=len(train_data)))
    if args.smoke and not args.resume:
        write_json(args.output/'initial_regression.json', initial_regression(model, train_data, device))
    # A deterministic online epoch0 baseline; eval RNG never advances training RNG.
    if not args.resume:
        previous = rng_state()
        seed_all(config['seed']+100000)
        baseline, predictions = run_pass(model, eval_data, device, config)
        restore_rng(previous)
        write_json(args.output/'initial_evaluation.json', dict(metrics=baseline, predictions=predictions))
    log = args.output/'metrics.jsonl'
    if args.resume and log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        # last.pt is authoritative; only interrupted, unsaved rows are discarded.
        log.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch'] < start))
    limit = min(config['epochs'], args.stop_after_epoch or config['epochs'])
    if args.smoke:
        limit = min(limit, start)  # one complete small epoch per invocation, resume is explicit
    for epoch in range(start, limit+1):
        train_data = VideoSequenceDataset(config, 'train', epoch, args.smoke)
        torch.cuda.reset_peak_memory_stats(device)
        train, _ = run_pass(model, train_data, device, config, optimizer)
        total_steps += train['steps']
        previous = rng_state()
        seed_all(config['seed']+100000)
        evaluation, predictions = run_pass(model, eval_data, device, config)
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
        write_json(args.output/'status.json', dict(completed_epoch=epoch, total_epochs=config['epochs'], best=best,
            status='complete' if epoch == config['epochs'] else 'paused' if epoch == limit else 'running'))
        emit('epoch', **item)
        if args.smoke:
            if train['continued_chunks'] < 1 or train['singleton_windows'] < 1:
                raise RuntimeError('Smoke must exercise consecutive chunks and singleton tail')
            restored = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), restored['model_state'][key], rtol=0, atol=0)
            if not restored['optimizer_state']['state']:
                raise RuntimeError('Missing optimizer state')
    emit('finished', completed_epoch=limit, smoke=args.smoke)


if __name__ == '__main__':
    main()
