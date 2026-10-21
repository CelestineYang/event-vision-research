"""Opt-in CPU/single-process fit/eval of routing modules on frozen evidence."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from fusion_router import Router, degeneracies, window_nll
from routing_data import (EvidenceCache, ObservationNormalizer, RoutingDataset,
                          OBSERVATIONS, collate_sequences, digest, fingerprint,
                          json_write, validate_training_roles, video_key)


def code_identity():
    return {name:digest(Path(__file__).parent/name) for name in
            ('routing_data.py', 'fusion_router.py', 'train_fusion_routing.py')}


def new_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Refusing nonempty output: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path, data):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def move(batch, device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


class RoutingMetrics:
    """Accept LOG PROBABILITIES only; arbitrary C; probability-mean video vote.

    Brier = mean_video sum_class (p-y)^2, without dividing by class count.
    ECE = 15 equal-width confidence bins by default, absolute accuracy gap,
    video-count weighted; confidence=1 belongs to the last bin.
    """
    def __init__(self, classes, bins=15):
        if classes<2 or bins<1:
            raise ValueError('Invalid metric dimensions')
        self.classes, self.bins = classes, bins
        self.videos = {}
        self.windows = 0
        self.loss_sum = 0.
        self.top1 = self.top5 = 0
        self.alphas, self.etas = [], []
        self.seen = set()

    def update_log_probs(self, log_probs, baseline_log_probs, labels, mask, rows, alpha, eta):
        for lp in (log_probs, baseline_log_probs):
            if lp.shape[-1] != self.classes or not torch.isfinite(lp).all():
                raise ValueError('Invalid log probabilities')
            if not torch.allclose(lp.logsumexp(-1), torch.zeros_like(lp[..., 0]), atol=2e-5, rtol=0):
                raise ValueError('Metrics require normalized LOG probabilities, not logits/probabilities')
        lp = log_probs.detach().cpu().double()
        base = baseline_log_probs.detach().cpu().double()
        mask, labels = mask.cpu(), labels.cpu()
        for b, metadata in enumerate(rows):
            if int(mask[b].sum()) != len(metadata):
                raise ValueError('Padding metadata mismatch')
            for t, row in enumerate(metadata):
                if not mask[b, t]:
                    raise ValueError('Rows must be a valid prefix followed by padding')
                key = (video_key(row), str(row['window_id']))
                if key in self.seen:
                    raise ValueError('Duplicate evaluation window')
                self.seen.add(key)
                label = int(labels[b, t])
                if label != int(row['label_id']):
                    raise ValueError('Metric label mismatch')
                value, old = lp[b, t], base[b, t]
                self.windows += 1
                self.loss_sum -= float(value[label])
                self.top1 += int(value.argmax() == label)
                self.top5 += int((value.topk(min(5, self.classes)).indices == label).any())
                vkey = video_key(row)
                if vkey not in self.videos:
                    self.videos[vkey] = dict(label=label, count=0, total=torch.full_like(value, -torch.inf),
                                             base=torch.full_like(old, -torch.inf))
                v = self.videos[vkey]
                if v['label'] != label:
                    raise ValueError('Video label changed')
                v['count'] += 1
                v['total'] = torch.logaddexp(v['total'], value)
                v['base'] = torch.logaddexp(v['base'], old)
        self.alphas.extend(alpha.detach().cpu()[mask].tolist())
        self.etas.extend(eta.detach().cpu()[mask].tolist())

    def finish(self):
        if not self.windows:
            raise ValueError('Empty evaluation')
        probabilities, nlls, correct, top5, brier = [], [], [], [], []
        rescued = damaged = baseline_correct = 0
        predictions = []
        for key in sorted(self.videos):
            v = self.videos[key]
            lp = v['total']-math.log(v['count'])
            old = v['base']-math.log(v['count'])
            p = lp.exp()
            label = v['label']
            pred, base_pred = int(p.argmax()), int(old.argmax())
            right, base_right = pred == label, base_pred == label
            probabilities.append(p)
            nlls.append(float(-lp[label]))
            correct.append(right)
            top5.append(bool((lp.topk(min(5, self.classes)).indices == label).any()))
            target = torch.zeros_like(p)
            target[label] = 1
            brier.append(float((p-target).square().sum()))
            rescued += int(right and not base_right)
            damaged += int(not right and base_right)
            baseline_correct += int(base_right)
            predictions.append(dict(dataset_id=key[0], source_split=key[1], video_id=key[2],
                                    label_id=label, prediction=pred, fusion_prediction=base_pred,
                                    windows=v['count'], nll=float(-lp[label])))
        confidence = torch.stack(probabilities).max(-1).values
        correctness = torch.tensor(correct, dtype=torch.float64)
        bins = (confidence*self.bins).long().clamp(max=self.bins-1)
        ece = 0.
        for idx in range(self.bins):
            selected = bins == idx
            if selected.any():
                ece += float(selected.double().mean()*(confidence[selected].mean()-correctness[selected].mean()).abs())
        def distribution(values):
            x = np.array(values)
            return dict(mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()),
                        p05=float(np.quantile(x, .05)), median=float(np.median(x)), p95=float(np.quantile(x, .95)))
        count = len(self.videos)
        metrics = dict(windows=self.windows, videos=count, window_nll=self.loss_sum/self.windows,
                       window_top1=self.top1/self.windows, window_top5=self.top5/self.windows,
                       video_top1=sum(correct)/count, video_top5=sum(top5)/count,
                       video_nll=sum(nlls)/count, video_brier=sum(brier)/count, video_ece=ece,
                       ece_bins=self.bins, ece_definition='equal_width_top1_confidence_abs_gap',
                       brier_definition='mean_video_sum_class_squared_error',
                       fusion_video_top1=baseline_correct/count, rescued=rescued, damaged=damaged,
                       net_change=rescued-damaged, alpha=distribution(self.alphas), eta=distribution(self.etas))
        return metrics, predictions


def batches(data, config, *, training=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(data, batch_size=int(config.get('batch_size', 8)), shuffle=training,
                      generator=generator, num_workers=0, collate_fn=collate_sequences)


def training_objective(log_probs, labels, mask, epoch_windows, epoch_steps):
    """Preserve WINDOW weighting even across unequal-size video batches.

    Every window has the same coefficient epoch_steps/epoch_windows. Averaging
    these batch objectives equals the epoch window-mean NLL (at fixed weights).
    This avoids implicitly giving a short video the weight of a long one.
    """
    return window_nll(log_probs, labels, mask)*(int(mask.sum())*epoch_steps/epoch_windows)


@torch.no_grad()
def evaluate(router, normalizer, data, config, device='cpu', ema_decay=None):
    router.eval()
    metric = RoutingMetrics(len(data.cache.identity['class_to_idx']), int(config.get('ece_bins', 15)))
    for batch in batches(data, config):
        batch = move(batch, device)
        output = router(batch['logits'], normalizer(batch['observations']), batch['mask'], batch['dt'],
                        state=None, ema_decay=ema_decay)
        metric.update_log_probs(output['log_probs'], batch['logits'][..., 0, :].log_softmax(-1),
                                batch['labels'], batch['mask'], batch['rows'], output['alpha'], output['eta'])
    return metric.finish()


def train(args):
    config = json.loads(Path(args.config).read_text())
    config['mode'] = args.mode
    if args.mode not in ('static', 'current', 'gru', 'conditional'):
        raise ValueError('Fit only static/current/gru/conditional; EMA is current-Gate evaluation only')
    if int(config.get('epochs', 0))<1 or int(config.get('batch_size', 0))<1 or float(config.get('lr', 0))<=0:
        raise ValueError('Invalid training budget')
    if config.get('batch_unit', 'video') not in ('video', 'window'):
        raise ValueError('batch_unit must be video or window')
    seed = int(config.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    expected = json.loads(Path(args.expect_identity).read_text()) if args.expect_identity else None
    cache = EvidenceCache(args.cache, expected)
    val_cache = EvidenceCache(args.val_cache, cache.identity) if args.val_cache else cache
    sequence = config.get('batch_unit', 'video') == 'video'
    if args.mode in ('gru', 'conditional') and not sequence:
        raise ValueError('Memory fit requires video sequence batches')
    train_data = RoutingDataset(cache, args.train_role, sequence=sequence)
    val_data = RoutingDataset(val_cache, args.val_role, sequence=sequence)
    validate_training_roles(train_data, val_data, allow_seen_experts=args.allow_seen_expert_validation)
    if args.mode != 'static':
        if args.static_init is None:
            raise ValueError('Dynamic fits require a shared --static-init checkpoint')
        static = load_checkpoint(args.static_init, cache.identity)
        if static['config']['mode'] != 'static' or static['config'].get('fusion_frame_only', False):
            raise ValueError('Dynamic initialization requires unrestricted static alpha/eta')
        if static['train_cache'] != cache.fingerprint or static['val_cache'] != val_cache.fingerprint:
            raise ValueError('Static initialization used different cache/roles')
        if (static['train_role'], static['val_role']) != (args.train_role, args.val_role):
            raise ValueError('Static initialization data roles differ')
        config.update(alpha0=static['weights']['alpha'], eta=static['weights']['eta'],
                      static_init_sha256=digest(args.static_init))
    router = Router(config).to(args.device)
    normalizer = ObservationNormalizer().fit(train_data).to(args.device)
    if args.mode != 'static':
        # Exact same normalization and b0/eta across all dynamic variants.
        normalizer.load_state_dict(static['normalizer_state'], strict=True)
    optimizer = torch.optim.Adam(router.parameters(), lr=float(config['lr']),
                                 weight_decay=float(config.get('weight_decay', 0.)))
    provenance = dict(train_exposure=sorted({cache.rows[i]['expert_fit_exposure'] for i in train_data.indices}),
                      validation_fit_exposure=sorted({val_cache.rows[i]['expert_fit_exposure'] for i in val_data.indices}),
                      validation_selection_exposure=sorted({val_cache.rows[i]['expert_selection_exposure'] for i in val_data.indices}),
                      exploratory_seen_expert_validation=args.allow_seen_expert_validation,
                      experts=cache.identity['expert_provenance'])
    config.update(observations=list(OBSERVATIONS), time_gap='log1p(seconds) in conditional write',
                  overlap='metadata only, not an additional Gate observation',
                  normalization_fit_role=args.train_role, expert_identity=cache.identity,
                  optimizer='Adam', loss='valid-window-mean-NLL', temperature=1.,
                  batch_loss_scaling='sum_window_NLL * epoch_steps / epoch_windows',
                  parameter_count=sum(p.numel() for p in router.parameters()),
                  train_role=args.train_role, val_role=args.val_role, provenance=provenance,
                  warnings=degeneracies(router), code=code_identity(),
                  effective_steps_per_epoch=len(batches(train_data, config)),
                  validation_selector='video_nll', independent_test=False if args.allow_seen_expert_validation else 'not_evaluated')
    output = new_output(args.output)
    json_write(output/'config.json', config)
    best = float('inf')
    steps = 0
    for epoch in range(1, int(config['epochs'])+1):
        started = time.perf_counter()
        router.train()
        train_loss, windows = 0., 0
        for batch in batches(train_data, config, training=True, seed=seed+epoch):
            batch = move(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            result = router(batch['logits'], normalizer(batch['observations']), batch['mask'], batch['dt'])
            loss = window_nll(result['log_probs'], batch['labels'], batch['mask'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite routing loss')
            objective = training_objective(result['log_probs'], batch['labels'], batch['mask'],
                                           len(train_data.indices), config['effective_steps_per_epoch'])
            objective.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), float(config.get('gradient_clip', 1.)), error_if_nonfinite=True)
            optimizer.step()
            count = int(batch['mask'].sum())
            train_loss += float(loss.detach())*count
            windows += count
            steps += 1
        training_metrics, _ = evaluate(router, normalizer, train_data, config, args.device)
        validation, predictions = evaluate(router, normalizer, val_data, config, args.device)
        improved = validation['video_nll'] < best
        best = min(best, validation['video_nll'])
        alpha, eta = router.weights()
        item = dict(epoch=epoch, train_online_window_nll=train_loss/windows,
                    train=training_metrics, validation=validation, optimizer_steps=steps,
                    parameter_count=config['parameter_count'], seconds=time.perf_counter()-started)
        saved = dict(schema=1, config=config, identity=cache.identity, router_state=router.state_dict(),
                     normalizer_state=normalizer.state_dict(), optimizer_state=optimizer.state_dict(),
                     epoch=epoch, optimizer_steps=steps, train_cache=cache.fingerprint, val_cache=val_cache.fingerprint,
                     train_role=args.train_role, val_role=args.val_role, metrics=item,
                     weights=dict(alpha=float(alpha.detach()), eta=float(eta.detach())),
                     rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate()))
        save_checkpoint(output/'last.pt', saved)
        if improved:
            save_checkpoint(output/'best_val_nll.pt', saved)
            json_write(output/'validation_predictions_best.json', predictions)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(item, allow_nan=False)+'\n')
        print(json.dumps(dict(epoch=epoch, validation=validation, optimizer_steps=steps)), flush=True)
    json_write(output/'status.json', dict(status='complete', epochs=config['epochs'], optimizer_steps=steps, best_validation_nll=best))
    return output


def load_checkpoint(path, identity):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('schema') != 1 or saved['identity'] != identity:
        raise ValueError('Router checkpoint/evidence identity mismatch')
    if saved['config'].get('code') != code_identity() or saved['config'].get('observations') != list(OBSERVATIONS):
        raise ValueError('Router code/observation identity changed')
    return saved


def eval_command(args):
    expected = json.loads(Path(args.expect_identity).read_text()) if args.expect_identity else None
    cache = EvidenceCache(args.cache, expected)
    if args.checkpoint:
        saved = load_checkpoint(args.checkpoint, cache.identity)
        config = saved['config']
        router = Router(config)
        router.load_state_dict(saved['router_state'], strict=True)
        normalizer = ObservationNormalizer()
        normalizer.load_state_dict(saved['normalizer_state'], strict=True)
    else:
        config = dict(mode='fixed', fixed=args.fixed, batch_size=8, ece_bins=15)
        router = Router(config)
        normalizer = ObservationNormalizer()  # Fixed weights do not use observations.
    if args.ema_decay is not None and (not args.checkpoint or router.mode != 'current'):
        raise ValueError('--ema-decay evaluates a fitted current Gate only; no Test tuning')
    sequence = router.sequence or args.ema_decay is not None or config.get('batch_unit', 'video') == 'video'
    data = RoutingDataset(cache, args.role, sequence=sequence)
    output = new_output(args.output)
    metrics, predictions = evaluate(router.to(args.device), normalizer.to(args.device), data, config,
                                    args.device, ema_decay=args.ema_decay)
    json_write(output/'metrics.json', dict(metrics=metrics, cache_fingerprint=cache.fingerprint,
        config=config, role=args.role, ema_decay=args.ema_decay,
        checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
        evaluation_only=True, expert_provenance=cache.identity['expert_provenance']))
    json_write(output/'video_predictions.json', predictions)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--config', required=True)
    fit.add_argument('--mode', required=True, choices=('static', 'current', 'gru', 'conditional'))
    fit.add_argument('--train-role', required=True, choices=('gate_train',))
    fit.add_argument('--val-role', required=True, choices=('gate_val',))
    fit.add_argument('--val-cache')
    fit.add_argument('--static-init', type=Path)
    fit.add_argument('--allow-seen-expert-validation', action='store_true',
                     help='Explicit exploratory validation on expert-seen/unknown TRAIN videos, never original Test')
    ev = commands.add_parser('eval')
    ev.add_argument('--role', required=True, choices=('gate_train', 'gate_val', 'test'))
    group = ev.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=Path)
    group.add_argument('--fixed', choices=('fusion', 'frame', 'frame_point', 'fusion_frame', 'three_way'))
    ev.add_argument('--ema-decay', type=float, help='Predeclared evaluation option, never selected on Test')
    for command in (fit, ev):
        command.add_argument('--cache', required=True)
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='cpu')
        command.add_argument('--expect-identity', help='Optional exact expected cache identity JSON')
    args = parser.parse_args()
    train(args) if args.command == 'fit' else eval_command(args)


if __name__ == '__main__':
    main()
