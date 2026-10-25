"""CPU synthetic correctness tests; real GPU smoke uses the real training entry.

These tiny encoders intentionally do NOT validate EventMamba CUDA numerics.
"""
import copy
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gcstr_feature_memory_fusion import FeatureMemoryFusion, detach_state, original_tokens
from gcstr_secnet_fusion import PrototypeStage, VIEWS
from gcstr_video_sequence_dataset import (apply_spatial_params, keyed_rng, make_schedule,
                                         sample_spatial_params, video_groups, VideoSequenceDataset)
from gcstr_representations import GCSTRConfig
from train_gcstr_feature_memory import (PredictionMetrics, atomic_save, objective_sum,
                                       restore_checkpoint, rng_state, seed_all, run_pass, initial_regression)


class TinyPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 8)
        self.bn = nn.BatchNorm1d(8)
        self.classifier = nn.Linear(8, 6)

    def forward_features(self, x):
        xyz = x.transpose(1, 2)
        f = self.bn(self.proj(xyz).transpose(1, 2)).transpose(1, 2)
        return [f, f, f], [xyz, xyz, xyz]

    def classify_features(self, features):
        return self.classifier(features[-1].mean(1))


class TinyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoders = nn.ModuleDict({v: nn.Linear(3, 8) for v in VIEWS})
        self.view_adapters = nn.ModuleDict({v: nn.Linear(8, 8) for v in VIEWS})
        self.point_encoder = TinyPoint()
        self.interactions = nn.ModuleList(PrototypeStage(8, 8, 2, 2) for _ in range(3))
        for stage in self.interactions:
            stage.position = nn.Linear(3, 8)
        self.classifier = nn.Linear(24, 6)

    def encode_views(self, views):
        return {v: self.view_adapters[v](self.image_encoders[v](views[v])) for v in VIEWS}

    def forward_batch_outputs(self, x):
        features, coords = self.point_encoder.forward_features(x['events'])
        views = self.encode_views(x)
        vectors = [s(f, c, views) for s, f, c in zip(self.interactions, features, coords)]
        return dict(logits=self.classifier(torch.cat(vectors, -1)),
                    point_logits=self.point_encoder.classify_features(features))


def batch(n=2):
    result = dict(events=torch.randn(n, 3, 7), gap_seconds=torch.ones(n)*.5,
                  valid=torch.ones(n, dtype=torch.bool))
    result.update({v: torch.randn(n, 5, 3) for v in VIEWS})
    return result


def activate(model):
    with torch.no_grad():
        for extra in model.extra:
            extra.delta[-1].weight.normal_(0, .05)
        if model.mode == 'GM':
            for layer in model.memory.output:
                layer.weight.normal_(0, .05)


class TinySequence(VideoSequenceDataset):
    def __init__(self, epoch):
        self.epoch = epoch
        self.rows = [dict(video_id=str(v), window_id=str(t), start_us=t*500000, label_id=v)
                     for v, n in enumerate((5, 3, 3)) for t in range(n)]
        self.groups = video_groups(self.rows)
        self.windows = len(self.rows)
        self.schedule = make_schedule(self.groups, 2, 2, 0, epoch, True)

    def window(self, index):
        row = self.rows[index]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(index + self.epoch*100)
            x = {k: v[0] for k, v in batch(1).items()}
        x.update(label=row['label_id'], video_id=row['video_id'], window_id=row['window_id'],
                 augmentation_fallbacks=0,
                 video_end=int(row['window_id']) == len(self.groups[row['video_id']])-1)
        return x


class FeatureTests(unittest.TestCase):
    def setUp(self):
        seed_all(7)
        self.base = TinyFusion()

    def model(self, mode='GM'):
        return FeatureMemoryFusion(copy.deepcopy(self.base), mode, seed=3)

    def test_common_initialization_freezing_and_parameter_groups(self):
        rng_before = torch.get_rng_state().clone()
        with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('Must not reseed caller CUDA RNG')):
            b, g, gm = [self.model(mode) for mode in ('B', 'G', 'GM')]
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        self.assertFalse(hasattr(b, 'extra'))
        self.assertFalse(hasattr(g, 'memory'))
        for key, value in b.base.state_dict().items():
            torch.testing.assert_close(value, gm.base.state_dict()[key], rtol=0, atol=0)
        for key, value in g.extra.state_dict().items():
            torch.testing.assert_close(value, gm.extra.state_dict()[key], rtol=0, atol=0)
        gm.train()
        self.assertFalse(gm.base.image_encoders.training)
        self.assertFalse(gm.base.point_encoder.bn.training)
        self.assertTrue(gm.base.point_encoder.bn.weight.requires_grad)
        self.assertTrue(gm.base.point_encoder.training)
        groups, _ = gm.parameter_groups(dict(lr_point=1e-5, lr_existing_fusion=5e-5, lr_new_modules=1e-4))
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in gm.parameters() if p.requires_grad})

    def test_zero_residual_recovers_original_multiple_windows(self):
        models = [self.model(mode).eval() for mode in ('B', 'G', 'GM')]
        states = [None]*3
        for _ in range(4):
            x = batch()
            outputs = []
            for i, model in enumerate(models):
                out, states[i] = model.forward_window(x, states[i])
                outputs.append(out)
            base = models[0].base
            features, coords = base.point_encoder.forward_features(x['events'])
            tokens = base.encode_views(x)
            reference = base.classifier(torch.cat([s(f, c, tokens) for s, f, c in
                zip(base.interactions, features, coords)], -1))
            for out in outputs:
                torch.testing.assert_close(out['logits'], reference, rtol=0, atol=0)
                torch.testing.assert_close(out['point_logits'], outputs[0]['point_logits'], rtol=0, atol=0)

    def test_frame_and_point_aux_independence(self):
        model = self.model().eval()
        x = batch()
        tokens = model.base.encode_views(x)
        q = model.base.interactions[0].prototypes[None].expand(2, -1, -1)
        f = model.extra[0].frame(q, tokens)
        altered = dict(x, events=x['events']*10)
        model.forward_window(altered)
        torch.testing.assert_close(f, model.extra[0].frame(q, tokens), rtol=0, atol=0)
        old, _ = model.forward_window(x)
        changed = {**x, **{v: x[v]*20 for v in VIEWS}}
        state = model.initial_state(2, 'cpu')
        state['memory'].normal_()
        state['history_valid'].fill_(True)
        new, _ = model.forward_window(changed, state)
        torch.testing.assert_close(old['point_logits'], new['point_logits'], rtol=0, atol=0)

    def test_gradients_after_zero_start_and_frozen_buffers(self):
        model = self.model()
        before = {k: v.clone() for k, v in model.base.image_encoders.state_dict().items()}
        bn = {k: v.clone() for k, v in model.base.point_encoder.bn.named_buffers()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.005)
        gradients = set()
        for _ in range(4):
            optimizer.zero_grad()
            state, loss = None, 0
            for _ in range(3):
                out, state = model.forward_window(batch(), state)
                loss = loss + objective_sum(out, torch.tensor([1, 3]), .5)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    gradients.add(name)
            optimizer.step()
        for prefix in ('base.point_encoder.proj', 'base.view_adapters', 'base.interactions', 'base.classifier',
                       'extra.0.attention', 'extra.0.merge', 'extra.0.gate', 'memory.read', 'memory.write',
                       'memory.query', 'memory.write_gate'):
            self.assertTrue(any(name.startswith(prefix) for name in gradients), prefix)
        for key, value in before.items():
            torch.testing.assert_close(value, model.base.image_encoders.state_dict()[key], rtol=0, atol=0)
        for key, value in bn.items():
            torch.testing.assert_close(value, dict(model.base.point_encoder.bn.named_buffers())[key], rtol=0, atol=0)

    def test_causality_write_after_predict_and_first_history(self):
        model = self.model().eval()
        activate(model)
        x, y = batch(), batch()
        a, state = model.forward_window(x)
        for key, value in a['diagnostics'].items():
            if 'read_norm' in key:
                self.assertEqual(float(value.sum()), 0.)
        changed = copy.deepcopy(model)
        with torch.no_grad():
            changed.memory.write_gate[0].bias.add_(10)
        b, new_state = changed.forward_window(x)
        torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
        c, _ = model.forward_window(y, state)
        d, _ = changed.forward_window(y, new_state)
        self.assertGreater(float((c['logits']-d['logits']).abs().max()), 0.)
        original = a['logits'].clone()
        model.forward_window(batch(), state)
        torch.testing.assert_close(a['logits'], original, rtol=0, atol=0)
        reset, _ = model.forward_window(x)
        torch.testing.assert_close(a['logits'], reset['logits'], rtol=0, atol=0)

    def test_padding_singleton_and_no_fake_history_bias(self):
        model = self.model().train()
        activate(model)
        for extra in model.extra:
            nn.init.constant_(extra.history_norm.bias, 10.)
        x = batch()
        x['valid'][1] = False
        out, state = model.forward_window(x)
        self.assertFalse(state['history_valid'][1])
        self.assertEqual(float(state['memory'][1].abs().sum()), 0.)
        for key, value in out['diagnostics'].items():
            if 'no_history_read_norm' in key:
                self.assertEqual(float(value.sum()), 0.)
        x['valid'].fill_(False)
        padded, same = model.forward_window(x, state)
        self.assertIs(same, state)
        self.assertEqual(float(padded['logits'].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(out['logits']).all())

    def test_chunk_equivalence_and_detach(self):
        model = self.model().eval()
        activate(model)
        windows = [batch() for _ in range(6)]
        def run(cut):
            state, result = None, []
            for i, x in enumerate(windows):
                if i == cut:
                    previous = state['memory']
                    state = detach_state(state)
                    self.assertIsNone(state['memory'].grad_fn)
                    torch.testing.assert_close(previous, state['memory'], rtol=0, atol=0)
                out, state = model.forward_window(x, state)
                result.append(out['logits'])
            return torch.stack(result)
        torch.testing.assert_close(run(-1), run(3), rtol=0, atol=0)

    def test_geometry_schedule_and_exact_tail_weights(self):
        params = sample_spatial_params(0, 2, 'video')
        self.assertEqual(params, sample_spatial_params(0, 2, 'video'))
        cfg = GCSTRConfig(sensor_width=320, sensor_height=240)
        for n in (50, 200):
            events = (np.arange(n), np.full(n, 150), np.full(n, 110), np.zeros(n, dtype=int))
            result, fallback = apply_spatial_params(events, params, keyed_rng('jitter', n), cfg)
            self.assertGreater(len(result[0]), 0)
            self.assertEqual(fallback, 0)
        rows = [dict(video_id=str(v), window_id=str(j), start_us=str(j*500000), label_id=str(v))
                for v, count in enumerate((1, 5, 9, 12, 2)) for j in range(count)]
        groups = video_groups(rows)
        schedule = make_schedule(groups, 2, 4, 0, 1, True)
        covered, weighted = [], 0.
        coefficient = len(schedule)/len(rows)
        for chunk in schedule:
            indices = [groups[v][t] for t in range(chunk['start'], chunk['stop'])
                       for v in chunk['videos'] if t < len(groups[v])]
            covered += indices
            weighted += len(indices)*coefficient
        self.assertEqual(sorted(covered), list(range(len(rows))))
        self.assertAlmostEqual(weighted/len(schedule), 1.)

    def test_metrics_vote_loss_and_resume_identity(self):
        metric = PredictionMetrics()
        logits = torch.tensor([[4., 0., 0., 0., 0., 0.], [0., 2., 0., 0., 0., 0.]])
        metric.update(logits, torch.tensor([0, 0]), ['a', 'a'])
        result, _ = metric.finish()
        meanp = logits.double().softmax(-1).mean(0)
        self.assertAlmostEqual(result['video_nll'], float(-meanp[0].log()))
        target = torch.tensor([0, 1])
        expected = nn.functional.cross_entropy(logits, target, reduction='sum')*1.5
        torch.testing.assert_close(objective_sum(dict(logits=logits, point_logits=logits), target, .5), expected)
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        out, _ = model.forward_window(batch())
        objective_sum(out, target, .5).backward()
        optimizer.step()
        ident = dict(architecture='synthetic', mode='GM')
        saved = dict(identity=ident, smoke=True, epoch=1, best={'nll': 1.}, total_steps=1,
                     model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), rng_state=rng_state())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'
            atomic_save(path, saved)
            loaded = torch.load(path, weights_only=False)
            clone = self.model()
            opt = torch.optim.AdamW(clone.parameters(), lr=.001)
            self.assertEqual(restore_checkpoint(loaded, clone, opt, ident, True)[0], 2)
            with self.assertRaises(ValueError):
                restore_checkpoint(loaded, clone, opt, dict(mode='G'), True)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)

    def test_real_training_loop_synthetic_resume_matches_uninterrupted(self):
        config = dict(seed=0, workers=0, point_aux_weight=.5, gradient_clip=1.)
        model = self.model()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        optimizer.param_groups[0]['name'] = 'all'
        with contextlib.redirect_stdout(io.StringIO()):
            first, _ = run_pass(model, TinySequence(1), torch.device('cpu'), config, optimizer)
            saved = copy.deepcopy(dict(identity={'test': 1}, smoke=True, epoch=1, data_epoch=1,
                best={'nll': 1.}, total_steps=first['steps'], rng_state=rng_state(),
                model_state=model.state_dict(), optimizer_state=optimizer.state_dict()))
            run_pass(model, TinySequence(2), torch.device('cpu'), config, optimizer)
            clone = self.model()
            opt = torch.optim.AdamW([p for p in clone.parameters() if p.requires_grad], lr=.001)
            opt.param_groups[0]['name'] = 'all'
            restore_checkpoint(saved, clone, opt, {'test': 1}, True)
            result, _ = run_pass(clone, TinySequence(2), torch.device('cpu'), config, opt)
        self.assertEqual(result['main']['windows'], 11)
        self.assertGreater(result['continued_chunks'], 0)
        self.assertGreater(result['singleton_windows'], 0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)

    def test_no_history_gate_observation_really_zero_after_norm_bias(self):
        model = self.model('G')
        extra = model.extra[0]
        nn.init.constant_(extra.history_norm.bias, 5.)
        captured = []
        hook = extra.gate[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        model.forward_window(batch())
        hook.remove()
        self.assertEqual(float(captured[0][:, 16:24].abs().sum()), 0.)

    def test_initial_regression_entry_synthetic_only(self):
        report = initial_regression(self.model(), TinySequence(1), torch.device('cpu'))
        self.assertTrue(report['parameters_buffers_exact'])
        self.assertEqual(len(report['windows']), 2)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
