"""Synthetic CPU checks, including two-epoch TBPTT/resume; no real CUDA claims."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

from gcstr_feature_memory_ablation import AblationFusion, INTERVENTIONS, policy, validate_policy
from gcstr_feature_memory_fusion import FeatureMemoryFusion, detach_state
from gcstr_video_sequence_dataset import make_schedule, video_groups
from feature_ablation_runtime import run_pass, paired, configuration, verify_evaluation_checkpoint
from train_gcstr_feature_memory import (objective_sum, rng_state, restore_rng, restore_checkpoint,
                                      atomic_save, initial_regression, seed_all)
from test_gcstr_feature_memory import TinyFusion, TinySequence, batch

VARIANTS = [('B', 'base'), ('G', 'base'), ('GM', 'base'), ('G', 'fixed_gate'),
            ('G', 'global_gate'), ('G', 'point_query'), ('GM', 'gate_only'),
            ('GM', 'feature_only'), ('GM', 'no_write_gate'), ('GM', 'fixed_update'),
            ('G', 'point_complement'), ('G', 'frame_complement')]


class Sequence(TinySequence):
    def __init__(self, epoch):
        self.epoch = epoch
        self.rows = [dict(video_id=str(v), window_id=str(t), start_us=t*500000, label_id=v)
                     for v, n in enumerate((9, 5, 1)) for t in range(n)]
        self.groups = video_groups(self.rows)
        self.windows = len(self.rows)
        self.schedule = make_schedule(self.groups, 2, 4, 0, epoch, True)


def activate(model):
    with torch.no_grad():
        for extra in getattr(model, 'extra', []):
            extra.delta[-1].weight.normal_(0, .05)
            extra.gate[-1].weight.normal_(0, .3)
            extra.history_norm.bias.fill_(.7)
        if model.mode == 'GM':
            for layer in model.memory.output:
                layer.weight.normal_(0, .05)
                layer.bias.fill_(.4)


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_all(17)
        self.base = TinyFusion()
        self.config = dict(seed=0, workers=0, point_aux_weight=.5, gradient_clip=1.,
                           lr_point=.001, lr_existing_fusion=.001, lr_new_modules=.001)

    def model(self, mode='GM', variant='base'):
        return AblationFusion(copy.deepcopy(self.base), mode, 3, variant)

    def test_policy_and_config_reject_contradictions(self):
        with self.assertRaises(ValueError):
            policy('G', 'gate_only')
        c = dict(mode='GM', base_mode='GM', ablation_id='feature_only', **policy('GM', 'feature_only'))
        validate_policy(c)
        c['history_to_gate'] = True
        with self.assertRaises(ValueError):
            validate_policy(c)
        paths = list(Path('configs/feature_memory_ablations').glob('*.json'))
        self.assertEqual(len(paths), len(VARIANTS))
        for path in paths:
            configuration(path)

    def test_evaluation_identity_only_allows_verified_path_relocation(self):
        from gcstr_feature_memory_ablation import ARCHITECTURE
        c = dict(experiment='/new/experiment', store='/new/store', source_checkpoint='/new/source', seed=0)
        expected = dict(architecture=ARCHITECTURE, config=c, code={'model.py': 'sha'}, manifest_sha256='manifest')
        actual = copy.deepcopy(expected)
        actual['config']['store'] = '/old/store'
        saved = dict(smoke=False, identity=actual, model_state={'example': torch.ones(1)})
        m = Mock()
        with patch('feature_ablation_runtime.identity', return_value=expected):
            verify_evaluation_checkpoint(saved, c, None, m)
            m.load_state_dict.assert_called_once_with(saved['model_state'], strict=True)
            saved['identity']['config']['seed'] = 1
            with self.assertRaises(ValueError):
                verify_evaluation_checkpoint(saved, c, None, m)
            saved['identity']['config']['seed'] = 0
            saved['identity']['code']['model.py'] = 'changed'
            with self.assertRaises(ValueError):
                verify_evaluation_checkpoint(saved, c, None, m)
            saved['smoke'] = True
            with self.assertRaises(ValueError):
                verify_evaluation_checkpoint(saved, c, None, m)

    def test_shared_initialization_rng_and_zero_residual_all_variants(self):
        legacy = {m: FeatureMemoryFusion(copy.deepcopy(self.base), m, 3).eval() for m in ('B', 'G', 'GM')}
        for mode, variant in VARIANTS:
            with self.subTest(variant=variant, mode=mode):
                before = torch.get_rng_state().clone()
                with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA RNG changed')):
                    model = self.model(mode, variant).eval()
                torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
                for name, value in legacy[mode].state_dict().items():
                    torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
                state = None
                for _ in range(3):
                    x = batch()
                    output, state = model.forward_window(x, state)
                    ref = legacy[mode].base.forward_batch_outputs(x)
                    torch.testing.assert_close(output['logits'], ref['logits'], rtol=0, atol=0)
                if mode != 'GM':
                    self.assertFalse(hasattr(model, 'memory'))

    def test_legacy_nonzero_weights_outputs_states_gradients_and_optimizer(self):
        for mode in ('B', 'G', 'GM'):
            model = self.model(mode).eval()
            activate(model)
            old = FeatureMemoryFusion(copy.deepcopy(self.base), mode, 3).eval()
            old.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(old.parameter_groups(self.config)[1], model.parameter_groups(self.config)[1])
            a, b = None, None
            for _ in range(3):
                x = batch()
                p, a = old.forward_window(x, a)
                q, b = model.forward_window(x, b)
                for key in ('logits', 'point_logits'):
                    torch.testing.assert_close(p[key], q[key], rtol=0, atol=0)
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            p['logits'].sum().backward()
            q['logits'].sum().backward()
            for (name, pa), (_, pb) in zip(old.named_parameters(), model.named_parameters()):
                if pa.grad is not None:
                    self.assertIsNotNone(pb.grad, name)
                    torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)

    def test_fixed_gate_and_optimizer_exclusions(self):
        for variant, expected in [('fixed_gate', .5), ('point_complement', 0.), ('frame_complement', 1.)]:
            m = self.model('G', variant)
            out, _ = m.forward_window(batch())
            for i in range(3):
                self.assertTrue((out['diagnostics'][f'stage{i}/gate'] == expected).all())
            names = sum(m.parameter_groups(self.config)[1].values(), [])
            self.assertFalse(any('.gate.' in n or '.history_norm.' in n for n in names))
            if variant == 'point_complement':
                self.assertFalse(any(n.startswith('extra.') and any(s in n for s in ('.attention.', '.merge.', '.frame_norm.')) for n in names))

    def test_point_query_only_changes_query_source(self):
        x, y = batch(), batch()
        y.update({v: x[v] for v in ('cstr3', 'xt_p', 'ty_p')})
        captured = {}
        for variant in ('base', 'point_query'):
            m = self.model('G', variant).eval()
            values = []
            handle = m.extra[0].attention['cstr3'].register_forward_pre_hook(
                lambda module, args: values.append(tuple(t.detach().clone() for t in args)))
            m.forward_window(x)
            m.forward_window(y)
            handle.remove()
            torch.testing.assert_close(values[0][1], values[1][1], rtol=0, atol=0)
            if variant == 'base':
                torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0)
            else:
                self.assertFalse(torch.equal(values[0][0], values[1][0]))
            captured[variant] = values
        torch.testing.assert_close(captured['base'][0][1], captured['point_query'][0][1], rtol=0, atol=0)

    def test_history_paths_are_independent_and_no_history_bias_is_masked(self):
        for variant in ('base', 'gate_only', 'feature_only'):
            m = self.model('GM', variant).eval()
            activate(m)
            x = batch()
            first, state = m.forward_window(x)
            out, _ = m.forward_window(x, state)
            for i in range(3):
                d = out['diagnostics']
                self.assertEqual(float(first['diagnostics'][f'stage{i}/memory_residual_ratio'].sum()), 0.)
                self.assertEqual(float(first['diagnostics'][f'stage{i}/gate_history_norm'].sum()), 0.)
                self.assertEqual(float(d[f'stage{i}/memory_residual_ratio'].sum()) == 0, variant == 'gate_only')
                self.assertEqual(float(d[f'stage{i}/gate_history_norm'].sum()) == 0, variant == 'feature_only')
                self.assertGreater(float(d[f'stage{i}/read_norm'].sum()), 0.)
            if variant == 'gate_only':
                self.assertTrue(all(not p.requires_grad for p in m.memory.output.parameters()))

    def test_fixed_write_formulas_and_candidate_preserves_history(self):
        for variant, value in [('no_write_gate', 1.), ('fixed_update', .5)]:
            m = self.model('GM', variant)
            state = m.initial_state(2, 'cpu')
            state['memory'].normal_()
            tokens = [torch.randn(2, 2, 8) for _ in range(3)]
            captures = []
            hook = m.memory.write.register_forward_hook(lambda mod, args, out: captures.append((args, out)))
            result, u = m.write_state(tokens, state, torch.ones(2, 1))
            hook.remove()
            self.assertTrue((u == value).all())
            self.assertEqual(captures[0][0][1].shape, (2, 6, 8))
            torch.testing.assert_close(result['memory'], (1-value)*state['memory']+value*captures[0][1])
            torch.testing.assert_close(captures[0][0][0], m.memory.query[None]+m.memory.state_norm(state['memory']))
            changed, _ = m.write_state(tokens, m.initial_state(2, 'cpu'), torch.ones(2, 1))
            self.assertFalse(torch.equal(changed['memory'], result['memory']))
            self.assertTrue(all(not p.requires_grad for p in m.memory.write_gate.parameters()))
            self.assertTrue(all(not p.requires_grad for p in m.memory.candidate_norm.parameters()))

    def test_interventions_reset_equivalence_with_bias_padding_and_real_gap(self):
        m = self.model().eval()
        activate(m)
        inputs = [batch() for _ in range(5)]
        inputs[2]['valid'][1] = False
        inputs[3]['valid'].fill_(False)
        for i, x in enumerate(inputs):
            x['gap_seconds'].fill_(i*.7)
        outputs = {}
        for mode in INTERVENTIONS:
            m.set_intervention(mode)
            state, values = None, []
            for x in inputs:
                before = {k: v.detach().clone() for k, v in state.items()} if state is not None else None
                out, state = m.forward_window(x, state)
                if before is not None and not x['valid'][1]:
                    torch.testing.assert_close(state['memory'][1], before['memory'][1], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(out['logits']).all())
                values.append(out['logits'])
            outputs[mode] = torch.stack(values)
        torch.testing.assert_close(outputs['reset_each_window'], outputs['disable_both_history_outputs'], rtol=0, atol=0)
        self.assertFalse(torch.equal(outputs['normal'], outputs['disable_both_history_outputs']))
        m.train()
        with self.assertRaises(ValueError):
            m.forward_window(inputs[0])

    def test_causality_and_chunk_detach(self):
        m = self.model().eval()
        activate(m)
        x, y = batch(), batch()
        old, state = m.forward_window(x)
        changed = copy.deepcopy(m)
        with torch.no_grad():
            changed.memory.write_gate[0].bias.add_(10)
        same, new_state = changed.forward_window(x)
        torch.testing.assert_close(old['logits'], same['logits'], rtol=0, atol=0)
        a, _ = m.forward_window(y, state)
        b, _ = changed.forward_window(y, new_state)
        self.assertFalse(torch.equal(a['logits'], b['logits']))
        c, _ = m.forward_window(y, detach_state(state))
        torch.testing.assert_close(a['logits'], c['logits'], rtol=0, atol=0)
        self.assertIsNone(detach_state(state)['memory'].grad_fn)

    def test_gradients_recover_all_variants_and_frozen_buffers(self):
        for mode, variant in VARIANTS:
            with self.subTest(mode=mode, variant=variant):
                m = self.model(mode, variant)
                before = {n: p.clone() for n, p in m.named_parameters() if not p.requires_grad}
                bn = {n: b.clone() for n, b in m.base.point_encoder.bn.named_buffers()}
                opt = torch.optim.AdamW(m.parameter_groups(self.config)[0], weight_decay=1e-4)
                reached = set()
                for _ in range(5):
                    opt.zero_grad(set_to_none=True)
                    state, loss = None, 0
                    for _ in range(3):
                        out, state = m.forward_window(batch(), state)
                        loss = loss+objective_sum(out, torch.tensor([1, 3]), .5)
                    loss.backward()
                    for name, p in m.named_parameters():
                        if p.grad is not None:
                            self.assertTrue(torch.isfinite(p.grad).all(), name)
                            if p.grad.abs().sum() > 0:
                                reached.add(name)
                    opt.step()
                expected = ['base.point_encoder.proj', 'base.classifier']
                if mode != 'B':
                    expected += ['extra.0.delta']
                    if variant != 'point_complement':
                        expected += ['extra.0.attention', 'extra.0.merge']
                    if m.settings['gate_policy'] == 'learned':
                        expected += ['extra.0.gate']
                    if variant == 'global_gate':
                        expected += ['extra.0.global_gate_logit']
                if mode == 'GM':
                    expected += ['memory.read', 'memory.write', 'memory.query']
                    if variant not in ('no_write_gate', 'fixed_update'):
                        expected += ['memory.write_gate']
                for prefix in expected:
                    self.assertTrue(any(n.startswith(prefix) for n in reached), prefix)
                for n, value in before.items():
                    torch.testing.assert_close(dict(m.named_parameters())[n], value, rtol=0, atol=0)
                for n, value in bn.items():
                    torch.testing.assert_close(dict(m.base.point_encoder.bn.named_buffers())[n], value, rtol=0, atol=0)

    def test_two_epoch_real_loop_and_disk_resume_all_variants(self):
        for mode, variant in VARIANTS:
            with self.subTest(mode=mode, variant=variant), contextlib.redirect_stdout(io.StringIO()):
                m = self.model(mode, variant)
                opt = torch.optim.AdamW(m.parameter_groups(self.config)[0])
                first, _, _ = run_pass(m, Sequence(1), torch.device('cpu'), self.config, opt)
                ident = dict(mode=mode, variant=variant)
                saved = dict(identity=ident, smoke=True, epoch=1, data_epoch=1, best={'nll': 1.},
                    total_steps=first['steps'], model_state=m.state_dict(), optimizer_state=opt.state_dict(), rng_state=rng_state())
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/'last.pt'
                    atomic_save(path, saved)
                    del saved
                    second, _, _ = run_pass(m, Sequence(2), torch.device('cpu'), self.config, opt)
                    clone = self.model(mode, variant)
                    opt2 = torch.optim.AdamW(clone.parameter_groups(self.config)[0])
                    loaded = torch.load(path, weights_only=False)
                    with self.assertRaises(ValueError):
                        restore_checkpoint(loaded, clone, opt2, {'variant': 'wrong'}, True)
                    restore_checkpoint(loaded, clone, opt2, ident, True)
                    result, _, _ = run_pass(clone, Sequence(2), torch.device('cpu'), self.config, opt2)
                self.assertEqual(result['main']['windows'], 15)
                self.assertGreater(first['continued_chunks'], 0)
                self.assertGreater(first['singleton_windows'], 0)
                for name, value in m.state_dict().items():
                    torch.testing.assert_close(value, clone.state_dict()[name], rtol=0, atol=0)
                self.assertEqual(second['main'], result['main'])

    def test_trace_pairs_and_null_fields(self):
        with contextlib.redirect_stdout(io.StringIO()):
            g = self.model('G').eval()
            metrics, _, trace = run_pass(g, Sequence(0), torch.device('cpu'), self.config, capture=True)
        self.assertIsNone(metrics['diagnostics']['memory/write_gate'])
        self.assertEqual(paired(trace, trace)['window_probability_max_abs'], 0.)
        altered = copy.deepcopy(trace)
        altered['rows'][0]['points_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            paired(trace, altered)


if __name__ == '__main__':
    unittest.main()
