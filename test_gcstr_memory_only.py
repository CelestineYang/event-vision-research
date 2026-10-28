"""M structural, causal, gradient and epoch-resume tests on synthetic CPU data."""
import copy
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import torch
from gcstr_memory_only_fusion import MemoryOnlyFusion
from gcstr_feature_memory_fusion import FeatureMemoryFusion
from thu_chl_memory_only_runtime import run_pass, configuration
from test_gcstr_feature_memory import TinyFusion, batch
from test_gcstr_feature_memory_ablation import Sequence
from train_gcstr_feature_memory import seed_all, objective_sum, atomic_save, restore_checkpoint, rng_state


class MemoryOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_all(17)
        self.base = TinyFusion()
        self.config = dict(seed=0, workers=0, point_aux_weight=.5, gradient_clip=1.,
                           lr_point=.001, lr_existing_fusion=.001, lr_new_modules=.001)

    def model(self):
        return MemoryOnlyFusion(copy.deepcopy(self.base), seed=3)

    def test_structure_shared_initialization_and_rng(self):
        rng = torch.get_rng_state().clone()
        m = self.model()
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        gm = FeatureMemoryFusion(copy.deepcopy(self.base), 'GM', 3)
        self.assertFalse(hasattr(m, 'extra'))
        for k,v in gm.memory.state_dict().items():
            torch.testing.assert_close(v, m.memory.state_dict()[k], rtol=0, atol=0)
        for i,norm in enumerate(m.history_norm):
            for k,v in norm.state_dict().items():
                torch.testing.assert_close(v, gm.extra[i].history_norm.state_dict()[k], rtol=0, atol=0)
        groups,_=m.parameter_groups(self.config)
        self.assertEqual(sum(len(g['params']) for g in groups),sum(p.requires_grad for p in m.parameters()))

    def test_reference_gm_without_second_term_and_modified_writer(self):
        m=self.model().eval();gm=FeatureMemoryFusion(copy.deepcopy(self.base),'GM',3).eval()
        with torch.no_grad():
            for layer in m.memory.output:
                layer.weight.normal_(0,.1);layer.bias.fill_(.3)
            gm.memory.load_state_dict(m.memory.state_dict())
            # GM PF heads remain zero; this is a reference forward with term2=0.
            state_a=state_b=None
            for _ in range(4):
                x=batch();a,state_a=m.forward_window(x,state_a);b,state_b=gm.forward_window(x,state_b)
                torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
                for k in state_a:torch.testing.assert_close(state_a[k],state_b[k],rtol=0,atol=0)

    def test_zero_init_first_window_padding_and_history_effect(self):
        m=self.model().eval();x=batch()
        with torch.no_grad():
            expected=m.base.forward_batch_outputs(x)['logits'];a,state=m.forward_window(x)
            torch.testing.assert_close(a['logits'],expected,rtol=0,atol=0)
            for layer in m.memory.output:
                layer.weight.normal_(0,.1);layer.bias.fill_(.8)
            for norm in m.history_norm:norm.bias.fill_(.7)
            first,_=m.forward_window(x)
            torch.testing.assert_close(first['logits'],expected,rtol=0,atol=0)
            later,_=m.forward_window(x,state)
            self.assertGreater((later['logits']-first['logits']).abs().max().item(),1e-6)
            masked=copy.deepcopy(x);masked['valid'][1]=False;masked['events'][1]=float('nan')
            out,next_state=m.forward_window(masked,state)
            self.assertTrue(torch.isfinite(out['logits']).all())
            for k in state:torch.testing.assert_close(next_state[k][1],state[k][1],rtol=0,atol=0)
            masked['valid'][:]=False;_,same=m.forward_window(masked,state)
            self.assertIs(same,state)

    def test_write_cannot_affect_current_prediction(self):
        m=self.model().eval();other=copy.deepcopy(m)
        with torch.no_grad():
            other.memory.query.add_(torch.randn_like(other.memory.query))
            x=batch();a,sa=m.forward_window(x);b,sb=other.forward_window(x)
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        self.assertFalse(torch.equal(sa['memory'],sb['memory']))

    def test_gradient_recovery_and_frozen_buffers(self):
        m=self.model();opt=torch.optim.AdamW(m.parameter_groups(self.config)[0]);reached=set()
        frozen={k:v.clone() for k,v in m.state_dict().items() if k.startswith('base.image_encoders.') or 'running_' in k or 'num_batches_tracked' in k}
        for _ in range(5):
            opt.zero_grad(set_to_none=True);state=None;loss=0
            for _ in range(3):
                out,state=m.forward_window(batch(),state);loss+=objective_sum(out,torch.tensor([1,3]),.5)
            loss.backward()
            for k,p in m.named_parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(),k)
                    if p.grad.abs().sum()>0:reached.add(k)
            opt.step()
        for prefix in ['base.point_encoder.proj','base.classifier','history_norm','memory.read','memory.output','memory.write.','memory.write_gate','memory.query']:
            self.assertTrue(any(k.startswith(prefix) for k in reached),prefix)
        for k,v in frozen.items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)

    def test_training_loop_resume_and_na_diagnostics(self):
        m=self.model();opt=torch.optim.AdamW(m.parameter_groups(self.config)[0]);ident={'mode':'M'}
        with contextlib.redirect_stdout(io.StringIO()),tempfile.TemporaryDirectory() as tmp:
            first,_,_=run_pass(m,Sequence(1),torch.device('cpu'),self.config,opt)
            path=Path(tmp)/'last.pt'
            atomic_save(path,dict(identity=ident,smoke=True,epoch=1,data_epoch=1,best={'top1':.1,'nll':2.},total_steps=first['steps'],model_state=m.state_dict(),optimizer_state=opt.state_dict(),rng_state=rng_state()))
            second,_,_=run_pass(m,Sequence(2),torch.device('cpu'),self.config,opt)
            clone=self.model();opt2=torch.optim.AdamW(clone.parameter_groups(self.config)[0]);saved=torch.load(path,weights_only=False)
            with self.assertRaises(ValueError):restore_checkpoint(saved,clone,opt2,{'mode':'GM'},True)
            restore_checkpoint(saved,clone,opt2,ident,True)
            result,_,_=run_pass(clone,Sequence(2),torch.device('cpu'),self.config,opt2)
        self.assertEqual(result['main'],second['main']);self.assertEqual(result['main']['windows'],15)
        self.assertGreater(first['continued_chunks'],0);self.assertGreater(first['singleton_windows'],0)
        self.assertIsNone(result['gate_quantiles']);self.assertIsNone(result['diagnostics']['stage0/gate'])
        self.assertIsNotNone(result['diagnostics']['memory/write_gate'])
        for k,v in m.state_dict().items():torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)

    def test_configuration(self):
        c=configuration('configs/examples/thu_chl_memory_only/M.json')
        self.assertEqual(c['video_batch_size_per_gpu'],48)
        self.assertFalse(c['history_to_gate']);self.assertEqual(c['lr_existing_fusion'],5e-5)

if __name__=='__main__':unittest.main()
