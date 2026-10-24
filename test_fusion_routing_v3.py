"""CPU mathematical, identity and synthetic end-to-end stage-3 regressions."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fusion_router import Router, mix_log_probs
from fusion_router_v3 import RouterV3, LOSSES, loss_terms, training_objective
from routing_data import digest, json_write, ObservationNormalizer
from train_fusion_routing import train, load_checkpoint as load_legacy, batches
from train_fusion_routing_v3 import fit, inputs, restore_checkpoint, evaluate_data, write_result
from evaluate_fusion_routing_v3 import evaluate, paired_tables, baseline_train
from test_fusion_routing import synthetic_cache


class MathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.logits=torch.randn(2,3,5,3)
        self.obs=torch.randn(2,3,17)
        self.mask=torch.tensor([[True,False,False],[True,True,True]])
        self.dt=torch.tensor([[0.,0.,0.],[0.,.5,1.5]])
        self.labels=torch.tensor([[0,0,0],[1,1,1]])
        self.args=(self.logits,self.obs,self.mask,self.dt)

    def test_static_init_saturation_and_common_rng(self):
        for a,e in [(1.33,-2.26),(40.,-40.),(-40.,40.),(.1,.2)]:
            state=dict(alpha_logit=torch.tensor(a),eta_logit=torch.tensor(e))
            weights=None
            for mode in ('fixed','global','current'):
                model=RouterV3(dict(eta_mode=mode,seed=4)).initialize_static(state)
                out=model(*self.args)
                expected=mix_log_probs(self.logits,torch.tensor(a).sigmoid(),torch.tensor(e).sigmoid())
                torch.testing.assert_close(out['log_probs'],expected,rtol=0,atol=0)
                for loss in LOSSES:torch.testing.assert_close(loss_terms(out['log_probs'],self.labels,self.mask,loss),loss_terms(expected,self.labels,self.mask,loss))
                self.assertFalse(bool(model.alpha_exact_endpoint))
                current=model.alpha_net.state_dict()
                if weights is not None:
                    for k in weights:torch.testing.assert_close(weights[k],current[k],atol=0,rtol=0)
                weights=current
                self.assertEqual(float(model.b_alpha),a if a in (40.,-40.) else float(torch.tensor(a)))
        state=torch.get_rng_state();RouterV3(dict(eta_mode='current'));self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_legacy_copy_and_explicit_endpoints(self):
        for alpha in (0.,1.,.2):
            old=Router(dict(mode='current',hidden=4,alpha0=alpha,eta=.3))
            with torch.no_grad():old.current[-1].weight.normal_()
            new=RouterV3(dict(eta_mode='fixed',hidden=4)).copy_legacy_current(old)
            torch.testing.assert_close(new(*self.args)['log_probs'],old(*self.args)['log_probs'])
        for a,e in [(0.,0.),(1.,0.),(0.,1.),(1.,1.)]:
            p=self.logits.softmax(-1)
            expected=(1-e)*p[...,0,:]+e*(a*p[...,2:,:].mean(-2)+(1-a)*p[...,1,:])
            torch.testing.assert_close(mix_log_probs(self.logits,a,e).exp(),expected)

    def test_eta_gradients_and_no_external_mutation(self):
        state=dict(alpha_logit=torch.tensor(1.3),eta_logit=torch.tensor(-2.))
        original=copy.deepcopy(state)
        for mode in ('fixed','global','current'):
            model=RouterV3(dict(eta_mode=mode,hidden=4)).initialize_static(state)
            optimizer=torch.optim.Adam(model.parameters(),lr=.03)
            for step in range(2):
                optimizer.zero_grad();out=model(*self.args)
                loss_terms(out['log_probs'],self.labels,self.mask,'video_nll').mean().backward()
                self.assertFalse(model.b_alpha.requires_grad)
                if mode=='fixed':self.assertIsNone(model.b_eta.grad)
                else:self.assertGreater(float(model.b_eta.grad.abs()),0)
                if mode=='current':
                    self.assertGreater(float(model.eta_net[-1].weight.grad.abs().sum()),0)
                    if step==0:self.assertEqual(float(model.eta_net[0].weight.grad.abs().sum()),0)
                    else:self.assertGreater(float(model.eta_net[0].weight.grad.abs().sum()),0)
                optimizer.step()
            for k in state:torch.testing.assert_close(state[k],original[k],atol=0,rtol=0)
            out=model(*self.args)
            torch.testing.assert_close(out['w_A']+out['w_F']+out['w_P'],torch.ones_like(out['alpha']))

    def test_video_math_jensen_and_gradients(self):
        p=torch.tensor([[[.9,.1],[.1,.9]],[[.8,.2],[.8,.2]]],dtype=torch.float64)
        mask=torch.tensor([[True,True],[True,False]]);labels=torch.zeros(2,2,dtype=torch.long)
        lp=p.log().requires_grad_()
        expected={'window_nll':-(np.log(.9)+np.log(.1)+np.log(.8))/3,
            'video_nll':-(np.log(.5)+np.log(.8))/2,
            'video_balanced_window_nll':-((np.log(.9)+np.log(.1))/2+np.log(.8))/2}
        for mode,value in expected.items():self.assertAlmostEqual(float(loss_terms(lp,labels,mask,mode).mean()),value)
        self.assertLessEqual(float(loss_terms(lp,labels,mask,'video_nll').mean()),float(loss_terms(lp,labels,mask,'video_balanced_window_nll').mean()))
        loss_terms(lp,labels,mask,'video_nll').mean().backward()
        self.assertGreater(float(lp.grad[mask].abs().sum()),0);self.assertEqual(float(lp.grad[~mask].abs().sum()),0)
        logits=self.logits.clone().requires_grad_();model=RouterV3(dict(eta_mode='current'))
        loss_terms(model(logits,self.obs,self.mask,self.dt)['log_probs'],self.labels,self.mask,'video_nll').mean().backward()
        self.assertGreater(float(logits.grad.abs().sum()),0)
        for mode in LOSSES:
            torch.testing.assert_close(loss_terms(lp[:,:1],labels[:,:1],mask[:,:1],mode),-lp[:,0,0])

    def test_padding_labels_and_epoch_scaling(self):
        lp=self.logits[...,0,:].log_softmax(-1).detach().requires_grad_()
        for mode in LOSSES:
            original=loss_terms(lp,self.labels,self.mask,mode).mean()
            padded=torch.cat([lp,torch.zeros(2,2,3)],1)
            labels=torch.cat([self.labels,torch.full((2,2),-999)],1)
            mask=torch.cat([self.mask,torch.zeros(2,2,dtype=torch.bool)],1)
            actual=loss_terms(padded,labels,mask,mode).mean()
            torch.testing.assert_close(original,actual)
            ga=torch.autograd.grad(original,lp,retain_graph=True)[0];gb=torch.autograd.grad(actual,lp,retain_graph=True)[0]
            torch.testing.assert_close(ga,gb)
            # Unequal videos, final batch of one; epoch-scale average matches global loss.
            expanded=torch.cat([lp,lp[:1]],0);labs=torch.cat([self.labels,self.labels[:1]],0);m=torch.cat([self.mask,self.mask[:1]],0)
            parts=[training_objective(expanded[s],labs[s],m[s],mode,5,3,2) for s in (slice(0,2),slice(2,3))]
            torch.testing.assert_close(sum(parts)/2,loss_terms(expanded,labs,m,mode).mean())
        bad=self.mask.clone();bad[0]=False
        with self.assertRaises(ValueError):loss_terms(lp,self.labels,bad,'video_nll')
        bad=self.labels.clone();bad[1,1]=2
        with self.assertRaises(ValueError):loss_terms(lp,bad,self.mask,'video_nll')


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.cache=synthetic_cache(cls.root/'cache',selection_protocol='original_test_as_val')
        cfg=cls.root/'legacy.json';json_write(cfg,dict(epochs=1,batch_size=2,hidden=4,lr=.001,seed=0,alpha0=.5,eta=.5))
        sources={k:dict(source=str(cls.root/'cache/manifest.json'),sha256=digest(cls.root/'cache/manifest.json')) for k in ('evidence_train','evidence_test')}
        with contextlib.redirect_stdout(io.StringIO()):
            for mode in ('static','current','gru','conditional'):
                train(argparse.Namespace(config=cfg,mode=mode,cache=cls.root/'cache',val_cache=None,
                    train_role='gate_train',val_role='gate_val',allow_seen_expert_validation=False,
                    static_init=None if mode=='static' else cls.root/'static/best_val_nll.pt',
                    output=cls.root/mode,device='cpu',expect_identity=None,selection_protocol='original_test_as_val'))
                for name in ('best_val_nll.pt','config.json','metrics.jsonl'):
                    p=cls.root/mode/name;sources[f'{mode}/{name}']=dict(source=str(p),sha256=digest(p))
        cls.sources=cls.root/'sources.json';json_write(cls.sources,sources)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def config(self,mode='global',loss='window_nll'):
        return dict(sources=str(self.sources),selection_protocol='original_test_as_val',eta_mode=mode,loss_mode=loss,
            epochs=2,batch_size=1,batch_unit='video',hidden=4,lr=.001,weight_decay=0.,gradient_clip=1.,seed=0,ece_bins=15)

    def test_four_smokes_restoration_and_no_refit(self):
        for eta in ('global','current'):
            for loss in ('window_nll','video_nll'):
                name=eta+'_'+loss;output=self.root/name
                with contextlib.redirect_stdout(io.StringIO()),patch.object(ObservationNormalizer,'fit',side_effect=AssertionError('Do not refit')):
                    fit(self.config(eta,loss),output)
                self.assertEqual(json.loads((output/'status.json').read_text())['optimizer_steps'],4)
                self.assertFalse(json.loads((output/'epoch0.json').read_text())['selected'])
                model,norm,saved=restore_checkpoint(output/'last.pt',self.cache)
                optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(saved['optimizer_state'])
                self.assertEqual(len(optimizer.state),len(saved['optimizer_state']['state']))
                for state in optimizer.state.values():self.assertEqual(float(state['step']),4.)
                for a,b in zip(optimizer.state.values(),saved['optimizer_state']['state'].values()):
                    for key in a:torch.testing.assert_close(a[key],b[key])
                _,data,static,_,_=inputs(self.config())
                for k in static['normalizer_state']:torch.testing.assert_close(norm.state_dict()[k],static['normalizer_state'][k])
                result=evaluate_data(model,norm,data,saved['config'])
                self.assertAlmostEqual(result['metrics']['video_nll'],saved['metrics']['validation']['video_nll'])
                actual=evaluate(output/'last.pt',self.root/'cache',self.root/(name+'_eval'),traces=True)
                np.testing.assert_array_equal(result['log_probs'],actual['log_probs'])
                with self.assertRaises(FileExistsError):fit(self.config(),output)
                with self.assertRaises(ValueError):load_legacy(output/'last.pt',self.cache.identity)
                # Actual tensors, not initialization placeholders, define restored eta.
                with torch.no_grad():model.b_eta.add_(.3)
                saved['router_state']=model.state_dict();torch.save(saved,output/'modified.pt')
                again,_,_=restore_checkpoint(output/'modified.pt',self.cache)
                torch.testing.assert_close(again.b_eta,model.b_eta)
                for k,v in again.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k])

    def test_identity_role_and_source_rejection(self):
        with self.assertRaises(ValueError):inputs(dict(self.config(),selection_protocol='strict'))
        with self.assertRaises(ValueError):inputs(dict(self.config(),train_role='gate_val'))
        with self.assertRaises(ValueError):fit(dict(self.config(),eta_mode='fixed'),self.root/'forbidden')
        config=self.config();_,_,static,norm,_=inputs(config)
        # Legacy identity guard remains active for class, expert and observation changes.
        for key in ('class_to_idx','expert_provenance','dataset_id'):
            identity=copy.deepcopy(self.cache.identity);identity[key]={} if key!='dataset_id' else 'different'
            with self.assertRaises(ValueError):load_legacy(self.root/'static/best_val_nll.pt',identity)
        from train_fusion_routing_v3 import code_identity
        model=RouterV3(config).initialize_static(static['router_state'])
        from routing_data import OBSERVATIONS, OBSERVATION_SCHEMA
        from fusion_router_v3 import ARCHITECTURE,MODEL_SCHEMA
        config.update(code=code_identity(),observations=list(OBSERVATIONS),observation_schema=OBSERVATION_SCHEMA,architecture=ARCHITECTURE,model_schema=MODEL_SCHEMA)
        saved=dict(architecture=ARCHITECTURE,model_schema=MODEL_SCHEMA,identity=self.cache.identity,val_cache=self.cache.fingerprint,
            config=config,router_state=model.state_dict(),normalizer_state=norm.state_dict())
        for field in ('model_schema','val_cache','identity'):
            bad=copy.deepcopy(saved);bad[field]='wrong';path=self.root/('bad_'+field+'.pt');torch.save(bad,path)
            with self.assertRaises(ValueError):restore_checkpoint(path,self.cache)
        bad=copy.deepcopy(saved);bad['config']['observations']=[];path=self.root/'badobs.pt';torch.save(bad,path)
        with self.assertRaises(ValueError):restore_checkpoint(path,self.cache)

    def test_baseline_and_pairing_commands(self):
        from routing_data import HEADS
        from fusion_router import FIXED
        from evaluate_fusion_routing_suite import evaluate_one
        config=self.config();_,val,_,norm,_=inputs(config)
        suite=self.root/'synthetic_suite'
        for name in list(HEADS)+[k for k in FIXED if k!='fusion']+['current']:
            model=Router(dict(mode='current',hidden=4)) if name=='current' else Router(dict(mode='fixed',fixed=name if name in FIXED else 'fusion'))
            result=evaluate_one(model,norm,val,config,head=name if name in HEADS else None)
            write_result(suite/f'{name}__normal__prefix_1',result,val,config)
        with contextlib.redirect_stdout(io.StringIO()):baseline_train(config,suite,self.root/'train_baselines')
        self.assertEqual(json.loads((self.root/'train_baselines/status.json').read_text())['methods'],9)
        result=paired_tables([suite/'current__normal__prefix_1'],suite,suite/'fusion_frame__normal__prefix_1',self.root/'pairing')
        comparisons=next(iter(result.values()))
        self.assertEqual(set(comparisons),{'fusion','frame','fusion_frame','current','static_frame'})
        for p in comparisons.values():
            self.assertAlmostEqual(sum(v['contribution_to_dataset_mean'] for v in p['nll_attribution'].values()),p['metrics_right_minus_left']['video_nll'])


if __name__=='__main__':unittest.main()
