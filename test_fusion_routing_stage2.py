"""CPU-only tests; no real expert evaluation or fitting."""
import copy
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from analyze_fusion_routing import compare, analyze
from evaluate_fusion_routing_suite import evaluate_one, run_suite, verify_reproduction
from fusion_router import Router, FIXED, mix_log_probs
from routing_data import ObservationNormalizer, RoutingDataset, collate_sequences
from routing_stage2 import forward_diagnostic, PrefixDataset, restored, observers, verified_artifacts
from test_fusion_routing import synthetic_cache
from train_fusion_routing import evaluate, load_checkpoint, code_identity, train
from routing_data import digest, json_write


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.cache=synthetic_cache(self.root/'cache',selection_protocol='original_test_as_val')
        self.data=RoutingDataset(self.cache,'gate_val',sequence=True)
        self.batch=collate_sequences([self.data[i] for i in range(len(self.data))])
        self.norm=ObservationNormalizer()

    def model(self, mode):
        m=Router(dict(mode=mode,hidden=4,alpha0=.7,eta=.1,ema_decay=.9))
        with torch.no_grad():
            for p in m.parameters():p.uniform_(-.5,.5)
        return m.eval()

    def forward(self,m,**kw):
        b=self.batch
        return forward_diagnostic(m,b['logits'],self.norm(b['observations']),b['mask'],b['dt'],**kw)

    def test_suite_matches_old_metrics_and_fixed_math(self):
        for mode in ('static','current','gru','conditional'):
            m=self.model(mode)
            old,_=evaluate(m,self.norm,self.data,dict(batch_size=2),'cpu')
            result=evaluate_one(m,self.norm,PrefixDataset(self.data),dict(batch_size=2))
            verify_reproduction(result['metrics'],old)
            np.testing.assert_allclose(np.exp(result['log_probs']).sum(-1),1,atol=1e-6)
        for fixed,(a,e) in FIXED.items():
            m=Router(dict(mode='fixed',fixed=fixed))
            out,_=self.forward(m,trace=True)
            torch.testing.assert_close(out['log_probs'],mix_log_probs(self.batch['logits'],a,e))

    def test_trace_step_whole_weights_and_causality(self):
        for mode in ('static','current','gru','conditional'):
            m=self.model(mode)
            whole,_=self.forward(m)
            stepped,trace=self.forward(m,trace=True)
            torch.testing.assert_close(whole['log_probs'],stepped['log_probs'])
            torch.testing.assert_close(trace['w_A']+trace['w_F']+trace['w_P'],torch.ones_like(trace['alpha']))
            torch.testing.assert_close(trace['w_A'],1-stepped['eta'])
            if whole['state'] is not None:torch.testing.assert_close(whole['state'],stepped['state'])
            b=self.batch;obs=self.norm(b['observations']).clone();obs[:,2]+=500
            changed=m(b['logits'],obs,b['mask'],b['dt'])
            torch.testing.assert_close(changed['log_probs'][:,:2],whole['log_probs'][:,:2])
            # Single-video forwards prove batch/video isolation, including padding.
            for i in range(2):
                n=int(b['mask'][i].sum())
                one=m(b['logits'][i:i+1,:n],self.norm(b['observations'][i:i+1,:n]),b['mask'][i:i+1,:n],b['dt'][i:i+1,:n])
                torch.testing.assert_close(one['log_probs'][0],whole['log_probs'][i,:n])
                if whole['state'] is not None:torch.testing.assert_close(one['state'][0],whole['state'][i])
            self.assertFalse(any(module._forward_hooks for module in m.modules()))

    def test_interventions_preserve_observation_dt_and_state_updates(self):
        for mode in ('static','current','gru','conditional'):
            m=self.model(mode);normal,_=self.forward(m,trace=True)
            seen=[]
            handle=m.register_forward_pre_hook(lambda module,args:seen.append((args[1].clone(),args[3].clone())))
            reset,tr=self.forward(m,trace=True,intervention='reset_each_window');handle.remove()
            torch.testing.assert_close(torch.cat([x[0] for x in seen],1),self.norm(self.batch['observations']))
            torch.testing.assert_close(torch.cat([x[1] for x in seen],1),self.batch['dt'])
            zero,zt=self.forward(m,trace=True,intervention='zero_memory_readout')
            if mode in ('static','current'):
                torch.testing.assert_close(normal['log_probs'],reset['log_probs'])
                torch.testing.assert_close(normal['log_probs'],zero['log_probs'])
            else:
                self.assertTrue((tr['state_before_norm']==0).all())
                self.assertTrue((zt['delta_memory']==0).all())
                self.assertTrue((zt['delta_memory_unmasked'].abs()>0).any())
                torch.testing.assert_close(normal['state'],zero['state'])
                self.assertFalse(torch.allclose(reset['alpha'],zero['alpha']))

    def test_ema_math_prefix_and_hook_cleanup(self):
        m=self.model('current');raw,_=self.forward(m)
        out,_=self.forward(m,ema_decay=.9,trace=True)
        whole,_=self.forward(m,ema_decay=.9)
        torch.testing.assert_close(out['log_probs'],whole['log_probs'])
        a=torch.full((2,),.7)
        for t in range(3):
            a=torch.where(self.batch['mask'][:,t],.9*a+.1*raw['alpha'][:,t],a)
            torch.testing.assert_close(out['alpha'][:,t][self.batch['mask'][:,t]],a[self.batch['mask'][:,t]])
        for ratio in (.25,.5,.75,1.):
            prefix=PrefixDataset(self.data,ratio)
            for i in range(2):
                count=max(1,int(np.ceil(ratio*len(self.data[i]['rows']))))
                self.assertEqual(prefix[i]['rows'],self.data[i]['rows'][:count])
        result=evaluate_one(m,self.norm,self.data,dict(batch_size=2))
        full=evaluate_one(m,self.norm,PrefixDataset(self.data,1),dict(batch_size=2),trajectory_path=self.root/'trace.jsonl')
        np.testing.assert_array_equal(result['log_probs'],full['log_probs'])
        with self.assertRaises(RuntimeError):
            with observers(m,{}):raise RuntimeError('fixture')
        self.assertFalse(any(module._forward_hooks for module in m.modules()))

    def test_state_dict_not_config_and_identity_rejected(self):
        m=self.model('current')
        with torch.no_grad():m.fixed_eta.fill_(.094);m.base_logit.fill_(1.3)
        saved=dict(config=m.config,router_state=m.state_dict(),normalizer_state=self.norm.state_dict())
        restored_model,_=restored(saved)
        self.assertAlmostEqual(float(restored_model.fixed_eta),.094,places=6)
        self.assertAlmostEqual(float(restored_model.base_logit),1.3,places=6)
        from routing_data import OBSERVATIONS, OBSERVATION_SCHEMA
        saved.update(schema=2,identity=self.cache.identity)
        saved['config']=dict(saved['config'],code=code_identity(),observations=list(OBSERVATIONS),
                             observation_schema=OBSERVATION_SCHEMA,selection_protocol='original_test_as_val')
        path=self.root/'fixture.pt';torch.save(saved,path)
        load_checkpoint(path,self.cache.identity)
        for field in ('class_to_idx','dataset_id'):
            identity=copy.deepcopy(self.cache.identity);identity[field]={} if field=='class_to_idx' else 'wrong'
            with self.assertRaises(ValueError):load_checkpoint(path,identity)
        saved['config']['observations']=[];torch.save(saved,path)
        with self.assertRaises(ValueError):load_checkpoint(path,self.cache.identity)

    def test_pairing_distinguishes_equal_accuracy_and_rejects_identity(self):
        metadata=dict(dataset_id='fixture',class_to_idx={'a':0,'b':1},cache_fingerprint='x',
                      selection_protocol='original_test_as_val',role='gate_val',prefix=1,
                      metrics=dict(video_top1=.5,video_nll=1,video_brier=1,video_ece=1))
        rows=[dict(dataset_id='fixture',source_split='test',video_id=str(i),label_id=0,
                   prediction=i,fusion_prediction=1,windows=1,window_index=[dict(window_id='0')]) for i in range(2)]
        left=dict(metadata=metadata,index=rows,log_probs=np.log([[.8,.2],[.2,.8]]))
        right=copy.deepcopy(left);right['log_probs']=left['log_probs'][::-1].copy()
        for row in right['index']:row['prediction']=1-row['prediction']
        result=compare(left,right)
        self.assertEqual(result['counts']['prediction_changed'],2)
        self.assertNotEqual(result['sets']['left_rescued'],result['sets']['right_rescued'])
        for field in ('class_to_idx','cache_fingerprint'):
            bad=copy.deepcopy(right);bad['metadata'][field]='wrong'
            with self.assertRaises(ValueError):compare(left,bad)
        bad=copy.deepcopy(right);bad['index'][0]['window_index']=[]
        with self.assertRaises(ValueError):compare(left,bad)
        for row in left['index']:row['prediction']=0
        self.assertEqual(compare(left,left)['counts']['both_wrong'],0)

    def test_synthetic_complete_suite_outputs_and_no_overwrite(self):
        saved={};audit=dict(models={})
        for mode in ('static','current','gru','conditional'):
            model=self.model(mode)
            metrics,_=evaluate(model,self.norm,self.data,dict(batch_size=2))
            saved[mode]=dict(config=model.config,router_state=model.state_dict(),normalizer_state=self.norm.state_dict(),
                             val_role='gate_val',metrics=dict(validation=metrics))
            audit['models'][mode]=dict(epoch=1)
        config=dict(sources='fixture',selection_protocol='original_test_as_val',role='gate_val',batch_size=2,
                    ece_bins=15,ema_decay=.9,prefixes=[1],interventions=['normal','reset_each_window','zero_memory_readout'])
        with patch('evaluate_fusion_routing_suite.verified_artifacts',return_value=(self.cache,self.cache,saved,audit)),contextlib.redirect_stdout(io.StringIO()):
            run_suite(config,self.root/'suite',traces=True)
            with self.assertRaises(FileExistsError):run_suite(config,self.root/'suite')
            with self.assertRaises(ValueError):run_suite(dict(config,ema_decay=.8),self.root/'invalid')
        pairs=analyze(self.root/'suite',self.root/'analysis')
        self.assertIn('current_vs_gru__prefix_1',pairs)
        with self.assertRaises(FileExistsError):analyze(self.root/'suite',self.root/'analysis')

    def test_artifact_audit_and_tampered_sources_rejected(self):
        config=self.root/'fit.json'
        json_write(config,dict(epochs=1,batch_size=2,hidden=4,lr=.001,seed=0,alpha0=.5,eta=.5))
        sources={name:dict(source=str(self.root/'cache'/'manifest.json'),sha256=digest(self.root/'cache'/'manifest.json'))
                 for name in ('evidence_train','evidence_test')}
        with contextlib.redirect_stdout(io.StringIO()):
            for mode in ('static','current','gru','conditional'):
                train(argparse.Namespace(config=config,mode=mode,cache=self.root/'cache',val_cache=None,
                    train_role='gate_train',val_role='gate_val',allow_seen_expert_validation=False,
                    static_init=None if mode=='static' else self.root/'static'/'best_val_nll.pt',
                    output=self.root/mode,device='cpu',expect_identity=None,selection_protocol='original_test_as_val'))
                for name in ('best_val_nll.pt','config.json','metrics.jsonl'):
                    path=self.root/mode/name
                    sources[f'{mode}/{name}']=dict(source=str(path),sha256=digest(path))
        index=self.root/'sources.json';json_write(index,sources)
        *_,report=verified_artifacts(index)
        self.assertEqual(report['models']['static']['epoch'],1)
        sources['current/best_val_nll.pt']['sha256']='tampered';json_write(index,sources)
        with self.assertRaises(ValueError):verified_artifacts(index)
        # Evidence payload modification is rejected even when manifest itself is unchanged.
        sources['current/best_val_nll.pt']['sha256']=digest(self.root/'current'/'best_val_nll.pt');json_write(index,sources)
        with (self.root/'cache'/'stats.npy').open('ab') as stream:stream.write(b'tampered')
        with self.assertRaises(ValueError):verified_artifacts(index)


if __name__=='__main__':unittest.main()
