"""CPU-only regression, routing math, data boundaries, causality and tiny fits.

Run: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -m unittest test_fusion_routing -v
Real-weight/CUDA regression is separate: export_fusion_evidence.py --integration-regression.
"""
import argparse
import contextlib
import copy
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from export_fusion_evidence import (FrozenEvidence, role_mapping, strict_fusion_state,
                                    make_export_dataset, load_trained_fusion, export_collate)
from fusion_router import FIXED, Router, mix_log_probs, window_nll
from routing_data import (EvidenceCache, EvidenceWriter, ObservationNormalizer, OBSERVATIONS,
                          RoutingDataset, collate_sequences, digest, fingerprint,
                          json_write, observations, row_key, validate_training_roles, validate_row, protocol_config)
from train_fusion_routing import RoutingMetrics, eval_command, evaluate, train, training_objective


def synthetic_cache(path, classes=3, selection_protocol='strict'):
    identity = dict(dataset_id='synthetic', class_to_idx={f'class{i}':i for i in range(classes)},
                    checkpoint_sha256='synthetic-not-a-real-model', data_manifest_sha256='synthetic',
                    code={'fixture':'synthetic'}, normalization={'fixture':'none'},
                    sampling={'fixture':'seed=12'}, export={'fixture':'CPU'}, time_unit='microseconds',
                    **protocol_config(selection_protocol),
                    expert_provenance={'fit_source_splits':['expert_train'], 'selection_source_splits':[],
                                       'verified_fit_unseen_source_splits':['train','holdout','test'],
                                       'verified_selection_unseen_source_splits':['train','holdout','test']})
    rows = []
    assignments = [('gate_train', 'train'), ('gate_val', 'holdout'), ('test', 'test')] if selection_protocol=='strict' else [('gate_train','train'),('gate_val','test')]
    for role, source in assignments:
        for video in range(2):
            # Nonuniform starts and a single-window video; metadata deliberately
            # reversed so grouping must sort by start_us, not CSV/window ID.
            for window, start in reversed(list(enumerate([0, 500000, 2000000][:1 if video==0 else 3]))):
                rows.append(dict(dataset_id='synthetic', video_id=f'{role}_{video}',
                                 window_id=str(window), source_split=source, role=role,
                                 start_us=start, end_us=start+1000000, label_id=video%classes,
                                 selection_protocol=selection_protocol,
                                 expert_fit_exposure='unknown' if source=='test' else 'unseen',
                                 expert_selection_exposure='seen' if source=='test' else 'unseen'))
    rng = np.random.default_rng(12)
    logits = rng.normal(size=(len(rows), 5, classes)).astype(np.float32)
    for i, row in enumerate(rows):
        logits[i, 2:, row['label_id']] += 2.
    stats = rng.random((len(rows), 4), dtype=np.float32)
    writer = EvidenceWriter(path, identity, [row_key(r) for r in rows])
    writer.append(rows, logits, stats)
    writer.finish()
    return EvidenceCache(path)


class RoutingMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.logits = torch.randn(2, 4, 5, 3, dtype=torch.float64)

    def test_arithmetic_mixture_endpoints_and_three_way(self):
        p = self.logits.softmax(-1)
        frame = p[..., 2:, :].mean(-2)
        for alpha, eta in [(0., 0.), (1., 0.), (0., 1.), (1., 1.), (.25, .7), (.5, 2/3)]:
            actual = mix_log_probs(self.logits, alpha, eta).exp()
            expected = (1-eta)*p[..., 0, :]+eta*(alpha*frame+(1-alpha)*p[..., 1, :])
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[..., 0]))
        torch.testing.assert_close(mix_log_probs(self.logits, .5, 2/3).exp(),
                                   (p[..., 0, :]+frame+p[..., 1, :])/3)
        # It is NOT softmax(mean logits).
        self.assertFalse(torch.allclose(frame, self.logits[..., 2:, :].mean(-2).softmax(-1)))

    def test_all_fixed_modes(self):
        mask = torch.ones(2, 4, dtype=torch.bool)
        obs = torch.zeros(2, 4, len(OBSERVATIONS), dtype=torch.float64)
        for mode, (a, e) in FIXED.items():
            router = Router(dict(mode='fixed', fixed=mode)).double()
            actual = router(self.logits, obs, mask, torch.zeros_like(mask, dtype=torch.float64))
            torch.testing.assert_close(actual['log_probs'], mix_log_probs(self.logits, a, e))

    def test_nll_gradient_and_extreme_logits(self):
        alpha = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
        eta = torch.tensor(.62, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a,e: mix_log_probs(self.logits, a, e), (alpha, eta)))
        labels = torch.tensor([[0,1,2,0], [1,2,0,1]])
        mask = torch.tensor([[True,False,False,False], [True,True,True,False]])
        lp = mix_log_probs(self.logits, alpha, eta)
        expected = -lp.gather(-1, labels[..., None]).squeeze(-1)[mask].mean()
        torch.testing.assert_close(window_nll(lp, labels, mask), expected)
        window_nll(lp, labels, mask).backward()
        self.assertTrue(torch.isfinite(alpha.grad) and torch.isfinite(eta.grad))
        extreme = self.logits*10000
        for a,e in [(0.,0.),(1.,1.),(.5,.5)]:
            self.assertTrue(torch.isfinite(mix_log_probs(extreme, a, e)).all())

    def test_unequal_video_batches_preserve_epoch_window_weights(self):
        labels=torch.tensor([[0,1,2,0],[1,2,0,1]])
        mask=torch.tensor([[True,False,False,False],[True,True,True,False]])
        lp=mix_log_probs(self.logits,.4,.6)
        parts=[training_objective(lp[b:b+1],labels[b:b+1],mask[b:b+1],4,2) for b in range(2)]
        torch.testing.assert_close(sum(parts)/2,window_nll(lp,labels,mask))

    def test_dynamic_initial_equals_static_and_frozen_eta(self):
        logits = self.logits.float()
        obs = torch.randn(2,4,len(OBSERVATIONS))
        mask = torch.ones(2,4,dtype=torch.bool)
        for mode in ('current','ema','gru','conditional'):
            model = Router(dict(mode=mode, alpha0=.3, eta=.7, hidden=8))
            out = model(logits, obs, mask, torch.ones(2,4))
            torch.testing.assert_close(out['log_probs'], mix_log_probs(logits, .3, .7))
            self.assertNotIn('fixed_eta', dict(model.named_parameters()))
        for a in (0.,1.):
            model = Router(dict(mode='current',alpha0=a,eta=0.))
            out = model(logits,obs,mask,torch.zeros(2,4))
            torch.testing.assert_close(out['log_probs'],logits[...,0,:].log_softmax(-1))

    def test_observations_ignore_labels_ids_and_have_fixed_dimension(self):
        for c in (2,3,11):
            logits = torch.randn(5,5,c)
            actual = observations(logits,torch.zeros(5,4),torch.zeros(5))
            self.assertEqual(actual.shape,(5,len(OBSERVATIONS)))
            # Uniform predictions have normalized entropy=1, margin=0.
            uniform = observations(torch.zeros(1,5,c),torch.zeros(1,4),torch.zeros(1))
            torch.testing.assert_close(uniform[0,:10:2],torch.ones(5))
            torch.testing.assert_close(uniform[0,1:10:2],torch.zeros(5))

    def test_raw_static_logit_extremes_ema_and_restore(self):
        for dtype in (torch.float32,torch.float64):
            logits=self.logits.to(dtype)
            obs=torch.randn(2,4,len(OBSERVATIONS),dtype=dtype)
            mask=torch.ones(2,4,dtype=torch.bool);dt=torch.ones(2,4,dtype=dtype)
            labels=torch.zeros(2,4,dtype=torch.long)
            for value in (0.7, math.log(1e-8/(1-1e-8)), 15., 100., -100.):
                static=Router(dict(mode='static')).to(dtype=dtype)
                with torch.no_grad():
                    static.alpha_logit.fill_(value);static.eta_logit.fill_(.8)
                expected=static(logits,obs,mask,dt)['log_probs']
                for mode in ('current','gru','conditional','ema'):
                    model=Router(dict(mode=mode,hidden=8)).initialize_from_static(static.state_dict())
                    self.assertEqual(model.base_logit.dtype,dtype)
                    torch.testing.assert_close(model.base_logit,static.alpha_logit,rtol=0,atol=0)
                    self.assertFalse(bool(model.exact_alpha_endpoint))
                    actual=model(logits,obs,mask,dt)['log_probs']
                    torch.testing.assert_close(actual,expected)
                    torch.testing.assert_close(window_nll(actual,labels,mask),window_nll(expected,labels,mask))
                    self.assertTrue({'base_logit','fixed_alpha','fixed_eta','exact_alpha_endpoint'}.isdisjoint(dict(model.named_parameters())))
                    # Roundtrip the real state, without reconstructing from probabilities.
                    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
                    restored=Router(dict(mode=mode,hidden=8)).to(dtype=dtype)
                    restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
                    torch.testing.assert_close(restored(logits,obs,mask,dt)['log_probs'],actual)
                    torch.testing.assert_close(restored.fixed_alpha,static.alpha_logit.sigmoid(),rtol=0,atol=0)
                    if mode=='ema':
                        torch.testing.assert_close(restored.initial_state(2,obs),static.alpha_logit.sigmoid().expand(2,1))
                    if abs(value)==100 and mode=='current':
                        with torch.no_grad():model.current[-1].bias.fill_(-value)
                        adjusted=model(logits,obs,mask,dt)
                        torch.testing.assert_close(adjusted['alpha'],torch.full((2,4),.5,dtype=dtype))
                        self.assertFalse(torch.allclose(adjusted['log_probs'],expected))

    def test_explicit_alpha_endpoints_with_nonzero_eta(self):
        for a in (0.,1.):
            for mode in ('current','gru','conditional','ema'):
                model=Router(dict(mode=mode,alpha0=a,eta=.7,hidden=8)).double()
                with torch.no_grad():model.current[-1].bias.fill_(5.)
                mask=torch.ones(2,4,dtype=torch.bool)
                out=model(self.logits,torch.randn(2,4,len(OBSERVATIONS),dtype=torch.float64),mask,torch.ones(2,4))
                torch.testing.assert_close(out['log_probs'],mix_log_probs(self.logits,model.fixed_alpha,model.fixed_eta))
                torch.testing.assert_close(out['alpha'],torch.full((2,4),a,dtype=torch.float64))


class SequenceTests(unittest.TestCase):
    def test_common_gap_input_has_nonzero_path_in_every_dynamic_mode(self):
        logits=torch.randn(1,3,5,3)
        for mode in ('current','gru','conditional'):
            router=Router(dict(mode=mode,hidden=8,alpha0=.4,eta=.7))
            with torch.no_grad():
                for p in router.parameters():p.fill_(.05)
            obs=torch.zeros(1,3,len(OBSERVATIONS),requires_grad=True)
            result=router(logits,obs,torch.ones(1,3,dtype=torch.bool),torch.zeros(1,3))
            result['alpha'].sum().backward()
            self.assertGreater(float(obs.grad[..., -1].abs().sum()),0.)
            changed=obs.detach().clone();changed[:,2,-1]=2.
            new=router(logits,changed,torch.ones(1,3,dtype=torch.bool),torch.tensor([[0.,0.,100.]]))
            torch.testing.assert_close(result['log_probs'][:,:2],new['log_probs'][:,:2])
            self.assertFalse(torch.allclose(result['log_probs'][:,2],new['log_probs'][:,2]))
            if mode=='conditional':
                self.assertEqual(router.write[0].in_features,len(OBSERVATIONS)+router.hidden)

    def test_future_isolation_video_reset_padding_and_explicit_state(self):
        torch.manual_seed(8)
        logits = torch.randn(2,5,5,3)
        obs = torch.randn(2,5,len(OBSERVATIONS))
        dt = torch.tensor([[0.,.5,1.5,.5,.5],[0.,.5,.5,0.,0.]])
        mask = torch.tensor([[True]*5,[True]*3+[False]*2])
        for mode in ('current','ema','gru','conditional'):
            model=Router(dict(mode=mode,hidden=8,alpha0=.4,eta=.7))
            # Nonzero readout is needed to test actual temporal dependency.
            with torch.no_grad():
                for name,p in model.named_parameters():
                    p.normal_(0,.15)
            original=model(logits,obs,mask,dt)
            modified=obs.clone();modified[:,3:]+=100
            changed=model(logits,modified,mask,dt)
            torch.testing.assert_close(original['log_probs'][:,:3],changed['log_probs'][:,:3])
            # Second video alone, with trailing padding removed, yields same
            # valid outputs and same state as its batched padded version.
            alone=model(logits[1:2,:3],obs[1:2,:3],mask[1:2,:3],dt[1:2,:3])
            torch.testing.assert_close(original['log_probs'][1,:3],alone['log_probs'][0])
            if alone['state'] is not None:
                torch.testing.assert_close(original['state'][1],alone['state'][0])
            first=model(logits[:1,:2],obs[:1,:2],mask[:1,:2],dt[:1,:2])
            rest=model(logits[:1,2:],obs[:1,2:],mask[:1,2:],dt[:1,2:],state=first['state'])
            torch.testing.assert_close(original['log_probs'][:1,2:],rest['log_probs'])
            # A new video starts from zero again, no global hidden dictionary.
            repeated=model(logits,obs,mask,dt)
            torch.testing.assert_close(original['log_probs'],repeated['log_probs'])

    def test_conditional_reads_old_state_before_current_write(self):
        model=Router(dict(mode='conditional',hidden=4,alpha0=.5,eta=.5))
        with torch.no_grad():
            for p in model.read.parameters():p.fill_(.1)
        logits=torch.randn(1,2,5,3);obs=torch.randn(1,2,len(OBSERVATIONS))
        mask=torch.ones(1,2,dtype=torch.bool);dt=torch.zeros(1,2)
        original=model(logits,obs,mask,dt)
        with torch.no_grad():
            for p in model.candidate.parameters():p.add_(1.)
        changed=model(logits,obs,mask,dt)
        torch.testing.assert_close(original['alpha'][:,0],changed['alpha'][:,0])
        self.assertFalse(torch.allclose(original['state'],changed['state']))


class CacheTests(unittest.TestCase):
    def test_original_test_as_val_explicit_protocol_and_training_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=synthetic_cache(root/'cache',selection_protocol='original_test_as_val')
            tr=RoutingDataset(cache,'gate_train');val=RoutingDataset(cache,'gate_val')
            validate_training_roles(tr,val,selection_protocol='original_test_as_val')
            with self.assertRaises(ValueError):validate_training_roles(tr,val)
            with self.assertRaises(ValueError):ObservationNormalizer().fit(val)
            cache.rows[tr.indices[0]]['source_split']='test'
            with self.assertRaises(ValueError):validate_training_roles(tr,val,selection_protocol='original_test_as_val')
            with self.assertRaises(ValueError):ObservationNormalizer().fit(tr)
            path=root/'roles.json'
            json_write(path,dict(schema=1,dataset_id='synthetic',selection_protocol='original_test_as_val',
                videos=[dict(source_split='test',video_id='x',role='gate_val')]))
            self.assertEqual(role_mapping(path,'synthetic','original_test_as_val')[('test','x')],'gate_val')
            with self.assertRaises(ValueError):role_mapping(path,'synthetic')
            obj=json.loads(path.read_text());obj['videos'][0]['role']='gate_train';json_write(path,obj)
            with self.assertRaises(ValueError):role_mapping(path,'synthetic','original_test_as_val')

    def test_window_sequence_gap_shuffle_and_normalization_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=synthetic_cache(Path(tmp)/'cache')
            window=RoutingDataset(cache,'gate_train',sequence=False)
            sequence=RoutingDataset(cache,'gate_train',sequence=True)
            expected={row_key(row):(item['observations'][t],item['dt'][t])
                for item in [sequence[i] for i in range(len(sequence))] for t,row in enumerate(item['rows'])}
            for size in (1,3):
                loader=torch.utils.data.DataLoader(window,batch_size=size,shuffle=True,collate_fn=collate_sequences)
                for batch in loader:
                    for b,rows in enumerate(batch['rows']):
                        obs,gap=expected[row_key(rows[0])]
                        torch.testing.assert_close(batch['observations'][b,0],obs)
                        torch.testing.assert_close(batch['dt'][b,0],gap)
                        torch.testing.assert_close(obs[-1],torch.log1p(gap))
            a=ObservationNormalizer().fit(window);b=ObservationNormalizer().fit(sequence)
            torch.testing.assert_close(a.mean,b.mean,rtol=0,atol=0)
            torch.testing.assert_close(a.scale,b.scale,rtol=0,atol=0)
            # All-source chronological gaps are computed before window batching.
            self.assertEqual(sorted(float(x[1]) for x in expected.values()),[0.,0.,.5,1.5])
            late=max(window.indices,key=lambda i:int(cache.rows[i]['start_us']))
            cache.rows[late]['start_us']+=3000000;cache.rows[late]['end_us']+=3000000
            changed=RoutingDataset(cache,'gate_train',sequence=True)
            for item in [changed[i] for i in range(len(changed))]:
                for t,row in enumerate(item['rows']):
                    if row_key(row)!=row_key(cache.rows[late]):
                        torch.testing.assert_close(item['observations'][t],expected[row_key(row)][0])
            old=json.loads((Path(tmp)/'cache'/'manifest.json').read_text());old['schema']=1
            json_write(Path(tmp)/'cache'/'manifest.json',old)
            with self.assertRaises(ValueError):EvidenceCache(Path(tmp)/'cache')
    def test_identity_completeness_integrity_and_role_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=synthetic_cache(root/'cache')
            seen_identity=copy.deepcopy(cache.identity)
            seen_identity['expert_provenance']['fit_source_splits']=['train']
            with self.assertRaisesRegex(ValueError,'cannot be relabeled'):
                validate_row(cache.rows[0],seen_identity)
            wrong=copy.deepcopy(cache.identity);wrong['class_to_idx']={'b':0,'a':1,'c':2}
            with self.assertRaises(ValueError):EvidenceCache(root/'cache',wrong)
            train_data=RoutingDataset(cache,'gate_train',sequence=True)
            val_data=RoutingDataset(cache,'gate_val',sequence=True)
            validate_training_roles(train_data,val_data)
            with self.assertRaises(ValueError):ObservationNormalizer().fit(val_data)
            old=cache.rows[val_data.indices[0]]['source_split']
            cache.rows[val_data.indices[0]]['source_split']='test'
            with self.assertRaises(ValueError):validate_training_roles(train_data,val_data,allow_seen_experts=True)
            cache.rows[val_data.indices[0]]['source_split']=old
            cache.rows[val_data.indices[0]]['expert_fit_exposure']='seen'
            with self.assertRaises(ValueError):validate_training_roles(train_data,val_data)
            validate_training_roles(train_data,val_data,allow_seen_experts=True)
            # On-disk content changes cannot be hidden behind a complete marker.
            p=root/'cache'/'stats.npy'
            with p.open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):EvidenceCache(root/'cache')

    def test_missing_duplicate_nonfinite_pending_and_cross_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=synthetic_cache(root/'good')
            rows=cache.rows[:2]
            writer=EvidenceWriter(root/'bad',cache.identity,[row_key(r) for r in rows])
            writer.append(rows[:1],np.array(cache.logits[:1]),np.array(cache.stats[:1]))
            with self.assertRaises(ValueError):writer.finish()
            with self.assertRaises(ValueError):EvidenceCache(writer.directory)
            with self.assertRaises(ValueError):writer.append(rows[:1],np.array(cache.logits[:1]),np.array(cache.stats[:1]))
            writer2=EvidenceWriter(root/'nan',cache.identity,[row_key(rows[0])])
            with self.assertRaises(ValueError):writer2.append(rows[:1],np.full((1,5,3),np.nan),np.zeros((1,4)))
            roles=root/'roles.json'
            json_write(roles,dict(schema=1,dataset_id='synthetic',videos=[dict(source_split='test',video_id='x',role='gate_val')]))
            with self.assertRaises(ValueError):role_mapping(roles,'synthetic')
            # Rehashing a deliberately altered row still fails semantic role checks.
            raw=[dict(r) for r in cache.rows]
            raw[2]['role']='gate_val'
            p=root/'good'/'rows.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in raw))
            m=json.loads((root/'good'/'manifest.json').read_text());m['files']['rows.jsonl']=digest(p)
            json_write(root/'good'/'manifest.json',m)
            with self.assertRaises(ValueError):EvidenceCache(root/'good')

    def test_sort_gaps_overlap_padding_and_label_free_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=synthetic_cache(Path(tmp)/'cache',classes=2)
            data=RoutingDataset(cache,'gate_train',sequence=True)
            items=[data[i] for i in range(len(data))]
            item=max(items,key=lambda i:len(i['rows']))
            torch.testing.assert_close(item['dt'],torch.tensor([0.,.5,1.5]))
            torch.testing.assert_close(item['overlap'],torch.tensor([0.,.5,0.]))
            before=item['observations'].clone()
            for row in cache.rows:row['label_id']=1-int(row['label_id'])
            after=max([data[i] for i in range(len(data))],key=lambda i:len(i['rows']))['observations']
            torch.testing.assert_close(before,after)
            batch=collate_sequences(items)
            self.assertEqual(int(batch['mask'].sum()),4)
            self.assertEqual(batch['mask'].shape,(2,3))


class EvaluationTests(unittest.TestCase):
    def test_metric_type_video_softvote_ece_brier_and_small_topk(self):
        probs=torch.tensor([[[.8,.2],[.4,.6]],[[.1,.9],[.5,.5]]],dtype=torch.float64)
        labels=torch.tensor([[0,0],[1,0]])
        mask=torch.tensor([[True,True],[True,False]])
        rows=[[dict(dataset_id='x',source_split='holdout',video_id='a',window_id=str(i),label_id=0) for i in range(2)],
              [dict(dataset_id='x',source_split='holdout',video_id='b',window_id='0',label_id=1)]]
        metric=RoutingMetrics(2)
        with self.assertRaises(ValueError):metric.update_log_probs(probs,probs,labels,mask,rows,mask.float(),mask.float())
        metric.update_log_probs(probs.log(),probs.log(),labels,mask,rows,mask.float()*.5,mask.float()*.5)
        result,_=metric.finish()
        self.assertAlmostEqual(result['video_nll'],(-np.log(.6)-np.log(.9))/2)
        self.assertAlmostEqual(result['video_brier'],(.32+.02)/2)
        self.assertAlmostEqual(result['video_ece'],(.4+.1)/2)
        self.assertEqual(result['video_top5'],1.)
        self.assertEqual(result['windows'],3)
        self.assertEqual(result['net_change'],0)


class RoleGenerationTests(unittest.TestCase):
    def test_complete_valid_video_mapping_without_new_split(self):
        from prepare_fusion_routing_roles import build_roles
        raw=[dict(video_id='a',split='train',label_id=0,group=1),
             dict(video_id='b',split='test',label_id=0,group=1),
             dict(video_id='excluded',split='train',label_id=1,group=2)]
        windows=[dict(video_id=v,window_id=str(t),split=s,label_id=0) for v,s,n in [('a','train',3),('b','test',1)] for t in range(n)]
        roles,report=build_roles(raw,raw,windows,dataset_id='synthetic')
        self.assertEqual(roles['videos'],[dict(source_split='train',video_id='a',role='gate_train'),
                                        dict(source_split='test',video_id='b',role='gate_val')])
        self.assertEqual(report['counts']['train']['excluded_no_valid_windows'],['excluded'])
        self.assertEqual(report['counts']['train']['valid_windows'],3)
        self.assertEqual(report['overlapping_class_groups'],1)
        self.assertFalse(report['independent_test'])
        with self.assertRaises(ValueError):build_roles(raw,raw,windows+[windows[0]])
        changed=copy.deepcopy(windows);changed[0]['split']='test'
        with self.assertRaises(ValueError):build_roles(raw,raw,changed)


class AnalysisTests(unittest.TestCase):
    def test_metric_directions_and_unique_rescue_numeric(self):
        from analyze_ucf101_complementarity import analyze,analysis_text,metric_change
        for name,base_correct,image_correct,base_nll,image_nll in [
            ('improved',False,True,2.,.1),('worsened',True,False,.1,2.),('unchanged',True,True,.1,.1)]:
            bank={}
            for head in ('fusion','point','cstr3','xt_p','ty_p'):
                correct=base_correct if head=='fusion' else image_correct
                bank[head]={'v':dict(label_id=0,prediction=0 if correct else 1,windows=1,
                                     top5=[0,1],nll=base_nll if head=='fusion' else image_nll)}
            result,cases=analyze(bank,'fusion');result['epoch']=None
            text=analysis_text(dict(results={'nll_selected':result,'latest':result}))
            self.assertIn('Top1 '+name,text);self.assertIn('NLL '+name,text)
            self.assertIn('Point-exclusive rescue after all image heads: 0/',text)
            self.assertIn('unknown',text)
            self.assertNotIn('very little',text)
        self.assertIn('unchanged',metric_change(.1,.1,higher_is_better=False))

    def test_zero_errors_csv_and_missing_checkpoint_metadata(self):
        from analyze_ucf101_complementarity import main,ERROR_FIELDS
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);fusion=root/'fusion';images=root/'images';fusion.mkdir();(images/'reports').mkdir(parents=True)
            row=dict(video_id='v',label_id=0,prediction=0,windows=1,top5='[0, 1]',nll=.1)
            paths=[fusion/f'{head}_predictions_{suffix}.csv' for head in ('fusion','point') for suffix in ('best_loss','last')]
            paths += [images/'reports'/f'{head}_test_video_predictions_best_loss.csv' for head in ('cstr3','xt_p','ty_p')]
            for path in paths:
                with path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            output=root/'report'
            argv=['analyze_ucf101_complementarity.py','--fusion',str(fusion),'--images',str(images),'--output',str(output)]
            with mock.patch.object(sys,'argv',argv):main()
            with (output/'latest_errors.csv').open() as stream:
                reader=csv.DictReader(stream);self.assertEqual(reader.fieldnames,list(ERROR_FIELDS));self.assertEqual(list(reader),[])
            text=(output/'analysis.md').read_text();self.assertIn('unknown',text);self.assertNotIn('epoch40',text)
            result=json.loads((output/'summary.json').read_text());self.assertIsNone(result['checkpoint_metadata']['nll_selected_epoch'])
            # A supplied epoch is read from metadata, not copied from old runs.
            scores=dict(videos=1,windows=1,video_top1=1.,video_nll=.1)
            (fusion/'metrics.jsonl').write_text(json.dumps(dict(epoch=13,test=dict(fusion=scores,point=scores)))+'\n')
            with mock.patch.object(sys,'argv',argv):main()
            result=json.loads((output/'summary.json').read_text())
            self.assertEqual(result['checkpoint_metadata']['nll_selected_epoch'],13)


class FrozenRegressionTests(unittest.TestCase):
    def test_full_checkpoint_code_hash_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fake.pt'
            torch.save(dict(epoch=40,model_state={'fake':torch.zeros(1)},config=dict(
                architecture='pretrained_filename_V2_prototype_A',
                code={'ucf101_pretrained_fusion.py':'not-the-real-hash'})),path)
            with self.assertRaisesRegex(ValueError,'code identity changed'):
                load_trained_fusion(path)

    def test_export_train_source_no_augmentation_and_fixed_sampling(self):
        # Small actual event-store fixture: test data wrapper without touching
        # real datasets, files or any GPU. Source identity remains 'train'.
        from preprocess_ucf101_event_store import prepare_video,video_directory
        from ucf101_window_data import write_csv,sha
        from ucf101_fusion_window_data import FusionWindowDataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);experiment=root/'experiment';experiment.mkdir();store=root/'store';store.mkdir()
            rng=np.random.default_rng(2);n=10002
            source=root/'source.npz'
            arrays=dict(t=np.arange(n,dtype=np.int64)*80,
                        x=rng.integers(0,32,n,dtype=np.int16),y=rng.integers(0,24,n,dtype=np.int16),
                        p=rng.integers(0,2,n,dtype=np.uint8))
            np.savez(source,**arrays)
            row=dict(video_id='real_fixture',window_id='0',split='train',start_us=0,end_us=1000000,
                     event_count=n,label_id=1,source_path=str(source),source_sha256=sha(source),
                     sensor_width=32,sensor_height=24,cache_key='fixture')
            other=dict(row,video_id='second_train',cache_key='second_train')
            test_row=dict(row,video_id='test_fixture',split='test',cache_key='test_fixture')
            write_csv(experiment/'window_manifest.csv',[row,other,test_row],list(row))
            index={}
            for entry in (row,other,test_row):
                prepare_video([entry],store)
                folder=store/video_directory(entry['video_id'])
                index[entry['video_id']]=dict(directory=video_directory(entry['video_id']),metadata_sha256=sha(folder/'metadata.json'))
            json_write(store/'videos.json',index)
            json_write(store/'summary.json',dict(status='complete',rendering_validation='passed',
                window_manifest_sha256=sha(experiment/'window_manifest.csv'),videos_index_sha256=sha(store/'videos.json')))
            roles={('train','real_fixture'):'gate_val'}
            exported=make_export_dataset(experiment,'train',store,roles,'synthetic',seed=0,sampling_epoch=0)
            self.assertEqual(exported.split,'train');self.assertEqual(exported.augmentation,'none')
            baseline=FusionWindowDataset(experiment,'train',store,seed=0)
            baseline.augmentation='none';baseline.set_epoch(0)
            actual=exported[0];expected=baseline[0]
            for key in ('events','cstr3','xt_p','ty_p'):
                torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
                torch.testing.assert_close(actual[key],exported[0][key],rtol=0,atol=0)
            self.assertEqual(actual['metadata']['source_split'],'train')
            self.assertEqual(actual['metadata']['expert_fit_exposure'],'seen')
            self.assertEqual(actual['statistics'].shape,(4,))
            selected=make_export_dataset(experiment,'train',store,
                {('train','real_fixture'):'gate_train',('train','second_train'):'gate_train'},
                'synthetic',selection_protocol='original_test_as_val')
            selected.rows=selected.rows[1:]  # Same mechanism as --max-videos.
            torch.testing.assert_close(selected[0]['events'],baseline[1]['events'],rtol=0,atol=0)
            self.assertEqual(selected[0]['metadata']['video_id'],'second_train')
            original_test=make_export_dataset(experiment,'test',store,{('test','test_fixture'):'gate_val'},
                'synthetic',selection_protocol='original_test_as_val')
            test_item=original_test[0]
            self.assertEqual(test_item['metadata']['source_split'],'test')
            self.assertEqual(test_item['metadata']['role'],'gate_val')
            self.assertEqual(test_item['metadata']['expert_fit_exposure'],'unknown')
            self.assertEqual(test_item['metadata']['expert_selection_exposure'],'seen')
            # Actual top-level export class must work under spawn, not just fork.
            batches=[]
            for workers in (0,1):
                loader=torch.utils.data.DataLoader(exported,batch_size=1,num_workers=workers,
                    collate_fn=export_collate,**({'multiprocessing_context':'spawn'} if workers else {}))
                batches.append(list(loader)[0])
            self.assertEqual(batches[0][1],batches[1][1])
            for key in ('events','cstr3','xt_p','ty_p','statistics'):
                torch.testing.assert_close(batches[0][0][key],batches[1][0][key],rtol=0,atol=0)
            # Test-as-Val also travels through the actual spawn export path.
            loader=torch.utils.data.DataLoader(original_test,batch_size=1,num_workers=1,
                multiprocessing_context='spawn',collate_fn=export_collate)
            batch,metadata=list(loader)[0]
            self.assertEqual(metadata[0],test_item['metadata'])
            torch.testing.assert_close(batch['statistics'][0],test_item['statistics'],rtol=0,atol=0)
            # Repeated sampling does not change or populate the old cache.
            self.assertFalse((experiment/'cache').exists())

    def test_original_interfaces_single_point_pass_frozen_bn_and_strict_state(self):
        # Actual unchanged project Fusion A and real ResNet branches, with only
        # the CUDA Point encoder replaced by a small synthetic CPU test double.
        from ucf101_pretrained_fusion import PretrainedWindowFusion
        class Encoder(nn.Module):
            stage_channels=(4,6,8)
            def __init__(self):
                super().__init__()
                self.layers=nn.ModuleList(nn.Linear(3,c) for c in self.stage_channels)
                self.classifier=nn.Linear(8,101)
                self.calls=0
            def forward_features(self,events):
                self.calls+=1
                coords=events.transpose(1,2)
                permutation=torch.randperm(coords.shape[1])
                coords=coords[:,permutation]
                return [layer(coords) for layer in self.layers],[coords]*3
            def classify_features(self,features):
                return self.classifier(features[-1].mean(1))
        torch.manual_seed(10)
        fusion=PretrainedWindowFusion(Encoder()).eval()
        evidence=FrozenEvidence(fusion)
        evidence.train(True)
        self.assertFalse(any(m.training for m in evidence.modules()))
        self.assertFalse(any(p.requires_grad for p in evidence.parameters()))
        batch={'events':torch.randn(1,3,8),**{v:torch.randn(1,3,32,32) for v in ('cstr3','xt_p','ty_p')}}
        buffers={k:v.clone() for k,v in fusion.named_buffers()}
        with torch.inference_mode():
            torch.manual_seed(17);original=fusion.forward_batch_outputs(batch)
            calls=fusion.point_encoder.calls
            torch.manual_seed(17);actual=evidence(batch)
            self.assertEqual(fusion.point_encoder.calls,calls+1)
            torch.testing.assert_close(actual['fusion_logits'],original['logits'],rtol=0,atol=0)
            torch.testing.assert_close(actual['point_logits'],original['point_logits'],rtol=0,atol=0)
            for view in ('cstr3','xt_p','ty_p'):
                torch.testing.assert_close(actual[view+'_logits'],fusion.image_encoders[view](batch[view]),rtol=0,atol=0)
        for name,value in fusion.named_buffers():torch.testing.assert_close(value,buffers[name],rtol=0,atol=0)
        state={'module.fusion.'+k:v for k,v in fusion.state_dict().items()}
        strict_fusion_state(fusion,state)
        state['module.fusion.nonexistent.weight']=torch.zeros(1)
        with self.assertRaises(RuntimeError):strict_fusion_state(fusion,state)
        state.pop('module.fusion.nonexistent.weight')
        state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError):strict_fusion_state(fusion,state)


class SmokeTests(unittest.TestCase):
    def test_synthetic_static_dynamic_memory_fit_and_explicit_test(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);cache=synthetic_cache(root/'cache',selection_protocol='original_test_as_val')
            config=root/'config.json'
            json_write(config,dict(epochs=2,batch_size=2,batch_unit='video',hidden=8,lr=.01,seed=4,alpha0=.5,eta=.5,ece_bins=15))
            def args(mode):
                return argparse.Namespace(config=config,mode=mode,cache=root/'cache',val_cache=None,
                    train_role='gate_train',val_role='gate_val',allow_seen_expert_validation=False,
                    static_init=None if mode=='static' else root/'static'/'best_val_nll.pt',
                    output=root/mode,device='cpu',expect_identity=None,selection_protocol='original_test_as_val')
            for mode in ('static','current','gru','conditional'):
                if mode=='static':
                    train(args(mode))
                else:
                    with mock.patch.object(ObservationNormalizer,'fit',side_effect=AssertionError('Dynamic fit must not refit normalization')):
                        train(args(mode))
                saved=torch.load(root/mode/'last.pt',map_location='cpu',weights_only=False)
                self.assertEqual(saved['optimizer_steps'],2)
                self.assertEqual(saved['metrics']['train']['windows'],4)
                self.assertEqual(saved['metrics']['validation']['videos'],2)
                self.assertFalse(saved['config']['independent_test'])
                self.assertEqual(saved['config']['selection_protocol'],'original_test_as_val')
                static=torch.load(root/'static'/'best_val_nll.pt',map_location='cpu',weights_only=False)
                for key in ('mean','scale'):
                    torch.testing.assert_close(saved['normalizer_state'][key],static['normalizer_state'][key],rtol=0,atol=0)
            ev=argparse.Namespace(cache=root/'cache',checkpoint=root/'current'/'best_val_nll.pt',
                 fixed=None,ema_decay=.9,role='gate_val',output=root/'test_eval',device='cpu',expect_identity=None,
                 selection_protocol='original_test_as_val')
            result=eval_command(ev)
            self.assertEqual(result['videos'],2)
            with self.assertRaises(FileExistsError):eval_command(ev)
            with self.assertRaises(FileExistsError):train(args('static'))
            # Restricted static fit and all fixed baselines also run.
            obj=json.loads(config.read_text());obj['fusion_frame_only']=True;json_write(config,obj)
            restricted=args('static');restricted.output=root/'restricted';train(restricted)
            saved=torch.load(root/'restricted'/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['weights']['alpha'],1.)
            for name in FIXED:
                ev.checkpoint=None;ev.fixed=name;ev.ema_decay=None;ev.output=root/('fixed_'+name)
                self.assertEqual(eval_command(ev)['windows'],4)


if __name__ == '__main__':
    unittest.main()
