"""CPU-only regression, routing math, data boundaries, causality and tiny fits.

Run: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -m unittest test_fusion_routing -v
Real-weight/CUDA regression is separate: export_fusion_evidence.py --integration-regression.
"""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from export_fusion_evidence import (FrozenEvidence, role_mapping, strict_fusion_state,
                                    make_export_dataset, load_trained_fusion)
from fusion_router import FIXED, Router, mix_log_probs, window_nll
from routing_data import (EvidenceCache, EvidenceWriter, ObservationNormalizer, OBSERVATIONS,
                          RoutingDataset, collate_sequences, digest, fingerprint,
                          json_write, observations, row_key, validate_training_roles, validate_row)
from train_fusion_routing import RoutingMetrics, eval_command, evaluate, train, training_objective


def synthetic_cache(path, classes=3):
    identity = dict(dataset_id='synthetic', class_to_idx={f'class{i}':i for i in range(classes)},
                    checkpoint_sha256='synthetic-not-a-real-model', data_manifest_sha256='synthetic',
                    code={'fixture':'synthetic'}, normalization={'fixture':'none'},
                    sampling={'fixture':'seed=12'}, export={'fixture':'CPU'}, time_unit='microseconds',
                    expert_provenance={'fit_source_splits':['expert_train'], 'selection_source_splits':[],
                                       'verified_fit_unseen_source_splits':['train','holdout','test'],
                                       'verified_selection_unseen_source_splits':['train','holdout','test']})
    rows = []
    for role, source in [('gate_train', 'train'), ('gate_val', 'holdout'), ('test', 'test')]:
        for video in range(2):
            # Nonuniform starts and a single-window video; metadata deliberately
            # reversed so grouping must sort by start_us, not CSV/window ID.
            for window, start in reversed(list(enumerate([0, 500000, 2000000][:1 if video==0 else 3]))):
                rows.append(dict(dataset_id='synthetic', video_id=f'{role}_{video}',
                                 window_id=str(window), source_split=source, role=role,
                                 start_us=start, end_us=start+1000000, label_id=video%classes,
                                 expert_fit_exposure='unseen', expert_selection_exposure='unseen'))
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
            actual = observations(logits,torch.zeros(5,4))
            self.assertEqual(actual.shape,(5,len(OBSERVATIONS)))
            # Uniform predictions have normalized entropy=1, margin=0.
            uniform = observations(torch.zeros(1,5,c),torch.zeros(1,4))
            torch.testing.assert_close(uniform[0,:10:2],torch.ones(5))
            torch.testing.assert_close(uniform[0,1:10:2],torch.zeros(5))


class SequenceTests(unittest.TestCase):
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
            write_csv(experiment/'window_manifest.csv',[row],list(row))
            metadata=prepare_video([row],store)
            folder=store/video_directory(row['video_id'])
            index={row['video_id']:dict(directory=video_directory(row['video_id']),metadata_sha256=sha(folder/'metadata.json'))}
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
            root=Path(tmp);cache=synthetic_cache(root/'cache')
            config=root/'config.json'
            json_write(config,dict(epochs=2,batch_size=2,batch_unit='video',hidden=8,lr=.01,seed=4,alpha0=.5,eta=.5,ece_bins=15))
            def args(mode):
                return argparse.Namespace(config=config,mode=mode,cache=root/'cache',val_cache=None,
                    train_role='gate_train',val_role='gate_val',allow_seen_expert_validation=False,
                    static_init=None if mode=='static' else root/'static'/'best_val_nll.pt',
                    output=root/mode,device='cpu',expect_identity=None)
            for mode in ('static','current','gru','conditional'):
                train(args(mode))
                saved=torch.load(root/mode/'last.pt',map_location='cpu',weights_only=False)
                self.assertEqual(saved['optimizer_steps'],2)
                self.assertEqual(saved['metrics']['train']['windows'],4)
                self.assertEqual(saved['metrics']['validation']['videos'],2)
            ev=argparse.Namespace(cache=root/'cache',checkpoint=root/'current'/'best_val_nll.pt',
                 fixed=None,ema_decay=.9,role='test',output=root/'test_eval',device='cpu',expect_identity=None)
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
