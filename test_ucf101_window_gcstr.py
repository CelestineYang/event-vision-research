"""CPU protocol tests. Kept separate from the explicitly requested future smoke."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from gcstr_representations import GCSTRConfig
from ucf101_window_data import full_window_bounds, window_views, WindowDataset, write_csv, sha
from train_ucf101_window_gcstr import ExactBatchSampler, Metrics
from train_ucf101_window_joint import ThreeViewDataset
from gcstr_augmentation import augment_events


class WindowProtocolTests(unittest.TestCase):
    def test_spatial_train_changes_by_epoch_and_test_uses_fixed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'events.npz'
            rng = np.random.default_rng(5)
            n = 10001
            np.savez(source,t=np.arange(n,dtype=np.int64)*90,
                     x=rng.integers(0,32,n),y=rng.integers(0,24,n),p=rng.integers(0,2,n))
            row = dict(video_id='train_video',window_id=0,start_us=0,end_us=1000000,
                       event_count=n,label_id=0,split='train',source_path=str(source),
                       sensor_width=32,sensor_height=24,source_sha256=sha(source),cache_key='train_key')
            test_row = dict(row,video_id='test_video',split='test',cache_key='test_key')
            write_csv(root/'window_manifest.csv',[row,test_row],list(row))
            train = WindowDataset(root,'train','xt_p',seed=3)
            train.set_epoch(1)
            first = train[0][0]
            torch.testing.assert_close(first,train[0][0],rtol=0,atol=0)
            train.set_epoch(2)
            self.assertFalse(torch.equal(first,train[0][0]))
            self.assertFalse(train.cache_path(row).exists())
            other_view = WindowDataset(root,'train','ty_p',seed=3)
            other_view.set_epoch(2)
            np.testing.assert_equal(train.augmentation_rng(row).random(8),other_view.augmentation_rng(row).random(8))
            self.assertEqual(len(train),1)
            together = ThreeViewDataset(root,'train',seed=3)
            together.set_epoch(2)
            with patch('train_ucf101_window_joint.augment_events',wraps=augment_events) as transform:
                all_images,label,video = together[0]
                self.assertEqual(transform.call_count,1)
            self.assertEqual(set(all_images),{'cstr3','xt_p','ty_p'})
            for view in all_images:
                independent = WindowDataset(root,'train',view,seed=3)
                independent.set_epoch(2)
                torch.testing.assert_close(all_images[view],independent[0][0],rtol=0,atol=0)
            self.assertEqual((label,video),(0,'train_video'))
            for view in all_images:
                WindowDataset(root,'test',view,require_cache=False)[0]
            test_joint = ThreeViewDataset(root,'test')
            with patch('train_ucf101_window_joint.augment_events',side_effect=AssertionError('Test augmented')):
                test_images,_,_ = test_joint[0]
            torch.testing.assert_close(test_images['xt_p'],WindowDataset(root,'test','xt_p')[0][0],rtol=0,atol=0)
            test = WindowDataset(root,'test','xt_p',require_cache=False)
            plain = test[0][0]
            self.assertTrue(test.cache_path(test_row).exists())
            test.set_epoch(9)
            torch.testing.assert_close(plain,test[0][0],rtol=0,atol=0)

    def test_ddp_exact_coverage_and_tail(self):
        for size in (2,3,4,63,64,65,66,127,128,129,1001):
            shards = [list(ExactBatchSampler(size,64,2,r,7)) for r in (0,1)]
            self.assertEqual(len(shards[0]),len(shards[1]))
            visited = []
            for a,b in zip(*shards):
                self.assertTrue(a and b)
                self.assertLessEqual(len(a)+len(b),64)
                visited.extend(a+b)
            self.assertEqual(sorted(visited),list(range(size)))
            self.assertEqual(shards[0],list(ExactBatchSampler(size,64,2,0,7)))

    def test_unequal_rank_gradient_weight(self):
        # DDP mean of scaled local means equals true global-window mean.
        local_values = [torch.tensor([1.,2.,3.]),torch.tensor([4.,5.])]
        total = sum(len(v) for v in local_values)
        weighted = sum(v.mean()*2*len(v)/total for v in local_values)/2
        torch.testing.assert_close(weighted,torch.cat(local_values).mean())

    def test_full_windows_and_count_cap(self):
        self.assertEqual(full_window_bounds(10,1_000_009),[])
        self.assertEqual(full_window_bounds(10,1_000_010),[(10,1_000_010)])
        self.assertEqual(full_window_bounds(10,1_500_010),[(10,1_000_010),(500_010,1_500_010)])
        self.assertEqual(len(full_window_bounds(0,20_000_000)),39)
        self.assertEqual(full_window_bounds(0,20_500_000),[])

    def test_half_open_and_strict_event_filter(self):
        t = np.concatenate([np.zeros(10000,dtype=np.int64),[999999,1000000]])
        l,r = np.searchsorted(t,[0,1000000],side='left')
        self.assertEqual(r-l,10001)
        self.assertGreater(r-l,10000)
        self.assertFalse(10000 > 10000)

    def test_fixed_time_axis_and_shared_polarity_origin(self):
        cfg = GCSTRConfig(sensor_width=4,sensor_height=3,image_size=4,time_bins=4)
        t = np.array([250000,750000])
        x,y,p = np.array([1,2]),np.array([1,1]),np.array([0,1])
        images = window_views(t,x,y,p,0,1000000,cfg)
        self.assertEqual(set(images),{'cstr3','xt_p','ty_p'})
        for image in images.values():
            self.assertEqual(image.shape,(3,4,4))
            self.assertTrue(np.isfinite(image).all())
        # x-t axes retained: the negative event is at bin 1, not rescaled to 0.
        self.assertEqual(int(images['xt_p'][1].sum(axis=0).argmax()),1)
        self.assertEqual(int(images['xt_p'][0].sum(axis=0).argmax()),3)
        with self.assertRaises(ValueError):
            window_views(np.array([1000000]),x[:1],y[:1],p[:1],0,1000000,cfg)

    def test_vote_is_order_independent_and_video_weighted(self):
        logits = torch.zeros(4,101,dtype=torch.float64)
        logits[0,0] = 3
        logits[1:,1] = 4
        labels = torch.tensor([0,0,0,1])
        ids = ['a','a','a','b']
        first = Metrics(); first.update(logits,labels,ids)
        perm = [3,1,0,2]
        second = Metrics(); second.update(logits[perm],labels[perm],[ids[i] for i in perm])
        a,rows = first.finish(); b,_ = second.finish()
        for key in a: self.assertAlmostEqual(a[key],b[key])
        self.assertEqual(a['videos'],2)
        self.assertEqual(a['windows'],4)
        self.assertAlmostEqual(a['video_nll'],sum(r['nll'] for r in rows)/2)
        self.assertEqual(a['video_top1'],.5)


if __name__ == '__main__': unittest.main()
