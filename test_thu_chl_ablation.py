"""Selection and complete split coverage regressions for the THU experiment."""
import unittest
from thu_chl_ablation_runtime import configuration, select_best
from thu_chl_sequence_dataset import THUVideoSequenceDataset, point_input
import numpy as np

class THUProtocolTests(unittest.TestCase):
    def test_selection(self):
        best=dict(top1=.6,nll=2.,epoch=1,nll_best=1.8,nll_epoch=2)
        b,p,n=select_best(best,dict(video_top1=.61,video_nll=2.5),3)
        self.assertTrue(p);self.assertFalse(n);self.assertEqual(b['epoch'],3)
        b,p,n=select_best(b,dict(video_top1=.60,video_nll=1.5),4)
        self.assertFalse(p);self.assertTrue(n);self.assertEqual(b['epoch'],3)
        b,p,n=select_best(b,dict(video_top1=.61,video_nll=2.),5)
        self.assertTrue(p);self.assertFalse(n)
        _,p,n=select_best(b,dict(video_top1=.61,video_nll=2.),6)
        self.assertFalse(p);self.assertFalse(n)
    def test_coverage_and_sparse_fallback(self):
        c=configuration('configs/thu_chl_ablations/g.json')
        a=THUVideoSequenceDataset(c,'train');b=THUVideoSequenceDataset(c,'test')
        self.assertEqual((a.windows,b.windows,len(a.groups),len(b.groups)),(7129,2392,1786,544))
        self.assertFalse(set(a.groups)&set(b.groups))
        self.assertEqual(sum(not r['valid'] for r in b.rows),58)
        events=(np.array([1,2]),np.array([0,345]),np.array([0,259]),np.array([0,1]))
        x,f=point_input(events,np.random.default_rng(0))
        self.assertEqual(f,1);self.assertEqual(x.shape,(3,2048));self.assertTrue(np.isfinite(x).all())
        self.assertTrue((np.diff(x[0])>=0).all())
    def test_pair_protocol(self):
        a=configuration('configs/thu_chl_ablations/g.json');b=configuration('configs/thu_chl_ablations/g_fixed_gate.json')
        self.assertEqual({k for k in a if a[k]!=b[k]}, {'ablation_id','gate_policy'})

if __name__=='__main__':unittest.main()
