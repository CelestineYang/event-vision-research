import tempfile
import unittest
from pathlib import Path
import numpy as np
from prepare_chl import official_windows, load_events, splits
from render_frames import render


def upstream_indices(events):
    # Literal control flow of public uti.get_window_index, independent reference.
    start=int(events[0]); starts=[]; ends=[]; idx=0
    while idx<len(events):
        if (events[idx]>=start) and (start+1000000<events[-1]):
            starts.append(idx);start+=500000
        else: idx+=1
    idx=len(events)-1;end=start-500000+1000000
    while idx>=0:
        if events[idx]<=end:
            ends.append(idx);end-=500000
        else: idx-=1
    return list(zip(starts,list(reversed(ends[:len(starts)]))))


class Tests(unittest.TestCase):
    def test_official_boundaries(self):
        rng=np.random.default_rng(7)
        examples=[np.array([0,1000000]),np.array([0,500000,1000000,1000000,1500001]),
                  np.array([0,3000000,5000000])]
        examples += [np.sort(rng.integers(0,7000000,size=300)) for _ in range(20)]
        for t in examples:
            self.assertEqual([(l,r) for _,_,l,r in official_windows(t)],upstream_indices(t))

    def test_labels_and_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('A19.npy','A39.npy'): np.save(root/name,np.ones((1,4)))
            (root/'train.txt').write_text('../DVS-action-data-npy/A19.npy 0\n')
            (root/'test.txt').write_text('../DVS-action-data-npy/A39.npy 1\n')
            result=splits(root,root/'train.txt',root/'test.txt')
            self.assertEqual(result[0]['label'],0)
            (root/'test.txt').write_text('A19.npy 1\n')
            with self.assertRaises(ValueError): splits(root,root/'train.txt',root/'test.txt')

    def test_raw_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.npy'
            a=np.array([[0,0,0,0],[345,259,100,1]],dtype='f4')
            np.save(p,a);np.testing.assert_array_equal(load_events(p),a)
            a[1,0]=346;np.save(p,a)
            with self.assertRaises(ValueError):load_events(p)
            a[1,0]=345;a[1,3]=-1;np.save(p,a)
            with self.assertRaises(ValueError):load_events(p)

    def test_frames(self):
        a=np.array([[0,0,0,0],[345,259,999999,1]],dtype='f4')
        views=render(a,0,1000000)
        self.assertEqual(set(views),{'cstr3','xt_p','ty_p'})
        for image in views.values():
            self.assertEqual(image.shape,(3,224,224))
            self.assertTrue(np.isfinite(image).all())


if __name__=='__main__':unittest.main()
